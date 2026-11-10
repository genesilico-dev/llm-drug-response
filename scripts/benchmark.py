"""Nested, patient-grouped CTRP-only RNA benchmark with resumable per-drug tasks."""

import argparse
import json
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from prepare import DATA, ROOT, SEED, digest
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

RIDGE = [10.0, 1.0, 0.1, 0.01, 0.001]
ENET = [(a, ratio) for a in (0.3, 0.1, 0.03) for ratio in (0.1, 0.5)]
CACHE = None


def features(train, test, top=None):
    valid = np.isfinite(train).all(axis=0) & (np.nanvar(train, axis=0) > 1e-12)
    indices = np.flatnonzero(valid)
    if top is not None:
        order = np.argsort(-np.var(train[:, indices], axis=0), kind="stable")
        indices = indices[order[:top]]
    if not len(indices):
        raise ValueError("No usable training genes")
    x = np.asarray(train[:, indices], dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    z = np.asarray(test[:, indices], dtype=float)
    z = np.where(np.isfinite(z), z, mean)
    return (x - mean) / scale, (z - mean) / scale


def rho(y, prediction):
    if len(y) < 3 or np.ptp(y) == 0:
        return float("nan")
    if np.ptp(prediction) == 0:
        return 0.0
    return float(spearmanr(y, prediction).statistic)


def ridge_path(x, y, z):
    kernel = x @ x.T / x.shape[1]
    values, vectors = np.linalg.eigh(kernel)
    projection = vectors.T @ (y - y.mean())
    cross = z @ x.T / x.shape[1] @ vectors
    return [y.mean() + cross @ (projection / (np.maximum(values, 0) + a)) for a in RIDGE]


def elastic_path(x, y, z):
    mean, scale = y.mean(), y.std()
    if scale <= 1e-12:
        return [np.full(len(z), mean) for _ in ENET], 0
    predictions, count = [], 0
    for alpha, ratio in ENET:
        model = ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=5000, tol=0.001)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(np.asfortranarray(x), (y - mean) / scale)
        count += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        predictions.append(mean + scale * model.predict(z))
    return predictions, count


def fit_models(x, y, groups, test):
    methods = ("ridge_all", "ridge_1000", "elastic_1000")
    scores = {name: [] for name in methods}
    warning_count = 0
    for a, b in GroupKFold(3).split(x, groups=groups):
        for top, names in (
            (None, ["ridge_all"]),
            (1000, ["ridge_1000", "elastic_1000"]),
        ):
            train, val = features(x[a], x[b], top)
            for name in names:
                if name.startswith("ridge"):
                    predictions = ridge_path(train, y[a], val)
                else:
                    predictions, count = elastic_path(train, y[a], val)
                    warning_count += count
                scores[name].append([rho(y[b], p) for p in predictions])
    result, choices = {}, {}
    for top, names in ((None, ["ridge_all"]), (1000, ["ridge_1000", "elastic_1000"])):
        train, val = features(x, test, top)
        for name in names:
            averaged = np.nan_to_num(np.array(scores[name]), nan=-1.0).mean(axis=0)
            best = int(np.argmax(averaged))
            if name.startswith("ridge"):
                predictions = ridge_path(train, y, val)
                parameter = RIDGE[best]
            else:
                predictions, count = elastic_path(train, y, val)
                warning_count += count
                parameter = ENET[best]
            result[name] = predictions[best]
            choices[name] = {"parameter": parameter, "cv_rho": float(averaged[best])}
    return result, choices, warning_count


def support32(rows, groups, seed):
    rng = np.random.default_rng(seed)
    selected = rng.permutation(np.unique(groups[rows]))[:32]
    return np.array([rng.choice(rows[groups[rows] == group]) for group in selected])


def load_data():
    global CACHE
    if CACHE is None:
        CACHE = (
            np.load(DATA / "expression.npy", mmap_mode="r"),
            pd.read_csv(DATA / "cells.csv", index_col="cell_id"),
            pd.read_parquet(DATA / "responses.parquet"),
        )
    return CACHE


def task(job):
    drug, out = job
    target = Path(out) / f"{drug}.json"
    if target.exists():
        return drug, "exists"
    start = time.perf_counter()
    x, cells, response = load_data()
    response = response.loc[response.drug_id.eq(drug)].set_index("cell_id")
    rows = cells.index.get_indexer(response.index)
    y = response.sensitivity.to_numpy()
    meta = cells.loc[response.index]
    groups, folds, lineage = (
        meta.group.to_numpy(),
        meta.fold.to_numpy(),
        meta.lineage.to_numpy(),
    )
    metrics, predictions, decisions, skipped = [], [], [], []
    warnings_total = 0
    with threadpool_limits(limits=1):
        for fold in range(3):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            if len(np.unique(groups[train])) < 60 or len(test) < 30:
                skipped.append({"fold": fold, "n_train": len(train), "n_test": len(test)})
                continue
            for budget in ("32", "full"):
                support = (
                    support32(train, groups, SEED + int(drug) + fold) if budget == "32" else train
                )
                if set(groups[support]) & set(groups[test]):
                    raise ValueError("Patient leakage")
                fitted, choices, count = fit_models(
                    x[rows[support]], y[support], groups[support], x[rows[test]]
                )
                warnings_total += count
                fitted["mean"] = np.full(len(test), y[support].mean())
                fitted["median"] = np.full(len(test), np.median(y[support]))
                means = pd.Series(y[support]).groupby(lineage[support]).mean()
                fitted["lineage_mean"] = np.array(
                    [means.get(v, y[support].mean()) for v in lineage[test]]
                )
                decisions.append(
                    {
                        "fold": fold,
                        "budget": budget,
                        "choices": choices,
                        "support": response.index[support].tolist(),
                        "test": response.index[test].tolist(),
                    }
                )
                for method, p in fitted.items():
                    metrics.append(
                        {
                            "drug_id": int(drug),
                            "fold": fold,
                            "budget": budget,
                            "method": method,
                            "rho": rho(y[test], p),
                            "mae": float(np.abs(y[test] - p).mean()),
                            "n_train": len(support),
                            "n_test": len(test),
                            "constant_prediction": bool(np.ptp(p) == 0),
                            "constant_observed": bool(np.ptp(y[test]) == 0),
                        }
                    )
                    predictions.append(
                        {
                            "fold": fold,
                            "budget": budget,
                            "method": method,
                            "observed": y[test].tolist(),
                            "predicted": p.tolist(),
                        }
                    )
    result = {
        "drug_id": int(drug),
        "metrics": metrics,
        "predictions": predictions,
        "decisions": decisions,
        "skipped": skipped,
        "convergence_warnings": warnings_total,
        "seconds": time.perf_counter() - start,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(target)
    return drug, f"{result['seconds']:.1f}s, warnings={warnings_total}"


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(SEED)
    boots = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(boots, [0.025, 0.975])
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "low": float(low),
        "high": float(high),
    }


def report(out, drugs):
    records = [json.loads((out / f"{d}.json").read_text()) for d in drugs]
    metrics = pd.DataFrame([r for record in records for r in record["metrics"]])
    metrics.to_csv(out / "per_drug_fold.csv", index=False)
    average = metrics.groupby(["drug_id", "budget", "method"])[["rho", "mae"]].mean().reset_index()
    summary = average.groupby(["budget", "method"]).agg(
        rho=("rho", "mean"),
        median_rho=("rho", "median"),
        mae=("mae", "mean"),
        drugs=("rho", "count"),
    )
    summary.to_csv(out / "summary.csv")
    contrasts = {}
    for method in ("ridge_all", "ridge_1000", "elastic_1000"):
        p = (
            average.loc[average.method.eq(method)]
            .pivot(index="drug_id", columns="budget", values="rho")
            .dropna()
        )
        contrasts[f"{method}_full_minus_32"] = interval(p["full"] - p["32"])
    full = average.loc[average.budget.eq("full")].pivot(
        index="drug_id", columns="method", values="rho"
    )
    for method in ("ridge_1000", "elastic_1000"):
        contrasts[f"full_{method}_minus_ridge_all"] = interval(full[method] - full.ridge_all)
    for budget in ("32", "full"):
        p = average.loc[average.budget.eq(budget)].pivot(
            index="drug_id", columns="method", values="mae"
        )
        for method in ("ridge_all", "ridge_1000", "elastic_1000"):
            for baseline in ("mean", "median", "lineage_mean"):
                contrasts[f"{budget}_{method}_mae_improvement_vs_{baseline}"] = interval(
                    p[baseline] - p[method]
                )
    evidence = {
        "status": "complete",
        "compounds": len(drugs),
        "contrasts": contrasts,
        "convergence_warnings": sum(r["convergence_warnings"] for r in records),
        "skipped_folds": sum(len(r["skipped"]) for r in records),
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    text = "# CTRP-native baseline results\n\nDevelopment evaluation, not independent confirmation.\n\n"
    text += summary.to_string() + "\n\n" + json.dumps(evidence, indent=2) + "\n"
    (out / "RESULT.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.run_id in ("", ".", "..") or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a folder name")
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(ROOT / "docs/BASELINE_PROTOCOL.md"):
        raise ValueError("Protocol changed after preparation")
    for name, sha in manifest["outputs"].items():
        if digest(DATA / name) != sha:
            raise ValueError(f"Prepared input changed: {name}")
    compounds = pd.read_csv(DATA / "compounds.csv")
    drugs = sorted(compounds.loc[compounds.eligible, "drug_id"].tolist())
    if args.smoke:
        drugs = drugs[:3]
    out = ROOT / "outputs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    signature = {
        "manifest": digest(DATA / "manifest.json"),
        "code": digest(__file__),
        "protocol": digest(ROOT / "docs/BASELINE_PROTOCOL.md"),
        "drugs": drugs,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "smoke": args.smoke,
    }
    config = out / "config.json"
    if config.exists() and json.loads(config.read_text()) != signature:
        raise ValueError("Existing run signature differs; use a new run-id")
    config.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(f"RUN {args.run_id}: {len(drugs)} compounds, {args.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (drug, status) in enumerate(
            pool.imap_unordered(task, [(d, str(out)) for d in drugs]), 1
        ):
            print(f"[{i}/{len(drugs)}] {drug}: {status}", flush=True)
    report(out, drugs)


if __name__ == "__main__":
    main()
