"""Double cold-start evaluation: unscreened compounds on unseen patient groups.

Phase `profiles` fits one ridge model per (cell fold, drug) using only training cells.
Phase `evaluate` builds drug folds, similarity sources and the stacked "tamed" models,
then scores every test compound on test cells.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark import RIDGE, interval, rho
from metrics import lineage_scores
from prepare import SEED, digest
from scipy import sparse
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/prepared"
DATA = ROOT / "data/experiment"
OUT = ROOT / "outputs"
LLMS = ["gemma", "kimi", "ember", "opus", "gpt"]
NEIGHBOURS = 10
STACK_ALPHA = 10.0
DRUG_FOLDS = 5
LINEAGE_METHODS = [
    "general",
    "knn_structure",
    "knn_curated",
    "tamed_nonllm",
    "tamed_all",
]


def load_inputs():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        assert digest(SOURCE / name) == expected, f"{name} changed"
    cells = pd.read_csv(SOURCE / "cells.csv", index_col="cell_id")
    expression = np.load(SOURCE / "expression.npy").astype(float)
    responses = pd.read_parquet(SOURCE / "responses.parquet")
    compounds = pd.read_csv(SOURCE / "compounds.csv")
    compounds = compounds.loc[compounds.eligible].sort_values("drug_id")
    compounds = compounds.set_index("drug_id")
    labels = responses.pivot(index="cell_id", columns="drug_id", values="sensitivity")
    labels = labels.reindex(index=cells.index, columns=compounds.index)
    return cells, expression, labels.to_numpy(), compounds


def standardize(train, test):
    mean, scale = train.mean(axis=0), train.std(axis=0)
    keep = scale > 1e-6
    return (train[:, keep] - mean[keep]) / scale[keep], (test[:, keep] - mean[keep]) / scale[keep]


def ridge_path(kernel, crosses, y):
    """Kernel ridge predictions for every RIDGE alpha, one eigendecomposition."""
    values, vectors = np.linalg.eigh(kernel)
    projected = vectors.T @ (y - y.mean())
    paths = []
    for alpha in RIDGE:
        coefficients = vectors @ (projected / (np.maximum(values, 0) + alpha))
        paths.append([y.mean() + cross @ coefficients for cross in crosses])
    return paths


def drug_profile(kernel, cross, y, observed, groups):
    """Inner-CV alpha, out-of-fold predictions at training cells, predictions at test cells."""
    rows = np.flatnonzero(observed)
    scores = np.zeros((3, len(RIDGE)))
    oof = np.full((len(rows), len(RIDGE)), np.nan)
    for split, (a, b) in enumerate(GroupKFold(3).split(rows, groups=groups[rows])):
        ra, rb = rows[a], rows[b]
        paths = ridge_path(kernel[np.ix_(ra, ra)], [kernel[np.ix_(rb, ra)]], y[ra])
        for i, (prediction,) in enumerate(paths):
            oof[b, i] = prediction
            scores[split, i] = rho(y[rb], prediction)
    best = int(np.argmax(np.nan_to_num(scores, nan=-1).mean(axis=0)))
    alpha = RIDGE[best]
    train_profile, test_profile = ridge_path(
        kernel[np.ix_(rows, rows)], [kernel[:, rows], cross[:, rows]], y[rows]
    )[best]
    # Observed training cells get out-of-fold values; unlabelled ones keep the full fit,
    # which is still out of sample for them.
    train_profile[rows] = oof[:, best]
    mean, scale = train_profile.mean(), train_profile.std()
    scale = scale if scale > 1e-12 else 1.0
    return (train_profile - mean) / scale, (test_profile - mean) / scale, alpha


def build_profiles():
    cells, expression, labels, _ = load_inputs()
    groups = cells.group.to_numpy()
    for fold in range(3):
        target = DATA / f"profiles_fold{fold}.npz"
        if target.exists():
            print(f"fold {fold}: exists", flush=True)
            continue
        train = np.flatnonzero(cells.fold.to_numpy() != fold)
        test = np.flatnonzero(cells.fold.to_numpy() == fold)
        z_train, z_test = standardize(expression[train], expression[test])
        kernel = z_train @ z_train.T / z_train.shape[1]
        cross = z_test @ z_train.T / z_train.shape[1]
        train_profiles = np.zeros((len(train), labels.shape[1]))
        test_profiles = np.zeros((len(test), labels.shape[1]))
        alphas = np.zeros(labels.shape[1])
        for j in range(labels.shape[1]):
            y = labels[train, j]
            observed = np.isfinite(y)
            train_profiles[:, j], test_profiles[:, j], alphas[j] = drug_profile(
                kernel, cross, np.nan_to_num(y), observed, groups[train]
            )
            if (j + 1) % 100 == 0:
                print(f"fold {fold}: {j + 1}/{labels.shape[1]}", flush=True)
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez(target, train=train_profiles, test=test_profiles, alphas=alphas)
        print(f"fold {fold}: saved", flush=True)


# ---------- drug similarity sources ----------


def fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        result.append(generator.GetFingerprint(molecule) if molecule is not None else None)
    return result


def tanimoto_matrix(prints):
    from rdkit import DataStructs

    n = len(prints)
    matrix = np.zeros((n, n))
    positions = [k for k, p in enumerate(prints) if p is not None]
    valid = [prints[k] for k in positions]
    for i in positions:
        matrix[i, positions] = DataStructs.BulkTanimotoSimilarity(prints[i], valid)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def jaccard_matrix(sets):
    n = len(sets)
    matrix = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1, n):
            union = sets[i] | sets[k]
            if union:
                matrix[i, k] = matrix[k, i] = len(sets[i] & sets[k]) / len(union)
    return matrix


def structure_folds(similarity, seed):
    from rdkit.ML.Cluster import Butina

    n = len(similarity)
    distances = [1 - similarity[i, k] for i in range(1, n) for k in range(i)]
    clusters = Butina.ClusterData(distances, n, 0.4, isDistData=True)
    order = np.random.default_rng(seed).permutation(len(clusters))
    folds, sizes = np.zeros(n, dtype=int), np.zeros(DRUG_FOLDS, dtype=int)
    for index in sorted(order, key=lambda c: -len(clusters[c])):
        # Largest clusters first, ties broken by the seeded shuffle above.
        smallest = int(np.argmin(sizes))
        folds[list(clusters[index])] = smallest
        sizes[smallest] += len(clusters[index])
    return folds, len(clusters)


def random_folds(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    folds = np.zeros(n, dtype=int)
    folds[order] = np.arange(n) % DRUG_FOLDS
    return folds


def load_annotations(model, drugs):
    result = []
    for drug in drugs:
        path = DATA / "annotations" / model / f"{drug}.json"
        record = json.loads(path.read_text()) if path.exists() else None
        if record is None or record["status"] != "valid":
            result.append(None)
        else:
            result.append(record["annotation"])
    return result


def annotation_digest(model, drugs):
    combined = hashlib.sha256()
    for drug in drugs:
        path = DATA / "annotations" / model / f"{drug}.json"
        if path.exists():
            combined.update(path.read_bytes())
    return combined.hexdigest()


def llm_similarity(annotations):
    targets = [set(a["targets"]) if a else set() for a in annotations]
    markers = [
        {m["gene"] + ("+" if m["direction"] == "sensitive_if_high" else "-") for m in a["markers"]}
        if a
        else set()
        for a in annotations
    ]
    classes = np.array([a["mechanism_class"] if a else "unknown" for a in annotations])
    informative = ~np.isin(classes, ["other", "unknown"])
    same = (classes[:, None] == classes[None, :]) & informative[:, None] & informative[None, :]
    matrix = 0.5 * jaccard_matrix(targets) + 0.25 * same + 0.25 * jaccard_matrix(markers)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def marker_weights(annotations, gene_index, n_genes):
    weights = sparse.lil_matrix((len(annotations), n_genes))
    for j, a in enumerate(annotations):
        if not a:
            continue
        valid = [m for m in a["markers"] if m["gene"] in gene_index]
        for m in valid:
            sign = 1.0 if m["direction"] == "sensitive_if_high" else -1.0
            weights[j, gene_index[m["gene"]]] += sign / len(valid)
    return weights.tocsr()


def derange(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(n)
        if not np.any(order == np.arange(n)):
            return order


# ---------- per-block methods ----------


def knn_scores(similarity, profiles, train_drugs, query_drugs, general):
    """Weighted mean profile of the most similar training drugs, excluding the drug itself."""
    sub = similarity[np.ix_(query_drugs, train_drugs)].copy()
    sub[query_drugs[:, None] == train_drugs[None, :]] = 0.0
    weights = np.zeros_like(sub)
    for row in range(len(query_drugs)):
        positive = np.flatnonzero(sub[row] > 0)
        top = positive[np.argsort(-sub[row, positive], kind="stable")[:NEIGHBOURS]]
        weights[row, top] = sub[row, top]
    totals = weights.sum(axis=1)
    scores = profiles[:, train_drugs] @ (weights / np.where(totals > 0, totals, 1)[:, None]).T
    missing = totals == 0
    scores[:, missing] = general[:, missing]
    return scores, sub.max(axis=1)


def general_scores(profiles, train_drugs, query_drugs):
    total = profiles[:, train_drugs].sum(axis=1, keepdims=True)
    counts = np.full(len(query_drugs), len(train_drugs), dtype=float)
    own = np.isin(query_drugs, train_drugs)
    scores = np.repeat(total, len(query_drugs), axis=1)
    scores[:, own] -= profiles[:, query_drugs[own]]
    counts[own] -= 1
    return scores / counts


def zscore_columns(matrix):
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    return (matrix - mean) / np.where(scale > 1e-12, scale, 1.0)


def nonllm_scores(context, train_drugs, query_drugs, side):
    profiles = context["profiles"][side]
    general = general_scores(profiles, train_drugs, query_drugs)
    scores, trust = {"general": general}, {}
    for name, similarity in context["similarity"].items():
        scores[f"knn_{name}"], trust[name] = knn_scores(
            similarity, profiles, train_drugs, query_drugs, general
        )
    return scores, trust


def llm_scores(context, variant, train_drugs, query_drugs, side, general):
    """kNN and marker scores for every LLM under one annotation variant."""
    profiles = context["profiles"][side]
    scores, trust = {}, {}
    for model, info in context["variants"][variant].items():
        scores[f"knn_llm_{model}"], trust[model] = knn_scores(
            info["similarity"], profiles, train_drugs, query_drugs, general
        )
        weights = info["markers"][query_drugs]
        marker = np.asarray((weights @ context["rna"][side].T).T)
        empty = weights.getnnz(axis=1) == 0
        marker[:, empty] = general[:, empty]
        scores[f"marker_{model}"] = marker
    return scores, trust


def design(base, base_trust, llm, llm_trust, info, llms, query_drugs):
    # Z-score each score within drug first, then multiply by per-drug trust. The other
    # order would cancel the trust weight, because a per-drug constant only rescales.
    general = zscore_columns(base["general"])
    structure = zscore_columns(base["knn_structure"])
    curated = zscore_columns(base["knn_curated"])
    columns = [
        general,
        structure,
        curated,
        structure * base_trust["structure"],
        curated * base_trust["curated"],
    ]
    for model in llms:
        knn = zscore_columns(llm[f"knn_llm_{model}"])
        marker = zscore_columns(llm[f"marker_{model}"])
        confidence = info[model]["confidence"][query_drugs]
        recognized = info[model]["recognized"][query_drugs]
        columns += [
            knn,
            marker,
            knn * confidence,
            knn * recognized,
            knn * llm_trust[model],
            marker * confidence,
        ]
    return np.stack(columns, axis=2)


def fit_stacker(features, target, observed):
    x, y = features[observed], target[observed]
    gram = x.T @ x + STACK_ALPHA * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def standardized_targets(labels):
    observed = np.isfinite(labels)
    mean, scale = np.nanmean(labels, axis=0), np.nanstd(labels, axis=0)
    target = (labels - mean) / np.where(scale > 1e-12, scale, 1.0)
    return np.nan_to_num(target), observed


def evaluate_block(context, cell_fold, train_drugs, test_drugs, sets):
    train_cells, _ = context["cells"][cell_fold]
    train_side, test_side = ("train", cell_fold), ("test", cell_fold)
    base_test, trust_test = nonllm_scores(context, train_drugs, test_drugs, test_side)
    base_train, trust_train = nonllm_scores(context, train_drugs, train_drugs, train_side)
    llm_test, llm_train = {}, {}
    for variant in context["variants"]:
        llm_test[variant] = llm_scores(
            context, variant, train_drugs, test_drugs, test_side, base_test["general"]
        )
        llm_train[variant] = llm_scores(
            context,
            variant,
            train_drugs,
            train_drugs,
            train_side,
            base_train["general"],
        )
    target, observed = standardized_targets(context["labels"][np.ix_(train_cells, train_drugs)])
    results = dict(base_test)
    results.update(llm_test["real"][0])
    weights = {}
    for name, (llms, variant) in sets.items():
        info = context["variants"][variant]
        x_train = design(base_train, trust_train, *llm_train[variant], info, llms, train_drugs)
        x_test = design(base_test, trust_test, *llm_test[variant], info, llms, test_drugs)
        coefficients = fit_stacker(x_train, target, observed)
        results[name] = x_test @ coefficients
        weights[name] = coefficients.tolist()
    return results, weights


def build_context(compounds, cells, expression, labels, shuffles):
    drugs = compounds.index.to_numpy()
    genes = pd.read_csv(SOURCE / "genes.csv").gene.str.split(" (", regex=False).str[0]
    gene_index = {g: i for i, g in enumerate(genes)}
    prints = fingerprints(compounds.cpd_smiles.astype(str))
    structure = tanimoto_matrix(prints)
    curated = jaccard_matrix(
        [
            set(str(t).split(";")) if isinstance(t, str) else set()
            for t in compounds.gene_symbol_of_protein_target
        ]
    )
    annotations = {model: load_annotations(model, drugs) for model in LLMS}
    variants = {"real": {}}
    for model in LLMS:
        variants["real"][model] = variant_info(annotations[model], gene_index, len(genes))
    for draw in range(shuffles):
        variants[f"shuffled{draw}"] = {}
        for model in LLMS:
            order = derange(len(drugs), SEED + 100 * draw + LLMS.index(model))
            records = [annotations[model][k] for k in order]
            variants[f"shuffled{draw}"][model] = variant_info(records, gene_index, len(genes))
    profiles, rna, fold_cells = {}, {}, {}
    for fold in range(3):
        saved = np.load(DATA / f"profiles_fold{fold}.npz")
        train = np.flatnonzero(cells.fold.to_numpy() != fold)
        test = np.flatnonzero(cells.fold.to_numpy() == fold)
        fold_cells[fold] = (train, test)
        profiles[("train", fold)], profiles[("test", fold)] = (
            saved["train"],
            saved["test"],
        )
        z_train, z_test = standardize_full(expression[train], expression[test])
        rna[("train", fold)], rna[("test", fold)] = z_train, z_test
    return {
        "drugs": drugs,
        "similarity": {"structure": structure, "curated": curated},
        "variants": variants,
        "annotations": annotations,
        "profiles": profiles,
        "rna": rna,
        "cells": fold_cells,
        "labels": labels,
    }


def variant_info(records, gene_index, n_genes):
    return {
        "similarity": llm_similarity(records),
        "markers": marker_weights(records, gene_index, n_genes),
        "confidence": np.array([a["confidence"] if a else 0.0 for a in records]),
        "recognized": np.array([float(a["recognized"]) if a else 0.0 for a in records]),
    }


def standardize_full(train, test):
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale = np.where(scale > 1e-6, scale, np.inf)
    return (train - mean) / scale, (test - mean) / scale


def stack_sets(shuffles):
    sets = {"tamed_nonllm": ([], "real")}
    for model in LLMS:
        sets[f"tamed_{model}"] = ([model], "real")
    sets["tamed_open"] = (["gemma", "kimi", "ember"], "real")
    sets["tamed_frontier"] = (["opus", "gpt"], "real")
    sets["tamed_all"] = (LLMS, "real")
    for draw in range(shuffles):
        sets[f"tamed_shuffled{draw}"] = (LLMS, f"shuffled{draw}")
    return sets


def run_split(context, folds, shuffles, keep_predictions):
    sets = stack_sets(shuffles)
    metrics, predictions, weights = [], [], []
    cell_ids = context["cell_ids"]
    for drug_fold in range(DRUG_FOLDS):
        train_drugs = np.flatnonzero(folds != drug_fold)
        test_drugs = np.flatnonzero(folds == drug_fold)
        for cell_fold in range(3):
            _, test_cells = context["cells"][cell_fold]
            results, block_weights = evaluate_block(
                context, cell_fold, train_drugs, test_drugs, sets
            )
            weights.append(
                {
                    "drug_fold": drug_fold,
                    "cell_fold": cell_fold,
                    "weights": block_weights,
                }
            )
            observed_block = context["labels"][np.ix_(test_cells, test_drugs)]
            for column, drug_position in enumerate(test_drugs):
                observed = observed_block[:, column]
                rows = np.isfinite(observed)
                drug_id = int(context["drugs"][drug_position])
                for method, scores in results.items():
                    predicted = scores[rows, column]
                    metrics.append(
                        {
                            "drug_id": drug_id,
                            "fold": cell_fold,
                            "method": method,
                            "rho": rho(observed[rows], predicted),
                        }
                    )
                    if method in keep_predictions:
                        predictions.append(
                            pd.DataFrame(
                                {
                                    "drug_id": drug_id,
                                    "fold": cell_fold,
                                    "method": method,
                                    "cell_id": cell_ids[test_cells[rows]],
                                    "observed": observed[rows],
                                    "predicted": predicted,
                                }
                            )
                        )
            print(f"drug fold {drug_fold}, cell fold {cell_fold}: done", flush=True)
    frame = pd.DataFrame(metrics)
    assert not frame.duplicated(["drug_id", "fold", "method"]).any()
    long = pd.concat(predictions, ignore_index=True) if predictions else None
    return frame, long, weights


def per_drug(metrics):
    return metrics.groupby(["drug_id", "method"]).rho.mean().unstack()


def contrast(table, a, b):
    return interval((table[a] - table[b]).dropna())


def contrasts(table, shuffles):
    table = table.copy()
    table["tamed_shuffled_mean"] = table[[f"tamed_shuffled{d}" for d in range(shuffles)]].mean(
        axis=1
    )
    result = {
        "P1_tamed_all_minus_tamed_nonllm": contrast(table, "tamed_all", "tamed_nonllm"),
        "P1_tamed_all_minus_shuffled_mean": contrast(table, "tamed_all", "tamed_shuffled_mean"),
        "tamed_nonllm_minus_general": contrast(table, "tamed_nonllm", "general"),
        "tamed_frontier_minus_tamed_open": contrast(table, "tamed_frontier", "tamed_open"),
        "knn_structure_minus_general": contrast(table, "knn_structure", "general"),
        "knn_curated_minus_general": contrast(table, "knn_curated", "general"),
    }
    for model in LLMS:
        result[f"P2_knn_llm_{model}_minus_knn_structure"] = contrast(
            table, f"knn_llm_{model}", "knn_structure"
        )
        result[f"P2_knn_llm_{model}_minus_knn_curated"] = contrast(
            table, f"knn_llm_{model}", "knn_curated"
        )
        result[f"tamed_{model}_minus_tamed_nonllm"] = contrast(
            table, f"tamed_{model}", "tamed_nonllm"
        )
        result[f"marker_{model}_minus_general"] = contrast(table, f"marker_{model}", "general")
    best = max(LLMS, key=lambda m: table[f"tamed_{m}"].mean())
    result["tamed_all_minus_best_single"] = {
        "best": best,
        **contrast(table, "tamed_all", f"tamed_{best}"),
    }
    return result


def subsets(context, compounds):
    recognized = np.zeros(len(compounds))
    for model in LLMS:
        recognized += context["variants"]["real"][model]["recognized"]
    status = compounds.cpd_status.to_numpy()
    return {
        "probe": compounds.index[status == "probe"],
        "clinical_or_fda": compounds.index[np.isin(status, ["clinical", "FDA"])],
        "recognized_by_3plus": compounds.index[recognized >= 3],
        "recognized_by_0to2": compounds.index[recognized < 3],
        "no_curated_target": compounds.index[compounds.gene_symbol_of_protein_target.isna()],
    }


def evaluate(shuffles=3):
    cells, expression, labels, compounds = load_inputs()
    context = build_context(compounds, cells, expression, labels, shuffles)
    context["cell_ids"] = cells.index.to_numpy()
    coverage = {
        model: int(sum(a is not None for a in context["annotations"][model])) for model in LLMS
    }
    folds, clusters = structure_folds(context["similarity"]["structure"], SEED)
    splits = {"structure_grouped": folds, "random": random_folds(len(compounds), SEED)}
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "drugs": len(compounds),
        "code": digest(Path(__file__)),
        "protocol": digest(ROOT / "docs/PROTOCOL.md"),
        "annotation_digests": {model: annotation_digest(model, context["drugs"]) for model in LLMS},
        "annotation_coverage": coverage,
        "structure_clusters": clusters,
        "fold_sizes": {k: np.bincount(v).tolist() for k, v in splits.items()},
    }
    for split, split_folds in splits.items():
        keep = LINEAGE_METHODS if split == "structure_grouped" else []
        metrics, long, weights = run_split(context, split_folds, shuffles, keep)
        metrics.to_csv(OUT / f"{split}_per_drug_fold.csv", index=False)
        (OUT / f"{split}_stacker_weights.json").write_text(json.dumps(weights), encoding="utf-8")
        table = per_drug(metrics)
        table.to_csv(OUT / f"{split}_per_drug.csv")
        entry = {
            "mean_rho": table.mean().round(6).sort_values(ascending=False).to_dict(),
            "contrasts": contrasts(table, shuffles),
            "subsets": {},
        }
        for name, ids in subsets(context, compounds).items():
            part = table.loc[table.index.isin(ids)]
            entry["subsets"][name] = {
                "n": len(part),
                "tamed_all_minus_tamed_nonllm": contrast(part, "tamed_all", "tamed_nonllm"),
                "knn_llm_opus_minus_knn_structure": contrast(part, "knn_llm_opus", "knn_structure"),
            }
        if long is not None:
            long.to_parquet(OUT / f"{split}_predictions.parquet", index=False)
            strata = lineage_scores(long, cells)
            within = (
                strata.loc[strata.status.eq("usable")]
                .groupby(["drug_id", "fold", "method"])
                .rho.mean()
                .groupby(["drug_id", "method"])
                .mean()
                .unstack()
            )
            entry["within_lineage"] = {
                "mean": within.mean().round(6).to_dict(),
                "tamed_all_minus_tamed_nonllm": contrast(within, "tamed_all", "tamed_nonllm"),
            }
        report[split] = entry
        print(json.dumps({split: entry["mean_rho"]}, indent=2), flush=True)
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k]["contrasts"] for k in splits}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("profiles", "evaluate"))
    args = parser.parse_args()
    if args.phase == "profiles":
        build_profiles()
    else:
        evaluate()


if __name__ == "__main__":
    main()
