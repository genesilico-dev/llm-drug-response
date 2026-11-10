"""Build an independent broad CTRPv2 dataset from raw responses and DepMap RNA."""

import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("DRP_RAW_DIR", ROOT / "data/raw"))
DATA = ROOT / "data/prepared"
ARCHIVE_SHA = "8f62b3b5ed70cfd367cf52ce0a99884dd0a674d1a8c301474b707648689bdee3"
SEED = 20261109


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalize(value):
    return re.sub("[^A-Z0-9]", "", str(value).upper()) if pd.notna(value) else ""


def aliases_for(models):
    aliases = {}
    for model, row in models.iterrows():
        values = [row.get(c) for c in ("CellLineName", "StrippedCellLineName", "CCLEName")]
        if pd.notna(row.get("CCLEName")):
            values.append(row.CCLEName.split("_")[0])
        for value in values:
            key = normalize(value)
            if key:
                aliases.setdefault(key, set()).add(model)
    return aliases


def assign_folds(groups):
    unique = np.unique(groups)
    np.random.default_rng(SEED).shuffle(unique)
    mapping = {group: i % 3 for i, group in enumerate(unique)}
    return np.array([mapping[group] for group in groups])


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    inputs = {
        "archive": RAW / "ctrpv2/CTRPv2.0_2015_ctd2_ExpandedDataset.zip",
        "model": RAW / "depmap/Model.csv",
        "expression": RAW / "depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    }
    hashes = {name: digest(path) for name, path in inputs.items()}
    if hashes["archive"] != ARCHIVE_SHA:
        raise ValueError("CTRP archive checksum mismatch")
    models = pd.read_csv(inputs["model"]).set_index("ModelID")
    expression = pd.read_csv(inputs["expression"], index_col=0)
    if not models.index.is_unique or not expression.index.is_unique:
        raise ValueError("Duplicate ModelID in raw inputs")
    aliases = aliases_for(models)
    members = {}
    with zipfile.ZipFile(inputs["archive"]) as archive:
        md5s = dict(
            line.split(" *", 1)[::-1] for line in archive.read("MANIFEST.txt").decode().splitlines()
        )

        def read(name):
            content = archive.read(name)
            if hashlib.md5(content).hexdigest() != md5s[name]:
                raise ValueError(f"Publisher checksum mismatch: {name}")
            members[name] = hashlib.sha256(content).hexdigest()
            return pd.read_csv(io.BytesIO(content), sep="\t")

        compounds = read("v20.meta.per_compound.txt")
        cells = read("v20.meta.per_cell_line.txt")
        experiments = read("v20.meta.per_experiment.txt")
        curves = read("v20.data.curves_post_qc.txt")
    mapping = []
    for row in cells.itertuples():
        matches = aliases.get(normalize(row.ccl_name), set())
        model = next(iter(matches)) if len(matches) == 1 else None
        status = "matched" if model in expression.index else "no_rna"
        if model is None:
            status = "ambiguous" if matches else "unmapped"
        mapping.append(
            {
                "master_ccl_id": row.master_ccl_id,
                "cell_id": model,
                "name": row.ccl_name,
                "status": status,
                "candidates": ";".join(sorted(matches)),
            }
        )
    mapping = pd.DataFrame(mapping)
    ex = experiments.loc[
        experiments.snp_fp_status.eq("SNP-matched-reference"),
        ["experiment_id", "master_ccl_id"],
    ].drop_duplicates()
    if not ex.experiment_id.is_unique:
        raise ValueError("Ambiguous experiment mapping")
    ex = ex.merge(
        mapping.loc[mapping.status.eq("matched"), ["master_ccl_id", "cell_id"]],
        validate="many_to_one",
    )
    joined = curves.merge(ex, on="experiment_id", validate="many_to_one")
    finite = np.isfinite(joined.area_under_curve)
    enough_points = joined.conc_pts_fit.ge(8)
    selected = joined.loc[finite & enough_points]
    responses = selected.groupby(["cell_id", "master_cpd_id"], as_index=False).agg(
        auc=("area_under_curve", "median"), n_curves=("area_under_curve", "size")
    )
    responses = responses.rename(columns={"master_cpd_id": "drug_id"})
    responses["sensitivity"] = -responses.auc
    ids = sorted(responses.cell_id.unique())
    meta = models.loc[ids, ["PatientID", "OncotreeLineage"]].copy()
    meta["group"] = meta.PatientID.fillna(pd.Series(meta.index, index=meta.index)).astype(str)
    meta["lineage"] = meta.OncotreeLineage.fillna("unknown")
    meta["fold"] = assign_folds(meta.group.to_numpy())
    counts = (
        responses.merge(meta[["group"]], left_on="cell_id", right_index=True)
        .groupby("drug_id")
        .group.nunique()
    )
    compounds = compounds.rename(columns={"master_cpd_id": "drug_id"}).set_index("drug_id")
    if not compounds.index.is_unique:
        raise ValueError("Duplicate native compound identifiers")
    compounds["n_groups"] = counts.reindex(compounds.index).fillna(0).astype(int)
    compounds["eligible"] = compounds.n_groups.ge(150)
    responses = responses.loc[responses.drug_id.isin(compounds.index[compounds.eligible])]
    np.save(DATA / "expression.npy", expression.loc[ids].to_numpy(dtype=np.float32))
    pd.Series(expression.columns, name="gene").to_csv(DATA / "genes.csv", index=False)
    meta.to_csv(DATA / "cells.csv", index_label="cell_id")
    compounds.to_csv(DATA / "compounds.csv")
    mapping.to_csv(DATA / "cell_mapping.csv", index=False)
    responses.to_parquet(DATA / "responses.parquet", index=False)
    audit = {
        "raw_compounds": len(compounds),
        "eligible_compounds": int(compounds.eligible.sum()),
        "mapped_rna_cells": len(meta),
        "rna_genes": expression.shape[1],
        "response_pairs": len(responses),
        "raw_curves": len(curves),
        "mapped_snp_matched_curves": len(joined),
        "excluded_nonfinite": int((~finite).sum()),
        "excluded_under_eight_points": int((~enough_points).sum()),
        "cell_mapping": mapping.status.value_counts().to_dict(),
        "experiment_qc": experiments.snp_fp_status.value_counts().to_dict(),
        "endpoint": "negative native CTRPv2 AUC",
        "prism_inputs": [],
    }
    outputs = {p.name: digest(p) for p in DATA.iterdir() if p.name != "manifest.json"}
    manifest = {
        "audit": audit,
        "inputs": hashes,
        "members": members,
        "outputs": outputs,
        "protocol_sha256": digest(ROOT / "docs/BASELINE_PROTOCOL.md"),
        "prepare_sha256": digest(__file__),
        "research_use": "CTD2 investigative reuse; commercial permission not asserted",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
