"""Annotation audit: validity, cost, telemetry and agreement with CTRP curated targets.

Uses no response labels, so it can run before or after evaluation.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from annotate import DATA, MODELS, SOURCE, load_ledger


def audit_model(key, compounds):
    curated = {
        int(d): set(str(t).split(";"))
        for d, t in compounds.gene_symbol_of_protein_target.items()
        if isinstance(t, str)
    }
    records = []
    for drug in compounds.index:
        path = DATA / "annotations" / key / f"{drug}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
    attempts = [a for r in records for a in r["attempts"]]
    answered = [a for a in attempts if "model" in a]
    valid = [r for r in records if r["status"] == "valid"]
    hits, predicted, truth, any_hit, scored = 0, 0, 0, 0, 0
    for record in valid:
        drug = record["drug_id"]
        if drug not in curated:
            continue
        targets = set(record["annotation"]["targets"])
        overlap = len(targets & curated[drug])
        hits += overlap
        predicted += len(targets)
        truth += len(curated[drug])
        any_hit += overlap > 0
        scored += 1
    ledger = load_ledger(key)
    return {
        "model": MODELS[key]["id"],
        "files": len(records),
        "valid": len(valid),
        "failed": len(records) - len(valid),
        "requests": len(ledger),
        "transport_errors": sum("error" in a for a in attempts),
        "returned_models": dict(Counter(a["model"] for a in answered)),
        "providers": dict(Counter(a["provider"] for a in answered)),
        "finish_reasons": dict(Counter(a["finish_reason"] for a in answered)),
        "reported_usd": round(sum(r["actual_usd"] or 0 for r in ledger), 6),
        "recognized": sum(r["annotation"]["recognized"] for r in valid),
        "unknown_class": sum(r["annotation"]["mechanism_class"] == "unknown" for r in valid),
        "invalid_symbols": sum(len(r["annotation"]["invalid_symbols"]) for r in valid),
        "curated_compounds_scored": scored,
        "target_precision": round(hits / predicted, 4) if predicted else None,
        "target_recall": round(hits / truth, 4) if truth else None,
        "any_curated_target_hit": round(any_hit / scored, 4) if scored else None,
    }


def main():
    compounds = pd.read_csv(SOURCE / "compounds.csv")
    compounds = compounds.loc[compounds.eligible].set_index("drug_id")
    result = {key: audit_model(key, compounds) for key in MODELS}
    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    (out / "annotation_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    columns = [
        "valid",
        "failed",
        "reported_usd",
        "recognized",
        "unknown_class",
        "target_precision",
        "target_recall",
        "any_curated_target_hit",
    ]
    print(pd.DataFrame(result).T[columns].to_string())


if __name__ == "__main__":
    main()
