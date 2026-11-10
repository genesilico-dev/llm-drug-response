"""Check the published macro correlations against the bundled per-drug tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    evidence = json.loads((root / "evidence.json").read_text())
    for split in ("structure_grouped", "random"):
        table = pd.read_csv(root / f"{split}_per_drug.csv", index_col=0)
        assert len(table) == evidence["drugs"] == 536
        assert table.index.is_unique
        for method, reported in evidence[split]["mean_rho"].items():
            np.testing.assert_allclose(table[method].mean(), reported, atol=1e-6)
        print(f"{split}: verified {len(table)} drugs and {len(evidence[split]['mean_rho'])} means")


if __name__ == "__main__":
    main()
