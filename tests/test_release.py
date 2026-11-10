"""No-data release checks for filesystem safety and bundled evidence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark
import check_results


@pytest.mark.parametrize("name", ["", ".", "..", "../outside"])
def test_run_id_cannot_escape_outputs(monkeypatch, name):
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--run-id", name])
    with pytest.raises(ValueError, match="folder name"):
        benchmark.main()


def test_published_means_match_tables():
    check_results.main()
