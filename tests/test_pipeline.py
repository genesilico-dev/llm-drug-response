import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import annotate
import evaluate as ev


def synthetic_context(seed=0, drugs=12, train=30, test=10, genes=40):
    rng = np.random.default_rng(seed)
    cells = train + test
    labels = rng.normal(size=(cells, drugs))
    labels[rng.random(labels.shape) < 0.1] = np.nan
    similarity = rng.random((drugs, drugs))
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, 0)
    records = [
        {
            "recognized": bool(i % 2),
            "mechanism_class": ["HDAC", "microtubule", "unknown"][i % 3],
            "confidence": 0.5,
            "targets": [f"G{i % 4}"],
            "markers": [{"gene": f"G{i % 5}", "direction": "sensitive_if_high"}],
        }
        for i in range(drugs)
    ]
    gene_index = {f"G{i}": i for i in range(genes)}
    info = ev.variant_info(records, gene_index, genes)
    return {
        "drugs": np.arange(drugs),
        "similarity": {"structure": similarity, "curated": similarity.T.copy()},
        "variants": {"real": {"m": info}},
        "profiles": {
            ("train", 0): rng.normal(size=(train, drugs)),
            ("test", 0): rng.normal(size=(test, drugs)),
        },
        "rna": {
            ("train", 0): rng.normal(size=(train, genes)),
            ("test", 0): rng.normal(size=(test, genes)),
        },
        "cells": {0: (np.arange(train), np.arange(train, cells))},
        "labels": labels,
    }


def test_block_never_reads_test_labels():
    context = synthetic_context()
    sets = {"tamed_nonllm": ([], "real"), "tamed_m": (["m"], "real")}
    train_drugs, test_drugs = np.arange(9), np.arange(9, 12)
    first, _ = ev.evaluate_block(context, 0, train_drugs, test_drugs, sets)
    corrupted = context["labels"].copy()
    corrupted[:, test_drugs] = 1e6
    corrupted[30:, :] = -1e6
    context["labels"] = corrupted
    second, _ = ev.evaluate_block(context, 0, train_drugs, test_drugs, sets)
    for method in first:
        np.testing.assert_allclose(first[method], second[method])


def test_trust_interaction_survives_scaling():
    rng = np.random.default_rng(1)
    scores = {k: rng.normal(size=(8, 2)) for k in ("general", "knn_structure", "knn_curated")}
    trust = {"structure": np.array([0.0, 1.0]), "curated": np.array([1.0, 1.0])}
    x = ev.design(scores, trust, {}, {}, {}, [], np.array([0, 1]))
    np.testing.assert_allclose(x[:, 0, 3], 0.0)
    np.testing.assert_allclose(x[:, 1, 3], x[:, 1, 1])


def test_knn_excludes_self_and_falls_back():
    similarity = np.array([[0, 0.9, 0], [0.9, 0, 0], [0, 0, 0]], dtype=float)
    profiles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    general = np.full((2, 3), -7.0)
    scores, trust = ev.knn_scores(
        similarity, profiles, np.array([0, 1, 2]), np.array([0, 2]), general[:, :2]
    )
    np.testing.assert_allclose(scores[:, 0], profiles[:, 1])
    np.testing.assert_allclose(scores[:, 1], -7.0)
    assert trust.tolist() == [0.9, 0.0]


def test_general_leaves_query_drug_out():
    profiles = np.array([[1.0, 2.0, 6.0]])
    scores = ev.general_scores(profiles, np.array([0, 1, 2]), np.array([2]))
    assert scores[0, 0] == pytest.approx(1.5)


def test_derangement_and_drug_folds():
    order = ev.derange(50, 3)
    assert not np.any(order == np.arange(50))
    folds = ev.random_folds(23, 1)
    assert sorted(np.bincount(folds).tolist()) == [4, 4, 5, 5, 5]


def test_validate_drops_unknown_symbols():
    content = (
        '{"recognized": true, "mechanism_class": "HDAC", "confidence": 1.4,'
        ' "targets": ["hdac1", "FAKE1"],'
        ' "markers": [{"gene": "TP53", "direction": "resistant_if_high"}]}'
    )
    clean = annotate.validate(content, {"HDAC1", "TP53"})
    assert clean["targets"] == ["HDAC1"]
    assert clean["invalid_symbols"] == ["FAKE1"]
    assert clean["confidence"] == 1.0


def test_reserve_respects_model_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(annotate, "DATA", tmp_path)
    monkeypatch.setitem(annotate.MODELS["gemma"], "ceiling", 0.005)
    body = {"x": "y"}
    annotate.reserve("gemma", body, 1)
    with pytest.raises(RuntimeError):
        annotate.reserve("gemma", body, 2)
