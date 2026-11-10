"""Synthetic checks independent of biological outcomes."""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark import RIDGE, features, fit_models, rho, ridge_path, support32
from prepare import aliases_for, assign_folds


def test_groups_stay_together():
    groups = np.repeat([f"patient{i}" for i in range(90)], 2)
    folds = assign_folds(groups)
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1
    assert set(folds) == {0, 1, 2}
    np.testing.assert_array_equal(folds, assign_folds(groups))


def test_support_uses_32_unique_training_groups():
    groups = np.repeat(np.arange(90), 2)
    train = np.arange(140)
    support = support32(train, groups, 1)
    assert len(support) == len(np.unique(groups[support])) == 32
    assert set(support).issubset(train)
    np.testing.assert_array_equal(support, support32(train, groups, 1))


def test_features_ignore_test_values():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(30, 20))
    test = rng.normal(size=(10, 20))
    a, _ = features(train, test, 5)
    b, _ = features(train, test * 1000, 5)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a.mean(axis=0), 0, atol=1e-12)
    assert a.shape == (30, 5)


def test_ridge_matches_sklearn():
    rng = np.random.default_rng(2)
    x, z = features(rng.normal(size=(30, 40)), rng.normal(size=(10, 40)))
    y = rng.normal(size=30)
    for alpha, prediction in zip(RIDGE, ridge_path(x, y, z)):
        expected = Ridge(alpha=alpha * x.shape[1]).fit(x, y).predict(z)
        np.testing.assert_allclose(prediction, expected, atol=1e-9)


def test_constant_metrics_are_explicit():
    assert np.isnan(rho(np.ones(5), np.arange(5)))
    assert rho(np.arange(5), np.ones(5)) == 0


def test_fit_is_repeatable_and_choices_ignore_test_features():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(36, 12))
    y = x[:, 0] + rng.normal(size=36) * 0.1
    groups = np.repeat(np.arange(18), 2)
    test = rng.normal(size=(8, 12))
    a, choices, warnings = fit_models(x, y, groups, test)
    b, changed_choices, _ = fit_models(x, y, groups, test + 500)
    assert choices == changed_choices
    assert warnings == 0
    assert set(a) == set(b) == {"ridge_all", "ridge_1000", "elastic_1000"}
    assert all(np.isfinite(p).all() for p in a.values())


def test_ambiguous_aliases_remain_ambiguous():
    import pandas as pd

    models = pd.DataFrame({"CellLineName": ["A-1", "A1"]}, index=["x", "y"])
    assert aliases_for(models)["A1"] == {"x", "y"}
