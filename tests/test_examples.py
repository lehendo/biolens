"""Tests for the selection-inflation example."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
import selection_inflation_demo as demo  # noqa: E402


def test_auroc_columns_matches_sklearn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 5))
    y = np.repeat([0, 1], 30)
    got = demo.auroc_columns(x, y)
    for j in range(5):
        assert got[j] == pytest.approx(roc_auc_score(y, x[:, j]))


def test_selection_inflates_but_heldout_stays_at_chance():
    selected_1, held_1 = demo.best_of_d_inflation(1, n_trials=60, seed=1)
    selected_500, held_500 = demo.best_of_d_inflation(500, n_trials=60, seed=1)
    assert selected_500 > selected_1 + 0.05
    assert selected_500 > 0.60
    assert held_1 == pytest.approx(0.5, abs=0.03)
    assert held_500 == pytest.approx(0.5, abs=0.03)


def test_example_registry_parses():
    from biolens.eval.feature_inspection import load_verified_annotations

    path = Path(__file__).parent.parent / "examples" / "verified_features_example.yaml"
    registry = load_verified_annotations(path)
    assert registry[7].status == "confirmed"
    assert registry[42].status == "spurious"
