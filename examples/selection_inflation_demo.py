"""
Why a high single-feature AUROC is only a candidate signal.

Searching many candidate features for the one with the best AUROC inflates the
reported value, even when no feature carries any signal. This script simulates
`d` candidate features that are pure noise with respect to a binary label, picks
the best one on a selection set, and re-measures it on disjoint held-out data.

This is a simulation that illustrates the selection effect; it says nothing about
the features of any particular SAE. For real features, use the held-out and
hard-negative baselines in `biolens.eval.probing` (`--held-out-baselines` in
scripts/run_eval.py) and check top-activating examples against reference data
(scripts/inspect_features.py).

Usage:
  python examples/selection_inflation_demo.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc_columns(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUROC of every column of `scores` (n, d) against binary `labels` (n,)."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    ranks = rankdata(scores, axis=0)
    return (ranks[labels == 1].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_of_d_inflation(
    d: int, n_select: int = 200, n_heldout: int = 200, n_trials: int = 200, seed: int = 0
) -> tuple[float, float]:
    """Mean AUROC of the selected feature on the selection set and on held-out data."""
    rng = np.random.default_rng(seed)
    selected, held_out = [], []
    for _ in range(n_trials):
        y_sel = np.repeat([0, 1], n_select // 2)
        y_ho = np.repeat([0, 1], n_heldout // 2)
        x_sel = rng.normal(size=(n_select, d))
        x_ho = rng.normal(size=(n_heldout, d))
        auc_sel = auroc_columns(x_sel, y_sel)
        best = int(np.argmax(auc_sel))
        selected.append(auc_sel[best])
        held_out.append(auroc_columns(x_ho[:, [best]], y_ho)[0])
    return float(np.mean(selected)), float(np.mean(held_out))


def main() -> None:
    print("Pure-noise features, binary label, 100 positives / 100 negatives per set")
    print(f"{'candidates d':>14} {'selected AUROC':>16} {'held-out AUROC':>16}")
    for d in (1, 10, 100, 1000, 10000):
        sel, ho = best_of_d_inflation(d, n_trials=100)
        print(f"{d:>14} {sel:>16.3f} {ho:>16.3f}")
    print("\nThe selected AUROC rises with the number of candidates searched; the held-out")
    print("AUROC stays at chance (0.5), because no feature has any signal.")


if __name__ == "__main__":
    main()
