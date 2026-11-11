"""
Cluster-robust inference for summaries over per-concept probing results.

A single SAE feature can be the best-scoring feature for several GO terms (or
concepts) at once, so rows of a probing table are not independent observations.
Bootstrapping individual rows would treat them as independent and understate the
variance of any summary. The bootstrap here resamples whole clusters (feature
indices) with replacement instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InflationBootstrapResult:
    """Cluster-bootstrap confidence interval for a mean AUROC-inflation summary."""

    mean: float
    ci: tuple[float, float]
    n_bootstrap_successful: int
    n_rows: int
    n_clusters: int

    def __str__(self) -> str:
        lo, hi = self.ci
        return (
            f"mean_auroc_inflation={self.mean:.4f}  "
            f"95% cluster-bootstrap CI=[{lo:.4f}, {hi:.4f}]  "
            f"(n_rows={self.n_rows}, n_clusters={self.n_clusters}, "
            f"{self.n_bootstrap_successful} successful resamples)"
        )


def bootstrap_mean_auroc_inflation(
    auroc_inflation: np.ndarray,
    feature_idx: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> InflationBootstrapResult:
    """
    Cluster (block) bootstrap CI for a mean AUROC-inflation value, clustering on
    the feature index.

    Args:
        auroc_inflation: (n_rows,) per-GO-term (or per-concept) inflation values,
            i.e. best-feature AUROC minus held-out AUROC, e.g. from
            go_probing_results.json.
        feature_idx: (n_rows,) each row's best feature index, the clustering variable.
        n_bootstrap: Number of cluster-bootstrap resamples.
        confidence: Confidence level (default 95%).
        random_state: Bootstrap RNG seed.

    Returns:
        InflationBootstrapResult with the point estimate and the cluster-bootstrap
        interval. With fewer than 10 distinct clusters the interval is still
        computed but a warning is logged, since it may be unreliable.
    """
    auroc_inflation = np.asarray(auroc_inflation, dtype=float)
    feature_idx = np.asarray(feature_idx)
    n_rows = len(auroc_inflation)
    if len(feature_idx) != n_rows:
        raise ValueError("auroc_inflation and feature_idx must be the same length")
    if n_rows == 0:
        raise ValueError("Cannot bootstrap a mean with zero rows")

    unique_clusters = np.unique(feature_idx)
    n_clusters = len(unique_clusters)
    if n_clusters < 10:
        logger.warning(
            "Only %d distinct feature clusters: the bootstrap interval may be unreliable.",
            n_clusters,
        )

    rows_by_cluster = {c: np.where(feature_idx == c)[0] for c in unique_clusters}
    rng = np.random.default_rng(random_state)

    means = []
    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in sampled_clusters])
        means.append(float(np.mean(auroc_inflation[idx])))

    alpha = 1.0 - confidence
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return InflationBootstrapResult(
        mean=float(np.mean(auroc_inflation)),
        ci=(float(lo), float(hi)),
        n_bootstrap_successful=len(means),
        n_rows=n_rows,
        n_clusters=n_clusters,
    )
