"""Tests for the cluster bootstrap of mean AUROC inflation."""

from __future__ import annotations

import numpy as np
import pytest


class TestBootstrapMeanAurocInflation:
    def test_ci_excludes_zero_for_clearly_positive_inflation(self):
        """A real, non-degenerate positive mean inflation should get a CI
        that excludes zero, given enough independent-ish clusters."""
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        rng = np.random.default_rng(0)
        n = 300
        auroc_inflation = rng.normal(loc=0.05, scale=0.02, size=n)
        feature_idx = np.arange(n)  # every row its own cluster

        result = bootstrap_mean_auroc_inflation(auroc_inflation, feature_idx, random_state=1)

        assert result.mean == pytest.approx(np.mean(auroc_inflation))
        lo, hi = result.ci
        assert lo > 0, f"95% CI should exclude 0 for clearly positive inflation: [{lo}, {hi}]"
        assert result.n_rows == n
        assert result.n_clusters == n

    def test_ci_includes_zero_for_noise_centered_at_zero(self):
        """Inflation values centered at zero should get a CI spanning zero."""
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        rng = np.random.default_rng(3)
        n = 300
        auroc_inflation = rng.normal(loc=0.0, scale=0.02, size=n)
        feature_idx = np.arange(n)

        result = bootstrap_mean_auroc_inflation(auroc_inflation, feature_idx, random_state=2)
        lo, hi = result.ci
        assert lo < 0 < hi, f"No real effect — CI should span 0: [{lo}, {hi}]"

    def test_clustering_widens_ci_under_intra_cluster_correlation(self):
        """If many rows share a feature_idx and that shared feature drives
        correlated inflation values, clustering must produce a wider CI than
        treating every row as its own independent cluster."""
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        rng = np.random.default_rng(3)
        n_clusters = 15
        rows_per_cluster = 10
        cluster_effects = rng.normal(loc=0.05, scale=0.03, size=n_clusters)
        feature_idx = np.repeat(np.arange(n_clusters), rows_per_cluster)
        auroc_inflation = np.repeat(cluster_effects, rows_per_cluster) + rng.normal(
            scale=0.005, size=n_clusters * rows_per_cluster
        )
        independent_idx = np.arange(len(auroc_inflation))

        clustered = bootstrap_mean_auroc_inflation(auroc_inflation, feature_idx, random_state=4)
        unclustered = bootstrap_mean_auroc_inflation(
            auroc_inflation, independent_idx, random_state=4
        )

        clustered_width = clustered.ci[1] - clustered.ci[0]
        unclustered_width = unclustered.ci[1] - unclustered.ci[0]
        assert clustered_width > unclustered_width, (
            f"Clustered CI ({clustered_width:.4f}) should be wider than treating "
            f"correlated rows as independent ({unclustered_width:.4f})"
        )
        assert clustered.n_clusters == n_clusters
        assert unclustered.n_clusters == len(auroc_inflation)

    def test_mismatched_lengths_raise(self):
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        with pytest.raises(ValueError):
            bootstrap_mean_auroc_inflation(np.array([0.1, 0.2]), np.array([0]))

    def test_empty_input_raises(self):
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        with pytest.raises(ValueError):
            bootstrap_mean_auroc_inflation(np.array([]), np.array([]))

    def test_str_output_includes_key_fields(self):
        from biolens.eval.statistics import InflationBootstrapResult

        result = InflationBootstrapResult(
            mean=0.05, ci=(0.02, 0.08), n_bootstrap_successful=1950, n_rows=200, n_clusters=40,
        )
        s = str(result)
        assert "0.0500" in s
        assert "n_clusters=40" in s
