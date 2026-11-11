"""
Tests for the standardized evaluation suite.

All tests use synthetic data — no model downloads or real GO annotations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ── Reconstruction metrics ────────────────────────────────────────────────────

class TestReconstructionMetrics:
    def test_perfect_reconstruction(self, topk_sae, d_model):
        """An SAE that perfectly reconstructs should have FVE = 1 (or very close)."""
        # Train briefly so reconstruction is decent
        import torch.optim as optim

        from biolens.eval.reconstruction import compute_reconstruction_metrics
        optimizer = optim.Adam(topk_sae.parameters(), lr=1e-2)
        x = torch.randn(512, d_model)
        for _ in range(200):
            out = topk_sae(x)
            loss = out.l2_loss + 1e-3 * out.auxiliary_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            topk_sae.normalize_decoder()

        metrics = compute_reconstruction_metrics(
            topk_sae, x, normalize=False, batch_size=256, device="cpu"
        )
        # After training on its own data, FVE should be reasonable
        assert metrics.frac_variance_explained > 0.3, (
            f"FVE too low: {metrics.frac_variance_explained}"
        )

    def test_untrained_metrics_shape(self, topk_sae, random_activations):
        from biolens.eval.reconstruction import compute_reconstruction_metrics

        metrics = compute_reconstruction_metrics(
            topk_sae, random_activations, normalize=False, batch_size=32, device="cpu"
        )
        assert 0.0 <= metrics.dead_fraction <= 1.0
        assert 0.0 <= metrics.mean_l0_frac <= 1.0
        assert metrics.n_samples == len(random_activations)
        assert len(metrics.per_feature_act_rate) == topk_sae.cfg.d_sae

    def test_mean_l0_matches_k(self, topk_sae, random_activations):
        from biolens.eval.reconstruction import compute_reconstruction_metrics

        metrics = compute_reconstruction_metrics(
            topk_sae, random_activations, normalize=False, batch_size=32, device="cpu"
        )
        # mean_l0 should be ≤ k (some might be zero due to ReLU clamp)
        assert metrics.mean_l0 <= topk_sae.k + 0.5

    def test_str_representation(self, topk_sae, random_activations):
        from biolens.eval.reconstruction import compute_reconstruction_metrics

        metrics = compute_reconstruction_metrics(
            topk_sae, random_activations, normalize=False, batch_size=32, device="cpu"
        )
        s = str(metrics)
        assert "FVE=" in s
        assert "L0=" in s

    def test_from_cache(self, topk_sae, tmp_cache_dir, d_model):
        from biolens.eval.reconstruction import compute_reconstruction_metrics
        from biolens.sae.dictionary import ActivationCache

        cache = ActivationCache(tmp_cache_dir)
        acts = torch.randn(100, d_model)
        ids = [f"P{i}" for i in range(100)]
        cache.write_shard(0, acts, ids)
        cache.finalize()

        metrics = compute_reconstruction_metrics(
            topk_sae, cache, batch_size=32, device="cpu", normalize=True
        )
        assert metrics.n_samples == 100


# ── GO probing ────────────────────────────────────────────────────────────────

class TestGOProbing:
    def test_probe_with_informative_features(self, d_sae):
        """Plant an informative feature: if feature_5 > 0 → label is 1."""
        from biolens.eval.probing import probe_go_terms

        N = 300
        d_sae_val = 64
        torch.manual_seed(1)

        Z = torch.randn(N, d_sae_val).abs()
        ids = [f"P{i:05d}" for i in range(N)]

        # Feature 5 perfectly predicts the label
        labels = (Z[:, 5] > Z[:, 5].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z,
            protein_ids=ids,
            go_labels=go_labels,
            go_names={"GO:0000001": "test_term"},
            min_positives=20,
            max_go_terms=10,
            train_fraction=0.8,
        )
        assert len(report.results) >= 1
        best = report.top_results(1)[0]
        assert best.single_feature_auroc > 0.75, (
            f"Informative feature not detected: AUROC={best.single_feature_auroc:.3f}"
        )
        assert best.best_feature_idx == 5

    def test_probe_random_features(self, d_sae, n_seqs):
        """Random features vs. random labels: mean AUROC should be ~0.5."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(2)
        N = 200
        d_sae_val = 32
        Z = torch.randn(N, d_sae_val)
        ids = [f"P{i:05d}" for i in range(N)]

        rng = np.random.default_rng(2)
        go_labels = {
            pid: ({"GO:0001234"} if rng.random() < 0.4 else set())
            for pid in ids
        }

        report = probe_go_terms(
            feature_acts=Z,
            protein_ids=ids,
            go_labels=go_labels,
            min_positives=30,
        )
        if report.results:
            mean_auroc = np.mean([r.single_feature_auroc for r in report.results])
            # Random features: reflected AUROC should be near 0.5..0.65
            assert mean_auroc < 0.85, (
                f"Random features achieved suspiciously high AUROC: {mean_auroc}"
            )

    def test_min_positives_filter(self):
        """GO terms with fewer than min_positives examples should be skipped."""
        from biolens.eval.probing import probe_go_terms

        N = 100
        Z = torch.randn(N, 32)
        ids = [f"P{i}" for i in range(N)]
        # Only 3 positives — below default min_positives=50
        go_labels = {ids[i]: {"GO:RARE"} for i in range(3)}

        report = probe_go_terms(
            feature_acts=Z,
            protein_ids=ids,
            go_labels=go_labels,
            min_positives=50,
        )
        assert report.n_go_terms_tested == 0

    def test_compute_multivariate_auroc_false_skips_the_fit(self):
        """A caller that only needs best_feature_idx shouldn't pay for the
        multivariate LogisticRegression fit at all: multivariate_auroc should come
        back NaN, not just be ignored after being computed anyway."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(5)
        N = 200
        Z = torch.randn(N, 32).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        labels = (Z[:, 3] > Z[:, 3].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_multivariate_auroc=False,
        )
        assert len(report.results) >= 1
        for r in report.results:
            assert np.isnan(r.multivariate_auroc)
            # single_feature_auroc/best_feature_idx must still be computed —
            # only the multivariate fit is skipped.
            assert 0.0 <= r.single_feature_auroc <= 1.0

    def test_report_fraction_above_threshold(self):
        from biolens.eval.probing import GOProbingReport, GOProbingResult

        results = [
            GOProbingResult("GO:0000001", "t1", 60, 200, 0.82, 0, 0.85),
            GOProbingResult("GO:0000002", "t2", 60, 200, 0.65, 1, 0.70),
            GOProbingResult("GO:0000003", "t3", 60, 200, 0.55, 2, 0.60),
        ]
        report = GOProbingReport(results, 3, "esm2_8m", 5, "topk")
        # [0.82, 0.65, 0.55] — only 0.82 is ≥ 0.70
        assert report.fraction_above_auroc(0.65) == pytest.approx(2 / 3)
        assert report.fraction_above_auroc(0.70) == pytest.approx(1 / 3)
        assert report.fraction_above_auroc(0.80) == pytest.approx(1 / 3)
        assert report.fraction_above_auroc(0.90) == pytest.approx(0.0)

    def test_vectorized_auroc_vs_sklearn(self):
        """Vectorized AUROC should match sklearn within floating-point tolerance."""
        from sklearn.metrics import roc_auc_score

        from biolens.eval.probing import _vectorized_auroc

        rng = np.random.default_rng(42)
        N, D = 200, 20
        scores = rng.standard_normal((N, D))
        labels = (rng.random(N) > 0.6).astype(int)

        vec_aurocs = _vectorized_auroc(scores, labels)
        for i in range(D):
            sk_auroc = roc_auc_score(labels, scores[:, i])
            # Both reflect below 0.5
            sk_reflected = max(sk_auroc, 1.0 - sk_auroc)
            assert abs(vec_aurocs[i] - sk_reflected) < 1e-6, (
                f"Feature {i}: vectorized={vec_aurocs[i]:.6f}, sklearn={sk_reflected:.6f}"
            )

    def test_str_output(self):
        from biolens.eval.probing import GOProbingReport, GOProbingResult

        results = [GOProbingResult("GO:0000001", "cytoplasm", 100, 500, 0.78, 42, 0.81)]
        report = GOProbingReport(results, 1, "esm2_8m", 5, "topk")
        s = str(report)
        assert "GO probing" in s
        assert "GO:0000001" in s
        assert "0.78" in s

    def test_multivariate_auroc_handles_wildly_scaled_features(self):
        """multivariate_auroc's LogisticRegression must standardize features
        before fitting: unscaled JumpReLU SAE activations can cause repeated
        `lbfgs failed to converge` warnings and very long runtimes, unlike a TopK SAE,
        whose fixed per-example sparsity tends to keep features well-conditioned. This test constructs
        features with deliberately wild per-feature scale differences (the
        actual mechanism suspected to cause the real failure) and only
        checks the fit still converges to a sensible, non-degenerate result
        — it does not attempt to detect a convergence *warning* directly,
        since that's an sklearn implementation detail, not part of this
        function's contract."""
        from biolens.eval.probing import probe_go_terms

        N = 300
        d_sae_val = 64
        rng = np.random.default_rng(3)

        Z = rng.standard_normal((N, d_sae_val)).astype(np.float64)
        # Wildly different per-feature scales — some tiny, some huge —
        # mimicking unscaled, dense JumpReLU-style activations rather than
        # TopK's uniform, sparse pattern.
        scale = 10.0 ** rng.uniform(-6, 6, size=d_sae_val)
        Z = torch.from_numpy((Z * scale).astype(np.float32))

        labels = (Z[:, 5] > Z[:, 5].median()).numpy().astype(int)
        ids = [f"P{i:05d}" for i in range(N)]
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z,
            protein_ids=ids,
            go_labels=go_labels,
            go_names={"GO:0000001": "test_term"},
            min_positives=20,
            max_go_terms=10,
            train_fraction=0.8,
        )
        assert len(report.results) == 1
        multi_auroc = report.results[0].multivariate_auroc
        assert not np.isnan(multi_auroc), "multivariate_auroc should not be NaN"
        assert 0.0 <= multi_auroc <= 1.0
        # An informative, if oddly-scaled, feature set should still separate
        # well once standardized — a value near chance (~0.5) would suggest
        # the fit degraded badly on the scale disparity instead of handling it.
        assert multi_auroc > 0.7, (
            f"multivariate_auroc unexpectedly low on wildly-scaled features: {multi_auroc}"
        )

    def test_multivariate_auroc_converges_when_n_train_near_d_sae(self):
        """Standardization alone (previous test) does not cover the case where
        n_train/d_sae is near 1 (e.g. d_sae=16384 against ~16000 training rows): at the
        n~=p interpolation threshold unregularized/weakly-regularized (C=1.0)
        logistic regression has no well-conditioned finite-norm solution regardless of
        feature scaling. This test reproduces that ratio at
        small scale (n_train ~= 0.85 * d_sae) with well-scaled features (so
        conditioning is not the confound) and checks the fit converges
        without a ConvergenceWarning after lowering C to 0.1."""
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        from biolens.eval.probing import probe_go_terms

        d_sae_val = 200
        N = 240  # train_fraction=0.8 -> n_train=192, n_train/d_sae_val = 0.96
        rng = np.random.default_rng(7)

        Z_np = rng.standard_normal((N, d_sae_val)).astype(np.float32)
        true_w = rng.standard_normal(d_sae_val).astype(np.float32)
        logits = Z_np @ true_w
        labels = (logits > np.median(logits)).astype(int)
        Z = torch.from_numpy(Z_np)

        ids = [f"P{i:05d}" for i in range(N)]
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = probe_go_terms(
                feature_acts=Z,
                protein_ids=ids,
                go_labels=go_labels,
                go_names={"GO:0000001": "test_term"},
                min_positives=20,
                max_go_terms=10,
                train_fraction=0.8,
            )

        convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        assert not convergence_warnings, (
            f"Expected no ConvergenceWarning at n_train~=d_sae, got: "
            f"{[str(w.message) for w in convergence_warnings]}"
        )
        assert len(report.results) == 1
        multi_auroc = report.results[0].multivariate_auroc
        assert not np.isnan(multi_auroc)
        assert 0.0 <= multi_auroc <= 1.0


class TestMaxGoTermsEligibleTracking:
    """n_go_terms_eligible reports the pre-cap eligible count alongside the tested
    count, so it is visible when max_go_terms truncates the ranked list."""

    def _make_go_labels(self, ids: list[str], n_terms: int, n_pos_per_term: int) -> dict:
        go_labels: dict[str, set[str]] = {pid: set() for pid in ids}
        for t in range(n_terms):
            go_id = f"GO:{t:07d}"
            for pid in ids[t * n_pos_per_term : (t + 1) * n_pos_per_term]:
                go_labels[pid].add(go_id)
        return go_labels

    def test_reports_eligible_greater_than_tested_when_capped(self, d_sae, n_seqs):
        from biolens.eval.probing import probe_go_terms

        N = 400
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        # 8 distinct eligible GO terms, each with enough positives, but cap at 3.
        go_labels = self._make_go_labels(ids, n_terms=8, n_pos_per_term=25)

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels, go_names={},
            min_positives=20, max_go_terms=3, train_fraction=0.8,
        )
        assert report.n_go_terms_eligible == 8
        assert report.n_go_terms_tested == 3
        assert report.n_go_terms_eligible > report.n_go_terms_tested
        assert "CAPPED" in str(report)

    def test_eligible_equals_tested_when_not_capped(self, d_sae, n_seqs):
        from biolens.eval.probing import probe_go_terms

        N = 400
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        go_labels = self._make_go_labels(ids, n_terms=3, n_pos_per_term=25)

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels, go_names={},
            min_positives=20, max_go_terms=500, train_fraction=0.8,
        )
        assert report.n_go_terms_eligible == 3
        assert report.n_go_terms_tested == 3
        assert "CAPPED" not in str(report)



class TestHeldOutBaselines:
    def test_default_behavior_unchanged_when_flag_off(self):
        """compute_held_out_baselines defaults to False — must not alter any
        existing single_feature_auroc numbers or leave the new fields unset
        as anything but None (backward compatibility)."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(1)
        N, d_sae = 300, 32
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        labels = (Z[:, 5] > Z[:, 5].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10,
        )
        r = report.top_results(1)[0]
        assert r.held_out_auroc is None
        assert r.hard_negative_held_out_auroc is None
        assert r.n_hard_negatives is None
        assert r.auroc_inflation is None

    def test_held_out_auroc_near_naive_for_strong_real_signal(self):
        """A genuinely informative feature (real separation, ample N) should
        show little winner's-curse inflation: held_out_auroc should track
        single_feature_auroc closely, not collapse toward 0.5."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(3)
        N, d_sae = 2000, 16
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        # Feature 2 is strongly, cleanly informative — not just noise that
        # happened to win a small-sample argmax.
        labels = (Z[:, 2] > Z[:, 2].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
            random_state=7,
        )
        r = report.top_results(1)[0]
        assert r.best_feature_idx == 2
        assert r.held_out_auroc is not None
        assert r.held_out_auroc > 0.9, (
            f"Strong real signal should survive held-out estimation: {r.held_out_auroc}"
        )
        assert r.auroc_inflation is not None
        assert r.auroc_inflation < 0.1, (
            f"Little winner's-curse inflation expected for strong real signal: "
            f"{r.auroc_inflation}"
        )

    def test_held_out_auroc_debiases_pure_noise_selection(self):
        """With zero real signal (labels independent of features, matching
        the winners_curse_simulation.py setup) and many candidate features,
        the naive single_feature_auroc (argmax over many noisy estimates)
        should be inflated above 0.5, while held_out_auroc — estimated on
        data disjoint from the selection step — should sit much closer to
        0.5. This is the core claim of the held-out baseline,
        exercised end-to-end here."""
        from biolens.eval.probing import probe_go_terms

        rng = np.random.default_rng(11)
        N, d_sae = 4000, 500
        Z = torch.from_numpy(rng.standard_normal((N, d_sae)).astype(np.float32))
        ids = [f"P{i:05d}" for i in range(N)]
        # Labels independent of every feature column by construction.
        label_arr = (rng.random(N) < 0.3).astype(int)
        go_labels = {
            pid: ({"GO:NULLSIGNAL"} if label_arr[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=1, compute_held_out_baselines=True,
            held_out_fraction=0.5, random_state=13,
        )
        r = report.results[0]
        assert r.single_feature_auroc > 0.55, (
            "Selection over 500 pure-noise features should show real inflation "
            f"(winner's curse): got {r.single_feature_auroc}"
        )
        assert r.held_out_auroc is not None
        assert r.held_out_auroc < r.single_feature_auroc, (
            "Held-out estimate must be lower than the inflated naive selection "
            f"estimate: held_out={r.held_out_auroc}, naive={r.single_feature_auroc}"
        )
        assert abs(r.held_out_auroc - 0.5) < 0.15, (
            f"Held-out AUROC should sit near the true null (0.5): {r.held_out_auroc}"
        )

    def test_hard_negative_auroc_none_without_go_dag(self):
        """No go_dag passed -> hard_negative_held_out_auroc must stay None,
        never silently computed against the wrong (or no) negative set."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(4)
        N, d_sae = 500, 16
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        labels = (Z[:, 1] > Z[:, 1].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
        )
        r = report.top_results(1)[0]
        assert r.hard_negative_held_out_auroc is None

    def test_hard_negative_auroc_exposes_structural_confounding(self):
        """The core W3 scenario: a feature that has learned a BROAD class
        ("membrane") rather than the SPECIFIC claimed concept
        ("ion channel", a sibling of "transporter" under
        "membrane") should ace the standard held-out AUROC (easy negatives:
        everything non-membrane) but collapse toward 0.5 on the hard-negative
        AUROC (negatives restricted to sibling membrane proteins) — the
        failure mode the hard-negative baseline is designed to catch."""
        from biolens.data.uniprot import GODag
        from biolens.eval.probing import probe_go_terms

        dag = GODag(
            parents={
                "GO:ACHR": {"GO:MEMBRANE"},
                "GO:TRANSPORTER": {"GO:MEMBRANE"},
            },
            children={"GO:MEMBRANE": {"GO:ACHR", "GO:TRANSPORTER"}},
        )

        rng = np.random.default_rng(21)
        # Large enough that the held-out hard-negative subset (~test_fraction
        # * held_out_fraction * n_transporter) has enough examples for a
        # statistically reliable AUROC estimate, not just directionally right.
        n_achr, n_transporter, n_other = 300, 3000, 8000
        N = n_achr + n_transporter + n_other
        d_sae = 8
        Z = np.zeros((N, d_sae), dtype=np.float32)

        # Feature 0 fires for ANY membrane protein (broad class), regardless
        # of which specific membrane sub-type — it has NOT learned the
        # channel-specific concept, only "membrane-ness".
        Z[: n_achr + n_transporter, 0] = rng.uniform(0.5, 1.5, n_achr + n_transporter)
        Z[n_achr + n_transporter :, 0] = rng.uniform(0.0, 0.1, n_other)
        Z[:, 1:] = rng.standard_normal((N, d_sae - 1)) * 0.01  # noise filler

        ids = [f"P{i:05d}" for i in range(N)]
        go_labels: dict[str, set[str]] = {}
        for i in range(N):
            if i < n_achr:
                go_labels[ids[i]] = {"GO:ACHR"}
            elif i < n_achr + n_transporter:
                go_labels[ids[i]] = {"GO:TRANSPORTER"}
            else:
                go_labels[ids[i]] = set()

        report = probe_go_terms(
            feature_acts=torch.from_numpy(Z), protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
            go_dag=dag, min_hard_negatives=10, random_state=5, held_out_fraction=0.5,
        )
        achr_result = next(r for r in report.results if r.go_id == "GO:ACHR")

        assert achr_result.best_feature_idx == 0
        assert achr_result.held_out_auroc is not None
        assert achr_result.held_out_auroc > 0.8, (
            "Standard held-out AUROC (mostly easy negatives) should look great — "
            f"this is exactly the false confidence the hard-negative test exposes: "
            f"{achr_result.held_out_auroc}"
        )
        assert achr_result.hard_negative_held_out_auroc is not None
        assert achr_result.hard_negative_held_out_auroc < 0.65, (
            "Hard-negative AUROC (sibling transporters as negatives) should "
            "collapse toward 0.5 — the feature can't distinguish the specific "
            f"claimed concept from its structural siblings: "
            f"{achr_result.hard_negative_held_out_auroc}"
        )

    def test_hard_negative_auroc_none_when_too_few_hard_negatives(self):
        """If fewer than min_hard_negatives sibling-annotated proteins land
        in the held-out split, report None rather than an unreliable
        estimate from a handful of examples."""
        from biolens.data.uniprot import GODag
        from biolens.eval.probing import probe_go_terms

        dag = GODag(
            parents={"GO:RARE_SIBLING": {"GO:PARENT"}, "GO:TARGET": {"GO:PARENT"}},
            children={"GO:PARENT": {"GO:RARE_SIBLING", "GO:TARGET"}},
        )
        torch.manual_seed(9)
        N, d_sae = 500, 8
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        labels = (Z[:, 0] > Z[:, 0].median()).numpy().astype(int)
        go_labels: dict[str, set[str]] = {}
        for i, pid in enumerate(ids):
            if labels[i] == 1:
                go_labels[pid] = {"GO:TARGET"}
            elif i % 200 == 0:  # only a handful of sibling-annotated negatives
                go_labels[pid] = {"GO:RARE_SIBLING"}
            else:
                go_labels[pid] = set()

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
            go_dag=dag, min_hard_negatives=50,
        )
        r = next(r for r in report.results if r.go_id == "GO:TARGET")
        assert r.hard_negative_held_out_auroc is None
        assert r.n_hard_negatives is not None
        assert r.n_hard_negatives < 50

    def test_auroc_inflation_property(self):
        from biolens.eval.probing import GOProbingResult

        r = GOProbingResult(
            "GO:0000001", "t", 100, 500, 0.85, 3, 0.80,
            held_out_auroc=0.60,
        )
        assert r.auroc_inflation == pytest.approx(0.25)

    def test_auroc_inflation_none_without_held_out(self):
        from biolens.eval.probing import GOProbingResult

        r = GOProbingResult("GO:0000001", "t", 100, 500, 0.85, 3, 0.80)
        assert r.auroc_inflation is None
