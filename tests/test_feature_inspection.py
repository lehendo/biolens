"""
Tests for feature-level inspection: pulling top-activating sequences for a
given SAE feature index, and cross-referencing GO probing results.

All tests use synthetic activations/sequences written to a temp ActivationCache
— no network access, no real model downloads.
"""

from __future__ import annotations

import json

import pytest
import torch


@pytest.fixture
def small_cache_and_sae(tmp_path, d_model, d_sae, k):
    """A finalized ActivationCache with known sequences + a matching TopKSAE,
    such that we can hand-verify which sequence should top-activate a feature."""
    from biolens.sae.architectures import SAEConfig, TopKSAE
    from biolens.sae.dictionary import ActivationCache

    cache = ActivationCache(tmp_path / "cache")
    torch.manual_seed(0)

    acts = torch.randn(40, d_model)
    ids = [f"P{i:03d}" for i in range(40)]
    seqs = [f"SEQ{i}" for i in range(40)]
    cache.write_shard(0, acts, ids, sequences=seqs)
    cache.finalize()

    cfg = SAEConfig(d_model=d_model, d_sae=d_sae)
    sae = TopKSAE(cfg, k=k)
    sae.eval()

    return cache, sae


class TestLoadCacheFeatures:
    def test_returns_all_ids_sequences_and_correct_shape(self, small_cache_and_sae):
        from biolens.eval.feature_inspection import load_cache_features

        cache, sae = small_cache_and_sae
        ids, seqs, feat_acts = load_cache_features(cache, sae, device="cpu")

        assert len(ids) == 40
        assert len(seqs) == 40
        assert seqs[0] == "SEQ0"
        assert feat_acts.shape == (40, sae.cfg.d_sae)

    def test_max_seqs_truncates_consistently(self, small_cache_and_sae):
        from biolens.eval.feature_inspection import load_cache_features

        cache, sae = small_cache_and_sae
        ids, seqs, feat_acts = load_cache_features(cache, sae, max_seqs=10, device="cpu")

        assert len(ids) == 10
        assert len(seqs) == 10
        assert feat_acts.shape[0] == 10

    def test_normalizes_before_encoding(self, small_cache_and_sae):
        """Feeding raw (un-normalized) activations directly to sae.encode()
        must NOT match load_cache_features' output — this guards against
        regressing back to the bug where the dashboard skipped normalization."""
        import h5py

        from biolens.eval.feature_inspection import load_cache_features

        cache, sae = small_cache_and_sae
        _, _, feat_acts = load_cache_features(cache, sae, device="cpu")

        # Manually replicate the buggy path: raw activations straight into encode().
        shard_path, _ = next(cache._iter_shards())
        with h5py.File(shard_path, "r") as f:
            raw = torch.from_numpy(f["activations"][:].astype("float32"))
        with torch.no_grad():
            unnormalized_acts = sae.encode(raw)

        assert not torch.allclose(feat_acts, unnormalized_acts), (
            "load_cache_features should differ from encoding raw activations "
            "directly — if this fails, normalization silently isn't happening."
        )

    def test_result_matches_manual_normalize_and_encode(self, small_cache_and_sae):
        """Positive check: load_cache_features' output must equal manually
        normalizing via cache.normalize() then encoding — the correct path."""
        import h5py

        from biolens.eval.feature_inspection import load_cache_features

        cache, sae = small_cache_and_sae
        _, _, feat_acts = load_cache_features(cache, sae, device="cpu")

        shard_path, _ = next(cache._iter_shards())
        with h5py.File(shard_path, "r") as f:
            raw = torch.from_numpy(f["activations"][:].astype("float32"))
        with torch.no_grad():
            expected = sae.encode(cache.normalize(raw))

        assert torch.allclose(feat_acts, expected)

    def test_feature_indices_gives_identical_columns_to_full_matrix(self, small_cache_and_sae):
        """The memory-efficient compacted path (feature_indices=[...]) must return exactly the
        same values as slicing the same columns out of the full matrix, not
        just run without erroring."""
        from biolens.eval.feature_inspection import load_cache_features

        cache, sae = small_cache_and_sae
        requested = [sae.cfg.d_sae - 1, 0, sae.cfg.d_sae // 2]  # deliberately out of order

        _, _, full = load_cache_features(cache, sae, device="cpu")
        _, _, compacted = load_cache_features(
            cache, sae, device="cpu", feature_indices=requested
        )

        assert compacted.shape == (full.shape[0], len(requested))
        for col, feat_idx in enumerate(requested):
            assert torch.allclose(compacted[:, col], full[:, feat_idx]), (
                f"compacted column {col} (SAE feature {feat_idx}) doesn't match "
                f"the full matrix's column {feat_idx}"
            )

    def test_feature_indices_ids_and_sequences_unaffected(self, small_cache_and_sae):
        from biolens.eval.feature_inspection import load_cache_features

        cache, sae = small_cache_and_sae
        ids_full, seqs_full, _ = load_cache_features(cache, sae, device="cpu")
        ids_compact, seqs_compact, _ = load_cache_features(
            cache, sae, device="cpu", feature_indices=[0, 1]
        )
        assert ids_full == ids_compact
        assert seqs_full == seqs_compact


class TestInspectFeature:
    def test_top_examples_are_actually_top_activating(self, d_sae):
        from biolens.eval.feature_inspection import inspect_feature

        n = 20
        feat_acts = torch.zeros(n, d_sae)
        # Make feature 3's activations a strictly increasing ramp so we know
        # exactly which rows should be "top".
        feat_acts[:, 3] = torch.arange(n, dtype=torch.float32)
        ids = [f"P{i}" for i in range(n)]
        seqs = [f"SEQ{i}" for i in range(n)]

        report = inspect_feature(3, ids, seqs, feat_acts, top_k=5)

        assert report.feature_idx == 3
        assert [ex.protein_id for ex in report.top_examples] == [
            "P19", "P18", "P17", "P16", "P15"
        ]
        assert report.max_activation == pytest.approx(19.0)

    def test_activation_rate_and_mean_active(self, d_sae):
        from biolens.eval.feature_inspection import inspect_feature

        feat_acts = torch.zeros(10, d_sae)
        feat_acts[:4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # 4/10 active

        report = inspect_feature(0, [f"P{i}" for i in range(10)], [""] * 10, feat_acts)
        assert report.activation_rate == pytest.approx(0.4)
        assert report.mean_activation_when_active == pytest.approx(2.5)

    def test_out_of_range_feature_raises(self, d_sae):
        from biolens.eval.feature_inspection import inspect_feature

        feat_acts = torch.zeros(5, d_sae)
        with pytest.raises(ValueError, match="out of range"):
            inspect_feature(d_sae, ["P0"] * 5, [""] * 5, feat_acts)
        with pytest.raises(ValueError, match="out of range"):
            inspect_feature(-1, ["P0"] * 5, [""] * 5, feat_acts)

    def test_column_idx_reads_a_compacted_matrix_correctly(self):
        """When feat_acts has been compacted (load_cache_features'
        feature_indices=[...] path), column_idx must select the right
        column while feature_idx keeps labeling the true SAE index in the
        returned report — this is what scripts/inspect_features.py relies
        on."""
        from biolens.eval.feature_inspection import inspect_feature

        n = 10
        # Compacted matrix for requested SAE features [50, 7] — only 2
        # columns, nowhere near d_sae wide, mirroring the real compacted
        # shape load_cache_features(feature_indices=[50, 7]) would produce.
        compacted = torch.zeros(n, 2)
        compacted[:, 1] = torch.arange(n, dtype=torch.float32)  # column 1 = SAE feature 7
        ids = [f"P{i}" for i in range(n)]
        seqs = [""] * n

        report = inspect_feature(7, ids, seqs, compacted, top_k=3, column_idx=1)

        assert report.feature_idx == 7  # true SAE index preserved as the label
        assert report.max_activation == pytest.approx(9.0)
        assert [ex.protein_id for ex in report.top_examples] == ["P9", "P8", "P7"]

    def test_column_idx_out_of_range_raises(self):
        from biolens.eval.feature_inspection import inspect_feature

        compacted = torch.zeros(5, 2)
        with pytest.raises(ValueError, match="out of range"):
            inspect_feature(50, ["P0"] * 5, [""] * 5, compacted, column_idx=2)

    def test_go_annotations_passed_through(self, d_sae):
        from biolens.eval.feature_inspection import inspect_feature

        feat_acts = torch.zeros(5, d_sae)
        report = inspect_feature(
            0, ["P0"] * 5, [""] * 5, feat_acts,
            go_annotations=["GO:0005515 protein binding (auroc=0.900)"],
        )
        assert report.go_annotations == ["GO:0005515 protein binding (auroc=0.900)"]
        assert "GO:0005515" in str(report)

    def test_str_with_no_annotations(self, d_sae):
        from biolens.eval.feature_inspection import inspect_feature

        feat_acts = torch.zeros(5, d_sae)
        report = inspect_feature(0, ["P0"] * 5, [""] * 5, feat_acts)
        assert "none" in str(report)


class TestLoadGoAnnotationsByFeature:
    def test_groups_by_feature_and_sorts_by_auroc(self, tmp_path):
        from biolens.eval.feature_inspection import load_go_annotations_by_feature

        results = [
            {
                "go_id": "GO:0005515", "go_name": "protein binding",
                "best_feature_idx": 42, "single_feature_auroc": 0.75,
            },
            {
                "go_id": "GO:0005886", "go_name": "plasma membrane",
                "best_feature_idx": 42, "single_feature_auroc": 0.95,
            },
            {
                "go_id": "GO:0005737", "go_name": "cytoplasm",
                "best_feature_idx": 7, "single_feature_auroc": 0.80,
            },
        ]
        path = tmp_path / "go_probing_results.json"
        path.write_text(json.dumps(results))

        by_feature = load_go_annotations_by_feature(path)

        assert set(by_feature.keys()) == {42, 7}
        # Feature 42 has two GO terms; higher AUROC (plasma membrane) must come first.
        assert by_feature[42][0].startswith("GO:0005886")
        assert by_feature[42][1].startswith("GO:0005515")
        assert by_feature[7] == ["GO:0005737 cytoplasm (auroc=0.800)"]

    def test_empty_results_file(self, tmp_path):
        from biolens.eval.feature_inspection import load_go_annotations_by_feature

        path = tmp_path / "empty.json"
        path.write_text("[]")
        assert load_go_annotations_by_feature(path) == {}


class TestVerifiedAnnotations:
    """A high single-feature AUROC is only a candidate signal. The
    verified-annotations registry records manual verification outcomes so a
    debunked claim never silently reappears as if it were an established
    finding."""

    def _write_registry(self, tmp_path, features_yaml: str):
        path = tmp_path / "verified.yaml"
        path.write_text(
            "model: esm2_8m\nlayer: 5\nsae_variant: topk\nk: 32\n"
            f"features:\n{features_yaml}"
        )
        return path

    def test_loads_confirmed_and_spurious_entries(self, tmp_path):
        from biolens.eval.feature_inspection import load_verified_annotations

        path = self._write_registry(
            tmp_path,
            """\
  9103:
    status: confirmed
    claimed_go: ["GO:0000159"]
    claimed_concept: "protein phosphatase type 2A complex"
    real_identity: "kinase regulatory subunit"
    evidence_accessions: ["P31383"]
  9862:
    status: spurious
    claimed_go: ["GO:0005892"]
    claimed_concept: "ion channel complex"
    real_identity: "Metal-ion transporter (CorA family)"
    evidence_accessions: ["P00001"]
""",
        )
        registry = load_verified_annotations(path)

        assert registry[9103].status == "confirmed"
        assert registry[9103].real_identity == "kinase regulatory subunit"
        assert registry[9862].status == "spurious"
        assert registry[9862].evidence_accessions == ["P00001"]

    def test_missing_optional_fields_default_sensibly(self, tmp_path):
        from biolens.eval.feature_inspection import load_verified_annotations

        path = self._write_registry(
            tmp_path,
            """\
  42:
    status: plausible
""",
        )
        registry = load_verified_annotations(path)
        assert registry[42].status == "plausible"
        assert registry[42].claimed_go == []
        assert registry[42].evidence_accessions == []

    def test_empty_features_section(self, tmp_path):
        from biolens.eval.feature_inspection import load_verified_annotations

        path = tmp_path / "empty.yaml"
        path.write_text("model: esm2_8m\nlayer: 5\nsae_variant: topk\nk: 32\n")
        assert load_verified_annotations(path) == {}

    def test_multi_claim_entry_keeps_per_claim_verdicts(self, tmp_path):
        from biolens.eval.feature_inspection import load_verified_annotations

        path = self._write_registry(
            tmp_path,
            """\
  1835:
    real_identity: "Homologous-recombination repair factor"
    evidence_accessions: ["P12345", "Q67890"]
    claims:
      - status: spurious
        claimed_go: ["GO:0065003"]
        claimed_concept: "protein-containing complex assembly"
      - status: confirmed
        claimed_go: ["GO:0000724"]
        claimed_concept: "double-strand break repair via homologous recombination"
""",
        )
        entry = load_verified_annotations(path)[1835]

        assert len(entry.claims) == 2
        assert entry.claim_for("GO:0065003").status == "spurious"
        assert entry.claim_for("GO:0000724").status == "confirmed"
        assert entry.claim_for("GO:9999999") is None  # never claimed on this feature
        # feature-level fields are shared by both claims
        assert entry.real_identity == "Homologous-recombination repair factor"
        assert entry.evidence_accessions == ["P12345", "Q67890"]
        # derived legacy fields: union of GO terms, "mixed" status
        assert entry.claimed_go == ["GO:0065003", "GO:0000724"]
        assert entry.status == "mixed"

    def test_multi_claim_with_agreeing_verdicts_keeps_shared_status(self, tmp_path):
        from biolens.eval.feature_inspection import load_verified_annotations

        path = self._write_registry(
            tmp_path,
            """\
  7:
    real_identity: "x"
    claims:
      - {status: spurious, claimed_go: ["GO:A"], claimed_concept: "a"}
      - {status: spurious, claimed_go: ["GO:B"], claimed_concept: "b"}
""",
        )
        assert load_verified_annotations(path)[7].status == "spurious"

    def test_legacy_entry_synthesizes_one_claim(self, tmp_path):
        from biolens.eval.feature_inspection import load_verified_annotations

        path = self._write_registry(
            tmp_path,
            """\
  9103:
    status: confirmed
    claimed_go: ["GO:0000159"]
    claimed_concept: "kinase complex"
    real_identity: "kinase subunit"
""",
        )
        entry = load_verified_annotations(path)[9103]
        assert len(entry.claims) == 1
        assert entry.claim_for("GO:0000159").status == "confirmed"

    def test_same_go_claimed_twice_on_one_feature_raises(self, tmp_path):
        """One (go_id, feature) pair can only carry one verdict -- a duplicate
        would silently let two possibly-conflicting verdicts coexist."""
        from biolens.eval.feature_inspection import load_verified_annotations

        path = self._write_registry(
            tmp_path,
            """\
  5:
    real_identity: "x"
    claims:
      - {status: confirmed, claimed_go: ["GO:A"], claimed_concept: "a"}
      - {status: spurious, claimed_go: ["GO:A"], claimed_concept: "a again"}
""",
        )
        with pytest.raises(ValueError, match="claimed more than once"):
            load_verified_annotations(path)

    def test_empty_or_incomplete_claims_raise(self, tmp_path):
        from biolens.eval.feature_inspection import load_verified_annotations

        empty = self._write_registry(tmp_path, "  5:\n    real_identity: x\n    claims: []\n")
        with pytest.raises(ValueError, match="empty"):
            load_verified_annotations(empty)

        missing = self._write_registry(
            tmp_path, "  5:\n    real_identity: x\n    claims:\n      - {claimed_go: [GO:A]}\n"
        )
        with pytest.raises(ValueError, match="needs both"):
            load_verified_annotations(missing)

    def test_comments_only_registry_file_loads_empty(self, tmp_path):
        """A registry that is just a header comment (yaml.safe_load -> None)
        must load as empty, not crash on None.get."""
        from biolens.eval.feature_inspection import load_verified_annotations

        path = tmp_path / "comments_only.yaml"
        path.write_text("# nothing here yet\n")
        assert load_verified_annotations(path) == {}


class TestInspectFeatureWithVerification:
    def test_verified_confirmed_shown_in_str(self, d_sae):
        from biolens.eval.feature_inspection import VerifiedAnnotation, inspect_feature

        feat_acts = torch.zeros(5, d_sae)
        verified = VerifiedAnnotation(
            status="confirmed",
            claimed_go=["GO:0000159"],
            claimed_concept="protein phosphatase type 2A complex",
            real_identity="kinase regulatory subunit",
            evidence_accessions=["P31383"],
        )
        report = inspect_feature(0, ["P0"] * 5, [""] * 5, feat_acts, verified=verified)

        s = str(report)
        assert "CONFIRMED" in s
        assert "kinase regulatory subunit" in s
        assert "P31383" in s

    def test_verified_spurious_flags_wrong_claim(self, d_sae):
        from biolens.eval.feature_inspection import VerifiedAnnotation, inspect_feature

        feat_acts = torch.zeros(5, d_sae)
        verified = VerifiedAnnotation(
            status="spurious",
            claimed_go=["GO:0005892"],
            claimed_concept="ion channel complex",
            real_identity="Metal-ion transporter (CorA family)",
            evidence_accessions=["P00001"],
        )
        report = inspect_feature(0, ["P0"] * 5, [""] * 5, feat_acts, verified=verified)

        s = str(report)
        assert "SPURIOUS" in s
        assert "WRONG" in s
        assert "Metal-ion transporter" in s

    def test_verified_takes_precedence_over_raw_go_annotations(self, d_sae):
        """When a feature has been verified, the raw AUROC-based GO claim
        must not be shown as if it were still an open/trustworthy candidate —
        the verified status is authoritative."""
        from biolens.eval.feature_inspection import VerifiedAnnotation, inspect_feature

        feat_acts = torch.zeros(5, d_sae)
        verified = VerifiedAnnotation(
            status="spurious",
            claimed_go=["GO:0005892"],
            claimed_concept="ion channel complex",
            real_identity="Metal-ion transporter (CorA family)",
        )
        report = inspect_feature(
            0, ["P0"] * 5, [""] * 5, feat_acts,
            go_annotations=["GO:0005892 ion channel complex (auroc=0.99)"],
            verified=verified,
        )

        s = str(report)
        assert "UNVERIFIED" not in s
        assert "SPURIOUS" in s

    def test_unverified_annotation_is_clearly_labeled(self, d_sae):
        """A feature with only raw AUROC-based GO annotations (no verified
        entry) must be labeled UNVERIFIED — never presented as fact."""
        from biolens.eval.feature_inspection import inspect_feature

        feat_acts = torch.zeros(5, d_sae)
        report = inspect_feature(
            0, ["P0"] * 5, [""] * 5, feat_acts,
            go_annotations=["GO:0005892 ion channel complex (auroc=0.99)"],
        )

        s = str(report)
        assert "UNVERIFIED" in s
        assert "GO:0005892" in s
