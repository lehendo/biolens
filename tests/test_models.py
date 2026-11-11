"""
Tests for the model registry and ESM2 adapter.

Fast tests use the registry and config loading only (no model weights).
Integration tests (marked integration) load actual ESM2 8M weights.
"""

from __future__ import annotations

import pytest
import torch

# ── ModelRegistry ─────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_list_supported_models(self):
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        models = registry.list_models(supported_only=True)
        assert "esm2_8m" in models
        assert "esm2_35m" in models

    def test_list_all_models(self):
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        all_models = registry.list_models(supported_only=False)
        assert "evo2_7b" in all_models
        assert "geneformer_12l" in all_models

    def test_family_filter(self):
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        protein = registry.list_models(family="protein")
        assert all(
            registry.get_config(m).family == "protein" for m in protein
        )

    def test_get_config(self):
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        cfg = registry.get_config("esm2_8m")
        assert cfg.num_layers == 6
        assert cfg.hidden_dim == 320
        assert cfg.family == "protein"
        assert cfg.supported is True

    def test_unknown_model_raises(self):
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        with pytest.raises(KeyError, match="Unknown model"):
            registry.get_config("nonexistent_model_xyz")

    def test_unsupported_model_raises_on_load(self):
        """geneformer_12l is still supported: false (single-cell); see
        TestEvo2Adapter below for the Evo 2 adapter's coverage."""
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        with pytest.raises(NotImplementedError, match="(?i)not yet implemented"):
            registry.load_model("geneformer_12l")

    def test_hyenadna_still_unimplemented(self):
        """Only evo2_* genomic models route to a real adapter; hyenadna_* must
        still raise clearly."""
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        # hyenadna_large_1m is still supported: false at the config level too,
        # so load_model raises before ever reaching _instantiate's genomic branch.
        with pytest.raises(NotImplementedError):
            registry.load_model("hyenadna_large_1m")


# ── ESM2Model (integration) ───────────────────────────────────────────────────

@pytest.mark.integration
class TestESM2Integration:
    """These tests download ESM2 8M (~30MB) — requires network access."""

    @pytest.fixture(scope="class")
    def esm2_model(self):
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        return registry.load_model("esm2_8m", device="cpu")

    def test_load_model(self, esm2_model):
        assert esm2_model.config.name == "esm2_8m"
        assert esm2_model.hidden_dim == 320

    def test_mean_pooling_shape(self, esm2_model):
        seqs = ["MKTAYIAKQRQISFVK", "ACDEFGHIKLMNPQRSTVWY"]
        acts = esm2_model.get_activations(seqs, layer=5, pooling="mean")
        assert acts.shape == (2, 320)
        assert acts.dtype == torch.float32

    def test_cls_pooling_shape(self, esm2_model):
        seqs = ["MKTAYIAKQRQISFVK"]
        acts = esm2_model.get_activations(seqs, layer=5, pooling="cls")
        assert acts.shape == (1, 320)

    def test_token_pooling_shape(self, esm2_model):
        seqs = ["ACDE"]  # 4 residues
        acts = esm2_model.get_activations(seqs, layer=5, pooling="none")
        # Shape: (1, L_padded, 320) — L_padded includes CLS + residues + EOS
        assert acts.shape[0] == 1
        assert acts.shape[2] == 320

    def test_all_layers_extraction(self, esm2_model):
        seqs = ["MKTAYIAKQRQISFVK"]
        all_acts = esm2_model.get_activations_all_layers(seqs, layers=[0, 2, 5])
        assert set(all_acts.keys()) == {0, 2, 5}
        for layer, acts in all_acts.items():
            assert acts.shape == (1, 320)

    def test_layer_out_of_range_raises(self, esm2_model):
        with pytest.raises(ValueError, match="out of range"):
            esm2_model.get_activations(["ACDE"], layer=99)

    def test_layer_negative_raises(self, esm2_model):
        with pytest.raises(ValueError, match="out of range"):
            esm2_model.get_activations(["ACDE"], layer=-1)

    def test_activations_finite(self, esm2_model):
        seqs = ["ACDEFGHIKLMNPQRSTVWY"] * 4
        acts = esm2_model.get_activations(seqs, layer=3, pooling="mean")
        assert torch.isfinite(acts).all()

    def test_output_on_cpu(self, esm2_model):
        """get_activations should always return CPU tensors regardless of model device."""
        seqs = ["ACDE"]
        acts = esm2_model.get_activations(seqs, layer=0, pooling="mean")
        assert acts.device == torch.device("cpu")

    def test_single_string_input(self, esm2_model):
        """Should accept a single string, not just a list."""
        acts = esm2_model.get_activations("ACDE", layer=0, pooling="mean")
        assert acts.shape == (1, 320)

    def test_consistent_across_calls(self, esm2_model):
        """Same input → same output (model is in eval mode, no dropout)."""
        seq = ["MKTAYIAKQRQISFVK"]
        a1 = esm2_model.get_activations(seq, layer=5, pooling="mean")
        a2 = esm2_model.get_activations(seq, layer=5, pooling="mean")
        assert torch.allclose(a1, a2)

    def test_layers_differ(self, esm2_model):
        """Different layers should give different representations."""
        seqs = ["MKTAYIAKQRQISFVK"]
        acts_0 = esm2_model.get_activations(seqs, layer=0, pooling="mean")
        acts_5 = esm2_model.get_activations(seqs, layer=5, pooling="mean")
        assert not torch.allclose(acts_0, acts_5)


# ── Activation hooks ──────────────────────────────────────────────────────────

class TestActivationHooks:
    def test_residual_stream_capture(self):
        """ResidualStreamCapture correctly captures output of a submodule."""
        import torch.nn as nn

        from biolens.models.activation_hooks import ResidualStreamCapture

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        model = TinyModel()
        x = torch.randn(2, 4)

        with ResidualStreamCapture(model.linear) as cap:
            _ = model(x)

        assert cap.output is not None
        assert cap.output.shape == (2, 4)

    def test_hook_removed_after_context(self):
        """Hook should be removed after the context manager exits."""
        import torch.nn as nn

        from biolens.models.activation_hooks import ResidualStreamCapture
        layer = nn.Linear(4, 4)

        with ResidualStreamCapture(layer):
            assert len(layer._forward_hooks) == 1
        assert len(layer._forward_hooks) == 0


# ── Evo2Model (mocked — the real `evo2` package needs CUDA/Flash Attention, ──
# unavailable in this environment; see src/biolens/models/evo2.py for the
# honest "untested on this machine" note and why a real cluster run is the
# first true validation) ──────────────────────────────────────────────────────

class TestEvo2Adapter:
    @pytest.fixture
    def evo2_config(self):
        from biolens.models.registry import ModelConfig

        return ModelConfig(
            name="evo2_7b", family="genomic", hf_name="arcinstitute/evo2_7b",
            tokenizer_name="arcinstitute/evo2_7b", num_layers=32, hidden_dim=4096,
            max_seq_len=8192, supported=True,
        )

    @pytest.fixture
    def fake_evo2_module(self, monkeypatch):
        """Inject a fake `evo2` module into sys.modules so Evo2Model._load's
        deferred `from evo2 import Evo2` succeeds without the real package."""
        import sys
        import types

        fake_module = types.ModuleType("evo2")

        class FakeTokenizer:
            def tokenize(self, sequence: str) -> list[int]:
                return [{"A": 0, "C": 1, "G": 2, "T": 3}[c] for c in sequence]

        class FakeInnerModel:
            def named_modules(self):
                yield "blocks.0.mlp.l3", object()
                yield "blocks.1.mlp.l3", object()
                yield "embedding_layer", object()

        class FakeEvo2:
            def __init__(self, name: str):
                self.name = name
                self.tokenizer = FakeTokenizer()
                self.model = FakeInnerModel()
                self.last_call = None

            def __call__(self, input_ids, return_embeddings=False, layer_names=None):
                self.last_call = {
                    "input_ids": input_ids, "return_embeddings": return_embeddings,
                    "layer_names": layer_names,
                }
                L = input_ids.shape[1]
                embeddings = {
                    name: torch.randn(1, L, 4096) for name in (layer_names or [])
                }
                return None, embeddings

        fake_module.Evo2 = FakeEvo2
        monkeypatch.setitem(sys.modules, "evo2", fake_module)
        return fake_module

    def test_load_calls_evo2_with_model_name(self, evo2_config, fake_evo2_module):
        from biolens.models.evo2 import Evo2Model

        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        model._ensure_loaded()
        assert model._model.name == "evo2_7b"

    def test_import_error_has_actionable_message_when_evo2_missing(
        self, evo2_config, monkeypatch
    ):
        import builtins
        import sys

        from biolens.models.evo2 import Evo2Model

        monkeypatch.delitem(sys.modules, "evo2", raising=False)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "evo2":
                raise ImportError("No module named 'evo2'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        with pytest.raises(ImportError, match="pip install evo2"):
            model._ensure_loaded()

    def test_mean_pooling_shape(self, evo2_config, fake_evo2_module):
        from biolens.models.evo2 import Evo2Model

        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        acts = model.get_activations(["ACGTACGT", "ACGT"], layer=0, pooling="mean")
        assert acts.shape == (2, 4096)

    def test_single_string_input(self, evo2_config, fake_evo2_module):
        from biolens.models.evo2 import Evo2Model

        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        acts = model.get_activations("ACGTACGT", layer=0, pooling="mean")
        assert acts.shape == (1, 4096)

    def test_uses_correct_layer_name_in_forward_call(self, evo2_config, fake_evo2_module):
        from biolens.models.evo2 import Evo2Model

        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        model.get_activations("ACGT", layer=5, pooling="mean")
        assert model._model.last_call["layer_names"] == ["blocks.5.mlp.l3"]
        assert model._model.last_call["return_embeddings"] is True

    def test_cls_pooling_raises(self, evo2_config, fake_evo2_module):
        from biolens.models.evo2 import Evo2Model

        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        with pytest.raises(ValueError, match="CLS"):
            model.get_activations("ACGT", layer=0, pooling="cls")

    def test_layer_out_of_range_raises(self, evo2_config, fake_evo2_module):
        from biolens.models.evo2 import Evo2Model

        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        with pytest.raises(ValueError):
            model.get_activations("ACGT", layer=999, pooling="mean")

    def test_none_pooling_returns_full_sequence(self, evo2_config, fake_evo2_module):
        from biolens.models.evo2 import Evo2Model

        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        acts = model.get_activations("ACGTACGT", layer=0, pooling="none")
        # Single sequence -> stacked to (1, L, D); L == len(sequence) here
        # since FakeTokenizer is 1 token per base.
        assert acts.shape == (1, 8, 4096)

    def test_list_available_layer_names(self, evo2_config, fake_evo2_module):
        from biolens.models.evo2 import Evo2Model

        model = Evo2Model(evo2_config, device="cpu", dtype=torch.float32)
        names = model.list_available_layer_names()
        assert "blocks.0.mlp.l3" in names
        assert "embedding_layer" in names

    def test_registry_routes_evo2_to_evo2_model(self, fake_evo2_module):
        from biolens.models.registry import ModelRegistry

        registry = ModelRegistry()
        model = registry.load_model("evo2_7b", device="cpu")
        from biolens.models.evo2 import Evo2Model
        assert isinstance(model, Evo2Model)
