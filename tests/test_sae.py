"""
Tests for SAE architectures and training loop.

All tests run on CPU with tiny dimensions — no GPU or model downloads required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ── TopKSAE ──────────────────────────────────────────────────────────────────

class TestTopKSAE:
    def test_forward_shape(self, topk_sae, random_activations):
        output = topk_sae(random_activations)
        N, D = random_activations.shape
        assert output.reconstruction.shape == (N, D)
        assert output.feature_acts.shape == (N, topk_sae.cfg.d_sae)

    def test_sparsity(self, topk_sae, random_activations):
        output = topk_sae(random_activations)
        # Each sample should have at most k active features
        n_active = (output.feature_acts > 0).sum(dim=-1)
        assert (n_active <= topk_sae.k).all(), "More than k features active"

    def test_feature_acts_nonnegative(self, topk_sae, random_activations):
        output = topk_sae(random_activations)
        assert (output.feature_acts >= 0).all(), "Negative feature activations"

    def test_l2_loss_positive(self, topk_sae, random_activations):
        output = topk_sae(random_activations)
        assert output.l2_loss.item() >= 0

    def test_decoder_norm(self, topk_sae):
        """Decoder columns should be unit-norm after normalization."""
        topk_sae.normalize_decoder()
        norms = topk_sae.W_dec.data.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_decode_roundtrip_shape(self, topk_sae, random_activations):
        z = topk_sae.encode(random_activations)
        x_hat = topk_sae.decode(z)
        assert x_hat.shape == random_activations.shape

    def test_encode_consistency(self, topk_sae, random_activations):
        """encode() should give same feature_acts as forward()."""
        z_encode = topk_sae.encode(random_activations)
        output = topk_sae(random_activations)
        assert torch.allclose(z_encode, output.feature_acts)

    def test_auxiliary_loss_finite(self, topk_sae, random_activations):
        output = topk_sae(random_activations)
        assert torch.isfinite(output.auxiliary_loss)

    def test_dead_fraction_initial(self, topk_sae):
        """Initially all features appear alive (EMA initialized to 0.1)."""
        # With EMA initialized to 0.1 and dead_threshold=1e-4, none should be dead yet
        assert topk_sae.dead_fraction == 0.0

    def test_gradient_flows(self, topk_sae, random_activations):
        output = topk_sae(random_activations)
        loss = output.l2_loss + 1e-3 * output.auxiliary_loss
        loss.backward()
        # Check that W_enc and W_dec have gradients
        assert topk_sae.W_enc.grad is not None
        assert topk_sae.W_dec.grad is not None
        assert topk_sae.b_enc.grad is not None
        assert topk_sae.b_dec.grad is not None

    def test_normalize_decoder_called_after_step(self, topk_sae, random_activations):
        """Decoder norms should be 1 after normalize_decoder()."""
        # Manually corrupt norms
        with torch.no_grad():
            topk_sae.W_dec.data *= 5.0
        topk_sae.normalize_decoder()
        norms = topk_sae.W_dec.data.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_k_equals_dsae(self, d_model):
        """k == d_sae should activate all features (no TopK filtering)."""
        from biolens.sae.architectures import SAEConfig, TopKSAE
        cfg = SAEConfig(d_model=d_model, d_sae=64)
        sae = TopKSAE(cfg, k=64)
        x = torch.randn(8, d_model)
        output = sae(x)
        # All positive pre-activations should be active
        n_active = (output.feature_acts > 0).sum(dim=-1)
        assert (n_active <= 64).all()

    def test_reconstruction_improves_with_training(self, d_model):
        """After a few gradient steps, reconstruction loss should decrease."""
        from biolens.sae.architectures import SAEConfig, TopKSAE
        cfg = SAEConfig(d_model=d_model, d_sae=d_model * 8)
        sae = TopKSAE(cfg, k=8)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

        torch.manual_seed(42)
        x = torch.randn(256, d_model)

        initial_loss = sae(x).l2_loss.item()

        for _ in range(50):
            output = sae(x)
            loss = output.l2_loss + 1e-3 * output.auxiliary_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

        final_loss = sae(x).l2_loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )


# ── SAEConfig ─────────────────────────────────────────────────────────────────

class TestSAEConfig:
    def test_d_sae_auto(self):
        from biolens.sae.architectures import SAEConfig
        cfg = SAEConfig(d_model=128, expansion_factor=8)
        assert cfg.d_sae == 1024

    def test_d_sae_override(self):
        from biolens.sae.architectures import SAEConfig
        cfg = SAEConfig(d_model=128, d_sae=512)
        assert cfg.d_sae == 512


# ── make_sae factory ──────────────────────────────────────────────────────────

class TestMakeSAE:
    def test_topk(self, d_model):
        from biolens.sae.architectures import TopKSAE, make_sae
        sae = make_sae("topk", d_model=d_model, expansion_factor=4, k=8)
        assert isinstance(sae, TopKSAE)

    def test_unknown_variant(self, d_model):
        from biolens.sae.architectures import make_sae
        with pytest.raises(ValueError, match="Unknown SAE variant"):
            make_sae("unknown", d_model=d_model)

    def test_unimplemented_variants_raise(self, d_model):
        from biolens.sae.architectures import make_sae
        with pytest.raises(NotImplementedError):
            make_sae("jumprelu", d_model=d_model)
        with pytest.raises(NotImplementedError):
            make_sae("gated", d_model=d_model)


# ── ActivationCache ───────────────────────────────────────────────────────────

class TestActivationCache:
    def test_write_and_read(self, tmp_cache_dir, d_model):
        from biolens.sae.dictionary import ActivationCache
        cache = ActivationCache(tmp_cache_dir)

        acts = torch.randn(50, d_model)
        ids = [f"P{i:05d}" for i in range(50)]
        cache.write_shard(0, acts, ids)

        assert len(cache) == 50
        assert cache.d_model == d_model
        assert cache.n_shards == 1

    def test_multiple_shards(self, tmp_cache_dir, d_model):
        from biolens.sae.dictionary import ActivationCache
        cache = ActivationCache(tmp_cache_dir)

        for shard_id in range(3):
            acts = torch.randn(20, d_model)
            ids = [f"P{shard_id}_{i}" for i in range(20)]
            cache.write_shard(shard_id, acts, ids)

        assert len(cache) == 60
        assert cache.n_shards == 3

    def test_iter_batches(self, tmp_cache_dir, d_model):
        from biolens.sae.dictionary import ActivationCache
        cache = ActivationCache(tmp_cache_dir)

        acts = torch.randn(100, d_model)
        ids = [f"P{i}" for i in range(100)]
        cache.write_shard(0, acts, ids, sequences=["A" * 20] * 100)
        cache.finalize()

        batches = list(cache.iter_batches(batch_size=32, normalize=True, device="cpu"))
        total = sum(b.shape[0] for b in batches)
        assert total == 100
        # Each batch has correct feature dimension
        for b in batches:
            assert b.shape[1] == d_model

    def test_finalize_computes_norm(self, tmp_cache_dir, d_model):

        from biolens.sae.dictionary import ActivationCache
        cache = ActivationCache(tmp_cache_dir)
        cache.write_shard(0, torch.randn(50, d_model), [f"P{i}" for i in range(50)])
        manifest = cache.finalize()
        assert "norm" in manifest
        assert "mean" in manifest["norm"]
        assert "scale" in manifest["norm"]
        assert manifest["norm"]["n_total"] == 50

    def test_finalize_multi_shard_matches_single_pass_reference(self, tmp_cache_dir, d_model):
        """finalize() streams shards in two passes (one for the mean, one for
        the centered-norm scale) to avoid materializing every shard
        simultaneously — see its docstring. None of the other
        finalize tests use more than one shard, so none of them would catch a
        bug in the streaming accumulation logic itself — this test writes
        several shards and checks the result against a straightforward
        single-pass numpy computation over the concatenated ground truth."""
        from biolens.sae.dictionary import ActivationCache

        torch.manual_seed(0)
        shards = [torch.randn(37, d_model), torch.randn(41, d_model), torch.randn(23, d_model)]
        cache = ActivationCache(tmp_cache_dir)
        for i, shard in enumerate(shards):
            cache.write_shard(i, shard, [f"P{i}_{j}" for j in range(shard.shape[0])])
        manifest = cache.finalize()

        # float16-quantized ground truth (write_shard stores as float16 — see
        # module docstring — so the reference must match that, not the
        # original float32 tensors, to compare like with like).
        reference = torch.cat(shards, dim=0).half().float().numpy()
        expected_mean = reference.mean(axis=0)
        expected_scale = float(np.linalg.norm(reference - expected_mean, axis=1).mean())

        assert manifest["norm"]["n_total"] == sum(s.shape[0] for s in shards)
        np.testing.assert_allclose(
            manifest["norm"]["mean"], expected_mean, atol=1e-3, rtol=1e-3,
        )
        assert manifest["norm"]["scale"] == pytest.approx(expected_scale, rel=1e-3)

    def test_atomic_write(self, tmp_cache_dir, d_model):
        """No .tmp files should remain after write_shard."""
        from biolens.sae.dictionary import ActivationCache
        cache = ActivationCache(tmp_cache_dir)
        cache.write_shard(0, torch.randn(10, d_model), [f"P{i}" for i in range(10)])
        tmp_files = list(tmp_cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Stale .tmp files: {tmp_files}"

    def test_iter_batches_without_finalize_raises(self, tmp_cache_dir, d_model):
        from biolens.sae.dictionary import ActivationCache
        cache = ActivationCache(tmp_cache_dir)
        cache.write_shard(0, torch.randn(10, d_model), [f"P{i}" for i in range(10)])
        with pytest.raises(RuntimeError, match="finalize"):
            list(cache.iter_batches(normalize=True))

    def test_get_norm_stats_matches_finalize(self, tmp_cache_dir, d_model):
        from biolens.sae.dictionary import ActivationCache
        cache = ActivationCache(tmp_cache_dir)
        cache.write_shard(0, torch.randn(50, d_model), [f"P{i}" for i in range(50)])
        manifest = cache.finalize()

        stats = cache.get_norm_stats()
        assert stats["mean"].shape == (d_model,)
        assert stats["scale"] == pytest.approx(manifest["norm"]["scale"])

    def test_normalize_matches_iter_batches_normalization(self, tmp_cache_dir, d_model):
        """cache.normalize(raw) must produce exactly what iter_batches(normalize=True)
        would yield for the same raw vectors — any drift here means dashboard/
        inspection tooling that calls normalize() directly would silently feed
        the SAE activations from a different distribution than it was trained on."""
        from biolens.sae.dictionary import ActivationCache

        cache = ActivationCache(tmp_cache_dir)
        torch.manual_seed(0)
        acts = torch.randn(64, d_model)
        cache.write_shard(0, acts, [f"P{i}" for i in range(64)])
        cache.finalize()

        via_normalize = cache.normalize(acts.clone())
        via_iter = torch.cat(
            list(cache.iter_batches(batch_size=64, normalize=True, device="cpu")), dim=0
        )
        # iter_batches stores as float16 then upcasts; allow float16 precision.
        assert torch.allclose(via_normalize, via_iter, atol=1e-2)

    def test_normalize_without_finalize_raises(self, tmp_cache_dir, d_model):
        from biolens.sae.dictionary import ActivationCache
        cache = ActivationCache(tmp_cache_dir)
        cache.write_shard(0, torch.randn(10, d_model), [f"P{i}" for i in range(10)])
        with pytest.raises(RuntimeError, match="finalize"):
            cache.normalize(torch.randn(5, d_model))

    def test_encode_matches_manual_normalize_then_sae_encode(self, tmp_cache_dir, d_model, topk_sae):
        """cache.encode(sae, acts) must be exactly equivalent to
        sae.encode(cache.normalize(acts)). This bundling exists so a call site
        can't feed raw activations to an SAE trained on normalized data by
        forgetting one of the two steps."""
        from biolens.sae.dictionary import ActivationCache

        cache = ActivationCache(tmp_cache_dir)
        torch.manual_seed(1)
        cache.write_shard(0, torch.randn(64, d_model), [f"P{i}" for i in range(64)])
        cache.finalize()

        raw_acts = torch.randn(10, d_model)
        via_bundled = cache.encode(topk_sae, raw_acts.clone())
        via_manual = topk_sae.encode(cache.normalize(raw_acts.clone()))
        assert torch.equal(via_bundled, via_manual)

    def test_encode_differs_from_unnormalized_encode(self, tmp_cache_dir, d_model, topk_sae):
        """Sanity check that the bug this bundling prevents is real and
        detectable: encoding raw (un-normalized) activations must NOT match
        cache.encode()'s output whenever the cache's normalization stats are
        non-trivial (non-zero mean or non-unit scale)."""
        from biolens.sae.dictionary import ActivationCache

        cache = ActivationCache(tmp_cache_dir)
        torch.manual_seed(2)
        # A large, non-zero mean shift makes normalization's effect unmistakable.
        shard = torch.randn(64, d_model) + 5.0
        cache.write_shard(0, shard, [f"P{i}" for i in range(64)])
        cache.finalize()

        raw_acts = torch.randn(10, d_model) + 5.0
        normalized_z = cache.encode(topk_sae, raw_acts.clone())
        unnormalized_z = topk_sae.encode(raw_acts.clone())
        assert not torch.equal(normalized_z, unnormalized_z)


# ── SAETrainer ────────────────────────────────────────────────────────────────

class TestSAETrainer:
    def test_train_from_tensors(self, d_model, d_sae, k):
        from biolens.sae.architectures import SAEConfig, TopKSAE
        from biolens.sae.train import SAETrainer, TrainingConfig

        cfg = SAEConfig(d_model=d_model, d_sae=d_sae)
        sae = TopKSAE(cfg, k=k)

        train_cfg = TrainingConfig(
            n_steps=10,
            batch_size=32,
            lr=1e-3,
            wandb_project=None,
            device="cpu",
        )
        trainer = SAETrainer(sae, train_cfg)

        x = torch.randn(256, d_model)
        trained = trainer.train_from_tensors(x)
        assert trained is sae  # in-place

    def test_checkpoint_save_load(self, tmp_cache_dir, d_model, d_sae, k):
        from biolens.sae.architectures import SAEConfig, TopKSAE
        from biolens.sae.train import SAETrainer, TrainingConfig, load_checkpoint

        cfg = SAEConfig(d_model=d_model, d_sae=d_sae)
        sae = TopKSAE(cfg, k=k)

        train_cfg = TrainingConfig(
            n_steps=5,
            batch_size=32,
            lr=1e-3,
            wandb_project=None,
            checkpoint_dir=tmp_cache_dir,
            device="cpu",
        )
        trainer = SAETrainer(sae, train_cfg)
        x = torch.randn(64, d_model)
        trainer.train_from_tensors(x)

        final_ckpt = tmp_cache_dir / "final.pt"
        assert final_ckpt.exists()

        # Load into fresh SAE
        sae2 = TopKSAE(cfg, k=k)
        sae2, step = load_checkpoint(final_ckpt, sae2)
        assert step == 5
        # Weights should match
        assert torch.allclose(sae.W_enc.data, sae2.W_enc.data)

    def test_load_checkpoint_restores_k_when_shell_mismatched(
        self, tmp_cache_dir, d_model, d_sae
    ):
        """
        TopKSAE.k is a plain Python attribute, not a state_dict buffer/param.
        If a caller reconstructs the SAE shell with the wrong k (e.g. forgot
        the training run used a non-default k), load_checkpoint must correct
        it rather than silently leaving the wrong sparsity in place.
        """
        from biolens.sae.architectures import SAEConfig, TopKSAE
        from biolens.sae.train import SAETrainer, TrainingConfig, load_checkpoint

        cfg = SAEConfig(d_model=d_model, d_sae=d_sae)
        trained_k = 4
        sae = TopKSAE(cfg, k=trained_k)

        train_cfg = TrainingConfig(
            n_steps=3, batch_size=32, lr=1e-3, wandb_project=None,
            checkpoint_dir=tmp_cache_dir, device="cpu",
        )
        SAETrainer(sae, train_cfg).train_from_tensors(torch.randn(64, d_model))

        # Caller builds the shell with the DEFAULT k (32), not knowing training used 4.
        wrong_shell = TopKSAE(cfg, k=32)
        assert wrong_shell.k == 32

        loaded, step = load_checkpoint(tmp_cache_dir / "final.pt", wrong_shell)
        assert loaded.k == trained_k, "load_checkpoint should override k to match training"

    def test_build_sae_from_checkpoint_infers_architecture(
        self, tmp_cache_dir, d_model, d_sae, k
    ):
        """New checkpoints carry sae_cfg/sae_extra — build_sae_from_checkpoint
        should reconstruct the exact architecture with no arguments beyond the path."""
        from biolens.sae.architectures import SAEConfig, TopKSAE
        from biolens.sae.train import (
            SAETrainer,
            TrainingConfig,
            build_sae_from_checkpoint,
        )

        cfg = SAEConfig(d_model=d_model, d_sae=d_sae)
        sae = TopKSAE(cfg, k=k)
        train_cfg = TrainingConfig(
            n_steps=3, batch_size=32, lr=1e-3, wandb_project=None,
            checkpoint_dir=tmp_cache_dir, device="cpu",
        )
        SAETrainer(sae, train_cfg).train_from_tensors(torch.randn(64, d_model))

        loaded, step = build_sae_from_checkpoint(tmp_cache_dir / "final.pt")
        assert step == 3
        assert loaded.cfg.d_model == d_model
        assert loaded.cfg.d_sae == d_sae
        assert loaded.k == k
        assert torch.allclose(sae.W_enc.data, loaded.W_enc.data)

    def test_build_sae_from_checkpoint_requires_args_for_legacy_checkpoint(
        self, tmp_cache_dir, d_model, d_sae, k
    ):
        """Checkpoints saved before sae_cfg/sae_extra existed must still load,
        but only if the caller supplies the shape explicitly."""
        from biolens.sae.architectures import SAEConfig, TopKSAE
        from biolens.sae.train import build_sae_from_checkpoint

        cfg = SAEConfig(d_model=d_model, d_sae=d_sae)
        sae = TopKSAE(cfg, k=k)
        legacy_path = tmp_cache_dir / "legacy.pt"
        # Simulate a pre-fix checkpoint: no sae_cfg / sae_extra keys at all.
        torch.save(
            {"step": 7, "model_state_dict": sae.state_dict(), "config": None},
            legacy_path,
        )

        with pytest.raises(ValueError, match="predates saved architecture metadata"):
            build_sae_from_checkpoint(legacy_path)

        loaded, step = build_sae_from_checkpoint(
            legacy_path, variant="topk", d_model=d_model,
            expansion_factor=cfg.expansion_factor, k=k,
        )
        assert step == 7
        assert loaded.cfg.d_model == d_model
        assert torch.allclose(sae.W_enc.data, loaded.W_enc.data)

    def test_build_sae_from_checkpoint_requires_variant_even_with_sae_cfg(
        self, tmp_cache_dir, d_model, d_sae, k
    ):
        """Defensive edge case: a checkpoint could in principle carry sae_cfg
        without sae_extra.variant (e.g. hand-edited, or a future save path
        bug) — must raise a clear error rather than passing None into
        make_sae() and producing a confusing 'Unknown SAE variant: None'."""
        from biolens.sae.architectures import SAEConfig, TopKSAE
        from biolens.sae.train import build_sae_from_checkpoint

        cfg = SAEConfig(d_model=d_model, d_sae=d_sae)
        sae = TopKSAE(cfg, k=k)
        path = tmp_cache_dir / "no_variant.pt"
        torch.save(
            {
                "step": 1,
                "model_state_dict": sae.state_dict(),
                "config": None,
                "sae_cfg": cfg,
                "sae_extra": {"k": k},  # variant missing
            },
            path,
        )

        with pytest.raises(ValueError, match="Could not determine SAE variant"):
            build_sae_from_checkpoint(path)

        # Passing variant= explicitly should still work.
        loaded, step = build_sae_from_checkpoint(path, variant="topk")
        assert step == 1
        assert torch.allclose(sae.W_enc.data, loaded.W_enc.data)
