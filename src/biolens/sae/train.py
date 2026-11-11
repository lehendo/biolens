"""
SAE training loop.

Supports:
  - Streaming from an ActivationCache (cluster mode — activations pre-cached)
  - Training directly from a list of tensors (dev/testing mode)
  - WandB logging, checkpointing, cosine LR decay, gradient clipping
  - Decoder column renormalization after each step

All SAE variants share this trainer; the variant-specific loss is computed
inside SAE.forward() and returned in SAEOutput.auxiliary_loss.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from biolens.sae.architectures import BaseSAE, SAEConfig, TopKSAE, make_sae
from biolens.sae.dictionary import ActivationCache

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # SAE variant — must match the sae passed to SAETrainer
    variant: str = "topk"
    # Optimization
    n_steps: int = 10_000
    batch_size: int = 4096
    lr: float = 2e-4
    lr_warmup_steps: int = 500
    lr_decay_start_fraction: float = 0.8
    grad_clip: float = 1.0
    # Aux loss weight (TopK); ignored by other variants
    aux_fraction: float = 1e-3
    # Normalization (applied before feeding to SAE — handled by ActivationCache)
    normalize_activations: bool = True
    # Logging and checkpointing
    wandb_project: str | None = "biolens"
    run_name: str | None = None
    checkpoint_dir: Path | None = None
    save_every: int = 2000
    eval_every: int = 500
    # Device
    device: str = "cuda"
    # For reproducibility
    seed: int = 42
    # Extra tags/config passed to wandb
    extra_config: dict = field(default_factory=dict)


class SAETrainer:
    """Trains any BaseSAE on activations from an ActivationCache or tensor iterator."""

    def __init__(self, sae: BaseSAE, cfg: TrainingConfig) -> None:
        self.sae = sae.to(cfg.device)
        self.cfg = cfg
        self._wandb_run: Any | None = None  # wandb.sdk.wandb_run.Run; wandb is optional

    def train_from_cache(self, cache: ActivationCache, preload: bool = True) -> BaseSAE:
        """Main entry point for cluster training (activations pre-extracted).

        preload=True (default): loads all vectors into RAM once, then shuffles
        in-memory each epoch. ~700 MB for 557K×320 — always fits in cluster alloc
        and avoids per-step HDF5 I/O on slow shared storage.
        preload=False: streams from disk (use only if dataset doesn't fit in RAM).
        """
        logger.info(
            "Training %s on %d cached vectors (d=%d), %d steps",
            self.cfg.variant,
            len(cache),
            cache.d_model,
            self.cfg.n_steps,
        )
        if preload:
            mb = len(cache) * cache.d_model * 4 / 1e6
            logger.info("Preloading %.0f MB into RAM (%d vectors)...", mb, len(cache))
            chunks = [
                b.cpu()
                for b in cache.iter_batches(
                    batch_size=8192,
                    normalize=self.cfg.normalize_activations,
                    device="cpu",
                )
            ]
            all_acts = torch.cat(chunks, dim=0)
            logger.info("Preloaded: %s", tuple(all_acts.shape))
            data_iter = _infinite_shuffled_iter(all_acts, self.cfg.batch_size, self.cfg.device)
            return self._train_loop(data_iter)

        data_iter = _infinite_iter(
            cache.iter_batches(
                batch_size=self.cfg.batch_size,
                normalize=self.cfg.normalize_activations,
                device=self.cfg.device,
            )
        )
        return self._train_loop(data_iter)

    def train_from_tensors(self, activations: Tensor) -> BaseSAE:
        """Dev/testing mode: train directly from an in-memory tensor."""
        activations = activations.float()
        if self.cfg.normalize_activations:
            mean = activations.mean(0, keepdim=True)
            scale = (activations - mean).norm(dim=1).mean().clamp(min=1e-8)
            activations = (activations - mean) / scale

        def _iter() -> Iterator[Tensor]:
            N = len(activations)
            while True:
                perm = torch.randperm(N)
                for start in range(0, N, self.cfg.batch_size):
                    yield activations[perm[start : start + self.cfg.batch_size]].to(
                        self.cfg.device
                    )

        return self._train_loop(_iter())

    def _train_loop(self, data_iter: Iterator[Tensor]) -> BaseSAE:
        torch.manual_seed(self.cfg.seed)
        self._init_wandb()

        optimizer = optim.Adam(self.sae.parameters(), lr=self.cfg.lr, betas=(0.9, 0.999))
        scheduler = _make_lr_scheduler(optimizer, self.cfg)

        self.sae.train()
        step = 0
        t0 = time.time()

        for batch in data_iter:
            if step >= self.cfg.n_steps:
                break

            batch = batch.to(self.cfg.device)
            output = self.sae(batch)

            if isinstance(self.sae, TopKSAE):
                loss = output.l2_loss + self.cfg.aux_fraction * output.auxiliary_loss
            else:
                loss = output.l2_loss + output.auxiliary_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Decoder column normalization — must happen after every optimizer step
            self.sae.normalize_decoder()

            step += 1

            if step % self.cfg.eval_every == 0 or step == 1:
                self._log_step(step, output, loss, optimizer, t0)
                t0 = time.time()

            if (
                self.cfg.checkpoint_dir is not None
                and step % self.cfg.save_every == 0
            ):
                self._save_checkpoint(step)

        self.sae.eval()
        if self.cfg.checkpoint_dir is not None:
            self._save_checkpoint(step, final=True)
        if self._wandb_run is not None:
            self._wandb_run.finish()

        logger.info("Training complete (%d steps).", step)
        return self.sae

    def _log_step(
        self, step: int, output, loss: Tensor, optimizer, t0: float
    ) -> None:
        with torch.no_grad():
            var_total = output.reconstruction.detach().var().clamp(min=1e-8)
            fve = 1.0 - output.l2_loss.item() / var_total.item()

            z = output.feature_acts
            mean_l0 = (z > 0).float().sum(-1).mean().item()

        dead_frac = (
            self.sae.dead_fraction
            if hasattr(self.sae, "dead_fraction")
            else float("nan")
        )
        lr_now = optimizer.param_groups[0]["lr"]
        steps_per_sec = self.cfg.eval_every / max(time.time() - t0, 1e-9)

        metrics = {
            "train/loss": loss.item(),
            "train/l2_loss": output.l2_loss.item(),
            "train/aux_loss": output.auxiliary_loss.item(),
            "train/frac_variance_explained": fve,
            "train/mean_l0": mean_l0,
            "train/dead_fraction": dead_frac,
            "train/lr": lr_now,
            "train/steps_per_sec": steps_per_sec,
            "step": step,
        }
        logger.info(
            "step=%d loss=%.4f fve=%.3f L0=%.1f dead=%.3f",
            step, loss.item(), fve, mean_l0, dead_frac,
        )
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        assert self.cfg.checkpoint_dir is not None
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fname = "final.pt" if final else f"step_{step:08d}.pt"
        path = self.cfg.checkpoint_dir / fname
        torch.save(
            {
                "step": step,
                "model_state_dict": self.sae.state_dict(),
                "config": self.cfg,
                # Architecture metadata needed to reconstruct the SAE without the
                # caller having to separately guess d_model/expansion_factor/k —
                # see build_sae_from_checkpoint(). Variant-specific hyperparams
                # like k aren't part of state_dict (plain Python attrs, not
                # buffers/params), so they must be saved explicitly here.
                "sae_cfg": self.sae.cfg,
                "sae_extra": {
                    "variant": self.cfg.variant,
                    "k": getattr(self.sae, "k", None),
                },
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

    def _init_wandb(self) -> None:
        if self.cfg.wandb_project is None:
            return
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                config={
                    "variant": self.cfg.variant,
                    "d_model": self.sae.cfg.d_model,
                    "d_sae": self.sae.cfg.d_sae,
                    "expansion_factor": self.sae.cfg.expansion_factor,
                    **self.cfg.extra_config,
                },
            )
        except ImportError:
            logger.warning("wandb not installed; training without logging.")


def _make_lr_scheduler(optimizer: optim.Optimizer, cfg: TrainingConfig) -> LambdaLR:
    """Linear warmup then cosine decay."""
    warmup = cfg.lr_warmup_steps
    decay_start = int(cfg.n_steps * cfg.lr_decay_start_fraction)
    total = cfg.n_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        if step < decay_start:
            return 1.0
        # Cosine decay from decay_start to total
        progress = (step - decay_start) / max(total - decay_start, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159265 * progress)).item())

    return LambdaLR(optimizer, lr_lambda)


def _infinite_iter(it: Iterator) -> Iterator:
    """Restart an iterator indefinitely."""
    while True:
        yield from it


def _infinite_shuffled_iter(acts: Tensor, batch_size: int, device: str) -> Iterator[Tensor]:
    """Yield randomly shuffled batches from an in-memory tensor indefinitely."""
    N = len(acts)
    while True:
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            yield acts[perm[start : start + batch_size]].to(device)


def load_checkpoint(path: str | Path, sae: BaseSAE) -> tuple[BaseSAE, int]:
    """Load a checkpoint into an existing SAE, returning (sae, step).

    If the checkpoint carries variant-specific hyperparameters that live
    outside state_dict (e.g. TopKSAE.k is a plain Python attribute, not a
    buffer or parameter), they're restored onto `sae` too — otherwise a
    caller who reconstructs the SAE shell with a different k than the one
    used at training time would silently get the wrong sparsity, with no
    error to signal it.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae.load_state_dict(ckpt["model_state_dict"])

    extra = ckpt.get("sae_extra", {})
    ckpt_k = extra.get("k")
    if ckpt_k is not None and hasattr(sae, "k") and sae.k != ckpt_k:
        logger.warning(
            "Checkpoint was trained with k=%d but the supplied SAE was constructed "
            "with k=%d; overriding sae.k to match the checkpoint.",
            ckpt_k, sae.k,
        )
        sae.k = ckpt_k

    return sae, ckpt["step"]


def build_sae_from_checkpoint(
    path: str | Path,
    device: str = "cpu",
    variant: str | None = None,
    d_model: int | None = None,
    expansion_factor: int | None = None,
    k: int | None = None,
) -> tuple[BaseSAE, int]:
    """
    Reconstruct a trained SAE directly from a checkpoint file, inferring its
    architecture (d_model, expansion_factor, variant, k) from metadata saved
    at training time — the caller doesn't need to separately know or guess
    the shape used during training.

    Checkpoints written before this metadata existed (no "sae_cfg" key) fall
    back to the variant/d_model/expansion_factor/k arguments given here,
    which are then required — there's no way to recover the shape from a
    bare state_dict alone without risking a silent mismatch.

    Returns:
        (sae, step) — sae is in eval() mode on `device`.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae_cfg: SAEConfig | None = ckpt.get("sae_cfg")
    extra = ckpt.get("sae_extra", {})
    ckpt_variant: str | None = extra.get("variant") or variant

    if sae_cfg is None:
        if d_model is None or expansion_factor is None or ckpt_variant is None:
            raise ValueError(
                "This checkpoint predates saved architecture metadata "
                "(no 'sae_cfg' key) — pass variant, d_model, and "
                "expansion_factor explicitly so the SAE shape can be "
                "reconstructed correctly."
            )
        sae_cfg = SAEConfig(d_model=d_model, expansion_factor=expansion_factor)

    if ckpt_variant is None:
        raise ValueError(
            "Could not determine SAE variant from checkpoint metadata "
            "(no 'sae_extra.variant' entry) — pass variant= explicitly."
        )

    sae: BaseSAE
    if ckpt_variant == "topk":
        sae = TopKSAE(sae_cfg, k=extra.get("k") or k or 32)
    else:
        sae = make_sae(
            ckpt_variant, d_model=sae_cfg.d_model, expansion_factor=sae_cfg.expansion_factor
        )

    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    return sae, ckpt["step"]
