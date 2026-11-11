"""
Reconstruction fidelity metrics for trained SAEs.

Metrics reported:
  frac_variance_explained (R²): 1 - MSE(x, x̂) / Var(x).  Goal: > 0.9.
  mean_l0:         Average number of active features per sample.
  mean_l0_frac:    mean_l0 / d_sae — fraction of features active on average.
  dead_fraction:   Fraction of features that activated for fewer than dead_threshold
                   of the evaluation samples.  Goal: < 0.02.
  per_feature_act_rate: (d_sae,) array — fraction of samples each feature fires on.

These metrics apply identically to every model family and SAE variant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from biolens.sae.architectures import BaseSAE
from biolens.sae.dictionary import ActivationCache

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionMetrics:
    frac_variance_explained: float      # R² — primary quality metric
    mse: float                          # raw MSE
    mean_l0: float                      # average active features per sample
    mean_l0_frac: float                 # mean_l0 / d_sae
    dead_fraction: float                # fraction of features with rate < dead_threshold
    per_feature_act_rate: np.ndarray    # (d_sae,) — activation rate per feature
    n_samples: int

    def __str__(self) -> str:
        return (
            f"FVE={self.frac_variance_explained:.4f}  "
            f"MSE={self.mse:.4f}  "
            f"L0={self.mean_l0:.1f} ({self.mean_l0_frac*100:.2f}%)  "
            f"dead={self.dead_fraction*100:.1f}%  "
            f"n={self.n_samples}"
        )


def compute_reconstruction_metrics(
    sae: BaseSAE,
    activations: Tensor | ActivationCache,
    *,
    normalize: bool,
    batch_size: int = 1024,
    device: str = "cuda",
    dead_threshold: float = 1e-3,
) -> ReconstructionMetrics:
    """
    Compute reconstruction metrics on a held-out set.

    Args:
        sae:            Trained SAE (any variant).
        activations:    Either an (N, D) tensor or an ActivationCache.
        normalize:      Whether to normalize activations before feeding to SAE.
                        Required, no default. Set to True if the SAE was trained on
                        normalized data (normalize_activations=True in
                        biolens/sae/train.py) and `activations` is raw; False only
                        when `activations` is already normalized or came from the
                        same distribution the SAE was trained on. See
                        ActivationCache.encode(), which bundles this step.
        batch_size:     Batch size for GPU eval.
        device:         Where to run eval.
        dead_threshold: Feature is "dead" in eval if its activation rate < this.

    Returns:
        ReconstructionMetrics dataclass.
    """
    sae = sae.to(device)
    sae.eval()

    data_iter = _get_data_iter(activations, batch_size, normalize, device)

    # Streamed accumulation — avoids holding the full (N, D) / (N, d_sae)
    # tensors in memory, since N can be 500K+ and d_sae = D * expansion_factor.
    n_total = 0
    n_elements = 0
    sum_sq_err = 0.0
    sum_x = 0.0
    sum_x_sq = 0.0
    sum_l0 = 0.0
    active_counts: Tensor | None = None
    d_sae: int | None = None

    with torch.no_grad():
        for batch in data_iter:
            batch = batch.to(device)
            output = sae(batch)
            x = batch
            x_hat = output.reconstruction
            z = output.feature_acts

            n_total += x.shape[0]
            n_elements += x.numel()

            sum_sq_err += float(((x - x_hat) ** 2).sum().item())
            sum_x += float(x.sum().item())
            sum_x_sq += float((x**2).sum().item())

            is_active = z > 0
            if active_counts is None:
                d_sae = z.shape[1]
                active_counts = torch.zeros(d_sae, dtype=torch.float64)
            active_counts += is_active.sum(0).double().cpu()
            sum_l0 += float(is_active.sum().item())

    assert d_sae is not None and active_counts is not None, "No batches in data_iter"

    mse = sum_sq_err / n_elements
    mean_x = sum_x / n_elements
    # Bessel-corrected variance over all elements, matching X.var() semantics.
    var_x = max(
        (sum_x_sq / n_elements - mean_x**2) * (n_elements / (n_elements - 1)), 1e-8
    )
    fve = 1.0 - mse / var_x

    per_feature_act_rate = (active_counts / n_total).numpy().astype(np.float32)
    dead_fraction = float((per_feature_act_rate < dead_threshold).mean())
    mean_l0 = sum_l0 / n_total
    mean_l0_frac = mean_l0 / d_sae

    metrics = ReconstructionMetrics(
        frac_variance_explained=fve,
        mse=mse,
        mean_l0=mean_l0,
        mean_l0_frac=mean_l0_frac,
        dead_fraction=dead_fraction,
        per_feature_act_rate=per_feature_act_rate,
        n_samples=n_total,
    )
    logger.info("Reconstruction metrics: %s", metrics)
    return metrics


def compute_per_layer_metrics(
    sae: BaseSAE,
    layer_activations: dict[int, Tensor],
    **kwargs,
) -> dict[int, ReconstructionMetrics]:
    """Convenience wrapper: compute metrics for multiple layers at once."""
    return {
        layer: compute_reconstruction_metrics(sae, acts, **kwargs)
        for layer, acts in layer_activations.items()
    }


def _get_data_iter(
    activations: Tensor | ActivationCache,
    batch_size: int,
    normalize: bool,
    device: str,
):
    if isinstance(activations, Tensor):
        acts = activations.float()
        if normalize:
            mean = acts.mean(0, keepdim=True)
            scale = (acts - mean).norm(dim=1).mean().clamp(min=1e-8)
            acts = (acts - mean) / scale
        for i in range(0, len(acts), batch_size):
            yield acts[i : i + batch_size]
    else:
        # ActivationCache handles normalization internally
        yield from activations.iter_batches(
            batch_size=batch_size, normalize=normalize, device=device
        )
