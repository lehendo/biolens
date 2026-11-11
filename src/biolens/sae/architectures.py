"""
Sparse autoencoder architectures.

TopKSAE is implemented. JumpReLUSAE and GatedSAE are placeholders that raise
NotImplementedError (see SAELens for reference implementations).

All variants share:
  - SAEOutput namedtuple
  - Decoder column normalization after each optimizer step
  - Consistent encode/decode/forward API

References:
  TopK:     Gao et al. (2024) "Scaling and evaluating sparse autoencoders"
  JumpReLU: Lieberum et al. (2024) "Gemma Scope: Open Sparse Autoencoders Everywhere"
  Gated:    Rajamanoharan et al. (2024) "Improving Dictionary Learning with Gated SAE"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SAEOutput(NamedTuple):
    """Outputs produced by any SAE forward pass."""
    reconstruction: Tensor    # (*, d_model) — reconstructed input
    feature_acts: Tensor      # (*, d_sae) — sparse feature activations
    l2_loss: Tensor           # scalar — reconstruction MSE
    auxiliary_loss: Tensor    # scalar — variant-specific aux loss (0 if unused)


@dataclass
class SAEConfig:
    d_model: int
    expansion_factor: int = 8
    # Populated from d_model * expansion_factor on post_init:
    d_sae: int = 0
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.d_sae == 0:
            self.d_sae = self.d_model * self.expansion_factor


class BaseSAE(nn.Module):
    """Shared initialization and decoder normalization for all SAE variants."""

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Renormalize decoder columns to unit L2 norm after each optimizer step.

        This prevents the trivial solution where the SAE reduces reconstruction
        loss by growing decoder columns without bound.
        """
        raise NotImplementedError

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> SAEOutput:
        raise NotImplementedError


# ── TopK SAE ────────────────────────────────────────────────────────────────

class TopKSAE(BaseSAE):
    """
    TopK sparse autoencoder.

    Sparsity is controlled directly by k (number of active features), not by
    an L1 coefficient — this eliminates the main tuning knob of L1-based SAEs
    and gives predictable L0.

    Dead feature prevention: features inactive for many steps receive an
    auxiliary reconstruction gradient via the aux_loss term.

    Forward pass:
        x_c = x - b_dec                          # center by decoder bias
        h   = x_c @ W_enc + b_enc                # pre-activations
        z   = topk(h, k).clamp(min=0)            # k-sparse non-negative acts
        x̂   = z @ W_dec + b_dec                  # reconstruction
        L   = MSE(x, x̂) + α * aux_loss(x, x̂, h)
    """

    def __init__(
        self,
        cfg: SAEConfig,
        k: int = 32,
        aux_fraction: float = 1e-3,
        k_aux_factor: int = 4,
        dead_threshold: float = 1e-4,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__(cfg)
        self.k = k
        self.aux_fraction = aux_fraction
        self.k_aux = max(1, cfg.d_sae // k_aux_factor)
        self.dead_threshold = dead_threshold
        self.ema_decay = ema_decay

        self.W_enc = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model))

        # Exponential moving average of per-feature activation frequency.
        # Shape: (d_sae,). Not a parameter — not trained, just tracked.
        self.register_buffer("feature_ema", torch.ones(cfg.d_sae) * 0.1)

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming uniform for encoder; initialize decoder as W_enc.T / norm
        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            W = self.W_enc.T.clone()
            norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data = W / norms

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: Tensor) -> Tensor:
        """(*, d_model) → (*, d_sae) sparse feature activations."""
        x_c = x - self.b_dec
        h = x_c @ self.W_enc + self.b_enc  # (*, d_sae)
        return _topk_relu(h, self.k)

    def decode(self, z: Tensor) -> Tensor:
        """(*, d_sae) → (*, d_model) reconstruction."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> SAEOutput:
        """
        Args:
            x: (N, d_model) activation vectors.

        Returns:
            SAEOutput with reconstruction, feature_acts, l2_loss, auxiliary_loss.
        """
        x_c = x - self.b_dec
        h = x_c @ self.W_enc + self.b_enc  # (N, d_sae)

        # Primary sparse activations
        z = _topk_relu(h, self.k)           # (N, d_sae)
        x_hat = z @ self.W_dec + self.b_dec  # (N, d_model)

        l2_loss = F.mse_loss(x_hat, x)

        # Update EMA of feature activation frequency (detached — no grad)
        with torch.no_grad():
            activated = (z > 0).float().mean(0)  # (d_sae,)
            self.feature_ema.mul_(self.ema_decay).add_(
                activated * (1 - self.ema_decay)
            )

        # Auxiliary loss: steer dead features toward residual reconstruction.
        aux_loss = self._aux_loss(x, x_hat, x_c)

        return SAEOutput(
            reconstruction=x_hat,
            feature_acts=z,
            l2_loss=l2_loss,
            auxiliary_loss=aux_loss,
        )

    def _aux_loss(self, x: Tensor, x_hat: Tensor, x_c: Tensor) -> Tensor:
        """
        Compute auxiliary reconstruction loss for dead features.

        Approach from Gao et al. (2024): run TopK_aux on the residual using only
        dead features, penalise the auxiliary reconstruction MSE.

        This prevents feature collapse by ensuring dead features receive
        gradient signal toward the residual the main SAE hasn't captured.
        """
        dead_mask = self.feature_ema < self.dead_threshold  # (d_sae,) bool
        if not dead_mask.any():
            return x.new_zeros(1).squeeze()

        residual = (x - x_hat).detach()  # stop gradient from aux path to main decoder

        # Compute auxiliary pre-activations on dead features only
        h_aux = x_c.detach() @ self.W_enc + self.b_enc  # (N, d_sae)
        h_aux = h_aux * dead_mask.float()  # zero out live features

        # TopK among dead features
        k_aux = min(self.k_aux, dead_mask.sum().item())
        z_aux = _topk_relu(h_aux, int(k_aux))  # (N, d_sae)

        x_hat_aux = z_aux @ self.W_dec  # (N, d_model) — no b_dec, targets residual
        return F.mse_loss(x_hat_aux, residual)

    @property
    def dead_features(self) -> Tensor:
        """Boolean mask (d_sae,): True where feature is considered dead."""
        return self.feature_ema < self.dead_threshold

    @property
    def dead_fraction(self) -> float:
        return self.dead_features.float().mean().item()

    def extra_repr(self) -> str:
        return (
            f"d_model={self.cfg.d_model}, d_sae={self.cfg.d_sae}, "
            f"k={self.k}, expansion={self.cfg.expansion_factor}x"
        )



class JumpReLUSAE(BaseSAE):
    """
    JumpReLU sparse autoencoder (not yet implemented).

    Implements a learned per-feature threshold θ:
        z_i = h_i * (h_i > θ_i)   via STE gradient for the step function.

    Target sparsity (L0) replaces k; sparsity loss penalises deviation from it.

    Reference: Lieberum et al. (2024), Gemma Scope; implementation in SAELens
    (github.com/jbloomAU/SAELens) is the recommended starting point.
    """

    def __init__(self, cfg: SAEConfig, target_l0: int = 32, bandwidth: float = 1e-3) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "JumpReLUSAE is not yet implemented. "
            "See SAELens for a reference implementation: "
            "https://github.com/jbloomAU/SAELens"
        )



class GatedSAE(BaseSAE):
    """
    Gated sparse autoencoder (not yet implemented).

    Two-stream design: a gate stream (sigmoid) controls which features activate;
    a magnitude stream scales the active features.  L1 penalty on gate activations.

    Reference: Rajamanoharan et al. (2024); implementation in SAELens.
    """

    def __init__(self, cfg: SAEConfig, sparsity_coeff: float = 2e-3) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "GatedSAE is not yet implemented. "
            "See SAELens for a reference implementation: "
            "https://github.com/jbloomAU/SAELens"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _topk_relu(h: Tensor, k: int) -> Tensor:
    """Apply TopK activation: keep top-k values (with ReLU), zero the rest.

    Args:
        h: Pre-activation tensor of shape (*, d_sae).
        k: Number of features to keep active.

    Returns:
        Sparse tensor of same shape — at most k non-zero entries per row,
        all non-negative (pre-activations below zero are treated as inactive).
    """
    if k >= h.shape[-1]:
        return h.clamp(min=0)

    topk_vals, topk_idx = h.topk(k, dim=-1)
    acts = torch.zeros_like(h)
    acts.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
    return acts


def make_sae(
    variant: str,
    d_model: int,
    expansion_factor: int = 8,
    **kwargs,
) -> BaseSAE:
    """Factory function — construct an SAE by variant name."""
    cfg = SAEConfig(d_model=d_model, expansion_factor=expansion_factor)
    if variant == "topk":
        return TopKSAE(cfg, **kwargs)
    elif variant == "jumprelu":
        return JumpReLUSAE(cfg, **kwargs)
    elif variant == "gated":
        return GatedSAE(cfg, **kwargs)
    else:
        raise ValueError(f"Unknown SAE variant: {variant!r}. Choose 'topk', 'jumprelu', 'gated'.")
