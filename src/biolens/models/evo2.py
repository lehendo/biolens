"""
Evo 2 (Arc Institute) genomic foundation model adapter.

Evo 2 has published SAE precedent and a long context suited to regulatory-variant
work.

Layer-tap point verified against the real evo2_7b checkpoint: `model.model.named_modules()` was
introspected directly and confirmed `blocks.{layer}.mlp.l3` exists on both
a hyena-filter block (block 0) and an attention block (block 3, one of
`attn_layer_idxs`), with identical `mlp.l1/l2/l3` structure across both
block types — this is no longer just a documentation-based guess (Arc
Institute's own GitHub issue ArcInstitute/evo2#78, asking for the same
clarification, was and remains open/unresolved upstream). A real
`get_activations` forward pass was also run for both layer 0 and layer 3,
confirming `return_embeddings=True, layer_names=[...]` returns
`embeddings[layer_name]` and correctly reduces to the expected (N, 4096)
shape after mean pooling. This adapter is now trusted end-to-end.

Key differences from ESM2Model (the pattern this otherwise follows):
  - Loaded via the `evo2` package's own `Evo2` class, not
    transformers.AutoModel — Evo 2 uses a custom StripedHyena2 architecture,
    not a standard HF transformer.
  - Activations extracted via Evo 2's own `return_embeddings=True,
    layer_names=[...]` forward-pass API, not output_hidden_states — no need
    for activation_hooks.py's manual forward-hook machinery here, since Evo
    2 already exposes an equivalent mechanism natively.
  - Tokenization is character-level (DNA bases), via the model's own
    tokenizer, with no confirmed HF-style attention_mask/padding contract —
    sequences are processed one at a time (not padded into a true batch) to
    avoid introducing padding-related bias into mean pooling under
    unverified tokenizer semantics. `batch_size` still chunks the Python-
    level loop for consistency with the BioModel interface and to bound
    memory, it does not enable padded batched forward passes.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from biolens.models.registry import BioModel, PoolingType

logger = logging.getLogger(__name__)


class Evo2Model(BioModel):
    """Thin wrapper around Arc Institute's `evo2` package exposing the same
    get_activations API as ESM2Model, for a unified cross-modality interface."""

    def _load(self) -> None:
        try:
            from evo2 import Evo2  # deferred import — heavy, GPU/FlashAttention-only
        except ImportError as exc:
            raise ImportError(
                "The 'evo2' package is required for Evo2Model but is not installed. "
                "Install with `pip install evo2` (requires a CUDA GPU and Flash "
                "Attention; see https://github.com/ArcInstitute/evo2 for the 7B "
                "'light install' that skips Transformer Engine/FP8 requirements)."
            ) from exc

        logger.info("Loading Evo 2 model: %s", self.config.name)
        # Evo2's own shortname convention ('evo2_7b', 'evo2_40b', ...) matches
        # this project's configs/models.yaml keys directly by construction.
        self._model = Evo2(self.config.name)
        logger.info(
            "Loaded %s (%d blocks, d_model=%d)",
            self.config.name, self.config.num_layers, self.config.hidden_dim,
        )
        # BioModel's dtype/device handling assumes a .to()-able nn.Module;
        # Evo2's own class manages device placement internally per its
        # documented usage (tokens moved to the target device explicitly in
        # get_activations below), so no extra .to() call here.

    def get_activations(
        self,
        sequences: list[str] | str,
        layer: int,
        pooling: PoolingType = "mean",
        batch_size: int = 8,
    ) -> Tensor:
        """
        Args:
            sequences:  One or more DNA sequences (A/C/G/T, uppercase).
            layer:      Block index (0-indexed, 0 = first StripedHyena2 block).
                        Evo 2 7B has 32 blocks (configs/models.yaml).
            pooling:    "mean" -> average over sequence positions, shape (N, D).
                        "cls"  -> NOT SUPPORTED (Evo 2 has no CLS token; a DNA
                                  autoregressive model has no equivalent
                                  concept) — raises ValueError.
                        "none" -> all positions, returned as a list of
                                  variable-length (L_i, D) tensors (DNA
                                  sequences vary in length far more than
                                  proteins do; no padding is applied, so this
                                  cannot be a single stacked tensor).
            batch_size: Sequences per chunk of the Python-level loop (NOT a
                        padded batched forward pass — see module docstring).

        Returns:
            CPU float32 tensor, shape (N, D) for "mean"/"cls" pooling.
        """
        self._ensure_loaded()
        self._validate_layer(layer)
        if pooling == "cls":
            raise ValueError(
                "Evo 2 has no CLS-token equivalent (autoregressive DNA model) — "
                "use pooling='mean' or 'none'."
            )

        if isinstance(sequences, str):
            sequences = [sequences]

        layer_name = self._layer_name(layer)
        all_acts: list[Tensor] = []
        none_pooled_chunks: list[Tensor] = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            for seq in batch_seqs:
                acts = self._forward_one(seq, layer_name)  # (L, D) on CPU
                if pooling == "mean":
                    all_acts.append(acts.mean(dim=0))
                elif pooling == "none":
                    none_pooled_chunks.append(acts)
                else:
                    raise ValueError(f"Unknown pooling: {pooling!r}")

        if pooling == "none":
            return torch.stack(none_pooled_chunks) if len(sequences) == 1 else none_pooled_chunks  # type: ignore[return-value]
        return torch.stack(all_acts).float().cpu()

    def _forward_one(self, sequence: str, layer_name: str) -> Tensor:
        """Tokenize and forward a single DNA sequence, returning its (L, D)
        activation at `layer_name` on CPU."""
        assert self._model is not None
        input_ids = torch.tensor(
            self._model.tokenizer.tokenize(sequence), dtype=torch.int
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, embeddings = self._model(
                input_ids, return_embeddings=True, layer_names=[layer_name]
            )

        acts: Tensor = embeddings[layer_name]  # (1, L, D) per the documented API
        if acts.dim() == 3:
            acts = acts.squeeze(0)
        return acts.float().cpu()

    def _layer_name(self, layer: int) -> str:
        """
        Map a 0-indexed block index to Evo 2's internal layer_names string.

        Uses `blocks.{layer}.mlp.l3` — confirmed to exist via direct
        introspection of `named_modules()` against the real evo2_7b
        checkpoint (both a hyena-filter block and an attention block have
        this exact submodule, with `l3` being the down-projection back to
        `hidden_size`, matching the SAE's expected d_model). This is an
        MLP-sublayer tap point, not a fully-documented "residual stream
        after block N" name — Arc Institute's own GitHub issue #78 asks for
        the full valid-name list and remains open/unresolved upstream. The
        module's existence is now verified; whether `mlp.l3`'s output is
        preferable to some other residual-stream tap point is a separate,
        still-open design question — not a correctness bug.
        """
        return f"blocks.{layer}.mlp.l3"

    def list_available_layer_names(self) -> list[str]:
        """
        Runtime introspection fallback: list every named submodule Evo 2's
        underlying nn.Module exposes, for finding valid `layer_names` values
        beyond the one documented pattern this adapter defaults to. Requires
        the model to already be loaded (calls _ensure_loaded).
        """
        self._ensure_loaded()
        assert self._model is not None
        return [name for name, _ in self._model.model.named_modules() if name]
