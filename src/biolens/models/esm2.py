"""
ESM2 protein language model adapter.

Uses HuggingFace transformers.  Activations are extracted from the residual
stream (output of each EsmLayer block) via output_hidden_states=True, which is
cleaner and faster than forward hooks for this architecture.

Hook-based extraction (for architectures that don't support output_hidden_states)
is in activation_hooks.py.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from biolens.models.registry import BioModel, PoolingType

logger = logging.getLogger(__name__)


class ESM2Model(BioModel):
    """Thin wrapper around facebook/esm2_* that exposes a consistent get_activations API."""

    def _load(self) -> None:
        from transformers import EsmModel, EsmTokenizer  # deferred import

        logger.info("Loading ESM2 model: %s", self.config.hf_name)
        self._tokenizer = EsmTokenizer.from_pretrained(self.config.tokenizer_name)
        self._model = EsmModel.from_pretrained(
            self.config.hf_name,
            torch_dtype=self.dtype,
        )
        self._model.eval()
        self._model.to(self.device)
        logger.info(
            "Loaded %s (%d layers, d_model=%d) on %s",
            self.config.name,
            self.config.num_layers,
            self.config.hidden_dim,
            self.device,
        )

    def get_activations(
        self,
        sequences: list[str] | str,
        layer: int,
        pooling: PoolingType = "mean",
        batch_size: int = 32,
    ) -> Tensor:
        """
        Args:
            sequences:  One or more amino-acid sequences.
            layer:      Transformer block index (0 = first block after embedding).
                        ESM2 8M has layers 0..5; 35M has 0..11; etc.
            pooling:    "mean" → average over non-padding residues, shape (N, D).
                        "cls"  → [CLS] token, shape (N, D).
                        "none" → all positions (padded), shape (N, L_max, D).
            batch_size: Sequences per GPU forward pass.

        Returns:
            CPU float32 tensor.
        """
        self._ensure_loaded()
        self._validate_layer(layer)

        if isinstance(sequences, str):
            sequences = [sequences]

        # hidden_states tuple has length num_layers + 1:
        # index 0 = embedding output, index i+1 = output of transformer block i
        hs_index = layer + 1

        all_acts: list[Tensor] = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            acts = self._forward_batch(batch_seqs, hs_index, pooling)
            all_acts.append(acts)

        return torch.cat(all_acts, dim=0).float().cpu()

    def get_activations_all_layers(
        self,
        sequences: list[str] | str,
        layers: list[int],
        pooling: PoolingType = "mean",
        batch_size: int = 32,
    ) -> dict[int, Tensor]:
        """Extract activations from multiple layers in one forward pass per batch."""
        self._ensure_loaded()
        for layer in layers:
            self._validate_layer(layer)

        if isinstance(sequences, str):
            sequences = [sequences]

        results: dict[int, list[Tensor]] = {layer: [] for layer in layers}
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            all_hs = self._forward_batch_all_layers(batch_seqs, pooling)
            for layer in layers:
                results[layer].append(all_hs[layer + 1])

        return {
            layer: torch.cat(chunks, dim=0).float().cpu()
            for layer, chunks in results.items()
        }

    def _forward_batch(
        self, sequences: list[str], hs_index: int, pooling: PoolingType
    ) -> Tensor:
        assert self._model is not None and self._tokenizer is not None

        encoded = self._tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[hs_index]  # (N, L, D)
        return _apply_pooling(hidden, attention_mask, pooling)

    def _forward_batch_all_layers(
        self, sequences: list[str], pooling: PoolingType
    ) -> tuple[Tensor, ...]:
        """Returns all hidden states (embedding + each transformer block)."""
        assert self._model is not None and self._tokenizer is not None

        encoded = self._tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        return tuple(
            _apply_pooling(hs, attention_mask, pooling)
            for hs in outputs.hidden_states
        )


def _apply_pooling(hidden: Tensor, attention_mask: Tensor, pooling: PoolingType) -> Tensor:
    """Apply pooling strategy to (N, L, D) hidden states."""
    if pooling == "cls":
        return hidden[:, 0, :]  # (N, D)
    elif pooling == "mean":
        # Exclude padding tokens from the average.
        # attention_mask is 1 for real tokens (including CLS/EOS), 0 for pad.
        # We exclude CLS (pos 0) and EOS from the average to match common protein-ML conventions.
        # Determine EOS position per sequence: last position where mask == 1.
        mask = attention_mask.clone().float()
        # Zero out CLS token (position 0)
        mask[:, 0] = 0.0
        # Zero out EOS token (last real position per sequence)
        seq_lens = attention_mask.sum(dim=1)  # (N,)
        for i, slen in enumerate(seq_lens):
            mask[i, slen - 1] = 0.0
        # Expand mask to (N, L, 1) for broadcasting
        mask = mask.unsqueeze(-1)
        sum_acts = (hidden * mask).sum(dim=1)  # (N, D)
        count = mask.sum(dim=1).clamp(min=1.0)  # (N, 1)
        return sum_acts / count  # (N, D)
    elif pooling == "none":
        return hidden  # (N, L, D) — caller handles variable lengths
    else:
        raise ValueError(f"Unknown pooling: {pooling!r}. Choose 'mean', 'cls', or 'none'.")
