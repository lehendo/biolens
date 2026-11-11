"""
Model registry and BioModel base class.

All supported biological foundation models expose:
  get_activations(sequences, layer, pooling) -> Tensor

Supported: ESM2 (protein, all sizes) and Evo 2 (genomic).
Planned: Geneformer / scGPT (single-cell).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml

logger = logging.getLogger(__name__)

PoolingType = Literal["none", "mean", "cls"]

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # biolens/ repo root


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: Literal["protein", "genomic", "single_cell"]
    hf_name: str
    tokenizer_name: str
    num_layers: int
    hidden_dim: int
    max_seq_len: int
    supported: bool = True


class BioModel(ABC):
    """Abstract base class for BioLens model wrappers.

    Every concrete adapter must implement get_activations; everything else
    (caching helpers, batch splitting) is handled here.
    """

    def __init__(self, config: ModelConfig, device: str, dtype: torch.dtype) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self._model: torch.nn.Module | None = None
        self._tokenizer = None

    @property
    def hidden_dim(self) -> int:
        return self.config.hidden_dim

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load()

    @abstractmethod
    def _load(self) -> None:
        """Load model weights and tokenizer into self._model / self._tokenizer."""

    @abstractmethod
    def get_activations(
        self,
        sequences: list[str] | str,
        layer: int,
        pooling: PoolingType = "mean",
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Extract residual-stream activations from the given layer.

        Args:
            sequences: Amino-acid / nucleotide sequences (strings).
            layer:     0-indexed transformer block (0 = first layer after embedding).
            pooling:   "mean" → (N, D); "cls" → (N, D); "none" → (N, L, D).
            batch_size: Number of sequences per forward pass.

        Returns:
            Float tensor on CPU.  Shape depends on pooling (see above).
        """

    def _validate_layer(self, layer: int) -> None:
        if not (0 <= layer < self.config.num_layers):
            raise ValueError(
                f"Layer {layer} out of range for {self.config.name} "
                f"(0..{self.config.num_layers - 1})."
            )


class ModelRegistry:
    """Central registry for all supported biological foundation models."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = _REPO_ROOT / "configs" / "models.yaml"
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        self._configs: dict[str, ModelConfig] = {}
        for name, cfg in raw["models"].items():
            self._configs[name] = ModelConfig(
                name=name,
                family=cfg["family"],
                hf_name=cfg["hf_name"] or "",
                tokenizer_name=cfg.get("tokenizer") or cfg.get("hf_name") or "",
                num_layers=cfg["num_layers"],
                hidden_dim=cfg["hidden_dim"],
                max_seq_len=cfg["max_seq_len"],
                supported=cfg.get("supported", True),
            )

    def get_config(self, model_name: str) -> ModelConfig:
        if model_name not in self._configs:
            raise KeyError(
                f"Unknown model '{model_name}'. Available: {list(self._configs)}"
            )
        return self._configs[model_name]

    def list_models(self, family: str | None = None, supported_only: bool = True) -> list[str]:
        models = self._configs.items()
        if supported_only:
            models = [(n, c) for n, c in models if c.supported]  # type: ignore[assignment]
        if family is not None:
            models = [(n, c) for n, c in models if c.family == family]  # type: ignore[assignment]
        return [n for n, _ in models]

    def load_model(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> BioModel:
        """Return a loaded BioModel wrapper for the given model name."""
        config = self.get_config(model_name)
        if not config.supported:
            raise NotImplementedError(
                f"Model '{model_name}' is not yet implemented."
            )
        return _instantiate(config, device=device, dtype=dtype)


def _instantiate(config: ModelConfig, device: str, dtype: torch.dtype) -> BioModel:
    if config.family == "protein":
        from biolens.models.esm2 import ESM2Model
        return ESM2Model(config, device=device, dtype=dtype)
    elif config.family == "genomic":
        if config.name.startswith("evo2_"):
            from biolens.models.evo2 import Evo2Model
            return Evo2Model(config, device=device, dtype=dtype)
        raise NotImplementedError(
            f"Genomic model '{config.name}' is not yet implemented — only Evo 2 "
            f"variants (evo2_*) are supported so far."
        )
    elif config.family == "single_cell":
        raise NotImplementedError("Single-cell models (Geneformer, scGPT) are not yet implemented.")
    else:
        raise ValueError(f"Unknown family: {config.family}")
