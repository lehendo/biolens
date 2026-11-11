"""
BioLens — SAE interpretability toolkit for biological foundation models.

Train sparse autoencoders on protein (ESM-2) and genomic (Evo 2) model activations,
evaluate reconstruction and GO-term probing, and verify claimed feature-concept
mappings against real reference data.
"""

__version__ = "0.1.0"
__all__ = [
    "ModelRegistry",
    "TopKSAE",
    "ActivationCache",
    "SAETrainer",
    "TrainingConfig",
]

# Guarded import: lightweight submodules (e.g. `biolens.provenance`) should stay
# importable in environments that lack the full torch stack.
try:
    from biolens.models.registry import ModelRegistry
    from biolens.sae.architectures import TopKSAE
    from biolens.sae.dictionary import ActivationCache
    from biolens.sae.train import SAETrainer, TrainingConfig
except ImportError:
    pass
