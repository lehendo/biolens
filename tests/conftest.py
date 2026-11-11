"""
Shared pytest fixtures.

Fast tests use synthetic data (random tensors, mock sequences) — no model downloads.
Integration tests (marked @pytest.mark.integration) require network and GPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Dimensions ────────────────────────────────────────────────────────────────
D_MODEL_SMALL = 32      # small d_model for fast CPU tests
D_SAE_SMALL = 256       # expansion 8×
K_SMALL = 8             # TopK k
N_SEQS = 64             # number of synthetic sequences
SEQ_LEN = 20            # synthetic sequence length


@pytest.fixture
def d_model():
    return D_MODEL_SMALL


@pytest.fixture
def d_sae():
    return D_SAE_SMALL


@pytest.fixture
def k():
    return K_SMALL


@pytest.fixture
def n_seqs():
    return N_SEQS


@pytest.fixture
def random_activations(n_seqs, d_model):
    """Gaussian activation vectors — stand-in for real ESM2 residual stream."""
    torch.manual_seed(0)
    return torch.randn(n_seqs, d_model)


@pytest.fixture
def topk_sae(d_model, d_sae, k):
    from biolens.sae.architectures import SAEConfig, TopKSAE
    cfg = SAEConfig(d_model=d_model, d_sae=d_sae)
    sae = TopKSAE(cfg, k=k)
    return sae


@pytest.fixture
def tmp_cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def small_go_labels():
    """Minimal GO label dict: 100 'proteins', two GO terms."""
    ids = [f"P{i:05d}" for i in range(100)]
    go_labels = {}
    for i, pid in enumerate(ids):
        terms = set()
        if i % 5 == 0:          # 20% positive for GO:0000001
            terms.add("GO:0000001")
        if i % 3 == 0:          # 33% positive for GO:0000002
            terms.add("GO:0000002")
        go_labels[pid] = terms
    return ids, go_labels


@pytest.fixture
def mock_protein_sequences():
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    rng = np.random.default_rng(42)
    return [
        "".join(rng.choice(amino_acids, size=rng.integers(20, 100)))
        for _ in range(20)
    ]
