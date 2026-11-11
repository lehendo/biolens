# BioLens

A toolkit for training sparse autoencoders (SAEs) on biological foundation models and for checking whether the features they find mean what they are claimed to mean.

- **Models:** ESM-2 (8M, 35M, 150M, 650M) and an Evo 2 7B adapter, behind one registry and one `get_activations` API.
- **SAEs:** TopK SAE with auxiliary dead-feature loss, a sharded activation cache with normalization, and a training loop.
- **Evaluation:** reconstruction metrics; GO-term probing of SAE features against Swiss-Prot annotations, with held-out and hard-negative AUROC baselines and cluster-robust bootstrap intervals.
- **Feature verification:** a CLI and a Gradio dashboard that show a feature's top-activating sequences next to its claimed GO terms and a recorded verification status.
- **Homology-aware splits:** MMseqs2 clustering of Swiss-Prot and cluster-disjoint training and evaluation pools.

## Why the verification tooling

The feature with the best AUROC out of many candidates is biased upward, because it was chosen for being the best. The simulation below uses features that are pure noise: the selected feature's AUROC rises with the number of candidates searched, while its AUROC on held-out data stays at chance.

| candidates searched | selected AUROC | held-out AUROC |
|---:|---:|---:|
| 1 | 0.508 | 0.506 |
| 10 | 0.565 | 0.491 |
| 100 | 0.604 | 0.494 |
| 1,000 | 0.633 | 0.498 |
| 10,000 | 0.657 | 0.498 |

Reproduce with `python examples/selection_inflation_demo.py` (100 positives and 100 negatives per set). This is a simulation that illustrates the selection effect. It says nothing about the features of any particular SAE. For real features, use the held-out and hard-negative baselines (`scripts/run_eval.py --held-out-baselines`) and check top-activating examples against reference data (`scripts/inspect_features.py`).

## Installation

```bash
git clone https://github.com/lehendo/biolens
cd biolens
pip install -e ".[dev]"
```

Requires Python 3.10+. Optional extras: `dashboard` (Gradio), `tracking` (Weights & Biases), `evo2` (Evo 2; needs a CUDA GPU and `flash-attn`, which must be built from source against your installed `torch`).

Run the tests (no GPU or network needed):

```bash
pytest -m "not integration"
```

## Quick start (ESM-2 8M)

```bash
# 1. Cache residual-stream activations for Swiss-Prot sequences
python scripts/extract_activations.py --model esm2_8m --layer 5 --pooling mean \
    --output-dir data/activations/esm2_8m_layer5 --max-seqs 20000 --sample-seed 0
python scripts/extract_activations.py --finalize --output-dir data/activations/esm2_8m_layer5

# 2. Train a TopK SAE
python scripts/train_sae.py --model esm2_8m --layer 5 --variant topk \
    --cache data/activations/esm2_8m_layer5 --output-dir checkpoints/esm2_8m_layer5_topk --no-wandb

# 3. Evaluate: reconstruction + GO-term probing (add --held-out-baselines for inflation baselines)
python scripts/run_eval.py --model esm2_8m --layer 5 \
    --sae-checkpoint checkpoints/esm2_8m_layer5_topk/final.pt \
    --cache data/activations/esm2_8m_layer5 --output-dir eval/esm2_8m_layer5 \
    --min-positives 100 --sample-seed 0 --no-wandb

# 4. Inspect specific features: top-activating sequences, claimed GO terms, verification status
python scripts/inspect_features.py --sae-checkpoint checkpoints/esm2_8m_layer5_topk/final.pt \
    --cache data/activations/esm2_8m_layer5 --feature-idx 10 20 \
    --go-probing-results eval/esm2_8m_layer5/go_probing_results.json
```

Use `--device cpu` where a script defaults to `cuda`. A verification registry records, per feature and claimed GO term, what a manual check of the top-activating sequences against UniProt found; see `examples/verified_features_example.yaml` for the format and pass it with `--verified-annotations`. The browser version is `python -m biolens.dashboard.app` (requires the `dashboard` extra).

### Python API

```python
import torch

from biolens.eval.reconstruction import compute_reconstruction_metrics
from biolens.sae.architectures import SAEConfig, TopKSAE
from biolens.sae.train import SAETrainer, TrainingConfig

acts = torch.randn(2000, 64)  # stand-in for cached model activations, shape (N, d_model)

sae = TopKSAE(SAEConfig(d_model=64, expansion_factor=4), k=8)
trainer = SAETrainer(sae, TrainingConfig(n_steps=200, batch_size=256, device="cpu", wandb_project=None))
trainer.train_from_tensors(acts)

print(compute_reconstruction_metrics(sae, acts, normalize=True, device="cpu"))
```

### Homology-aware pools

```bash
python scripts/cluster_full_swissprot.py --output-dir data/swissprot_clustering      # needs `mmseqs` on PATH
python scripts/partition_swissprot_pools.py --cluster-dir data/swissprot_clustering --output-dir data/pools
```

This clusters Swiss-Prot at 30% sequence identity and 80% coverage, then assigns whole clusters to disjoint training and evaluation pools, so no close homolog of an evaluation protein appears in the homology-excluded training pool.

## Reference run

ESM-2 8M, layer 5, TopK SAE (expansion 8, `d_sae` = 2560, k = 32), evaluated on Swiss-Prot sequence activations: fraction of variance explained 0.970, mean L0 32.0 (equal to k by construction), 23.4% dead features under a strict single-pass criterion. These are from the author's run and are not reproduced by the test suite.

## Status and limitations

- Alpha software. Implemented: ESM-2 and Evo 2 adapters, TopK SAE. Not implemented: JumpReLU and Gated SAEs (the classes raise `NotImplementedError`) and single-cell models (Geneformer, scGPT).
- Activation extraction from the command line covers protein sequences (Swiss-Prot). The Evo 2 adapter (`biolens.models.evo2`) is available through the Python API for your own sequences; it needs a CUDA GPU with `flash-attn` and is not covered by the CPU test suite.
- GO-term probing uses single-feature and logistic-regression AUROC on pooled sequence activations; labels come from Swiss-Prot experimental GO annotations.
- A feature's recorded verification status is a manual judgement, not ground truth.

## Prior work

BioLens does not claim to be the first to apply SAEs to biology. Related work includes SAEs on ESM-2 that recover GO terms and other properties (Adams et al., 2024, "From Mechanistic Interpretability to Mechanistic Biology"; Simon & Zou, 2024, "InterPLM"), SAEs on Evo 2 (Arc Institute), and SAEs on single-cell foundation models (Geneformer, scGPT). What this repository adds is a single toolkit across model families and tooling that treats an AUROC as a claim to be verified.

## License

MIT. See [LICENSE](LICENSE). Model weights carry their own licenses; check each before redistributing derived SAE weights.
