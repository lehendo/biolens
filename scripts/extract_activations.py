"""
Extract and cache residual-stream activations from a biological foundation model.

Designed for SLURM array jobs: each task handles a disjoint slice of the dataset.
Writes shards atomically and registers them in a filelock-protected manifest.

Usage (single process):
  python scripts/extract_activations.py \\
      --model esm2_8m --layer 5 --pooling mean \\
      --output-dir data/activations/esm2_8m_layer5 \\
      --data-source swissprot --max-seq-len 1024

Usage (sharded, e.g. as a SLURM array job):
  python scripts/extract_activations.py \\
      --model esm2_8m --layer 5 --pooling mean \\
      --output-dir ... \\
      --shard-id $SLURM_ARRAY_TASK_ID --n-shards $N_SHARDS

After all shards are written, run:
  python scripts/extract_activations.py --finalize \\
      --output-dir /path/to/cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src/ to path so biolens is importable without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.data.uniprot import load_swissprot, swissprot_fasta_path
from biolens.models.registry import ModelRegistry
from biolens.provenance import capture, sha256_file, sha256_strings
from biolens.sae.dictionary import ActivationCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="esm2_8m", help="Model name in configs/models.yaml")
    p.add_argument("--layer", type=int, default=-1, help="Layer index (-1 = last layer)")
    p.add_argument("--pooling", default="mean", choices=["mean", "cls", "none"])
    p.add_argument("--output-dir", required=False, help="Cache directory path")
    p.add_argument("--data-source", default="swissprot", choices=["swissprot"])
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32, help="Sequences per GPU forward pass")
    # SLURM array sharding
    p.add_argument("--shard-id", type=int, default=0, help="This task's shard index")
    p.add_argument("--n-shards", type=int, default=1, help="Total number of shards")
    # Finalization
    p.add_argument("--finalize", action="store_true", help="Compute norm stats; run after all shards written")
    # Device
    p.add_argument("--device", default="cuda")
    # Max sequences (for dev/testing)
    p.add_argument("--max-seqs", type=int, default=None)
    p.add_argument(
        "--sample-seed", type=int, default=None,
        help="If set with --max-seqs (swissprot only), draws a seeded random "
             "sample instead of the first N entries in FASTA file order, which "
             "is measurably skewed (see biolens.data.uniprot.load_swissprot's "
             "docstring). Record the seed you use.",
    )
    p.add_argument(
        "--protein-ids-file", default=None,
        help="Path to a JSON file containing a list of protein IDs to extract "
             "exactly (swissprot only) -- for a pre-computed pool (e.g. from "
             "scripts/partition_swissprot_pools.py's cluster-disjoint pools) "
             "rather than a fresh "
             "--max-seqs/--sample-seed draw. Overrides --max-seqs/--sample-seed "
             "when set; the resulting pool is exactly the given ID list, order "
             "preserved for determinism, not a re-sample.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.finalize:
        if not args.output_dir:
            raise ValueError("--output-dir required for --finalize")
        cache = ActivationCache(args.output_dir)
        manifest = cache.finalize()
        logger.info(
            "Finalization complete: %d total vectors, scale=%.4f",
            manifest["norm"]["n_total"], manifest["norm"]["scale"],
        )
        return

    if not args.output_dir:
        raise ValueError("--output-dir required")

    # ── Load model ────────────────────────────────────────────────────────────
    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    layer = args.layer if args.layer >= 0 else model_cfg.num_layers - 1
    logger.info(
        "Extracting %s layer %d (%s pooling) → shard %d/%d",
        args.model, layer, args.pooling, args.shard_id, args.n_shards,
    )

    model = registry.load_model(args.model, device=args.device)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading %s sequences...", args.data_source)
    if args.data_source == "swissprot":
        if args.protein_ids_file:
            with open(args.protein_ids_file) as f:
                wanted_ids = json.load(f)
            logger.info(
                "Loading full Swiss-Prot pool to extract exactly %d IDs from %s...",
                len(wanted_ids), args.protein_ids_file,
            )
            all_ids, all_seqs = load_swissprot(max_seqs=None, max_seq_len=args.max_seq_len)
            id_to_seq = dict(zip(all_ids, all_seqs))
            missing = [pid for pid in wanted_ids if pid not in id_to_seq]
            if missing:
                raise ValueError(
                    f"{len(missing)}/{len(wanted_ids)} protein IDs from "
                    f"{args.protein_ids_file} not found in the pinned Swiss-Prot "
                    f"pool (max_seq_len={args.max_seq_len}) -- e.g. {missing[:5]}"
                )
            ids = wanted_ids
            sequences = [id_to_seq[pid] for pid in wanted_ids]
        else:
            ids, sequences = load_swissprot(
                max_seqs=args.max_seqs,
                max_seq_len=args.max_seq_len,
                random_seed=args.sample_seed,
            )
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    # Assign this shard's slice
    shard_ids_seq = _shard_slice(ids, sequences, args.shard_id, args.n_shards)
    if not shard_ids_seq:
        logger.info("Shard %d: nothing to do (empty slice).", args.shard_id)
        return

    shard_protein_ids = [pid for pid, _ in shard_ids_seq]
    shard_sequences = [seq for _, seq in shard_ids_seq]
    logger.info("Shard %d: %d sequences", args.shard_id, len(shard_sequences))

    # ── Extract activations ───────────────────────────────────────────────────
    all_acts = model.get_activations(
        shard_sequences,
        layer=layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
    )
    logger.info("Extracted activations: shape %s", tuple(all_acts.shape))

    metadata = capture(
        model=args.model,
        layer=layer,
        pooling=args.pooling,
        data_source=args.data_source,
        max_seqs=args.max_seqs,
        sample_seed=args.sample_seed,
        protein_ids_file=args.protein_ids_file,
        n_pool_sequences=len(ids),
        pool_ids_sha256=sha256_strings(sorted(ids)),
    )
    if args.data_source == "swissprot":
        metadata["swissprot_fasta_sha256"] = sha256_file(swissprot_fasta_path())

    cache = ActivationCache(args.output_dir)
    cache.write_shard(
        shard_id=args.shard_id,
        activations=all_acts,
        ids=shard_protein_ids,
        sequences=shard_sequences,
        metadata=metadata,
    )
    logger.info("Shard %d written successfully.", args.shard_id)


def _shard_slice(
    ids: list[str], sequences: list[str], shard_id: int, n_shards: int
) -> list[tuple[str, str]]:
    """Return the subset of (id, seq) pairs assigned to this shard."""
    pairs = list(zip(ids, sequences))
    size = len(pairs)
    shard_size = (size + n_shards - 1) // n_shards
    start = shard_id * shard_size
    end = min(start + shard_size, size)
    return pairs[start:end]


if __name__ == "__main__":
    main()
