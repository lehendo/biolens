"""
Run the standardized BioLens evaluation suite on a trained SAE.

Computes and logs:
  1. Reconstruction metrics (FVE, dead fraction, L0)
  2. GO term linear probing (single-feature and multivariate AUROC), optionally
     with held-out and hard-negative AUROC baselines (--held-out-baselines) that
     show how much of the best-feature AUROC is selection inflation

Usage:
  python scripts/run_eval.py \\
      --model esm2_8m --layer 5 \\
      --sae-checkpoint /path/to/checkpoint/final.pt \\
      --cache /path/to/activations \\
      --output-dir /path/to/eval_results \\
      [--wandb-project biolens]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.data.uniprot import (
    load_go_annotations,
    load_swissprot,
    resolve_go_dag,
    resolve_go_names,
    swissprot_fasta_path,
)
from biolens.eval.probing import probe_go_terms
from biolens.eval.reconstruction import compute_reconstruction_metrics
from biolens.eval.statistics import bootstrap_mean_auroc_inflation
from biolens.models.registry import ModelRegistry
from biolens.provenance import capture, sha256_file, sha256_strings
from biolens.sae.dictionary import ActivationCache
from biolens.sae.train import build_sae_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="esm2_8m")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--sae-checkpoint", required=True)
    p.add_argument("--cache", required=True, help="ActivationCache directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-positives", type=int, default=50)
    p.add_argument("--max-go-terms", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--expansion-factor", type=int, default=None,
        help="Only needed for checkpoints that predate saved sae_cfg metadata "
             "(build_sae_from_checkpoint raises a clear ValueError naming this "
             "flag if one is passed without it) — e.g. an "
             "early esm2_8m_layer5_topk checkpoint. Ignored (harmless) for any "
             "checkpoint that already carries its own sae_cfg.",
    )
    p.add_argument(
        "--held-out-baselines", action="store_true",
        help=(
            "Compute held-out and hard-negative AUROC baselines alongside the "
            "naive max AUROC. Off by default (it needs the GO DAG and extra "
            "compute)."
        ),
    )
    p.add_argument(
        "--held-out-fraction", type=float, default=0.5,
        help="Fraction of the test split reserved for held-out estimation "
             "(only used with --held-out-baselines)",
    )
    p.add_argument(
        "--min-hard-negatives", type=int, default=10,
        help="Skip hard-negative AUROC below this many sibling-annotated "
             "negatives in the held-out set (only used with --held-out-baselines)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    p.add_argument("--run-name", default=None)
    # Probing eval requires sequences — use eval subset of the cache
    p.add_argument("--max-eval-seqs", type=int, default=10000,
                   help="Max sequences for probing eval (subset of cache)")
    p.add_argument(
        "--sample-seed", type=int, default=None,
        help="Must match whatever --sample-seed extract_activations.py was "
             "run with for this --cache, so the protein-ID sample here "
             "aligns with what's actually cached. The file-order default is "
             "measurably skewed; pass a seed and record it.",
    )
    p.add_argument(
        "--protein-ids-file", default=None,
        help="Path to a JSON file containing the exact protein IDs to probe -- for "
             "a pre-computed evaluation pool (e.g. from scripts/partition_swissprot_pools.py) "
             "rather than a fresh --max-eval-seqs/--sample-seed draw. Overrides "
             "--max-eval-seqs/--sample-seed when set. Note that --cache is only used for "
             "reconstruction metrics; the probing step draws its own sequences.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model config + SAE ───────────────────────────────────────────────
    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    layer = args.layer if args.layer >= 0 else model_cfg.num_layers - 1
    d_model = model_cfg.hidden_dim

    sae, step = build_sae_from_checkpoint(
        args.sae_checkpoint, device=args.device, variant="topk",
        d_model=d_model, expansion_factor=args.expansion_factor,
    )
    logger.info("Loaded SAE from step %d (d_sae=%d)", step, sae.cfg.d_sae)

    # ── Load cache ────────────────────────────────────────────────────────────
    cache = ActivationCache(args.cache)
    logger.info("Cache: %d vectors, d=%d", len(cache), cache.d_model)

    # ── 1. Reconstruction metrics ─────────────────────────────────────────────
    logger.info("Computing reconstruction metrics...")
    recon = compute_reconstruction_metrics(
        sae=sae,
        activations=cache,
        batch_size=args.batch_size,
        device=args.device,
        normalize=True,
    )
    logger.info(str(recon))
    _save_json(
        {
            "frac_variance_explained": recon.frac_variance_explained,
            "mse": recon.mse,
            "mean_l0": recon.mean_l0,
            "mean_l0_frac": recon.mean_l0_frac,
            "dead_fraction": recon.dead_fraction,
            "n_samples": recon.n_samples,
        },
        output_dir / "reconstruction_metrics.json",
    )

    # ── 2. GO term linear probing ─────────────────────────────────────────────
    logger.info("Loading sequences and GO annotations for probing eval...")
    if args.protein_ids_file:
        with open(args.protein_ids_file) as f:
            wanted_ids = json.load(f)
        logger.info(
            "Loading full Swiss-Prot pool to probe exactly %d IDs from %s...",
            len(wanted_ids), args.protein_ids_file,
        )
        all_ids, all_seqs = load_swissprot(max_seqs=None)
        id_to_seq = dict(zip(all_ids, all_seqs))
        missing = [pid for pid in wanted_ids if pid not in id_to_seq]
        if missing:
            raise ValueError(
                f"{len(missing)}/{len(wanted_ids)} protein IDs from "
                f"{args.protein_ids_file} not found in the pinned Swiss-Prot pool "
                f"-- e.g. {missing[:5]}"
            )
        ids_all = wanted_ids
        seqs_all = [id_to_seq[pid] for pid in wanted_ids]
    else:
        ids_all, seqs_all = load_swissprot(
            max_seqs=args.max_eval_seqs, random_seed=args.sample_seed
        )
    protein_id_set = set(ids_all)

    # Provenance: which exact data and code produced this run's outputs.
    provenance = capture(
        script="run_eval.py",
        args={k: v for k, v in vars(args).items()},
        n_eval_sequences=len(ids_all),
        eval_ids_sha256=sha256_strings(sorted(ids_all)),
        swissprot_fasta_sha256=sha256_file(swissprot_fasta_path()),
    )

    go_labels, go_names = load_go_annotations(
        protein_ids=protein_id_set,
        evidence_codes={"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"},  # experimental only
    )
    logger.info("Resolving human-readable GO term names from the ontology...")
    go_names = resolve_go_names(go_names.keys())

    go_dag = None
    if args.held_out_baselines:
        logger.info("Resolving GO DAG for hard-negative sibling lookup...")
        go_dag = resolve_go_dag()

    # Extract SAE features for the eval proteins
    logger.info("Extracting SAE features for %d eval proteins...", len(ids_all))
    import torch

    model = registry.load_model(args.model, device=args.device)

    all_z_chunks = []
    eval_ids_with_acts = []
    BS = args.batch_size
    for i in range(0, len(seqs_all), BS):
        batch_seqs = seqs_all[i : i + BS]
        batch_ids = ids_all[i : i + BS]
        acts = model.get_activations(batch_seqs, layer=layer, pooling="mean")
        acts = acts.to(args.device)
        with torch.no_grad():
            z = cache.encode(sae, acts)
        all_z_chunks.append(z.cpu())
        eval_ids_with_acts.extend(batch_ids)

    Z = torch.cat(all_z_chunks, dim=0)  # (N, d_sae)
    logger.info("Feature activations shape: %s", tuple(Z.shape))

    run_name = args.run_name or f"{args.model}_L{layer}"
    report = probe_go_terms(
        feature_acts=Z,
        protein_ids=eval_ids_with_acts,
        go_labels=go_labels,
        go_names=go_names,
        min_positives=args.min_positives,
        max_go_terms=args.max_go_terms,
        model_name=args.model,
        layer=layer,
        sae_variant="topk",
        return_per_feature_aurocs=False,
        compute_held_out_baselines=args.held_out_baselines,
        held_out_fraction=args.held_out_fraction,
        go_dag=go_dag,
        min_hard_negatives=args.min_hard_negatives,
    )

    auroc_threshold = 0.70
    passed = report.fraction_above_auroc(auroc_threshold) > 0
    logger.info(
        "Any GO term with single-feature AUROC > %.2f: %s",
        auroc_threshold, "yes" if passed else "no",
    )

    # Serialize report
    results_data = [
        {
            "go_id": r.go_id,
            "go_name": r.go_name,
            "n_positives": r.n_positives,
            "n_total": r.n_total,
            "single_feature_auroc": r.single_feature_auroc,
            "best_feature_idx": r.best_feature_idx,
            "multivariate_auroc": r.multivariate_auroc,
            "held_out_auroc": r.held_out_auroc,
            "hard_negative_held_out_auroc": r.hard_negative_held_out_auroc,
            "n_hard_negatives": r.n_hard_negatives,
            "auroc_inflation": r.auroc_inflation,
        }
        for r in report.results
    ]
    _save_json(results_data, output_dir / "go_probing_results.json")

    summary = {
        "n_go_terms_tested": report.n_go_terms_tested,
        "n_go_terms_eligible": report.n_go_terms_eligible,
        "max_go_terms_capped": report.n_go_terms_eligible > report.n_go_terms_tested,
        "fraction_above_0.65": report.fraction_above_auroc(0.65),
        "fraction_above_0.70": report.fraction_above_auroc(0.70),
        "fraction_above_0.75": report.fraction_above_auroc(0.75),
        "fraction_above_0.80": report.fraction_above_auroc(0.80),
        "any_term_above_0.70": passed,
        "model": args.model,
        "layer": layer,
        "step": step,
        "held_out_baselines_computed": args.held_out_baselines,
    }
    if args.held_out_baselines:
        inflation_rows = [
            (r.auroc_inflation, r.best_feature_idx)
            for r in report.results if r.auroc_inflation is not None
        ]
        hard_neg_aurocs = [
            r.hard_negative_held_out_auroc for r in report.results
            if r.hard_negative_held_out_auroc is not None
        ]
        if inflation_rows:
            inflations, inflation_feat_idx = zip(*inflation_rows)
            summary["mean_auroc_inflation"] = float(sum(inflations) / len(inflations))
            ci_result = bootstrap_mean_auroc_inflation(
                np.array(inflations), np.array(inflation_feat_idx),
            )
            summary["mean_auroc_inflation_ci95"] = list(ci_result.ci)
            summary["mean_auroc_inflation_ci_n_clusters"] = ci_result.n_clusters
        else:
            summary["mean_auroc_inflation"] = None
            summary["mean_auroc_inflation_ci95"] = None
            summary["mean_auroc_inflation_ci_n_clusters"] = 0
        summary["n_go_terms_with_held_out_auroc"] = len(inflation_rows)
        summary["mean_hard_negative_held_out_auroc"] = (
            float(sum(hard_neg_aurocs) / len(hard_neg_aurocs)) if hard_neg_aurocs else None
        )
        summary["n_go_terms_with_hard_negative_auroc"] = len(hard_neg_aurocs)
    _save_json(summary, output_dir / "go_probing_summary.json")

    print("\n" + str(report))

    # ── WandB logging ─────────────────────────────────────────────────────────
    if args.wandb_project and not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=run_name + "_eval")
            provenance["wandb_run_id"] = wandb.run.id
            provenance["wandb_run_url"] = wandb.run.url
            wandb.log({
                "eval/frac_variance_explained": recon.frac_variance_explained,
                "eval/dead_fraction": recon.dead_fraction,
                "eval/mean_l0": recon.mean_l0,
                "eval/go_auroc_frac_above_0.70": report.fraction_above_auroc(0.70),
                "eval/go_auroc_frac_above_0.75": report.fraction_above_auroc(0.75),
                "eval/go_auroc_frac_above_0.80": report.fraction_above_auroc(0.80),
                "eval/any_term_above_0.70": int(passed),
            })
            wandb.finish()
        except ImportError:
            pass

    # Written once per output_dir as a sidecar, so the result files keep a
    # stable schema.
    _save_json(provenance, output_dir / "provenance.json")

    logger.info("Eval results written to %s", output_dir)


def _save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
