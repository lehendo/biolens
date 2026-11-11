"""
Inspect specific SAE feature indices: print their top-activating sequences,
activation statistics, and any GO term annotations they were the best match
for in a prior run_eval.py run.

Headless (no Gradio, no browser, no port-forwarding) — the right tool for a
remote SLURM cluster. For interactive browsing, use biolens.dashboard.app
instead, which uses the same underlying biolens.eval.feature_inspection logic.

Usage:
  python scripts/inspect_features.py \\
      --sae-checkpoint checkpoints/esm2_8m_layer5_topk/.../final.pt \\
      --cache data/activations/esm2_8m_layer5 \\
      --feature-idx 10 20 30 \\
      --top-k 15 \\
      --go-probing-results eval/esm2_8m_layer5/go_probing_results.json \\
      --verified-annotations examples/verified_features_example.yaml

Note: a high single-feature AUROC is a CANDIDATE signal, not proof — always
check the printed top-activating sequences against real ground truth before
trusting a GO annotation. Once checked, record the outcome in a
configs/verified_features/*.yaml file (see that directory for the format)
so the verification survives beyond this one conversation; pass it via
--verified-annotations and any already-checked feature will show its real
confirmed/spurious/plausible status instead of the raw unverified AUROC claim.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.feature_inspection import (
    inspect_feature,
    load_cache_features,
    load_go_annotations_by_feature,
    load_verified_annotations,
)
from biolens.sae.dictionary import ActivationCache
from biolens.sae.train import build_sae_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-checkpoint", required=True, help="Path to SAE checkpoint .pt")
    p.add_argument("--cache", required=True, help="ActivationCache directory")
    p.add_argument(
        "--feature-idx", type=int, nargs="+", required=True,
        help="One or more SAE feature indices to inspect",
    )
    p.add_argument("--top-k", type=int, default=15, help="Top-activating sequences per feature")
    p.add_argument(
        "--max-seqs", type=int, default=None,
        help="Limit how much of the cache to scan (default: all)",
    )
    p.add_argument(
        "--go-probing-results", default=None,
        help="Path to go_probing_results.json from run_eval.py — cross-references "
        "which GO terms (if any) this feature was the best single-feature match for. "
        "These are UNVERIFIED AUROC-based candidates unless overridden by --verified-annotations.",
    )
    p.add_argument(
        "--verified-annotations", default=None,
        help="Path to a configs/verified_features/*.yaml registry — if a requested "
        "feature has an entry there, shows its confirmed/spurious/plausible status "
        "instead of the raw unverified GO-probing claim",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None, help="Optional path to also write results as JSON")
    # Fallback args for checkpoints saved before architecture metadata existed
    # (see build_sae_from_checkpoint) — only needed for pre-fix checkpoints.
    p.add_argument("--variant", default=None, help="SAE variant (only needed for legacy checkpoints)")
    p.add_argument("--d-model", type=int, default=None, help="Only needed for legacy checkpoints")
    p.add_argument("--expansion-factor", type=int, default=None, help="Only needed for legacy checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sae, step = build_sae_from_checkpoint(
        args.sae_checkpoint,
        device=args.device,
        variant=args.variant,
        d_model=args.d_model,
        expansion_factor=args.expansion_factor,
    )
    logger.info("Loaded SAE from step %d (d_sae=%d, k=%s)", step, sae.cfg.d_sae, getattr(sae, "k", "?"))

    cache = ActivationCache(args.cache)
    logger.info("Scanning cache (%d total vectors)...", len(cache))
    ids, sequences, feat_acts = load_cache_features(
        cache, sae, max_seqs=args.max_seqs, device=args.device,
        feature_indices=args.feature_idx,
    )
    logger.info("Loaded features for %d sequences", len(ids))

    go_by_feature: dict[int, list[str]] = {}
    if args.go_probing_results:
        go_by_feature = load_go_annotations_by_feature(args.go_probing_results)

    verified_by_feature = {}
    if args.verified_annotations:
        verified_by_feature = load_verified_annotations(args.verified_annotations)
        logger.info(
            "Loaded %d verified feature annotations from %s",
            len(verified_by_feature), args.verified_annotations,
        )

    reports = []
    for column_idx, feat_idx in enumerate(args.feature_idx):
        report = inspect_feature(
            feat_idx, ids, sequences, feat_acts,
            top_k=args.top_k,
            go_annotations=go_by_feature.get(feat_idx),
            verified=verified_by_feature.get(feat_idx),
            column_idx=column_idx,
        )
        reports.append(report)
        print("\n" + str(report))

    if args.output:
        import json

        out_data = [
            {
                "feature_idx": r.feature_idx,
                "activation_rate": r.activation_rate,
                "max_activation": r.max_activation,
                "mean_activation_when_active": r.mean_activation_when_active,
                "go_annotations": r.go_annotations,
                "verified": (
                    {
                        "status": r.verified.status,
                        "claimed_go": r.verified.claimed_go,
                        "claimed_concept": r.verified.claimed_concept,
                        "real_identity": r.verified.real_identity,
                        "evidence_accessions": r.verified.evidence_accessions,
                    }
                    if r.verified is not None
                    else None
                ),
                "top_examples": [
                    {"protein_id": e.protein_id, "sequence": e.sequence, "activation": e.activation}
                    for e in r.top_examples
                ],
            }
            for r in reports
        ]
        Path(args.output).write_text(json.dumps(out_data, indent=2))
        logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
