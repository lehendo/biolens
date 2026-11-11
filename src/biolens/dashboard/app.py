"""
BioLens feature browser dashboard (Gradio).

Relation to InterPLM (Simon & Zou 2024): InterPLM covers ESM2 specifically and already
has a public browser. This dashboard supports the verification workflow in this
repository: it shows each feature's top-activating sequences next to its claimed GO
annotations and its recorded verification status.

Usage:
  python -m biolens.dashboard.app --model esm2_8m --layer 5 \\
      --sae-checkpoint /path/to/final.pt --cache /path/to/cache \\
      [--go-probing-results /path/to/go_probing_results.json] \\
      [--verified-annotations configs/verified_features/esm2_8m_layer5_topk_k32.yaml]
  biolens-dashboard --model esm2_8m --layer 5 --cache /path/to/cache ...

Note: launching this on a remote SLURM cluster requires either --share
(public Gradio link) or SSH port-forwarding to view it in a local browser.
For headless feature investigation with no browser at all, use
scripts/inspect_features.py instead — it shares the same underlying
biolens.eval.feature_inspection logic as this dashboard's "Feature Detail" tab.

IMPORTANT: GO annotations shown here are single-feature-AUROC candidates, not
proof of what a feature encodes: the best-AUROC feature among many candidates is
biased upward by the search. Pass --verified-annotations to
show a feature's real confirmed/spurious/plausible status wherever it's been
checked; annotations without a verified-annotations entry are always labeled
UNVERIFIED in the UI.
"""

from __future__ import annotations

import argparse
import logging

from biolens.eval.feature_inspection import (
    VerifiedAnnotation,
    inspect_feature,
    load_cache_features,
    load_go_annotations_by_feature,
    load_verified_annotations,
)

logger = logging.getLogger(__name__)


def build_dashboard(
    sae,
    model,
    cache,
    layer: int,
    feature_annotations: dict[int, list[str]] | None = None,
    verified_annotations: dict[int, VerifiedAnnotation] | None = None,
    top_k_seqs: int = 10,
    max_cache_seqs: int = 5000,
):
    """
    Build and return a Gradio Blocks app.

    Args:
        sae:                  Trained SAE (any variant).
        model:                Loaded BioModel (for encoding user-input sequences).
        cache:                ActivationCache — source of pre-extracted activations,
                               sequences, and normalization stats for both the
                               "Analyze Sequence" and "Feature Detail" tabs.
        layer:                Layer index the SAE was trained on (used as the
                               default in the "Analyze Sequence" tab).
        feature_annotations:  Optional feature_idx -> list of GO annotation
                               strings, from load_go_annotations_by_feature().
                               These are UNVERIFIED AUROC-based candidates —
                               always labeled as such unless overridden by
                               verified_annotations.
        verified_annotations: Optional feature_idx -> VerifiedAnnotation, from
                               load_verified_annotations(). Takes precedence
                               over feature_annotations wherever present —
                               shows a feature's real confirmed/spurious/
                               plausible status instead of a raw AUROC claim.
        top_k_seqs:           Number of top-activating sequences to display per feature.
        max_cache_seqs:       Cap on how much of the cache to preload for display
                               (the whole cache can be hundreds of thousands of
                               vectors; this keeps dashboard startup fast).
    """
    import gradio as gr

    feature_annotations = feature_annotations or {}
    verified_annotations = verified_annotations or {}

    logger.info("Preloading up to %d cached sequences for the feature browser...", max_cache_seqs)
    cached_ids, cached_seqs, cached_acts = load_cache_features(
        cache, sae, max_seqs=max_cache_seqs, device="cpu"
    )
    logger.info("Loaded %d cached sequences.", len(cached_ids))

    # ── Sequence → feature activations ───────────────────────────────────────
    def analyze_sequence(sequence: str, layer_str: str) -> tuple[str, str]:
        sequence = sequence.strip().upper()
        if not sequence:
            return "Please enter a sequence.", ""

        try:
            seq_layer = int(layer_str)
            acts = model.get_activations([sequence], layer=seq_layer, pooling="mean")
            acts = acts.to(next(sae.parameters()).device)
            z = cache.encode(sae, acts).squeeze(0).cpu()
        except Exception as exc:
            return f"Error: {exc}", ""

        active_mask = z > 0
        n_active = int(active_mask.sum().item())

        rows = []
        top_vals, top_idx = z.topk(min(20, max(n_active, 1)))
        for rank, (feat_idx, val) in enumerate(
            zip(top_idx.tolist(), top_vals.tolist()), start=1
        ):
            v = verified_annotations.get(feat_idx)
            if v is not None:
                info = f"[{v.status.upper()}] {v.real_identity}"
            else:
                anns = feature_annotations.get(feat_idx)
                info = f"[UNVERIFIED] {anns[0]}" if anns else ""
            rows.append(
                f"  #{rank:2d}  Feature {feat_idx:5d}  activation={val:.3f}  {info}"
            )

        summary = (
            f"Sequence length: {len(sequence)}\n"
            f"Active features (L0): {n_active} / {sae.cfg.d_sae} "
            f"({100 * n_active / sae.cfg.d_sae:.1f}%)\n"
            f"Top active features:\n" + "\n".join(rows)
        )
        return summary, str(top_idx[0].item()) if n_active > 0 else ""

    # ── Feature detail: top-activating sequences ──────────────────────────────
    def show_feature(feature_idx_str: str) -> str:
        try:
            feat_idx = int(feature_idx_str.strip())
        except ValueError:
            return "Please enter a valid feature index."
        if feat_idx < 0 or feat_idx >= sae.cfg.d_sae:
            return f"Feature index out of range (0..{sae.cfg.d_sae - 1})."

        report = inspect_feature(
            feat_idx,
            cached_ids,
            cached_seqs,
            cached_acts,
            top_k=top_k_seqs,
            go_annotations=feature_annotations.get(feat_idx),
            verified=verified_annotations.get(feat_idx),
        )
        return str(report)

    # ── Gradio layout ─────────────────────────────────────────────────────────
    model_cfg = getattr(model, "config", None)
    model_name = model_cfg.name if model_cfg else "unknown"

    with gr.Blocks(title="BioLens Feature Browser") as demo:
        gr.Markdown(
            f"# BioLens Feature Browser\n"
            f"**Model:** `{model_name}` (layer {layer})  |  "
            f"**SAE:** d_sae={sae.cfg.d_sae}, k={getattr(sae, 'k', '?')}\n\n"
            "Explore sparse autoencoder features learned from biological foundation models.\n"
            "[GitHub](https://github.com/lehendo/biolens)"
        )

        with gr.Tab("Analyze Sequence"):
            with gr.Row():
                seq_input = gr.Textbox(
                    label="Protein sequence (amino acids)",
                    placeholder="MKTAYIAKQRQISFVK...",
                    lines=4,
                )
                layer_input = gr.Textbox(
                    label="Layer index", value=str(layer), max_lines=1
                )
            analyze_btn = gr.Button("Analyze")
            feature_output = gr.Textbox(label="Active features", lines=25, interactive=False)
            top_feature_id = gr.Textbox(label="Jump to top feature", visible=False)
            analyze_btn.click(
                analyze_sequence,
                inputs=[seq_input, layer_input],
                outputs=[feature_output, top_feature_id],
            )

        with gr.Tab("Feature Detail"):
            feat_idx_input = gr.Textbox(
                label="Feature index", placeholder="e.g. 42", max_lines=1
            )
            feat_detail_btn = gr.Button("Show top-activating sequences")
            feat_detail_output = gr.Textbox(
                label="Top-activating sequences", lines=20, interactive=False
            )
            feat_detail_btn.click(
                show_feature,
                inputs=[feat_idx_input],
                outputs=[feat_detail_output],
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="BioLens feature browser dashboard")
    parser.add_argument("--model", default="esm2_8m", help="Model name from registry")
    parser.add_argument("--layer", type=int, default=-1, help="Layer (-1 = last)")
    parser.add_argument("--sae-checkpoint", required=True, help="Path to SAE checkpoint .pt")
    parser.add_argument("--cache", required=True, help="Path to ActivationCache directory")
    parser.add_argument(
        "--go-probing-results", default=None,
        help="Path to go_probing_results.json from run_eval.py — annotates "
        "features in the dashboard with the GO terms they best match "
        "(UNVERIFIED AUROC-based candidates unless overridden by --verified-annotations)",
    )
    parser.add_argument(
        "--verified-annotations", default=None,
        help="Path to a configs/verified_features/*.yaml registry — shows a "
        "feature's real confirmed/spurious/plausible status wherever it's "
        "been manually checked, instead of the raw unverified GO-probing claim",
    )
    parser.add_argument("--max-cache-seqs", type=int, default=5000)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    # Fallback args for checkpoints saved before architecture metadata existed.
    parser.add_argument("--variant", default=None)
    parser.add_argument("--expansion-factor", type=int, default=None)
    args = parser.parse_args()

    from biolens.models.registry import ModelRegistry
    from biolens.sae.dictionary import ActivationCache
    from biolens.sae.train import build_sae_from_checkpoint

    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    layer = args.layer if args.layer >= 0 else model_cfg.num_layers - 1

    model = registry.load_model(args.model)
    cache = ActivationCache(args.cache)

    sae, step = build_sae_from_checkpoint(
        args.sae_checkpoint,
        variant=args.variant,
        d_model=model_cfg.hidden_dim,
        expansion_factor=args.expansion_factor,
    )
    logger.info("Loaded SAE checkpoint from step %d", step)

    feature_annotations: dict[int, list[str]] = {}
    if args.go_probing_results:
        feature_annotations = load_go_annotations_by_feature(args.go_probing_results)
        logger.info(
            "Loaded GO annotations for %d features from %s",
            len(feature_annotations), args.go_probing_results,
        )

    verified_annotations: dict[int, VerifiedAnnotation] = {}
    if args.verified_annotations:
        verified_annotations = load_verified_annotations(args.verified_annotations)
        logger.info(
            "Loaded %d verified feature annotations from %s",
            len(verified_annotations), args.verified_annotations,
        )

    demo = build_dashboard(
        sae, model, cache, layer,
        feature_annotations=feature_annotations,
        verified_annotations=verified_annotations,
        max_cache_seqs=args.max_cache_seqs,
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
