"""
Feature-level inspection: given specific SAE feature indices, find their
top-activating examples in a cached activation set, and cross-reference GO
term annotations from a probing run.

This is the same analysis the dashboard's "Feature Detail" tab performs
interactively — extracted here as plain, testable functions so it can also
run headlessly (no Gradio, no port-forwarding needed on a remote cluster)
via scripts/inspect_features.py, and so the dashboard and CLI tool share one
correct code path instead of maintaining two.

IMPORTANT — single-feature AUROC is a candidate signal, not proof: the feature with
the best AUROC among many candidates is biased upward by the search, and a feature can
win an AUROC by tracking a broad class that overlaps the claimed concept without
encoding it. This module surfaces that distinction directly: a feature's raw
AUROC-based GO annotation is always labeled "unverified" unless a verified_features
registry entry says otherwise, so a debunked claim can never silently look like an
established finding in the dashboard or CLI output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import torch
import yaml
from torch import Tensor

from biolens.sae.architectures import BaseSAE
from biolens.sae.dictionary import ActivationCache


@dataclass
class FeatureExample:
    protein_id: str
    sequence: str
    activation: float


@dataclass
class VerifiedClaim:
    """One (GO term(s), feature) claim and its verdict. The unit that gets
    verified is the CLAIM, not the feature: the same feature backing two
    different GO terms is two distinct claims that can legitimately get
    different verdicts (e.g. a kinase-like feature confirmed for one term and
    spurious for another)."""

    status: str  # "confirmed" | "spurious" | "plausible" | "unverifiable"
    claimed_go: list[str]
    claimed_concept: str
    confidence: str = ""  # "high" | "medium" | "low"
    justification: str = ""
    dominant_cluster: str = ""
    plausible_subcase: str | None = None  # "a" | "b" | "c" (only when status == plausible)
    spurious_category: str | None = None  # "S1" | "S2" | "S3" -- there is deliberately no S4
    verification_claim_id: str = ""  # opaque blinded id, links to the verdict file


@dataclass
class VerifiedAnnotation:
    """The outcome of manually checking a feature's real top-activating
    sequences against ground truth (e.g. UniProt), for one specific trained
    SAE checkpoint. See configs/verified_features/*.yaml.

    A feature may carry one or more claims (`claims`). `real_identity` and
    `evidence_accessions` are feature-level -- the exemplar sequences a
    verifier inspects are a property of the feature, shared by every claim
    on it -- while `status`/`claimed_go`/`claimed_concept` are per-claim.

    The top-level `status`/`claimed_go`/`claimed_concept` fields are kept for
    backward compatibility with legacy single-claim entries and existing
    consumers. For a multi-claim feature they are DERIVED: `claimed_go` is
    the union of every claim's GO terms, `claimed_concept` joins the
    concepts, and `status` is the shared verdict if all claims agree, else
    "mixed". "mixed" is deliberately NOT in statistics.STATUS_TO_TARGET, so a
    consumer that reads only the legacy field excludes such a feature rather
    than silently attributing one claim's verdict to another -- use
    `claim_for(go_id)` to get the per-claim verdict."""

    status: str  # "confirmed" | "spurious" | "plausible" | "unverifiable" | "mixed"
    claimed_go: list[str]
    claimed_concept: str
    real_identity: str
    evidence_accessions: list[str] = field(default_factory=list)
    claims: list[VerifiedClaim] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.claims:
            # Legacy single-claim construction: synthesize the one claim.
            self.claims = [VerifiedClaim(self.status, list(self.claimed_go), self.claimed_concept)]

    def claim_for(self, go_id: str) -> VerifiedClaim | None:
        """The claim that covers `go_id`, or None if this feature was never
        verified for that GO term (even if it was verified for another)."""
        for claim in self.claims:
            if go_id in claim.claimed_go:
                return claim
        return None


@dataclass
class FeatureReport:
    feature_idx: int
    activation_rate: float  # fraction of the evaluated set firing on this feature
    max_activation: float
    mean_activation_when_active: float
    top_examples: list[FeatureExample]
    go_annotations: list[str] = field(default_factory=list)
    verified: VerifiedAnnotation | None = None

    def __str__(self) -> str:
        lines = [
            f"Feature {self.feature_idx}",
            f"  activation rate:      {self.activation_rate * 100:.2f}%",
            f"  max activation:       {self.max_activation:.3f}",
            f"  mean | active:        {self.mean_activation_when_active:.3f}",
        ]

        if self.verified is not None:
            v = self.verified
            badge = {
                "confirmed": "CONFIRMED",
                "spurious": "SPURIOUS — claimed GO term is WRONG",
                "plausible": "PLAUSIBLE (not fully confirmed)",
            }.get(v.status, v.status.upper())
            lines.append(f"  verification status:  [{badge}]")
            if len(v.claims) > 1:
                for claim in v.claims:
                    claim_badge = {
                        "confirmed": "CONFIRMED",
                        "spurious": "SPURIOUS",
                        "plausible": "PLAUSIBLE",
                    }.get(claim.status, claim.status.upper())
                    lines.append(
                        f"  claim [{claim_badge}]: {claim.claimed_concept} "
                        f"({', '.join(claim.claimed_go)})"
                    )
            else:
                lines.append(f"  claimed concept:       {v.claimed_concept} ({', '.join(v.claimed_go)})")
            lines.append(f"  real identity:         {v.real_identity}")
            if v.evidence_accessions:
                lines.append(f"  evidence (UniProt):    {', '.join(v.evidence_accessions)}")
        elif self.go_annotations:
            lines.append(
                "  GO annotations (UNVERIFIED — AUROC-based candidates only, not yet "
                "checked against real sequences):"
            )
            for ann in self.go_annotations:
                lines.append(f"    - {ann}")
        else:
            lines.append("  GO annotations:        none (not the top feature for any tested GO term)")

        lines.append(f"  top {len(self.top_examples)} activating sequences:")
        for rank, ex in enumerate(self.top_examples, start=1):
            preview = ex.sequence[:70] + ("..." if len(ex.sequence) > 70 else "")
            lines.append(
                f"    #{rank:2d}  [{ex.activation:.3f}]  {ex.protein_id:14s}  {preview}"
            )
        return "\n".join(lines)


def load_cache_features(
    cache: ActivationCache,
    sae: BaseSAE,
    max_seqs: int | None = None,
    device: str = "cpu",
    feature_indices: list[int] | None = None,
) -> tuple[list[str], list[str], Tensor]:
    """
    Load (protein_ids, sequences, feature_acts) for every vector in the cache
    (or the first max_seqs, if given).

    Raw activations are normalized with the cache's stored (mean, scale)
    stats before encoding — the SAE was trained on normalized activations
    (see ActivationCache.iter_batches(normalize=True)), so feeding it raw
    activations directly would silently produce wrong feature activations.

    Args:
        cache, sae, max_seqs, device: as before.
        feature_indices: If given, only keep these specific SAE feature
            columns in the returned tensor (column i of the result
            corresponds to feature_indices[i], NOT SAE feature index i —
            see the Returns note below). For large d_sae, materializing the
            full (N, d_sae) matrix across the whole cache can exceed
            available memory. Discarding unwanted columns immediately after
            each shard's encode() call cuts peak memory from
            O(N * d_sae) to O(N * len(feature_indices)) — for a typical
            inspect_features.py call (a handful of requested indices), a
            reduction of two to three orders of magnitude. Pass None only
            when the caller genuinely needs arbitrary/all features
            afterward (e.g. the interactive dashboard, which doesn't know
            which feature a user will request until browse time);
            scripts/inspect_features.py always knows its exact
            --feature-idx list upfront and should always pass it here.

    Returns:
        ids:       list[str], length N
        sequences: list[str], length N (empty strings if the shard didn't
                   store raw sequences)
        feat_acts: (N, d_sae) tensor of SAE feature activations, on CPU —
                   or (N, len(feature_indices)) if feature_indices was
                   given, in which case column i holds SAE feature
                   feature_indices[i], NOT column feature_indices[i]
                   itself. Callers using a compacted matrix must track
                   this remapping themselves (see inspect_feature's
                   column_idx parameter, and scripts/inspect_features.py
                   for the reference usage).
    """
    sae = sae.to(device)
    sae.eval()

    ids: list[str] = []
    sequences: list[str] = []
    acts_chunks: list[Tensor] = []

    idx_tensor = (
        torch.tensor(feature_indices, dtype=torch.long)
        if feature_indices is not None
        else None
    )

    with torch.no_grad():
        for shard_path, _ in cache._iter_shards():
            if max_seqs is not None and len(ids) >= max_seqs:
                break

            with h5py.File(shard_path, "r") as f:
                raw = torch.from_numpy(f["activations"][:].astype("float32"))
                shard_ids = [s.decode() for s in f["ids"][:]]
                shard_seqs = (
                    [s.decode() for s in f["sequences"][:]]
                    if "sequences" in f
                    else [""] * len(raw)
                )

            z = cache.encode(sae, raw.to(device)).cpu()
            if idx_tensor is not None:
                # Discard unwanted columns NOW, before this shard's z ever
                # reaches acts_chunks — the whole point of feature_indices
                # is to never let the full-width tensor accumulate.
                z = z.index_select(1, idx_tensor)

            ids.extend(shard_ids)
            sequences.extend(shard_seqs)
            acts_chunks.append(z)

    default_width = len(feature_indices) if feature_indices is not None else sae.cfg.d_sae
    feat_acts = (
        torch.cat(acts_chunks, dim=0) if acts_chunks else torch.zeros(0, default_width)
    )
    if max_seqs is not None and len(ids) > max_seqs:
        ids = ids[:max_seqs]
        sequences = sequences[:max_seqs]
        feat_acts = feat_acts[:max_seqs]

    return ids, sequences, feat_acts


def inspect_feature(
    feature_idx: int,
    ids: list[str],
    sequences: list[str],
    feat_acts: Tensor,
    top_k: int = 15,
    go_annotations: list[str] | None = None,
    verified: VerifiedAnnotation | None = None,
    column_idx: int | None = None,
) -> FeatureReport:
    """
    Build a FeatureReport for one feature index from precomputed activations.

    Args:
        feature_idx: The feature's true SAE index — always used as the
            returned FeatureReport's label, and (when column_idx is None)
            also which column of feat_acts to read.
        column_idx: Which column of feat_acts actually holds this feature's
            activations. Defaults to feature_idx — correct whenever
            feat_acts is the full (N, d_sae) matrix (the historical
            behavior, and still what the dashboard does). Pass a different
            value when feat_acts has been compacted to only a requested
            subset of columns via load_cache_features'
            feature_indices=... — e.g. column_idx=i when feat_acts's
            column i corresponds to SAE feature feature_idx (see
            scripts/inspect_features.py for the reference usage).
    """
    if column_idx is None:
        column_idx = feature_idx
    width = feat_acts.shape[1]
    if not (0 <= column_idx < width):
        raise ValueError(f"column_idx {column_idx} out of range (0..{width - 1})")

    col = feat_acts[:, column_idx]
    active_mask = col > 0
    activation_rate = active_mask.float().mean().item() if len(col) else 0.0
    max_activation = col.max().item() if len(col) else 0.0
    mean_active = col[active_mask].mean().item() if active_mask.any() else 0.0

    top_n = min(top_k, len(col))
    if top_n > 0:
        top_vals, top_idx = col.topk(top_n)
    else:
        top_vals, top_idx = col, col.long()

    examples = [
        FeatureExample(
            protein_id=ids[i] if i < len(ids) else "?",
            sequence=sequences[i] if i < len(sequences) else "",
            activation=v.item(),
        )
        for v, i in zip(top_vals, top_idx.tolist())
    ]

    return FeatureReport(
        feature_idx=feature_idx,
        activation_rate=activation_rate,
        max_activation=max_activation,
        mean_activation_when_active=mean_active,
        top_examples=examples,
        go_annotations=go_annotations or [],
        verified=verified,
    )


def load_go_annotations_by_feature(go_probing_results_path: str | Path) -> dict[int, list[str]]:
    """
    Build a reverse index: feature_idx -> list of human-readable GO annotation
    strings, from a run_eval.py `go_probing_results.json` file.

    A single feature can be the best_feature_idx for multiple GO terms — all
    are kept (sorted by AUROC descending), not just the single best, since a
    feature being the top hit for several *related* GO terms (e.g. multiple
    cholinergic-signaling terms) is itself a meaningful interpretability signal.
    """
    with open(go_probing_results_path) as f:
        results = json.load(f)

    by_feature: dict[int, list[tuple[float, str]]] = {}
    for r in results:
        feat = r["best_feature_idx"]
        label = f"{r['go_id']} {r['go_name']} (auroc={r['single_feature_auroc']:.3f})"
        by_feature.setdefault(feat, []).append((r["single_feature_auroc"], label))

    return {
        feat: [label for _, label in sorted(entries, key=lambda t: t[0], reverse=True)]
        for feat, entries in by_feature.items()
    }


def load_verified_annotations(path: str | Path) -> dict[int, VerifiedAnnotation]:
    """
    Load a verified-features registry (see configs/verified_features/*.yaml)
    mapping feature_idx -> VerifiedAnnotation, for one specific trained SAE.

    These files record the outcome of manually checking a feature's real
    top-activating sequences against ground truth — they are NOT regenerated
    automatically and must be scoped to the exact (model, layer, SAE variant,
    k) checkpoint they were built from, since feature indices are meaningless
    across different trained SAEs.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    return {
        int(feat_idx): _parse_registry_entry(int(feat_idx), entry)
        for feat_idx, entry in ((data or {}).get("features") or {}).items()
    }


def _parse_registry_entry(feat_idx: int, entry: dict) -> VerifiedAnnotation:
    """Parse one registry entry, in either schema form:

    Legacy single-claim (top-level `status`/`claimed_go`/`claimed_concept`):
        7: {status: confirmed, claimed_go: [GO:...], claimed_concept: ..., real_identity: ...}

    Multi-claim (`claims:` list; `real_identity`/`evidence_accessions` stay
    feature-level because the exemplar sequences belong to the feature):
        1835: {real_identity: ..., evidence_accessions: [...],
               claims: [{status: spurious, claimed_go: [GO:A], claimed_concept: ...},
                        {status: confirmed, claimed_go: [GO:B], claimed_concept: ...}]}

    Raises ValueError on a malformed multi-claim entry rather than guessing:
    a GO term claimed twice on one feature (that (go_id, feature) pair would
    carry two possibly-conflicting verdicts), an empty `claims` list, or a
    claim missing `status`/`claimed_go`.
    """
    if "claims" not in entry:
        return VerifiedAnnotation(
            status=entry["status"],
            claimed_go=entry.get("claimed_go", []),
            claimed_concept=entry.get("claimed_concept", ""),
            real_identity=entry.get("real_identity", ""),
            evidence_accessions=entry.get("evidence_accessions", []),
        )

    raw_claims = entry["claims"]
    if not raw_claims:
        raise ValueError(f"feature {feat_idx}: `claims` is empty")
    claims: list[VerifiedClaim] = []
    seen_go: set[str] = set()
    for i, c in enumerate(raw_claims):
        if "status" not in c or not c.get("claimed_go"):
            raise ValueError(
                f"feature {feat_idx}: claim #{i} needs both `status` and a non-empty `claimed_go`"
            )
        dupes = seen_go & set(c["claimed_go"])
        if dupes:
            raise ValueError(
                f"feature {feat_idx}: GO term(s) {sorted(dupes)} claimed more than once -- "
                f"one (go_id, feature) pair can only carry one verdict"
            )
        seen_go.update(c["claimed_go"])
        if c.get("spurious_category") not in (None, "S1", "S2", "S3"):
            raise ValueError(
                f"feature {feat_idx}: spurious_category {c['spurious_category']!r} is not S1/S2/S3 -- "
                f"a spurious call resting on pure absence (S4) is unverifiable, never spurious"
            )
        claims.append(
            VerifiedClaim(
                c["status"], list(c["claimed_go"]), c.get("claimed_concept", ""),
                confidence=c.get("confidence", ""), justification=c.get("justification", ""),
                dominant_cluster=c.get("dominant_cluster", ""),
                plausible_subcase=c.get("plausible_subcase"),
                spurious_category=c.get("spurious_category"),
                verification_claim_id=c.get("verification_claim_id", ""),
            )
        )

    statuses = {c.status for c in claims}
    return VerifiedAnnotation(
        status=statuses.pop() if len(statuses) == 1 else "mixed",
        claimed_go=[g for c in claims for g in c.claimed_go],
        claimed_concept="; ".join(c.claimed_concept for c in claims if c.claimed_concept),
        real_identity=entry.get("real_identity", ""),
        evidence_accessions=entry.get("evidence_accessions", []),
        claims=claims,
    )
