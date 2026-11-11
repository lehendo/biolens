"""
Linear probing: do SAE features linearly predict known biological labels?

Methodology follows Adams et al. (2024) and InterPLM (Simon & Zou 2024),
applied uniformly across all model families so results are directly comparable.

For each GO term with >= min_positives examples in the eval set:
  1. single_feature_auroc: best AUROC achieved by any single SAE feature alone.
     A value >> 0.5 is evidence of a monosemantic, interpretable feature.
  2. multivariate_auroc:   AUROC of logistic regression on all SAE features.
     Upper bound on linear separability.

Reference threshold (configs/eval.yaml::probing.reference_auroc): a GO term
with single_feature_auroc > 0.70.

This module also supports subcellular localization and thermostability probing
(for proteins), motif enrichment probing (for genomic models) — added as the
corresponding annotation datasets are integrated.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import Tensor

if TYPE_CHECKING:
    from biolens.data.uniprot import GODag

logger = logging.getLogger(__name__)


@dataclass
class GOProbingResult:
    """Results for a single GO term."""
    go_id: str
    go_name: str
    n_positives: int
    n_total: int
    single_feature_auroc: float       # max AUROC across individual SAE features
    best_feature_idx: int             # which feature achieved it
    multivariate_auroc: float         # logistic regression on all features
    # Distribution of single-feature AUROCs (for diagnostic plots)
    per_feature_aurocs: np.ndarray | None = field(default=None, repr=False)
    held_out_auroc: float | None = None
    hard_negative_held_out_auroc: float | None = None
    n_hard_negatives: int | None = None

    @property
    def auroc_inflation(self) -> float | None:
        """`single_feature_auroc - held_out_auroc`: how much of the best
        feature's AUROC is attributable to selecting it from many candidates
        on the same data. None if held_out_auroc wasn't computed."""
        if self.held_out_auroc is None:
            return None
        return self.single_feature_auroc - self.held_out_auroc


@dataclass
class GOProbingReport:
    """Aggregate probing results across all GO terms."""
    results: list[GOProbingResult]
    n_go_terms_tested: int
    model_name: str
    layer: int
    sae_variant: str
    n_go_terms_eligible: int = 0
    """Terms meeting min_positives BEFORE the --max-go-terms cap, if any, is
    applied -- 0 for callers/tests that don't pass it, so this stays optional
    without breaking existing construction sites. Reporting both counts side by
    side makes it visible when the max_go_terms cap truncates the ranked list
    regardless of threshold."""

    def fraction_above_auroc(self, threshold: float) -> float:
        if not self.results:
            return 0.0
        above = sum(r.single_feature_auroc >= threshold for r in self.results)
        return above / len(self.results)

    def top_results(self, n: int = 10) -> list[GOProbingResult]:
        return sorted(self.results, key=lambda r: r.single_feature_auroc, reverse=True)[:n]

    def __str__(self) -> str:
        cap_note = ""
        if self.n_go_terms_eligible and self.n_go_terms_eligible > self.n_go_terms_tested:
            cap_note = f"  (--max-go-terms CAPPED: {self.n_go_terms_eligible} eligible)"
        lines = [
            f"GO probing — {self.model_name} layer {self.layer} ({self.sae_variant})",
            f"  GO terms tested:        {self.n_go_terms_tested}{cap_note}",
            f"  AUROC > 0.65:           {self.fraction_above_auroc(0.65)*100:.1f}%",
            f"  AUROC > 0.70:           {self.fraction_above_auroc(0.70)*100:.1f}%",
            f"  AUROC > 0.75:           {self.fraction_above_auroc(0.75)*100:.1f}%",
            f"  AUROC > 0.80:           {self.fraction_above_auroc(0.80)*100:.1f}%",
            "",
            "  Top 10 GO terms:",
        ]
        for r in self.top_results(10):
            lines.append(
                f"    {r.go_id:12s} {r.go_name[:35]:35s}  "
                f"feat={r.single_feature_auroc:.3f}  multi={r.multivariate_auroc:.3f}  "
                f"n+={r.n_positives}"
            )
        return "\n".join(lines)


def _probe_single_term(
    Z: np.ndarray,
    protein_ids: list[str],
    y: np.ndarray,
    go_id: str,
    go_name: str,
    go_labels: dict[str, set[str]],
    splitter: StratifiedShuffleSplit,
    held_out_splitter: StratifiedShuffleSplit,
    min_positives: int,
    compute_held_out_baselines: bool,
    go_dag: GODag | None,
    min_hard_negatives: int,
    return_per_feature_aurocs: bool,
    random_state: int,
    compute_multivariate_auroc: bool = True,
) -> GOProbingResult | None:
    """
    Run the full single-feature/multivariate/held-out/hard-negative probing
    pipeline for ONE already-constructed label array `y` against one term
    identity (`go_id`/`go_name`, used only for logging/attribution — the
    actual positives are whatever `y` says, not re-derived from go_labels).

    compute_multivariate_auroc: Set False to skip the LogisticRegression fit
        entirely (multivariate_auroc comes back NaN). The multivariate fit is
        expensive for large d_sae; callers that only need best_feature_idx can
        skip it. Defaults True.


    Returns None if `y` doesn't have enough positives/negatives for a valid
    split (mirrors probe_go_terms' previous inline `continue` behavior).
    """
    n_pos = int(y.sum())
    if n_pos < min_positives or n_pos >= len(y) - min_positives:
        return None

    # Train/test split stratified by label
    try:
        train_idx, test_idx = next(splitter.split(Z, y))
    except ValueError:
        return None

    Z_train, Z_test = Z[train_idx], Z[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    held_out_auroc: float | None = None
    hard_negative_auroc: float | None = None
    n_hard_negatives: int | None = None

    if compute_held_out_baselines:
        try:
            sel_idx, ho_idx = next(held_out_splitter.split(Z_test, y_test))
        except ValueError:
            sel_idx = np.arange(len(y_test))
            ho_idx = np.array([], dtype=int)
        Z_sel, y_sel = Z_test[sel_idx], y_test[sel_idx]
        Z_ho, y_ho = Z_test[ho_idx], y_test[ho_idx]
        ho_protein_ids = [protein_ids[test_idx[i]] for i in ho_idx]
    else:
        Z_sel, y_sel = Z_test, y_test
        Z_ho, y_ho, ho_protein_ids = None, None, []

    # ── Single-feature AUROC (selection step) ──────────────────────────
    per_feature_aurocs = _vectorized_auroc(Z_sel, y_sel)
    best_feat_idx = int(np.argmax(per_feature_aurocs))
    single_feat_auroc = float(per_feature_aurocs[best_feat_idx])

    if compute_held_out_baselines and Z_ho is not None and len(y_ho) > 0:
        # Direction is fixed by the selection step, never re-optimized on
        # held-out data — re-reflecting here would silently reintroduce
        # the exact selection bias this baseline exists to measure.
        positive_direction = _raw_auroc(
            Z_sel[:, best_feat_idx], y_sel
        ) >= 0.5

        if y_ho.sum() > 0 and (len(y_ho) - y_ho.sum()) > 0:
            raw = _raw_auroc(Z_ho[:, best_feat_idx], y_ho)
            held_out_auroc = float(raw if positive_direction else 1.0 - raw)

        if go_dag is not None:
            hard_negative_auroc, n_hard_negatives = _hard_negative_auroc(
                scores=Z_ho[:, best_feat_idx],
                labels=y_ho,
                protein_ids=ho_protein_ids,
                go_id=go_id,
                go_labels=go_labels,
                go_dag=go_dag,
                positive_direction=positive_direction,
                min_hard_negatives=min_hard_negatives,
            )

    if not compute_multivariate_auroc:
        multi_auroc = float("nan")
    else:
        scaler = StandardScaler()
        Z_train_scaled = scaler.fit_transform(Z_train)
        Z_test_scaled = scaler.transform(Z_test)

        lr = LogisticRegression(
            max_iter=500,
            C=0.01,
            solver="lbfgs",
            random_state=random_state,
        )
        old_showwarning = warnings.showwarning

        def _log_and_show(message, category, filename, lineno, file=None, line=None):
            if issubclass(category, ConvergenceWarning):
                logger.warning(
                    "multivariate_auroc: lbfgs did not converge for %s "
                    "(n_positives=%d, n_train=%d, d_sae=%d)",
                    go_id, n_pos, len(y_train), Z.shape[1],
                )
            old_showwarning(message, category, filename, lineno, file, line)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("always", category=ConvergenceWarning)
                warnings.showwarning = _log_and_show
                lr.fit(Z_train_scaled, y_train)
            y_score = lr.predict_proba(Z_test_scaled)[:, 1]
            multi_auroc = float(roc_auc_score(y_test, y_score))
        except Exception as exc:
            logger.debug("Logistic regression failed for %s: %s", go_id, exc)
            multi_auroc = float("nan")

    return GOProbingResult(
        go_id=go_id,
        go_name=go_name,
        n_positives=n_pos,
        n_total=len(y),
        single_feature_auroc=single_feat_auroc,
        best_feature_idx=best_feat_idx,
        multivariate_auroc=multi_auroc,
        per_feature_aurocs=(
            per_feature_aurocs if return_per_feature_aurocs else None
        ),
        held_out_auroc=held_out_auroc,
        hard_negative_held_out_auroc=hard_negative_auroc,
        n_hard_negatives=n_hard_negatives,
    )


def probe_go_terms(
    feature_acts: Tensor,
    protein_ids: list[str],
    go_labels: dict[str, set[str]],
    go_names: dict[str, str] | None = None,
    min_positives: int = 50,
    max_go_terms: int = 500,
    train_fraction: float = 0.8,
    model_name: str = "",
    layer: int = -1,
    sae_variant: str = "",
    return_per_feature_aurocs: bool = False,
    random_state: int = 42,
    compute_held_out_baselines: bool = False,
    held_out_fraction: float = 0.5,
    go_dag: GODag | None = None,
    min_hard_negatives: int = 10,
    compute_multivariate_auroc: bool = True,
) -> GOProbingReport:
    """
    Run GO term linear probing on SAE feature activations.

    Args:
        feature_acts:   (N, d_sae) SAE feature activation matrix.
        protein_ids:    N UniProt accession IDs corresponding to each row.
        go_labels:      protein_id → set of GO term IDs (e.g. {"GO:0005737", ...}).
        go_names:       Optional GO ID → human-readable name (for reporting).
        min_positives:  Skip GO terms with fewer positive examples.
        max_go_terms:   Cap number of GO terms (sample most-common if exceeded).
        train_fraction: Train/test split.
        return_per_feature_aurocs: If True, store (d_sae,) AUROC array per result.
        compute_multivariate_auroc: Set False to skip the (potentially expensive
            — a full LogisticRegression fit per term, d_sae-dimensional) multivariate
            fit entirely; multivariate_auroc comes back NaN. Defaults True.
        compute_held_out_baselines: If True, further split the test set into a
            "selection" subset (used exactly as `single_feature_auroc` already
            is — unchanged when this flag is off) and a disjoint "held-out"
            subset, then report the selected feature's AUROC on the held-out
            subset (the classical sample-split de-biasing baseline) and, if `go_dag` is
            given, its AUROC against a matched hard-negative set drawn from GO-DAG
            siblings of `go_id`. Off by default — a distinct,
            opt-in run mode, not a change to existing single_feature_auroc
            numbers when unset.
        held_out_fraction: Of the test split, the fraction reserved as the
            disjoint held-out-estimation subset (the rest remains the
            "selection" subset). Only used when compute_held_out_baselines.
        go_dag: Optional GODag (see biolens.data.uniprot.resolve_go_dag) for
            hard-negative sibling lookup. If None, hard_negative_held_out_auroc
            is left unset even when compute_held_out_baselines=True.
        min_hard_negatives: Skip the hard-negative AUROC for a GO term if
            fewer than this many sibling-annotated hard negatives are
            available in the held-out subset — an unreliable estimate from a
            handful of hard negatives is worse than reporting "not enough
            data" honestly.

    Returns:
        GOProbingReport with per-GO-term results.
    """
    Z = feature_acts.float().numpy()   # (N, d_sae)
    go_names = go_names or {}

    # Collect all GO terms and their positive counts
    all_go_ids: dict[str, int] = {}
    for gos in go_labels.values():
        for go_id in gos:
            all_go_ids[go_id] = all_go_ids.get(go_id, 0) + 1

    # Filter to GO terms with enough positives and that appear in our protein set
    eligible = sorted(
        [(go_id, cnt) for go_id, cnt in all_go_ids.items() if cnt >= min_positives],
        key=lambda x: -x[1],
    )
    n_eligible_before_cap = len(eligible)
    if len(eligible) > max_go_terms:
        eligible = eligible[:max_go_terms]

    logger.info(
        "Probing %d GO terms (min_positives=%d) across %d proteins, d_sae=%d%s",
        len(eligible), min_positives, len(protein_ids), Z.shape[1],
        f" ({n_eligible_before_cap} eligible, --max-go-terms capped)"
        if n_eligible_before_cap > len(eligible) else "",
    )

    results = []
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - train_fraction, random_state=random_state
    )
    held_out_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=held_out_fraction, random_state=random_state
    )

    for go_id, _ in eligible:
        y = np.array([
            1 if pid in go_labels and go_id in go_labels[pid] else 0
            for pid in protein_ids
        ])
        result = _probe_single_term(
            Z=Z,
            protein_ids=protein_ids,
            y=y,
            go_id=go_id,
            go_name=go_names.get(go_id, go_id),
            go_labels=go_labels,
            splitter=splitter,
            held_out_splitter=held_out_splitter,
            min_positives=min_positives,
            compute_held_out_baselines=compute_held_out_baselines,
            go_dag=go_dag,
            min_hard_negatives=min_hard_negatives,
            return_per_feature_aurocs=return_per_feature_aurocs,
            random_state=random_state,
            compute_multivariate_auroc=compute_multivariate_auroc,
        )
        if result is not None:
            results.append(result)

    report = GOProbingReport(
        results=results,
        n_go_terms_tested=len(results),
        model_name=model_name,
        layer=layer,
        sae_variant=sae_variant,
        n_go_terms_eligible=n_eligible_before_cap,
    )
    logger.info("\n%s", report)
    return report


def _raw_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUROC for a single feature column, WITHOUT the below-0.5 reflection
    `_vectorized_auroc` applies. Held-out estimation must evaluate the exact
    direction chosen during selection — reflecting again on held-out data
    would silently let a mediocre feature "flip direction" to look good,
    reintroducing the same selection bias this baseline exists to measure.
    """
    pos_mask = labels == 1
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    from scipy.stats import rankdata

    ranks = rankdata(scores)
    rank_sum_pos = ranks[pos_mask].sum()
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


def _hard_negative_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    protein_ids: list[str],
    go_id: str,
    go_labels: dict[str, set[str]],
    go_dag: GODag,
    positive_direction: bool,
    min_hard_negatives: int,
) -> tuple[float | None, int | None]:
    """
    AUROC of the selected feature restricted to a matched hard-negative set:
    positives unchanged (proteins annotated with go_id), negatives restricted
    to proteins annotated with a GO-DAG sibling of go_id but NOT go_id itself
    (e.g. other transmembrane proteins as negatives for an ion-channel feature,
    rather than the generic negative pool).

    This is a stress test of whether a feature has learned the specific claimed
    concept or a broader structural class it happens to overlap with: a feature
    that only distinguishes the claimed class from *unrelated* proteins, but not
    from its own siblings, is likely tracking the broader class.

    Returns (auroc, n_hard_negatives), or (None, n_hard_negatives) if there
    aren't enough hard negatives in the held-out set to estimate reliably.
    """
    siblings = go_dag.siblings(go_id)
    if not siblings:
        return None, 0

    positive_mask = labels == 1
    hard_negative_mask = np.array(
        [
            (not positive_mask[i])
            and bool(go_labels.get(pid, set()) & siblings)
            for i, pid in enumerate(protein_ids)
        ]
    )
    n_hard_negatives = int(hard_negative_mask.sum())
    if n_hard_negatives < min_hard_negatives or positive_mask.sum() == 0:
        return None, n_hard_negatives

    mask = positive_mask | hard_negative_mask
    raw = _raw_auroc(scores[mask], labels[mask])
    auroc = raw if positive_direction else 1.0 - raw
    return float(auroc), n_hard_negatives


def _vectorized_auroc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute AUROC for every column of scores independently.

    Uses the Wilcoxon rank-sum / Mann-Whitney U relationship:
        AUROC = U / (n_pos * n_neg)
    where U is computed via argsort of each feature column.

    Much faster than calling sklearn.roc_auc_score in a loop for large d_sae.

    Args:
        scores: (N, d_sae) float array — one column per feature.
        labels: (N,) binary int array.

    Returns:
        (d_sae,) float array of AUROCs, one per feature.
        Values < 0.5 are reflected to [0.5, 1.0] (we care about any linear
        separation, positive or negative correlation).
    """
    pos_mask = labels == 1
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], 0.5)

    # Rank each column independently.
    # NOTE: scipy.stats.rankdata(scores, axis=0) — ranking all columns in one
    # call — was benchmarked as a "vectorization" here and measured ~1.7x
    # SLOWER than this per-column loop at realistic scale (d_sae=2560,
    # N=50000, ~1.25% nonzero per column matching TopK's k/d_sae sparsity):
    # scipy's axis-wise tie-ranking path is less optimized than its 1-D path.
    # Keep the loop — it's the faster option here, confirmed empirically.
    from scipy.stats import rankdata

    d_sae = scores.shape[1]
    aurocs = np.empty(d_sae)

    for i in range(d_sae):
        ranks = rankdata(scores[:, i])
        rank_sum_pos = ranks[pos_mask].sum()
        U = rank_sum_pos - n_pos * (n_pos + 1) / 2
        auroc = U / (n_pos * n_neg)
        aurocs[i] = max(auroc, 1.0 - auroc)  # reflect below 0.5

    return aurocs
