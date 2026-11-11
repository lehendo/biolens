"""
Regression tests for scripts/partition_swissprot_pools.py's cluster-to-bucket
assignment and sampling logic -- the correctness-critical piece of
the pool design (no homolog of an evaluation protein
may appear in the homology-excluded training pool, or vice versa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from partition_swissprot_pools import (  # noqa: E402
    assign_clusters_to_buckets,
    draw_from_clusters,
)


def _synthetic_clusters(n_clusters: int, min_size: int, max_size: int, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    member_to_rep: dict[str, str] = {}
    pid = 0
    for c in range(n_clusters):
        size = int(rng.integers(min_size, max_size + 1))
        rep = f"rep_{c}"
        for _ in range(size):
            member_to_rep[f"p{pid}"] = rep
            pid += 1
    return member_to_rep


def test_assign_clusters_to_buckets_are_disjoint():
    member_to_rep = _synthetic_clusters(5000, 1, 20, seed=42)
    eval_clusters, train_clusters, _ = assign_clusters_to_buckets(member_to_rep, pool_size=10_000, seed=1)
    assert eval_clusters.isdisjoint(train_clusters)
    assert len(eval_clusters) > 0
    assert len(train_clusters) > 0


def test_assign_clusters_to_buckets_meets_pool_size():
    member_to_rep = _synthetic_clusters(5000, 1, 20, seed=42)
    eval_clusters, train_clusters, cluster_to_members = assign_clusters_to_buckets(
        member_to_rep, pool_size=10_000, seed=1
    )
    eval_count = sum(len(cluster_to_members[c]) for c in eval_clusters)
    train_count = sum(len(cluster_to_members[c]) for c in train_clusters)
    assert eval_count >= 10_000
    assert train_count >= 10_000


def test_assign_clusters_to_buckets_raises_if_pool_too_small():
    member_to_rep = _synthetic_clusters(10, 1, 5, seed=42)  # far too few proteins
    with pytest.raises(ValueError):
        assign_clusters_to_buckets(member_to_rep, pool_size=10_000, seed=1)


def test_draw_from_clusters_exact_size_no_duplicates():
    member_to_rep = _synthetic_clusters(5000, 1, 20, seed=42)
    eval_clusters, _, cluster_to_members = assign_clusters_to_buckets(member_to_rep, pool_size=10_000, seed=1)
    drawn = draw_from_clusters(eval_clusters, cluster_to_members, pool_size=10_000, seed=2)
    assert len(drawn) == 10_000
    assert len(set(drawn)) == 10_000


def test_full_partition_is_cluster_level_disjoint_end_to_end():
    """The property that actually matters: after drawing both pools, no protein in
    either pool shares a cluster with any protein in the other pool."""
    member_to_rep = _synthetic_clusters(5000, 1, 20, seed=7)
    eval_clusters, train_clusters, cluster_to_members = assign_clusters_to_buckets(
        member_to_rep, pool_size=10_000, seed=3
    )
    eval_ids = draw_from_clusters(eval_clusters, cluster_to_members, pool_size=10_000, seed=4)
    train_ids = draw_from_clusters(train_clusters, cluster_to_members, pool_size=10_000, seed=5)

    assert set(eval_ids).isdisjoint(train_ids)
    eval_pool_clusters = {member_to_rep[pid] for pid in eval_ids}
    train_pool_clusters = {member_to_rep[pid] for pid in train_ids}
    assert eval_pool_clusters.isdisjoint(train_pool_clusters)


def test_draw_from_clusters_is_seed_reproducible():
    member_to_rep = _synthetic_clusters(2000, 1, 20, seed=1)
    eval_clusters, _, cluster_to_members = assign_clusters_to_buckets(member_to_rep, pool_size=5_000, seed=1)
    drawn_a = draw_from_clusters(eval_clusters, cluster_to_members, pool_size=5_000, seed=99)
    drawn_b = draw_from_clusters(eval_clusters, cluster_to_members, pool_size=5_000, seed=99)
    assert drawn_a == drawn_b
