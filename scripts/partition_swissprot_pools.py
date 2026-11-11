"""
Partition the full Swiss-Prot pool into three protein pools using the cluster
assignment from scripts/cluster_full_swissprot.py:

  1. evaluation pool (200K)             -- seeded random draw from clusters
                                            assigned to the "evaluation" bucket.
  2. homology-excluded training pool     -- seeded random draw from clusters
     (200K)                               assigned to the "training" bucket
                                            (disjoint clusters from the eval pool,
                                            by construction).
  3. broad-random training pool (200K)  -- an ordinary seeded load_swissprot()
                                            draw from the full pool, independent
                                            of cluster assignment (may share
                                            sequences or close homologs with the
                                            eval pool; that is the point of this arm).

Cluster-to-bucket assignment: shuffle the distinct clusters (seeded), then walk
the shuffled list assigning whole clusters greedily to "evaluation" until its
running protein count reaches the target, then to "training" until it reaches
the target; any remaining clusters go unused. Assigning whole clusters (not
individual proteins) is what guarantees no homolog of an evaluation protein
ends up in the homology-excluded training pool, or vice versa.

Usage:
  python scripts/partition_swissprot_pools.py \\
      --cluster-dir data/swissprot_clustering --output-dir data/pools
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from biolens.data.uniprot import load_swissprot  # noqa: E402
from biolens.provenance import capture  # noqa: E402

POOL_SIZE = 200_000
PARTITION_SEED = 20260918  # shuffling which clusters go to which bucket
BROAD_RANDOM_SEED = 20260919  # the broad-random training arm's own independent draw


def load_cluster_assignment(cluster_tsv: Path) -> dict[str, str]:
    """protein_id -> representative_cluster_id, from an MMseqs2 easy-cluster TSV."""
    member_to_rep: dict[str, str] = {}
    with open(cluster_tsv) as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            member_to_rep[member] = rep
    return member_to_rep


def assign_clusters_to_buckets(
    member_to_rep: dict[str, str], pool_size: int, seed: int,
) -> tuple[set[str], set[str]]:
    """Returns (evaluation_cluster_ids, training_cluster_ids), each bucket's
    total member count >= pool_size (trimmed to exactly pool_size later by a
    separate seeded subsample within the bucket)."""
    cluster_to_members: dict[str, list[str]] = {}
    for member, rep in member_to_rep.items():
        cluster_to_members.setdefault(rep, []).append(member)

    cluster_ids = list(cluster_to_members.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_ids)

    eval_clusters: set[str] = set()
    train_clusters: set[str] = set()
    eval_count = 0
    train_count = 0
    for cid in cluster_ids:
        n = len(cluster_to_members[cid])
        if eval_count < pool_size:
            eval_clusters.add(cid)
            eval_count += n
        elif train_count < pool_size:
            train_clusters.add(cid)
            train_count += n
        else:
            break  # remaining clusters unused

    if eval_count < pool_size or train_count < pool_size:
        raise ValueError(
            f"Full pool too small to fill both {pool_size}-protein buckets: "
            f"eval_count={eval_count}, train_count={train_count}. "
            f"Total proteins clustered: {sum(len(m) for m in cluster_to_members.values())}."
        )

    return eval_clusters, train_clusters, cluster_to_members


def draw_from_clusters(
    cluster_ids: set[str], cluster_to_members: dict[str, list[str]], pool_size: int, seed: int,
) -> list[str]:
    """All members of the assigned clusters, seeded-subsampled down to exactly pool_size."""
    members = [pid for cid in cluster_ids for pid in cluster_to_members[cid]]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(members), size=pool_size, replace=False)
    return [members[i] for i in idx]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster-dir", type=Path, default=Path("data/swissprot_clustering"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/pools"))
    args = ap.parse_args()
    CLUSTER_DIR, OUT_DIR = args.cluster_dir, args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_tsv = CLUSTER_DIR / "cluster_result_cluster.tsv"
    print(f"Loading cluster assignment from {cluster_tsv}...", flush=True)
    member_to_rep = load_cluster_assignment(cluster_tsv)
    print(f"  {len(member_to_rep)} proteins, "
          f"{len(set(member_to_rep.values()))} distinct clusters", flush=True)

    eval_clusters, train_clusters, cluster_to_members = assign_clusters_to_buckets(
        member_to_rep, POOL_SIZE, PARTITION_SEED
    )
    print(f"Evaluation bucket: {len(eval_clusters)} clusters, "
          f"{sum(len(cluster_to_members[c]) for c in eval_clusters)} proteins available", flush=True)
    print(f"Homology-excluded training bucket: {len(train_clusters)} clusters, "
          f"{sum(len(cluster_to_members[c]) for c in train_clusters)} proteins available", flush=True)
    assert eval_clusters.isdisjoint(train_clusters), "eval/train cluster buckets must be disjoint"

    eval_ids = draw_from_clusters(eval_clusters, cluster_to_members, POOL_SIZE, seed=PARTITION_SEED + 1)
    homology_excluded_train_ids = draw_from_clusters(
        train_clusters, cluster_to_members, POOL_SIZE, seed=PARTITION_SEED + 2
    )
    assert set(eval_ids).isdisjoint(homology_excluded_train_ids)
    # Stronger check: no protein in either drawn pool shares a cluster with the other pool.
    eval_pool_clusters = {member_to_rep[pid] for pid in eval_ids}
    train_pool_clusters = {member_to_rep[pid] for pid in homology_excluded_train_ids}
    assert eval_pool_clusters.isdisjoint(train_pool_clusters), (
        "cluster-level disjointness violated after subsampling -- should be impossible "
        "since subsampling only removes members, never moves a cluster between buckets"
    )

    print(f"Drawing broad-random training pool (seed={BROAD_RANDOM_SEED}, independent of "
          f"cluster assignment, may overlap the evaluation pool at the sequence level)...",
          flush=True)
    broad_random_ids, _ = load_swissprot(max_seqs=POOL_SIZE, random_seed=BROAD_RANDOM_SEED)
    n_overlap_with_eval = len(set(broad_random_ids) & set(eval_ids))
    print(f"  {len(broad_random_ids)} proteins, {n_overlap_with_eval} exact-ID overlap "
          f"with the evaluation pool (expected/harmless for this arm; the whole point "
          f"is that it does NOT enforce disjointness)", flush=True)

    for name, ids in [
        ("evaluation_pool", eval_ids),
        ("homology_excluded_training_pool", homology_excluded_train_ids),
        ("broad_random_training_pool", broad_random_ids),
    ]:
        path = OUT_DIR / f"{name}_ids.json"
        path.write_text(json.dumps(ids))
        print(f"Wrote {len(ids)} protein IDs to {path}", flush=True)

    provenance = capture(
        script="partition_swissprot_pools.py",
        pool_size=POOL_SIZE,
        partition_seed=PARTITION_SEED,
        broad_random_seed=BROAD_RANDOM_SEED,
        n_eval_clusters=len(eval_clusters),
        n_train_clusters=len(train_clusters),
        n_eval_ids=len(eval_ids),
        n_homology_excluded_train_ids=len(homology_excluded_train_ids),
        n_broad_random_ids=len(broad_random_ids),
        broad_random_overlap_with_eval=n_overlap_with_eval,
    )
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Provenance written to {OUT_DIR / 'provenance.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
