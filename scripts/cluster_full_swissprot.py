"""
MMseqs2 clustering over the full Swiss-Prot snapshot (no sampling): 30% sequence
identity, 80% coverage, --cov-mode 0.

Cluster membership for every sequence is the input to scripts/partition_swissprot_pools.py,
which assigns whole clusters to disjoint training and evaluation pools so that no close
homolog of an evaluation protein appears in the homology-excluded training pool. Clustering
the full set (rather than a pre-drawn sample) matters here: a sequence outside a given
sample could still be a close homolog of one inside it.

Requires the `mmseqs` binary on PATH. CPU only; ~570K sequences, so allow time and memory.

Usage:
  python scripts/cluster_full_swissprot.py --output-dir data/swissprot_clustering
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from biolens.data.uniprot import load_swissprot  # noqa: E402
from biolens.provenance import capture, sha256_file  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("data/swissprot_clustering"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading the FULL pinned Swiss-Prot pool (max_seqs=None, no truncation)...", flush=True)
    ids, seqs = load_swissprot(max_seqs=None, random_seed=None)
    print(f"  {len(ids)} proteins (max_seq_len=1024 filter already applied)", flush=True)

    fasta_path = OUT_DIR / "swissprot_full.fasta"
    print(f"Writing FASTA to {fasta_path}...", flush=True)
    with open(fasta_path, "w") as f:
        for pid, seq in zip(ids, seqs):
            f.write(f">{pid}\n{seq}\n")

    cluster_out = OUT_DIR / "cluster_result"
    tmp_dir = OUT_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print("Running mmseqs easy-cluster at 30% identity, 80% coverage ("
          "for homology exclusion)...", flush=True)
    result = subprocess.run(
        [
            "mmseqs", "easy-cluster", str(fasta_path), str(cluster_out), str(tmp_dir),
            "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0", "--threads", str(args.threads),
        ],
        capture_output=True, text=True,
    )
    print("mmseqs stdout tail:", result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        print("mmseqs stderr:", result.stderr[-3000:], flush=True)
        raise SystemExit(f"mmseqs failed with code {result.returncode}")

    cluster_tsv = Path(f"{cluster_out}_cluster.tsv")
    member_to_rep: dict[str, str] = {}
    with open(cluster_tsv) as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            member_to_rep[member] = rep

    n_clusters = len(set(member_to_rep.values()))
    print(f"\n{len(ids)} proteins clustered into {n_clusters} distinct sequence clusters "
          f"(30% identity, 80% coverage).", flush=True)

    provenance = capture(
        script="cluster_full_swissprot.py",
        n_proteins=len(ids),
        n_clusters=n_clusters,
        mmseqs_params={"min_seq_id": 0.3, "coverage": 0.8, "cov_mode": 0},
        fasta_sha256=sha256_file(fasta_path),
        cluster_tsv_sha256=sha256_file(cluster_tsv),
    )
    (OUT_DIR / "provenance.json").write_text(__import__("json").dumps(provenance, indent=2))
    print(f"Provenance written to {OUT_DIR / 'provenance.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
