"""Provenance capture for experiment output files: which commit, data, seed and run
produced a given result, recorded alongside the result rather than inferred later."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit_hash() -> str | None:
    """Current git commit hash, with a '-dirty' suffix if there are uncommitted
    changes. None if not in a git repo or git is unavailable (never raises)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return None


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str | None:
    """SHA-256 of a file's contents. None if the path doesn't exist or can't be
    read (never raises) -- a missing checksum should not crash an eval run."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_strings(items: list[str]) -> str:
    """SHA-256 fingerprint of an ordered list of strings (e.g. the exact set of
    protein IDs evaluated). Cheap, deterministic identity check for 'was this
    the same sample as that other run' without storing the full ID list."""
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def capture(**extra) -> dict:
    """Base provenance block (git commit + UTC timestamp) merged with caller-
    supplied fields -- script args, data checksums, seeds, wandb run id, etc.
    This function doesn't guess what matters for a given script; the caller
    decides what's worth recording."""
    block = {
        "git_commit": git_commit_hash(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    block.update(extra)
    return block
