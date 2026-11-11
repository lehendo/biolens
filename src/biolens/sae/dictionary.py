"""
HDF5-backed activation cache for SAE training.

Design:
  - Activations are stored in sharded HDF5 files under a single directory.
  - Each shard: 'activations' (N, D) float16, 'ids' (N,) str, 'sequences' (N,) str.
  - A JSON manifest tracks shard paths, sizes, and normalization stats.
  - FileLock guards manifest writes so concurrent SLURM array tasks are safe.
  - Normalization statistics (mean, scale) are computed across all shards and
    stored in the manifest; the training loop can apply them before feeding
    activations to the SAE.

SLURM usage:
  Each array task calls write_shard() with a unique shard_id.
  After all tasks complete, finalize() computes normalization stats.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
from filelock import FileLock
from torch import Tensor

if TYPE_CHECKING:
    from biolens.sae.architectures import BaseSAE

logger = logging.getLogger(__name__)

_MANIFEST_FNAME = "manifest.json"
_LOCK_FNAME = "manifest.lock"
_SHARD_PATTERN = "shard_{:06d}.h5"


class ActivationCache:
    """Read/write interface for cached activation shards."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.cache_dir / _MANIFEST_FNAME
        self._lock_path = self.cache_dir / _LOCK_FNAME

    # ── Writing ──────────────────────────────────────────────────────────────

    def write_shard(
        self,
        shard_id: int,
        activations: Tensor | np.ndarray,
        ids: list[str],
        sequences: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Path:
        """
        Write a shard of activations to disk and register it in the manifest.

        Thread/process-safe via FileLock.

        Args:
            shard_id:    Unique integer ID for this shard (used in filename).
            activations: (N, D) tensor or numpy array; stored as float16.
            ids:         Unique string identifiers (e.g. UniProt accessions).
            sequences:   Optional raw sequences — useful for dashboard display.
            metadata:    Optional dict (model_name, layer, pooling, etc.).

        Returns:
            Path to the written shard file.
        """
        if isinstance(activations, Tensor):
            activations_np = activations.float().cpu().numpy().astype(np.float16)
        else:
            activations_np = activations.astype(np.float16)

        N = len(activations_np)
        assert len(ids) == N, f"ids length {len(ids)} != activations length {N}"

        shard_path = self.cache_dir / _SHARD_PATTERN.format(shard_id)
        _write_hdf5_shard(shard_path, activations_np, ids, sequences, metadata)
        logger.info("Wrote shard %d (%d vectors) → %s", shard_id, N, shard_path)

        with FileLock(str(self._lock_path)):
            manifest = self._read_manifest()
            manifest["shards"][str(shard_id)] = {
                "path": str(shard_path),
                "n": N,
                "d": int(activations_np.shape[1]),
            }
            if metadata:
                manifest.setdefault("metadata", {}).update(metadata)
            self._write_manifest(manifest)

        return shard_path

    def finalize(self) -> dict:
        """
        Compute and store normalization statistics across all shards.

        Call this once after all extraction jobs complete.  Adds:
            manifest["norm"]["mean"]:  (D,) list — mean across all vectors
            manifest["norm"]["scale"]: scalar — mean L2 norm of centered vectors

        Streams shards one at a time (two passes: sum for the mean, then
        centered-norm for the scale) rather than materializing every shard's
        float32 upcast simultaneously — the previous implementation held the
        full per-shard list, the concatenated array, AND the centered array
        in memory at once, multiple times the dataset's total size. That
        can exhaust memory on large corpora (e.g. ~1M Evo 2 windows at d_model=4096).
        Peak memory here is one shard at a time.

        Returns the updated manifest.
        """
        shard_paths = [shard_path for shard_path, _ in self._iter_shards()]
        if not shard_paths:
            raise RuntimeError("No shards found; nothing to finalize.")

        d_model: int | None = None
        n_total = 0
        running_sum: np.ndarray | None = None
        for shard_path in shard_paths:
            with h5py.File(shard_path, "r") as f:
                acts = f["activations"][:].astype(np.float32)  # (n_shard, D)
            if running_sum is None:
                d_model = acts.shape[1]
                running_sum = np.zeros(d_model, dtype=np.float64)
            running_sum += acts.sum(axis=0, dtype=np.float64)
            n_total += acts.shape[0]
        assert running_sum is not None
        mean = (running_sum / n_total).astype(np.float32)  # (D,)

        running_norm_sum = 0.0
        for shard_path in shard_paths:
            with h5py.File(shard_path, "r") as f:
                acts = f["activations"][:].astype(np.float32)  # (n_shard, D)
            centered = acts - mean
            running_norm_sum += float(np.linalg.norm(centered, axis=1).sum())
        scale = running_norm_sum / n_total

        with FileLock(str(self._lock_path)):
            manifest = self._read_manifest()
            manifest["norm"] = {
                "mean": mean.tolist(),
                "scale": scale,
                "n_total": int(n_total),
            }
            self._write_manifest(manifest)

        logger.info(
            "Finalized cache: %d vectors, scale=%.4f", n_total, scale
        )
        return manifest

    # ── Reading ───────────────────────────────────────────────────────────────

    def iter_batches(
        self,
        batch_size: int = 4096,
        normalize: bool = True,
        device: str = "cpu",
    ) -> Iterator[Tensor]:
        """
        Stream activation batches across all shards.

        Args:
            batch_size: Vectors per yielded tensor.
            normalize:  If True, subtract mean and divide by scale (see finalize()).
            device:     "cpu" or "cuda".

        Yields:
            (batch_size_actual, D) float32 tensor.
        """
        norm = self._get_norm_stats() if normalize else None
        leftover: np.ndarray | None = None

        for shard_path, _ in self._iter_shards():
            with h5py.File(shard_path, "r") as f:
                acts = f["activations"][:].astype(np.float32)  # (N_shard, D)

            if norm is not None:
                acts = (acts - norm["mean"]) / max(norm["scale"], 1e-8)

            if leftover is not None:
                acts = np.concatenate([leftover, acts], axis=0)
                leftover = None

            n = len(acts)
            n_full = (n // batch_size) * batch_size
            for start in range(0, n_full, batch_size):
                yield torch.from_numpy(acts[start : start + batch_size]).to(device)

            if n_full < n:
                leftover = acts[n_full:]

        if leftover is not None and len(leftover) > 0:
            yield torch.from_numpy(leftover).to(device)

    def get_activations_by_ids(self, ids: list[str]) -> Tensor:
        """Retrieve activations for specific IDs (linear scan; for eval use only)."""
        target = set(ids)
        found: dict[str, np.ndarray] = {}

        for shard_path, _ in self._iter_shards():
            with h5py.File(shard_path, "r") as f:
                shard_ids = [s.decode() for s in f["ids"][:]]
                acts = f["activations"][:]
                for i, sid in enumerate(shard_ids):
                    if sid in target:
                        found[sid] = acts[i].astype(np.float32)

        result = np.stack([found[i] for i in ids])
        return torch.from_numpy(result)

    def __len__(self) -> int:
        manifest = self._read_manifest()
        return sum(s["n"] for s in manifest.get("shards", {}).values())

    @property
    def d_model(self) -> int:
        manifest = self._read_manifest()
        for shard in manifest.get("shards", {}).values():
            return shard["d"]
        raise RuntimeError("Cache is empty — no shards written yet.")

    @property
    def n_shards(self) -> int:
        return len(self._read_manifest().get("shards", {}))

    def get_norm_stats(self) -> dict:
        """
        Public accessor for the (mean, scale) normalization stats computed by
        finalize(). Raises the same RuntimeError as iter_batches(normalize=True)
        if finalize() hasn't been run yet.

        Any code that feeds raw cached activations into a trained SAE (the
        dashboard, feature-inspection tooling, etc.) must apply these same
        stats first — SAEs are trained on normalized activations via
        iter_batches(normalize=True), so raw activations are out of the
        distribution the encoder was fit on.
        """
        return self._get_norm_stats()

    def normalize(self, acts: Tensor) -> Tensor:
        """Apply this cache's stored (mean, scale) normalization to raw activations.

        Args:
            acts: (N, D) float tensor of raw (un-normalized) activations, on any device.

        Returns:
            (N, D) float32 tensor, same device as input, normalized identically
            to what iter_batches(normalize=True) yields.
        """
        norm = self._get_norm_stats()
        mean = torch.from_numpy(norm["mean"]).to(device=acts.device, dtype=acts.dtype)
        return (acts - mean) / max(norm["scale"], 1e-8)

    def encode(self, sae: BaseSAE, acts: Tensor) -> Tensor:
        """Normalize `acts` with this cache's stats, then run them through
        `sae.encode()` -- the only correct way to feed this cache's raw
        activations to an SAE trained on it (normalize_activations=True
        default, biolens/sae/train.py).

        Calling `sae.encode(acts)` directly on raw activations silently feeds the
        SAE an out-of-distribution input; this method bundles the normalization
        step so a call site cannot do one and forget the other. `sae.encode()` and
        `cache.normalize()` remain available separately.
        """
        return sae.encode(self.normalize(acts))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {"shards": {}}
        with open(self._manifest_path) as f:
            return json.load(f)

    def _write_manifest(self, manifest: dict) -> None:
        self._manifest_path.write_text(json.dumps(manifest, indent=2))

    def _iter_shards(self) -> Iterator[tuple[Path, dict]]:
        manifest = self._read_manifest()
        for shard_id in sorted(manifest.get("shards", {}), key=int):
            info = manifest["shards"][shard_id]
            yield Path(info["path"]), info

    def _get_norm_stats(self) -> dict:
        manifest = self._read_manifest()
        if "norm" not in manifest:
            raise RuntimeError(
                "Normalization stats not found. Run cache.finalize() after all shards are written."
            )
        norm = manifest["norm"]
        return {
            "mean": np.array(norm["mean"], dtype=np.float32),
            "scale": float(norm["scale"]),
        }


def _write_hdf5_shard(
    path: Path,
    activations: np.ndarray,
    ids: list[str],
    sequences: list[str] | None,
    metadata: dict | None,
) -> None:
    """Write a single HDF5 shard atomically (write to .tmp then rename)."""
    tmp_path = path.with_suffix(".h5.tmp")
    dt_str = h5py.string_dtype(encoding="utf-8")

    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("activations", data=activations, compression="lzf", chunks=True)
        f.create_dataset("ids", data=np.array(ids, dtype=object), dtype=dt_str)
        if sequences is not None:
            f.create_dataset(
                "sequences", data=np.array(sequences, dtype=object), dtype=dt_str
            )
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = str(v)

    tmp_path.rename(path)  # atomic on POSIX filesystems
