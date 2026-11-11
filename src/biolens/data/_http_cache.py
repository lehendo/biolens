"""
Shared HTTP download-and-cache helper, used by every biolens.data loader
that fetches a public annotation file (UniProt/GO, ENCODE, GENCODE, ...).

Not public API — internal to biolens.data. Each loader module re-exports
what it needs (e.g. `from biolens.data._http_cache import get_data_dir,
_get_or_download`), so callers still import from the domain-specific module
(biolens.data.uniprot, biolens.data.genomic_annotations, ...), not this one.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # biolens/ repo root
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "annotations"


def get_data_dir() -> Path:
    d = Path(os.environ.get("BIOLENS_DATA_DIR", _DEFAULT_DATA_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_or_download(url: str, filename: str, data_dir: Path | None) -> Path:
    """Return path to cached file, downloading if not present."""
    d = data_dir if data_dir is not None else get_data_dir()
    path = d / filename
    if path.exists():
        logger.debug("Using cached %s", path)
        return path

    logger.info("Downloading %s → %s", url, path)
    _download_with_progress(url, path)
    return path


def _download_with_progress(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "biolens/0.1 (+github.com/biolens)"})
    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        chunk_size = 1 << 20  # 1 MB

        with open(tmp_path, "wb") as out, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            leave=False,
        ) as bar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))

    tmp_path.rename(dest)
