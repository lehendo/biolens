"""
UniProt Swiss-Prot + Gene Ontology annotation data loading.

Downloads:
  - Swiss-Prot FASTA: UniProt FTP (reviewed sequences, ~570K proteins, ~80MB gz)
  - GO annotations: Gene Ontology Consortium GAF file (goa_uniprot_all.gaf.gz)

All downloads are cached under data/annotations/.
Set BIOLENS_DATA_DIR env var to override the default cache location.

Note: These are all publicly available, no authentication required.
"""

from __future__ import annotations

import gzip
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biolens.data._http_cache import _get_or_download

logger = logging.getLogger(__name__)

# Public UniProt FTP — stable, no auth required
_SWISSPROT_FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.fasta.gz"
)
# Gene Ontology Consortium: UniProt-GOA annotations
_GOA_GAF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz"
)
# Gene Ontology Consortium: full ontology (ID → human-readable name, is_a graph, etc.)
_GO_OBO_URL = "https://purl.obolibrary.org/obo/go.obo"


# ── Swiss-Prot FASTA ─────────────────────────────────────────────────────────

def load_swissprot(
    max_seqs: int | None = None,
    max_seq_len: int = 1024,
    data_dir: Path | None = None,
    random_seed: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    Load Swiss-Prot protein sequences.

    With `random_seed=None` (the default), `max_seqs` takes the first N entries in
    raw FASTA file order, which is not a representative sample. Swiss-Prot's file
    order groups entries by accession-adjacent naming (the file opens with
    `001R_FRG3G`, a frog virus protein), so a file-order truncation is measurably
    skewed: in the first 10,000 entries, two viral organisms appear in the top 15
    by count that are absent from a representative whole-file sample's top 15, and
    several major model organisms (S. cerevisiae, S. pombe, B. subtilis) are
    missing from the skewed slice's top 15 despite being prominent file-wide.

    When `random_seed` is set, the full filtered sequence set is loaded (reading
    past `max_seqs` is unavoidable to sample from the whole pool), then
    seeded-shuffled before truncating to `max_seqs`. Pass a seed and record it.
    `random_seed=None` preserves the file-order, early-break behavior exactly.

    Returns:
        ids:       List of UniProt accession IDs (e.g. "P12345").
        sequences: Corresponding amino-acid sequences (uppercased, non-standard
                   amino acids replaced with 'X').
    """
    fasta_gz = _get_or_download(
        _SWISSPROT_FASTA_URL, "uniprot_sprot.fasta.gz", data_dir
    )
    ids, sequences = [], []
    # Only early-break on max_seqs when NOT randomly sampling -- a random
    # sample needs the whole filtered pool available to shuffle from.
    early_break_limit = max_seqs if random_seed is None else None

    with gzip.open(fasta_gz, "rt") as f:
        current_id: str | None = None
        current_seq: list[str] = []

        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None and current_seq:
                    seq = "".join(current_seq)
                    if len(seq) <= max_seq_len:
                        ids.append(current_id)
                        sequences.append(seq)
                    if early_break_limit and len(ids) >= early_break_limit:
                        break
                # Header format: >sp|P12345|GENE_HUMAN ...
                current_id = _parse_uniprot_id(line)
                current_seq = []
            else:
                current_seq.append(_sanitize_aa(line))

        # Last entry
        if current_id is not None and current_seq and (
            not early_break_limit or len(ids) < early_break_limit
        ):
            seq = "".join(current_seq)
            if len(seq) <= max_seq_len:
                ids.append(current_id)
                sequences.append(seq)

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        order = rng.permutation(len(ids))
        ids = [ids[i] for i in order]
        sequences = [sequences[i] for i in order]
        if max_seqs:
            ids, sequences = ids[:max_seqs], sequences[:max_seqs]
        logger.info(
            "Loaded %d Swiss-Prot sequences (max_seq_len=%d), seeded random "
            "sample (seed=%d) from the full %d-entry filtered pool",
            len(ids), max_seq_len, random_seed, len(order),
        )
    else:
        logger.info("Loaded %d Swiss-Prot sequences (max_seq_len=%d)", len(ids), max_seq_len)
    return ids, sequences


def swissprot_fasta_path(data_dir: Path | None = None) -> Path:
    """Resolve (downloading if needed) the local path of the pinned Swiss-Prot
    FASTA, without parsing it -- for callers that just need to checksum the
    exact data file a run used (see biolens.provenance.sha256_file)."""
    return _get_or_download(_SWISSPROT_FASTA_URL, "uniprot_sprot.fasta.gz", data_dir)


def _parse_uniprot_id(header: str) -> str:
    """Extract accession from >sp|P12345|GENE_HUMAN ... or >tr|... or >P12345 ..."""
    m = re.match(r">[a-z]+\|([A-Z0-9]+)\|", header)
    if m:
        return m.group(1)
    # Fallback: first token after >
    return header[1:].split()[0].split("|")[0]


_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _sanitize_aa(seq: str) -> str:
    """Replace non-standard residues with X, uppercase everything."""
    return "".join(c if c.upper() in _VALID_AA else "X" for c in seq.upper())


# ── GO Annotations ───────────────────────────────────────────────────────────

def load_go_annotations(
    protein_ids: set[str] | None = None,
    evidence_codes: set[str] | None = None,
    aspects: set[str] | None = None,
    data_dir: Path | None = None,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """
    Load Gene Ontology annotations for UniProt proteins.

    Args:
        protein_ids:   If given, restrict to these UniProt IDs (faster).
        evidence_codes: Evidence code filter (e.g. {"EXP","IDA","IPI","IMP","IGI","IEP"}).
                        None = all codes including IEA (electronic annotation).
        aspects:       GO aspect filter: {"P"=biological process, "F"=molecular function,
                        "C"=cellular component}.  None = all three.
        data_dir:      Override default annotation cache directory.

    Returns:
        go_labels:  protein_id → set of GO term IDs.
        go_names:   GO term ID → human-readable name (from GAF description field).
    """
    gaf_gz = _get_or_download(_GOA_GAF_URL, "goa_uniprot_all.gaf.gz", data_dir)
    go_labels: dict[str, set[str]] = {}
    go_names: dict[str, str] = {}

    for line in _filtered_gaf_lines(gaf_gz, protein_ids):
        if line.startswith("!"):  # comment
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9:
            continue

        # GAF 2.x columns:
        # 0=DB, 1=DB_Object_ID, 2=DB_Object_Symbol, 3=Qualifier, 4=GO_ID,
        # 5=DB_Ref, 6=Evidence_Code, 7=With, 8=Aspect, ...
        db, obj_id, _sym, qualifier, go_id, _ref, evidence, _with, aspect = (
            parts[i] for i in range(9)
        )

        # Skip negative annotations
        if "NOT" in qualifier.upper():
            continue
        if evidence_codes is not None and evidence not in evidence_codes:
            continue
        if aspects is not None and aspect not in aspects:
            continue
        if protein_ids is not None and obj_id not in protein_ids:
            continue

        go_labels.setdefault(obj_id, set()).add(go_id)
        if go_id not in go_names:
            go_names[go_id] = go_id

    logger.info(
        "Loaded GO annotations for %d proteins (%d unique GO terms)",
        len(go_labels), len(go_names),
    )
    return go_labels, go_names


def _filtered_gaf_lines(gaf_gz: Path, protein_ids: set[str] | None) -> Iterable[str]:
    """
    Yield GAF lines, pre-filtered to those containing one of protein_ids via
    `zgrep -F` (fixed-string multi-pattern match in C) when protein_ids is given.

    goa_uniprot_all.gaf.gz covers every UniProtKB entry (250M+ lines) — scanning
    it line-by-line in Python to keep ~10K proteins is the dominant cost of GO
    annotation loading (tens of minutes). zgrep does the same byte-level scan
    in seconds to minutes. Falls back to a plain Python scan if protein_ids is
    unset or zgrep is unavailable/fails.
    """
    if not protein_ids:
        with gzip.open(gaf_gz, "rt", errors="replace") as f:
            yield from f
        return

    pattern_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as pf:
            pf.write("\n".join(sorted(protein_ids)))
            pattern_path = pf.name

        result = subprocess.run(
            ["zgrep", "-F", "-f", pattern_path, str(gaf_gz)],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if result.returncode not in (0, 1):  # 1 = no matches, still a valid result
            raise RuntimeError(
                f"zgrep exited {result.returncode}: {result.stderr[:500]}"
            )

        logger.info(
            "zgrep prefiltered GOA GAF to %d candidate lines",
            result.stdout.count("\n"),
        )
        yield from result.stdout.splitlines(keepends=True)
        return
    except (FileNotFoundError, RuntimeError, OSError) as e:
        logger.warning(
            "zgrep prefilter unavailable (%s); falling back to full Python scan", e
        )
    finally:
        if pattern_path is not None:
            os.unlink(pattern_path)

    with gzip.open(gaf_gz, "rt", errors="replace") as f:
        yield from f


def load_go_obo_names(obo_path: Path) -> dict[str, str]:
    """
    Parse a .obo file and return GO ID → human-readable name mapping.

    Handles alt_id lines so deprecated/merged GO IDs still resolve to the
    name of the term they were merged into.
    """
    names: dict[str, str] = {}
    in_term_stanza = False
    current_id: str | None = None
    current_alt_ids: list[str] = []
    current_name: str | None = None

    def _flush() -> None:
        if current_id is None or current_name is None:
            return
        names[current_id] = current_name
        for alt_id in current_alt_ids:
            names.setdefault(alt_id, current_name)

    with open(obo_path) as f:
        for line in f:
            line = line.rstrip()
            # Every stanza (not just [Term] — also [Typedef], [Instance], etc.)
            # begins with a bracketed header; any of them ends the previous one.
            if line.startswith("["):
                _flush()
                in_term_stanza = line == "[Term]"
                current_id = None
                current_alt_ids = []
                current_name = None
            elif in_term_stanza and line.startswith("id: GO:"):
                current_id = line[4:]
            elif in_term_stanza and line.startswith("name: ") and current_id is not None:
                current_name = line[6:]
            elif in_term_stanza and line.startswith("alt_id: GO:"):
                current_alt_ids.append(line[8:])
        _flush()  # last stanza in the file

    return names


@dataclass
class GODag:
    """
    Minimal Gene Ontology DAG: parent (`is_a`) edges, for computing sibling
    terms (terms sharing a parent) — needed to build hard-negative sets for
    the hard-negative held-out AUROC baseline: a feature that has learned a broad
    structural class (e.g. "transmembrane transporter") should fail to discriminate
    a specific sibling concept (e.g. "ion channel complex") from OTHER siblings
    under the same parent, even though it easily beats the generic negative
    pool. Deliberately minimal — parents/children only, not the full OBO
    relationship model (part_of, regulates, etc.), since sibling lookup is
    all this use case needs.
    """
    parents: dict[str, set[str]]   # GO ID -> set of direct parent GO IDs
    children: dict[str, set[str]]  # GO ID -> set of direct child GO IDs

    def siblings(self, go_id: str) -> set[str]:
        """Terms sharing at least one direct parent with go_id (excluding itself)."""
        sibs: set[str] = set()
        for parent in self.parents.get(go_id, ()):
            sibs |= self.children.get(parent, set())
        sibs.discard(go_id)
        return sibs


def load_go_dag(obo_path: Path) -> GODag:
    """
    Parse `is_a:` relationships from a .obo file into a minimal parent/child
    DAG. Handles alt_id lines the same way load_go_obo_names does, so sibling
    lookups work correctly for deprecated/merged GO IDs too.

    GO OBO format for is_a lines: `is_a: GO:0005886 ! plasma membrane` — the
    GO ID is the token immediately after "is_a: ", before the "!" comment.
    """
    # alt_ids are aliases for the canonical id, not separate DAG nodes — kept
    # in a *separate* map and merged into `parents` only (so a parent lookup
    # on an alt_id still works), never into `children` (so an alt_id doesn't
    # appear as a phantom extra sibling alongside its own canonical id, the
    # same bug class load_go_obo_names avoids for name resolution).
    canonical_parents: dict[str, set[str]] = {}
    alt_id_map: dict[str, str] = {}  # alt_id -> canonical id
    in_term_stanza = False
    current_id: str | None = None
    current_alt_ids: list[str] = []
    current_parents: set[str] = set()

    def _flush() -> None:
        if current_id is None:
            return
        canonical_parents.setdefault(current_id, set()).update(current_parents)
        for alt_id in current_alt_ids:
            alt_id_map[alt_id] = current_id

    with open(obo_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("["):
                _flush()
                in_term_stanza = line == "[Term]"
                current_id = None
                current_alt_ids = []
                current_parents = set()
            elif in_term_stanza and line.startswith("id: GO:"):
                current_id = line[4:]
            elif in_term_stanza and line.startswith("alt_id: GO:"):
                current_alt_ids.append(line[8:])
            elif in_term_stanza and line.startswith("is_a: GO:"):
                parent_id = line[6:].split(" !", 1)[0].strip()
                current_parents.add(parent_id)
        _flush()

    children: dict[str, set[str]] = {}
    for child_id, parent_ids in canonical_parents.items():
        for parent_id in parent_ids:
            children.setdefault(parent_id, set()).add(child_id)

    parents: dict[str, set[str]] = dict(canonical_parents)
    for alt_id, canonical_id in alt_id_map.items():
        parents[alt_id] = canonical_parents.get(canonical_id, set())

    return GODag(parents=parents, children=children)


def resolve_go_dag(data_dir: Path | None = None) -> GODag:
    """
    Download (if needed) and parse go.obo into a GODag, using the same cache
    location and graceful-fallback convention as resolve_go_names.

    Returns an empty GODag (no parents/children) if the ontology can't be
    obtained — callers should treat "no siblings found" as "hard-negative
    baseline not applicable to this term" rather than crashing an eval run.
    """
    try:
        obo_path = _get_or_download(_GO_OBO_URL, "go.obo", data_dir)
        return load_go_dag(obo_path)
    except Exception as e:  # noqa: BLE001 — network/parse failures must not block eval
        logger.warning("Could not load GO DAG (%s); sibling lookups will be empty", e)
        return GODag(parents={}, children={})


def resolve_go_names(
    go_ids: Iterable[str], data_dir: Path | None = None
) -> dict[str, str]:
    """
    Map GO term IDs to human-readable names via the full Gene Ontology (.obo file).

    Downloads and caches go.obo on first use (~50MB). Falls back to echoing the
    GO ID back as its own name for any ID not found in the ontology (e.g. if the
    download fails or the GAF references a since-removed term), so callers never
    get a missing key.

    Args:
        go_ids:   GO term IDs to resolve, e.g. {"GO:0005515", "GO:0005886"}.
        data_dir: Override default annotation cache directory.

    Returns:
        Dict mapping every input GO ID to its human-readable name (or itself
        if no name could be resolved).
    """
    go_ids = list(go_ids)
    try:
        obo_path = _get_or_download(_GO_OBO_URL, "go.obo", data_dir)
        all_names = load_go_obo_names(obo_path)
    except Exception as e:  # noqa: BLE001 — network/parse failures must not block eval
        logger.warning(
            "Could not load GO ontology names (%s); falling back to raw GO IDs", e
        )
        all_names = {}

    resolved = {go_id: all_names.get(go_id, go_id) for go_id in go_ids}
    n_resolved = sum(1 for go_id in go_ids if go_id in all_names)
    logger.info(
        "Resolved %d/%d GO term names from ontology", n_resolved, len(go_ids)
    )
    return resolved
