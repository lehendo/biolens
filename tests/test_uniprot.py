"""
Tests for UniProt Swiss-Prot / Gene Ontology data loading.

All tests use synthetic FASTA/GAF/OBO files written to temp dirs — no network
access, no real UniProt/GO downloads. Covers:
  - Swiss-Prot FASTA parsing (multi-line sequences, max_seq_len, max_seqs, ID formats)
  - GO annotation loading (evidence code / aspect / NOT-qualifier filtering)
  - The zgrep prefilter path and its Python fallback (must agree exactly)
  - GO OBO ontology name parsing (including alt_id merges)
  - resolve_go_names end-to-end, including graceful fallback on missing ontology
"""

from __future__ import annotations

import gzip
import textwrap
from pathlib import Path
from unittest.mock import patch

# ── Swiss-Prot FASTA ──────────────────────────────────────────────────────────

class TestLoadSwissprot:
    def _write_fasta(self, tmp_path: Path, records: list[tuple[str, str]]) -> Path:
        data_dir = tmp_path / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = data_dir / "uniprot_sprot.fasta.gz"
        lines = []
        for header, seq in records:
            lines.append(header)
            # wrap sequence across multiple lines like real UniProt FASTA
            for i in range(0, len(seq), 60):
                lines.append(seq[i : i + 60])
        with gzip.open(fasta_path, "wt") as f:
            f.write("\n".join(lines) + "\n")
        return data_dir

    def test_parses_standard_sp_header(self, tmp_path):
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path, [(">sp|P12345|GENE1_HUMAN Some protein", "MKTAYIAKQRQISFVK")]
        )
        ids, seqs = load_swissprot(data_dir=data_dir)
        assert ids == ["P12345"]
        assert seqs == ["MKTAYIAKQRQISFVK"]

    def test_multiline_sequence_reassembled(self, tmp_path):
        from biolens.data.uniprot import load_swissprot

        seq = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 residues, spans multiple 60-char lines
        data_dir = self._write_fasta(tmp_path, [(">sp|Q99999|GENE2_HUMAN desc", seq)])
        ids, seqs = load_swissprot(data_dir=data_dir)
        assert seqs == [seq]

    def test_multiple_records(self, tmp_path):
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path,
            [
                (">sp|P00001|A_HUMAN d", "ACDEFG"),
                (">sp|P00002|B_HUMAN d", "HIKLMN"),
                (">sp|P00003|C_HUMAN d", "PQRSTV"),
            ],
        )
        ids, seqs = load_swissprot(data_dir=data_dir)
        assert ids == ["P00001", "P00002", "P00003"]
        assert seqs == ["ACDEFG", "HIKLMN", "PQRSTV"]

    def test_max_seqs_truncates(self, tmp_path):
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path, [(f">sp|P0000{i}|X_HUMAN d", "ACDE") for i in range(5)]
        )
        ids, seqs = load_swissprot(max_seqs=2, data_dir=data_dir)
        assert len(ids) == 2

    def test_random_seed_none_preserves_file_order(self, tmp_path):
        """Default behavior (random_seed=None) must stay byte-for-byte the
        historical file-order truncation -- existing runs depend on this
        not changing."""
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path, [(f">sp|P0000{i}|X_HUMAN d", "ACDE") for i in range(5)]
        )
        ids, _ = load_swissprot(max_seqs=3, data_dir=data_dir)
        assert ids == ["P00000", "P00001", "P00002"]

    def test_random_seed_set_does_not_return_file_order(self, tmp_path):
        """File order is measurably taxonomy-skewed. A seeded sample must actually shuffle, not just accept the
        parameter and ignore it."""
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path, [(f">sp|P{i:05d}|X_HUMAN d", "ACDE") for i in range(200)]
        )
        ids, _ = load_swissprot(max_seqs=50, data_dir=data_dir, random_seed=7)
        assert len(ids) == 50
        assert ids != [f"P{i:05d}" for i in range(50)]

    def test_random_seed_is_deterministic(self, tmp_path):
        """Same seed, same input -> byte-identical sample, every time --
        this is what makes a sampled sweep reproducible/reportable at all."""
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path, [(f">sp|P{i:05d}|X_HUMAN d", "ACDE") for i in range(200)]
        )
        ids1, seqs1 = load_swissprot(max_seqs=50, data_dir=data_dir, random_seed=7)
        ids2, seqs2 = load_swissprot(max_seqs=50, data_dir=data_dir, random_seed=7)
        assert ids1 == ids2
        assert seqs1 == seqs2

    def test_different_seeds_give_different_samples(self, tmp_path):
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path, [(f">sp|P{i:05d}|X_HUMAN d", "ACDE") for i in range(200)]
        )
        ids_a, _ = load_swissprot(max_seqs=50, data_dir=data_dir, random_seed=1)
        ids_b, _ = load_swissprot(max_seqs=50, data_dir=data_dir, random_seed=2)
        assert ids_a != ids_b

    def test_random_seed_samples_from_whole_pool_not_just_prefix(self, tmp_path):
        """The whole point: a seeded sample must be able to include entries
        well past position max_seqs in file order, not just reshuffle a
        file-order-truncated prefix."""
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path, [(f">sp|P{i:05d}|X_HUMAN d", "ACDE") for i in range(1000)]
        )
        ids, _ = load_swissprot(max_seqs=50, data_dir=data_dir, random_seed=7)
        late_entries = {f"P{i:05d}" for i in range(500, 1000)}
        assert any(pid in late_entries for pid in ids), (
            "seeded sample never drew from beyond the first half of the file "
            "-- random_seed is not actually sampling the whole pool"
        )

    def test_max_seq_len_excludes_long_sequences(self, tmp_path):
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path,
            [
                (">sp|P00001|SHORT_HUMAN d", "ACDE"),
                (">sp|P00002|LONG_HUMAN d", "A" * 2000),
            ],
        )
        ids, seqs = load_swissprot(max_seq_len=1024, data_dir=data_dir)
        assert ids == ["P00001"]

    def test_non_standard_residues_replaced_with_x(self, tmp_path):
        from biolens.data.uniprot import load_swissprot

        # 'U' (selenocysteine) and 'B' are non-standard/ambiguous residues.
        data_dir = self._write_fasta(
            tmp_path, [(">sp|P00001|X_HUMAN d", "ACDEUB")]
        )
        ids, seqs = load_swissprot(data_dir=data_dir)
        assert seqs == ["ACDEXX"]

    def test_last_record_without_trailing_newline_included(self, tmp_path):
        """The final FASTA record (no header follows) must not be dropped."""
        from biolens.data.uniprot import load_swissprot

        data_dir = self._write_fasta(
            tmp_path,
            [
                (">sp|P00001|A_HUMAN d", "ACDE"),
                (">sp|P00002|B_HUMAN d", "FGHI"),
            ],
        )
        ids, _ = load_swissprot(data_dir=data_dir)
        assert "P00002" in ids  # last record present, not just first


# ── GO annotation loading ─────────────────────────────────────────────────────

_GAF_HEADER = "!gaf-version: 2.2\n"


def _gaf_line(
    obj_id: str,
    go_id: str,
    evidence: str = "EXP",
    aspect: str = "F",
    qualifier: str = "",
) -> str:
    # GAF 2.x: DB, DB_Object_ID, Symbol, Qualifier, GO_ID, DB_Ref, Evidence,
    #          With, Aspect, Name, Synonym, Type, Taxon, Date, Assigned_By
    return (
        f"UniProtKB\t{obj_id}\tSYM\t{qualifier}\t{go_id}\tGO_REF:1\t{evidence}\t"
        f"\t{aspect}\tdesc\tsyn\tprotein\ttaxon:9606\t20200101\tUniProt\t\t\n"
    )


class TestLoadGoAnnotations:
    def _write_gaf(self, tmp_path: Path, lines: list[str]) -> Path:
        data_dir = tmp_path / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)
        gaf_path = data_dir / "goa_uniprot_all.gaf.gz"
        with gzip.open(gaf_path, "wt") as f:
            f.write(_GAF_HEADER)
            f.writelines(lines)
        return data_dir

    def test_basic_filtering_by_protein_id(self, tmp_path):
        from biolens.data.uniprot import load_go_annotations

        data_dir = self._write_gaf(
            tmp_path,
            [
                _gaf_line("P12345", "GO:0005524"),
                _gaf_line("Q99999", "GO:0003677"),
            ],
        )
        go_labels, go_names = load_go_annotations(
            protein_ids={"P12345"}, data_dir=data_dir
        )
        assert go_labels == {"P12345": {"GO:0005524"}}
        assert "GO:0005524" in go_names

    def test_not_qualifier_excluded(self, tmp_path):
        from biolens.data.uniprot import load_go_annotations

        data_dir = self._write_gaf(
            tmp_path,
            [
                _gaf_line("P12345", "GO:0005524"),
                _gaf_line("P12345", "GO:0016301", qualifier="NOT"),
            ],
        )
        go_labels, _ = load_go_annotations(protein_ids={"P12345"}, data_dir=data_dir)
        assert go_labels["P12345"] == {"GO:0005524"}

    def test_evidence_code_filter(self, tmp_path):
        from biolens.data.uniprot import load_go_annotations

        data_dir = self._write_gaf(
            tmp_path,
            [
                _gaf_line("P12345", "GO:0005524", evidence="EXP"),
                _gaf_line("P12345", "GO:0003677", evidence="IEA"),
            ],
        )
        go_labels, _ = load_go_annotations(
            protein_ids={"P12345"}, evidence_codes={"EXP"}, data_dir=data_dir
        )
        assert go_labels["P12345"] == {"GO:0005524"}

    def test_aspect_filter(self, tmp_path):
        from biolens.data.uniprot import load_go_annotations

        data_dir = self._write_gaf(
            tmp_path,
            [
                _gaf_line("P12345", "GO:0005524", aspect="F"),
                _gaf_line("P12345", "GO:0008150", aspect="P"),
            ],
        )
        go_labels, _ = load_go_annotations(
            protein_ids={"P12345"}, aspects={"F"}, data_dir=data_dir
        )
        assert go_labels["P12345"] == {"GO:0005524"}

    def test_no_protein_id_filter_loads_everything(self, tmp_path):
        from biolens.data.uniprot import load_go_annotations

        data_dir = self._write_gaf(
            tmp_path,
            [_gaf_line("P12345", "GO:0005524"), _gaf_line("Q99999", "GO:0003677")],
        )
        go_labels, _ = load_go_annotations(data_dir=data_dir)
        assert set(go_labels.keys()) == {"P12345", "Q99999"}

    def test_zgrep_prefilter_matches_python_fallback(self, tmp_path):
        """The fast zgrep path and the pure-Python fallback must agree exactly."""
        from biolens.data.uniprot import load_go_annotations

        data_dir = self._write_gaf(
            tmp_path,
            [
                _gaf_line("P00001", "GO:0000001"),
                _gaf_line("P00002", "GO:0000002"),
                _gaf_line("P00003", "GO:0000003", qualifier="NOT"),
                _gaf_line("P00004", "GO:0000004", evidence="IEA"),
            ],
        )
        protein_ids = {"P00001", "P00002", "P00003", "P00004"}

        via_zgrep, _ = load_go_annotations(protein_ids=protein_ids, data_dir=data_dir)

        with patch(
            "biolens.data.uniprot.subprocess.run",
            side_effect=FileNotFoundError("zgrep not found (simulated)"),
        ):
            via_fallback, _ = load_go_annotations(
                protein_ids=protein_ids, data_dir=data_dir
            )

        assert via_zgrep == via_fallback

    def test_empty_protein_ids_set_returns_nothing(self, tmp_path):
        """An empty (but non-None) protein_ids set should match nothing, not everything."""
        from biolens.data.uniprot import load_go_annotations

        data_dir = self._write_gaf(tmp_path, [_gaf_line("P12345", "GO:0005524")])
        go_labels, _ = load_go_annotations(protein_ids=set(), data_dir=data_dir)
        assert go_labels == {}


# ── GO OBO ontology parsing ───────────────────────────────────────────────────

_SAMPLE_OBO = textwrap.dedent(
    """\
    format-version: 1.2

    [Term]
    id: GO:0000001
    name: mitochondrion inheritance
    namespace: biological_process

    [Term]
    id: GO:0000002
    name: mitochondrial genome maintenance
    namespace: biological_process
    alt_id: GO:0000099

    [Typedef]
    id: part_of
    name: part of
    """
)


class TestLoadGoOboNames:
    def test_parses_id_name_pairs(self, tmp_path):
        from biolens.data.uniprot import load_go_obo_names

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO)
        names = load_go_obo_names(obo_path)
        assert names["GO:0000001"] == "mitochondrion inheritance"
        assert names["GO:0000002"] == "mitochondrial genome maintenance"

    def test_alt_id_resolves_to_same_name(self, tmp_path):
        from biolens.data.uniprot import load_go_obo_names

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO)
        names = load_go_obo_names(obo_path)
        assert names["GO:0000099"] == "mitochondrial genome maintenance"

    def test_typedef_stanza_not_treated_as_term(self, tmp_path):
        """[Typedef] stanzas use the same 'id:'/'name:' keys but aren't GO terms."""
        from biolens.data.uniprot import load_go_obo_names

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO)
        names = load_go_obo_names(obo_path)
        assert "part_of" not in names

    def test_last_stanza_in_file_is_included(self, tmp_path):
        """Regression guard: a stanza with no following [Term]/[Typedef] must
        still be flushed (there's no natural boundary after EOF)."""
        from biolens.data.uniprot import load_go_obo_names

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "[Term]\nid: GO:0000005\nname: last term in file\n"
        )
        names = load_go_obo_names(obo_path)
        assert names["GO:0000005"] == "last term in file"


# ── GO DAG (is_a) parsing ─────────────────────────────────────────────────────

_SAMPLE_OBO_WITH_HIERARCHY = textwrap.dedent(
    """\
    format-version: 1.2

    [Term]
    id: GO:0016020
    name: membrane

    [Term]
    id: GO:0022857
    name: transmembrane transporter activity
    is_a: GO:0016020 ! membrane

    [Term]
    id: GO:0005886
    name: acetylcholine-gated channel complex
    is_a: GO:0016020 ! membrane
    alt_id: GO:0009999

    [Term]
    id: GO:0046872
    name: metal ion transmembrane transporter activity
    is_a: GO:0016020 ! membrane
    is_a: GO:0022857 ! transmembrane transporter activity

    [Term]
    id: GO:0008152
    name: unrelated metabolic process
    """
)


class TestLoadGoDag:
    def test_parses_direct_parent(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert dag.parents["GO:0022857"] == {"GO:0016020"}

    def test_parses_multiple_parents(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert dag.parents["GO:0046872"] == {"GO:0016020", "GO:0022857"}

    def test_term_with_no_is_a_has_no_parents(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert dag.parents.get("GO:0016020", set()) == set()

    def test_children_index_is_inverse_of_parents(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert dag.children["GO:0016020"] == {
            "GO:0022857", "GO:0005886", "GO:0046872",
        }

    def test_alt_id_gets_same_parents_as_canonical_id(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert dag.parents["GO:0009999"] == dag.parents["GO:0005886"]

    def test_siblings_share_direct_parent(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert "GO:0022857" in dag.siblings("GO:0005886")
        assert "GO:0046872" in dag.siblings("GO:0005886")

    def test_siblings_excludes_self(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert "GO:0005886" not in dag.siblings("GO:0005886")

    def test_unrelated_term_has_no_siblings_in_common(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert dag.siblings("GO:0008152") == set()

    def test_unknown_term_has_no_siblings(self, tmp_path):
        from biolens.data.uniprot import load_go_dag

        obo_path = tmp_path / "go.obo"
        obo_path.write_text(_SAMPLE_OBO_WITH_HIERARCHY)
        dag = load_go_dag(obo_path)
        assert dag.siblings("GO:9999999") == set()


class TestResolveGoDag:
    def test_resolves_and_parses_cached_file(self, tmp_path):
        from biolens.data.uniprot import resolve_go_dag

        data_dir = tmp_path / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "go.obo").write_text(_SAMPLE_OBO_WITH_HIERARCHY)

        dag = resolve_go_dag(data_dir=data_dir)
        assert dag.parents["GO:0022857"] == {"GO:0016020"}

    def test_missing_ontology_file_falls_back_to_empty_dag(self, tmp_path):
        """A network failure must not crash the eval — hard-negative sampling
        simply becomes unavailable for that run, same fallback philosophy as
        resolve_go_names."""
        from biolens.data.uniprot import resolve_go_dag

        data_dir = tmp_path / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "biolens.data._http_cache._download_with_progress",
            side_effect=OSError("simulated network failure"),
        ):
            dag = resolve_go_dag(data_dir=data_dir)

        assert dag.parents == {}
        assert dag.children == {}
        assert dag.siblings("GO:0005886") == set()


# ── resolve_go_names ───────────────────────────────────────────────────────────

class TestResolveGoNames:
    def test_resolves_known_ids(self, tmp_path):
        from biolens.data.uniprot import resolve_go_names

        data_dir = tmp_path / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "go.obo").write_text(_SAMPLE_OBO)

        resolved = resolve_go_names(["GO:0000001", "GO:0000002"], data_dir=data_dir)
        assert resolved["GO:0000001"] == "mitochondrion inheritance"
        assert resolved["GO:0000002"] == "mitochondrial genome maintenance"

    def test_unknown_id_falls_back_to_itself(self, tmp_path):
        from biolens.data.uniprot import resolve_go_names

        data_dir = tmp_path / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "go.obo").write_text(_SAMPLE_OBO)

        resolved = resolve_go_names(["GO:9999999"], data_dir=data_dir)
        assert resolved["GO:9999999"] == "GO:9999999"

    def test_missing_ontology_file_falls_back_gracefully(self, tmp_path):
        """If the ontology can't be obtained at all (download fails, no
        network), every GO ID should still resolve — to itself — rather
        than raising and aborting the whole eval run."""
        from biolens.data.uniprot import resolve_go_names

        data_dir = tmp_path / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "biolens.data._http_cache._download_with_progress",
            side_effect=OSError("simulated network failure"),
        ):
            resolved = resolve_go_names(["GO:0005515", "GO:0005886"], data_dir=data_dir)

        assert resolved == {"GO:0005515": "GO:0005515", "GO:0005886": "GO:0005886"}

    def test_empty_input_returns_empty_dict(self, tmp_path):
        from biolens.data.uniprot import resolve_go_names

        data_dir = tmp_path / "annotations"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "go.obo").write_text(_SAMPLE_OBO)

        assert resolve_go_names([], data_dir=data_dir) == {}
