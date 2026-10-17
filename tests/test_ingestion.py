from __future__ import annotations

import pytest

from app.graph.tables import infer_type, load_tables
from app.ingestion import load_documents, parse_frontmatter, split_documents
from tests.conftest import FIXTURES

DOCS = FIXTURES / "documents"
TABLES = FIXTURES / "tables"


def test_frontmatter_is_parsed_and_stripped():
    meta, body = parse_frontmatter("---\njudul: Contoh\ntahun: 2025\n---\n\n# Isi\nteks")
    assert meta == {"judul": "Contoh", "tahun": "2025"}
    assert body.startswith("# Isi")


def test_documents_load_with_titles_from_frontmatter():
    docs = load_documents(DOCS)
    assert len(docs) == 3
    titles = {d.metadata["source"]: d.metadata["title"] for d in docs}
    assert titles["kebijakan-cuti.md"] == "Kebijakan Cuti Pegawai"
    assert all("---" not in d.page_content[:5] for d in docs)


def test_chunks_respect_size_and_have_unique_ids():
    docs = load_documents(DOCS)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    longest_title = max(len(d.metadata["title"]) for d in docs)
    assert len(chunks) > len(docs)
    # A chunk is at most chunk_size plus the "title\n\n" context prefix.
    assert all(len(c.page_content) <= 300 + longest_title + 2 for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_non_first_chunks_carry_the_document_title():
    docs = load_documents(DOCS)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    later = [c for c in chunks if c.metadata["chunk_index"] > 0]
    assert later and all(c.page_content.startswith(c.metadata["title"]) for c in later)


@pytest.mark.parametrize("values,expected", [
    (["1", "2", ""], "INTEGER"),
    (["1.5", "2"], "REAL"),
    (["a", "1"], "TEXT"),
    (["", None], "TEXT"),
])
def test_type_inference(values, expected):
    assert infer_type(values) == expected


def test_foreign_keys_inferred_by_exact_and_suffix_match():
    fks = {t.name: t.fks for t in load_tables(TABLES)}
    assert fks["pegawai"] == {"unit_id": "unit"}
    assert fks["proyek"] == {"unit_id": "unit", "penanggung_jawab_pegawai_id": "pegawai"}
    assert fks["unit"] == {"kepala_pegawai_id": "pegawai"}


def test_fk_override_applies(tmp_path):
    (tmp_path / "a.csv").write_text("a_id,owner\nA1,B1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("b_id,name\nB1,x\n", encoding="utf-8")
    assert load_tables(tmp_path)[0].fks == {}
    tables = load_tables(tmp_path, fk_overrides={"a.owner": "b"})
    assert tables[0].fks == {"owner": "b"}
