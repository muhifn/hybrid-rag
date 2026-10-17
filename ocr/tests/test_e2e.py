"""Uji end-to-end ocrpipe (bisa dijalankan dengan pytest atau langsung).

    python tests/test_e2e.py        # mode mandiri (cetak PASS/FAIL)
    pytest tests/test_e2e.py        # mode pytest

Menguji jalur yang tidak bergantung engine OCR eksternal (engine `native`)
secara menyeluruh, plus kebenaran matematika CER/WER. Jalur OCR (tesseract/
paddle/docling/marker) diuji terpisah bila engine-nya terpasang.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _make_pdfs(dirpath: Path) -> None:
    from tests.make_test_pdfs import build

    build(str(dirpath))


def test_page_routing() -> None:
    """Deteksi per halaman: digital vs scan vs campuran."""
    from ocrpipe import Config
    from ocrpipe.detect import probe_document

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_pdfs(tmp)
        cfg = Config()

        digital = probe_document(tmp / "digital_bertabel.pdf", cfg.detect)
        assert digital.kind == "digital", digital.kind

        scan = probe_document(tmp / "hasil_scan.pdf", cfg.detect)
        assert scan.kind == "scan", scan.kind

        mix = probe_document(tmp / "campuran.pdf", cfg.detect)
        assert mix.kind == "campuran", mix.kind


def test_native_extraction_and_outputs() -> None:
    """Ekstraksi digital + semua keluaran (md/jsonl/csv/xlsx)."""
    import json

    from ocrpipe import Config, Pipeline

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_pdfs(tmp)
        out = tmp / "keluaran"

        cfg = Config()
        cfg.detect.force_route = "native"  # jalur deterministik, tanpa OCR
        cfg.verbose = False
        run = Pipeline(cfg).run([tmp / "digital_bertabel.pdf"], out)

        assert len(run.documents) == 1
        doc = run.documents[0]

        # heading terdeteksi
        headings = [b for b in doc.blocks if b.kind == "heading"]
        assert len(headings) >= 3, f"heading terdeteksi: {len(headings)}"

        # tepat satu tabel, 5 baris x 4 kolom
        assert len(doc.tables) == 1
        t = doc.tables[0]
        assert t.n_rows == 5 and t.n_cols == 4, f"{t.n_rows}x{t.n_cols}"
        assert t.header == ["No", "Kegiatan", "Tanggal Mulai", "Tanggal Selesai"]

        # berkas keluaran ada
        md = list((out / "markdown").glob("*.md"))
        assert md, "markdown tidak ditulis"
        md_text = md[0].read_text(encoding="utf-8")
        # caption tabel TIDAK ganda (perbaikan bug)
        assert md_text.count("Jadwal kegiatan semester ganjil") <= 2

        assert list((out / "tabel" / "csv").rglob("*.csv")), "csv tabel tak ada"
        assert list((out / "tabel" / "xlsx").glob("*.xlsx")), "xlsx tabel tak ada"

        combined = out / "jsonl" / "semua-chunk.jsonl"
        assert combined.exists()
        chunks = [json.loads(l) for l in combined.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert chunks, "tidak ada chunk"

        # setiap chunk membawa metadata sitasi
        for c in chunks:
            assert c["doc_id"] and c["page_start"] >= 1
            assert "doc_version" in c and "checksum" in c

        # ada chunk tabel dengan table_refs
        tbl_chunks = [c for c in chunks if c["content_type"] == "table"]
        assert tbl_chunks and tbl_chunks[0]["table_refs"], "chunk tabel tak lengkap"

        # manifest berisi versi pustaka + konfigurasi
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert "versi_pustaka" in manifest and "konfigurasi" in manifest


def test_table_markdown_roundtrip() -> None:
    """Normalisasi grid + render Markdown + sharding header-berulang."""
    from ocrpipe.tables import build_table, shard_table, table_to_markdown

    rows = [
        ["No", "Nama", "Nilai"],
        ["1", "Andi", "80"],
        ["2", "Budi", "90"],
        ["3", "Citra", "", ],  # sel kosong
    ]
    t = build_table(rows, page=1, index=1, engine="x")
    md = table_to_markdown(t)
    assert md.count("\n") >= 4
    assert "| No | Nama | Nilai |" in md

    # sharding mengulang header
    big = build_table([["H1", "H2"]] + [[str(i), f"v{i}"] for i in range(50)],
                      page=1, index=1, engine="x")
    shards = shard_table(big, rows_per_shard=20)
    assert len(shards) == 3
    for s in shards:
        assert s.rows[0] == ["H1", "H2"], "header tidak diulang di shard"


def test_html_table_spans() -> None:
    """Parsing HTML tabel dengan colspan/rowspan (jalur paddle/marker)."""
    from ocrpipe.tables import html_table_to_rows

    html = """<table>
      <tr><td colspan="2">Gabung</td><td>C</td></tr>
      <tr><td rowspan="2">Kiri</td><td>B2</td><td>C2</td></tr>
      <tr><td>B3</td><td>C3</td></tr>
    </table>"""
    rows = html_table_to_rows(html)
    assert len(rows) == 3
    assert rows[0] == ["Gabung", "Gabung", "C"]  # colspan dibentangkan
    assert rows[1][0] == "Kiri" and rows[2][0] == "Kiri"  # rowspan diisi-turun


def test_cer_wer_math() -> None:
    """Kebenaran CER/WER + agregasi mikro."""
    from ocrpipe.qa import aggregate_eval, compute_cer_wer

    assert compute_cer_wer("abc", "abc").cer == 0.0
    r = compute_cer_wer("abcde", "abXde")
    assert abs(r.cer - 0.2) < 1e-9, r.cer
    r2 = compute_cer_wer("satu dua tiga", "satu dua tiga")
    assert r2.wer == 0.0
    r3 = compute_cer_wer("satu dua tiga", "satu dua empat")
    assert abs(r3.wer - 1 / 3) < 1e-9, r3.wer

    agg = aggregate_eval([compute_cer_wer("abcde", "abXde"), compute_cer_wer("fg", "fg")])
    # 1 edit dari total 7 karakter acuan
    assert abs(agg.cer - 1 / 7) < 1e-9, agg.cer


def test_chunker_respects_structure() -> None:
    """Chunk tidak memotong tabel; menyisipkan jejak heading."""
    from ocrpipe.config import ChunkConfig
    from ocrpipe.chunker import Chunker
    from ocrpipe.models import (
        KIND_HEADING, KIND_PARAGRAPH, KIND_TABLE, Block, DocMeta, DocResult,
        PageResult, TableData,
    )

    doc = DocResult(meta=DocMeta(doc_id="d1", title="Doc", version="1"), n_pages=1)
    page = PageResult(page=1, route="native", engine="native")
    tbl = TableData(rows=[["A", "B"], ["1", "2"]], n_header_rows=1, page=1, index=1)
    page.blocks = [
        Block(kind=KIND_HEADING, text="BAB I", page=1, order=0, level=1),
        Block(kind=KIND_PARAGRAPH, text="Isi paragraf satu." * 5, page=1, order=1),
        Block(kind=KIND_TABLE, text="| A | B |\n| --- | --- |\n| 1 | 2 |",
              page=1, order=2, table=tbl),
    ]
    doc.pages = [page]

    chunks = Chunker(ChunkConfig()).chunk_document(doc)
    assert chunks
    tbl_chunks = [c for c in chunks if c.content_type == "table"]
    assert len(tbl_chunks) == 1, "tabel harus jadi chunk sendiri"
    txt_chunks = [c for c in chunks if c.content_type == "text"]
    assert any("BAB I" in c.text for c in txt_chunks), "jejak heading hilang"


def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} lulus")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
