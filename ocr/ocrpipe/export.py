"""Penulisan keluaran: Markdown, JSONL, CSV tabel, XLSX tabel, manifest.

Struktur folder keluaran:

    keluaran/
      markdown/<doc_id>.md          <- untuk dibaca & dikoreksi manusia
      jsonl/<doc_id>.jsonl          <- chunk per dokumen
      jsonl/semua-chunk.jsonl       <- gabungan, siap di-ingest ke Qdrant
      tabel/csv/<doc_id>/p0003_t01.csv
      tabel/xlsx/<doc_id>.xlsx      <- multi-sheet + sheet INDEKS
      qa/laporan-qa.md, qa/qa-per-halaman.csv
      manifest.json                 <- jejak reproduktibilitas satu run
      konfigurasi-efektif.yaml
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import Config, OutputConfig
from .models import (
    KIND_CAPTION,
    KIND_FOOTER,
    KIND_HEADER,
    KIND_HEADING,
    KIND_TABLE,
    Chunk,
    DocResult,
)
from .tables import write_table_csv, write_tables_xlsx
from .utils import get_logger


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def _yaml_front_matter(doc: DocResult) -> str:
    m = doc.meta
    rows = {
        "doc_id": m.doc_id,
        "judul": m.title,
        "kategori": m.category,
        "versi": m.version,
        "tanggal_berlaku": m.effective_date or "",
        "unit": m.unit or "",
        "sumber": Path(m.source_file).name,
        "url": m.url or "",
        "bahasa": m.language,
        "jumlah_halaman": doc.n_pages,
        "halaman_diproses": len(doc.pages),
        "rute": ", ".join(f"{k}={v}" for k, v in sorted(doc.route_counts().items())),
        "engine": ", ".join(f"{k}={v}" for k, v in sorted(doc.engine_counts().items())),
        "jumlah_tabel": len(doc.tables),
        "halaman_perlu_tinjauan": ", ".join(str(p) for p in doc.pages_needing_review),
        "diekstrak_pada": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    lines = ["---"]
    for k, v in rows.items():
        text = str(v).replace('"', '\\"')
        lines.append(f'{k}: "{text}"')
    lines.append("---\n")
    return "\n".join(lines)


def doc_to_markdown(doc: DocResult, cfg: OutputConfig) -> str:
    parts: list[str] = []
    if cfg.front_matter:
        parts.append(_yaml_front_matter(doc))

    if doc.meta.title:
        parts.append(f"# {doc.meta.title}\n")

    last_page = None
    for page in sorted(doc.pages, key=lambda p: p.page):
        if page.page != last_page:
            if cfg.page_marker == "comment":
                parts.append(f"<!-- halaman: {page.page} -->")
            elif cfg.page_marker == "heading":
                parts.append(f"\n**[Halaman {page.page}]**\n")
            last_page = page.page

        if page.needs_review:
            parts.append(
                f"<!-- PERLU TINJAUAN hlm {page.page}: "
                + "; ".join(page.review_reasons).replace("--", "- -")
                + " -->"
            )

        for block in sorted(page.blocks, key=lambda b: b.order):
            if block.kind == KIND_HEADING:
                level = max(2, min(6, block.level or 2))
                parts.append(f"\n{'#' * level} {block.text.strip()}\n")
            elif block.kind == KIND_TABLE:
                parts.append(f"\n{block.text}\n")
            elif block.kind == KIND_CAPTION:
                parts.append(f"\n*{block.text.strip()}*\n")
            elif block.kind in (KIND_HEADER, KIND_FOOTER):
                # Dipertahankan sebagai komentar: tidak mencemari teks yang
                # dibaca, tetapi tetap bisa diaudit saat memeriksa hasil OCR.
                parts.append(
                    f"<!-- {block.kind}: {block.text.strip()[:160]} -->"
                )
            else:
                parts.append(f"\n{block.text.strip()}\n")

    text = "\n".join(parts)
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text.strip() + "\n"


def write_markdown(doc: DocResult, out_dir: str | Path, cfg: OutputConfig) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc.meta.doc_id}.md"
    path.write_text(doc_to_markdown(doc, cfg), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #


def write_jsonl(chunks: Sequence[Chunk], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(ch.to_jsonl())
            fh.write("\n")
    return p


def append_jsonl(chunks: Sequence[Chunk], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(ch.to_jsonl())
            fh.write("\n")
    return p


# --------------------------------------------------------------------------- #
# Tabel
# --------------------------------------------------------------------------- #


def write_tables(
    doc: DocResult, out_root: str | Path, cfg: OutputConfig
) -> dict[str, Any]:
    out_root = Path(out_root)
    info: dict[str, Any] = {"csv": [], "xlsx": None, "jumlah": 0}
    tables = [t for t in doc.tables if not t.is_degenerate()]
    info["jumlah"] = len(tables)
    if not tables:
        return info

    if cfg.csv_tables:
        csv_dir = out_root / "csv" / doc.meta.doc_id
        for t in tables:
            path = write_table_csv(t, csv_dir / f"{t.slug}.csv")
            info["csv"].append(str(path))

    if cfg.xlsx_tables:
        xlsx_path = out_root / "xlsx" / f"{doc.meta.doc_id}.xlsx"
        written = write_tables_xlsx(
            tables, xlsx_path, doc_title=doc.meta.title, doc_id=doc.meta.doc_id
        )
        info["xlsx"] = str(written) if written else None

    return info


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def write_manifest(
    path: str | Path,
    *,
    config: Config,
    documents: Sequence[dict[str, Any]],
    engine_notes: Sequence[str],
    started_at: str,
    finished_at: str,
    elapsed_s: float,
) -> Path:
    """Simpan jejak satu kali eksekusi.

    Manifest ini yang membuat hasil ekstraksi dapat diaudit dan direproduksi:
    versi pustaka, engine yang benar-benar dipakai, serta konfigurasi efektif
    semuanya tercatat. Lampirkan di skripsi bila diminta bukti prosedur.
    """
    from .engines.base import version_of

    payload = {
        "dibuat_pada": finished_at,
        "mulai": started_at,
        "durasi_detik": round(elapsed_s, 2),
        "lingkungan": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mesin": platform.machine(),
        },
        "versi_pustaka": {
            name: version_of(name)
            for name in (
                "pymupdf",
                "pdfplumber",
                "pypdf",
                "pytesseract",
                "opencv-python",
                "openpyxl",
                "pandas",
                "paddleocr",
                "paddlepaddle",
                "docling",
                "marker-pdf",
            )
        },
        "pemilihan_engine": list(engine_notes),
        "konfigurasi": config.to_dict(),
        "dokumen": list(documents),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    get_logger().debug("manifest ditulis: %s", p)
    return p


# --------------------------------------------------------------------------- #
# Panduan ingest Qdrant
# --------------------------------------------------------------------------- #

_INGEST_SNIPPET = '''"""Contoh ingest hasil ocrpipe ke Qdrant.

Dibuat otomatis oleh ocrpipe. Sesuaikan nama koleksi dan model embedding.
Skrip ini sengaja TIDAK dijalankan otomatis oleh pipeline: pemilihan model
embedding adalah keputusan penelitian, bukan detail implementasi.
"""

import json
from pathlib import Path

JSONL = Path("keluaran/jsonl/semua-chunk.jsonl")
COLLECTION = "dokumen_akademik"
EMBED_MODEL = "intfloat/multilingual-e5-base"  # ganti sesuai hasil perbandingan

def load_chunks(path=JSONL):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def main():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks()
    print(f"memuat {len(chunks)} chunk")

    # Chunk bertanda needs_review sebaiknya ditahan sampai dikoreksi manual,
    # agar hasil OCR yang meragukan tidak ikut menjawab pertanyaan pengguna.
    chunks = [c for c in chunks if not c.get("needs_review")]
    print(f"{len(chunks)} chunk lolos saringan QA")

    model = SentenceTransformer(EMBED_MODEL)
    # Model keluarga E5 mensyaratkan prefiks "passage: " pada sisi dokumen.
    texts = [f"passage: {c['text']}" for c in chunks]
    vectors = model.encode(texts, batch_size=16, show_progress_bar=True,
                           normalize_embeddings=True)

    client = QdrantClient(path="./qdrant_local")  # atau url="http://localhost:6333"
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
            for i in range(len(chunks))
        ],
    )
    print("selesai. payload sudah memuat doc_title/page_start/doc_version "
          "sehingga sitasi dapat dibangun tanpa query tambahan.")

if __name__ == "__main__":
    main()
'''


def write_ingest_example(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_INGEST_SNIPPET, encoding="utf-8")
    return p
