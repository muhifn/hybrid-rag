"""Struktur data inti pipeline OCR.

Semua objek di sini sengaja dibuat sebagai dataclass biasa (tanpa pydantic)
supaya dependensi tetap minimal dan mudah di-serialize ke JSON/JSONL.

Konvensi penting:
  * Nomor halaman SELALU 1-based di seluruh pipeline (page=1 adalah halaman
    pertama). Konversi ke 0-based hanya dilakukan di dalam adapter engine
    yang memerlukannya (PyMuPDF, Marker).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Konstanta jenis blok
# --------------------------------------------------------------------------- #

KIND_HEADING = "heading"
KIND_PARAGRAPH = "paragraph"
KIND_LIST = "list"
KIND_TABLE = "table"
KIND_FIGURE = "figure"
KIND_FORMULA = "formula"
KIND_CAPTION = "caption"
KIND_HEADER = "page_header"
KIND_FOOTER = "page_footer"

#: Blok yang secara default TIDAK dimasukkan ke chunk RAG (noise kop surat /
#: nomor halaman berulang yang merusak retrieval).
NOISE_KINDS = frozenset({KIND_HEADER, KIND_FOOTER})

#: Rute pemrosesan halaman.
ROUTE_NATIVE = "native"  # halaman punya text layer tepercaya -> ekstraksi langsung
ROUTE_OCR = "ocr"  # halaman hasil scan / text layer rusak -> wajib OCR
ROUTE_EMPTY = "empty"  # halaman kosong (tidak ada teks & tidak ada gambar berarti)


# --------------------------------------------------------------------------- #
# Tabel
# --------------------------------------------------------------------------- #


@dataclass
class TableData:
    """Satu tabel hasil ekstraksi, sudah dinormalisasi menjadi grid persegi."""

    rows: list[list[str]] = field(default_factory=list)
    n_header_rows: int = 1
    page: int = 0
    index: int = 0  # indeks tabel dalam satu dokumen (1-based)
    caption: str | None = None
    engine: str = ""
    bbox: tuple[float, float, float, float] | None = None
    #: Catatan kualitas, mis. "jumlah kolom tidak konsisten sebelum normalisasi".
    warnings: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> list[str]:
        if not self.rows or self.n_header_rows <= 0:
            return []
        if self.n_header_rows == 1:
            return list(self.rows[0])
        # Header multi-baris digabung per kolom dengan spasi.
        merged: list[str] = []
        for col in range(self.n_cols):
            parts = [
                self.rows[r][col]
                for r in range(min(self.n_header_rows, self.n_rows))
                if col < len(self.rows[r]) and self.rows[r][col]
            ]
            merged.append(" ".join(dict.fromkeys(parts)).strip())
        return merged

    @property
    def body(self) -> list[list[str]]:
        return self.rows[self.n_header_rows :] if self.n_header_rows else self.rows

    @property
    def slug(self) -> str:
        return f"p{self.page:04d}_t{self.index:02d}"

    def is_degenerate(self) -> bool:
        """Tabel yang tidak layak diekspor (1 sel, atau semua sel kosong)."""
        if self.n_rows == 0 or self.n_cols == 0:
            return True
        if self.n_rows * self.n_cols <= 1:
            return True
        return not any(cell.strip() for row in self.rows for cell in row)


# --------------------------------------------------------------------------- #
# Blok konten
# --------------------------------------------------------------------------- #


@dataclass
class Block:
    """Satu unit konten dalam urutan baca (reading order)."""

    kind: str = KIND_PARAGRAPH
    text: str = ""  # untuk KIND_TABLE: tabel dalam format Markdown
    page: int = 0
    order: int = 0
    level: int | None = None  # kedalaman heading (1..6)
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None  # 0..100, None jika engine tidak melaporkan
    table: TableData | None = None
    engine: str = ""

    @property
    def is_table(self) -> bool:
        return self.kind == KIND_TABLE and self.table is not None

    @property
    def char_count(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------- #
# Hasil per halaman
# --------------------------------------------------------------------------- #


@dataclass
class PageResult:
    page: int = 0
    route: str = ROUTE_NATIVE
    engine: str = ""
    blocks: list[Block] = field(default_factory=list)
    #: Metrik kualitas (diisi oleh ocrpipe.qa).
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def tables(self) -> list[TableData]:
        return [b.table for b in self.blocks if b.is_table and b.table is not None]

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    @property
    def mean_confidence(self) -> float | None:
        vals = [b.confidence for b in self.blocks if b.confidence is not None]
        return round(sum(vals) / len(vals), 2) if vals else None


# --------------------------------------------------------------------------- #
# Metadata & hasil per dokumen
# --------------------------------------------------------------------------- #


@dataclass
class DocMeta:
    """Metadata dokumen sumber.

    Field `version` dan `effective_date` sengaja dijadikan first-class karena
    sistem RAG target mewajibkan sitasi berbentuk (dokumen, halaman, versi).
    """

    doc_id: str = ""
    title: str = ""
    source_file: str = ""
    category: str = "lain-lain"  # mis. panduan-akademik, kalender, sk, biaya
    version: str = "tidak-diketahui"
    effective_date: str | None = None  # ISO-8601 (YYYY-MM-DD)
    unit: str | None = None  # prodi / biro penerbit
    url: str | None = None
    language: str = "id"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocResult:
    meta: DocMeta = field(default_factory=DocMeta)
    n_pages: int = 0
    pages: list[PageResult] = field(default_factory=list)
    doc_warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def tables(self) -> list[TableData]:
        out: list[TableData] = []
        for p in self.pages:
            out.extend(p.tables)
        return out

    @property
    def blocks(self) -> list[Block]:
        """Seluruh blok dokumen dalam urutan halaman lalu urutan baca."""
        out: list[Block] = []
        for p in sorted(self.pages, key=lambda x: x.page):
            out.extend(sorted(p.blocks, key=lambda b: b.order))
        return out

    def route_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.pages:
            counts[p.route] = counts.get(p.route, 0) + 1
        return counts

    def engine_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.pages:
            counts[p.engine] = counts.get(p.engine, 0) + 1
        return counts

    @property
    def pages_needing_review(self) -> list[int]:
        return [p.page for p in self.pages if p.needs_review]


# --------------------------------------------------------------------------- #
# Chunk untuk ingest ke vector store
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """Satu unit yang akan di-embed dan disimpan di Qdrant.

    Skema ini dirancang agar payload-nya cukup untuk membangun sitasi
    "Dokumen X, halaman N, versi V" tanpa perlu query balik ke sumber.
    """

    chunk_id: str = ""
    doc_id: str = ""
    text: str = ""
    content_type: str = "text"  # text | table | mixed
    page_start: int = 0
    page_end: int = 0
    section_path: list[str] = field(default_factory=list)
    heading: str | None = None
    chunk_index: int = 0
    char_count: int = 0
    token_estimate: int = 0
    # --- metadata dokumen (didenormalisasi supaya sitasi mandiri) ---
    doc_title: str = ""
    doc_category: str = ""
    doc_version: str = ""
    doc_date: str | None = None
    source_file: str = ""
    unit: str | None = None
    url: str | None = None
    language: str = "id"
    # --- provenans teknis (untuk audit & analisis skripsi) ---
    engines: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    mean_confidence: float | None = None
    needs_review: bool = False
    table_refs: list[str] = field(default_factory=list)  # slug tabel terkait
    checksum: str = ""

    def finalize(self) -> "Chunk":
        """Hitung field turunan. Dipanggil sekali sebelum ditulis."""
        self.char_count = len(self.text)
        # Perkiraan kasar token untuk bahasa Indonesia: ~4 karakter per token.
        self.token_estimate = max(1, round(self.char_count / 4))
        self.checksum = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_id}::c{self.chunk_index:05d}::{self.checksum[:8]}"
        return self

    def citation(self) -> str:
        """String sitasi siap pakai untuk prompt/jawaban chatbot."""
        halaman = (
            f"hlm. {self.page_start}"
            if self.page_start == self.page_end
            else f"hlm. {self.page_start}-{self.page_end}"
        )
        return f"{self.doc_title or self.doc_id}, {halaman}, versi {self.doc_version}"

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
