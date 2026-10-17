"""Engine `native`: ekstraksi halaman yang SUDAH punya text layer.

Tidak ada OCR di sini. Untuk halaman digital, membaca text layer memberi
akurasi karakter 100% dengan biaya komputasi nol — jauh lebih baik daripada
merasterisasi lalu meng-OCR halaman yang teksnya sudah tersedia. Inilah
sebabnya pipeline melakukan rute per halaman: pada panduan akademik yang
diekspor dari Word, 90% halaman ditangani di sini dan hanya lampiran
bertanda tangan yang dikirim ke engine OCR yang berat.

Implementasi memakai PyMuPDF untuk TEKS maupun TABEL. Memakai satu pustaka
untuk keduanya menghindari masalah ketidaksesuaian sistem koordinat yang
muncul bila bbox tabel berasal dari pdfplumber sementara bbox teks berasal
dari PyMuPDF (keduanya memakai origin kiri-atas, tetapi penanganan halaman
yang dirotasi berbeda). pdfplumber tetap dipakai sebagai cadangan untuk
tabel tanpa garis.
"""

from __future__ import annotations

import re
import statistics
import time
from pathlib import Path
from typing import Any

from ..models import (
    KIND_CAPTION,
    KIND_FOOTER,
    KIND_HEADER,
    KIND_HEADING,
    KIND_LIST,
    KIND_PARAGRAPH,
    KIND_TABLE,
    ROUTE_NATIVE,
    Block,
    PageResult,
)
from ..tables import build_table, table_to_markdown
from ..utils import clean_text, collapse_to_single_line, get_logger
from .base import Availability, Engine, EngineError, module_installed, version_of

# Pola heading khas dokumen akademik Indonesia.
_HEADING_PATTERNS = (
    re.compile(r"^BAB\s+[IVXLC]+\b", re.IGNORECASE),
    re.compile(r"^(?:LAMPIRAN|DAFTAR\s+\w+|KATA\s+PENGANTAR|PENDAHULUAN)\b", re.IGNORECASE),
    re.compile(r"^\d+(?:\.\d+){0,3}\.?\s+\S"),  # 2.1.3 Judul
    re.compile(r"^[A-Z]\.\s+\S"),  # A. Judul
    re.compile(r"^(?:Pasal|BAGIAN|Bagian)\s+\d+", re.IGNORECASE),
)
_LIST_PATTERN = re.compile(r"^\s*(?:[-•▪o*]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+")
_CAPTION_PATTERN = re.compile(r"^(Tabel|Gambar|Grafik|Lampiran)\s*\.?\s*[\d.IVXLC]+", re.IGNORECASE)

_BOLD_FLAG = 1 << 4  # bit 4 pada span["flags"] PyMuPDF = bold


class NativeEngine(Engine):
    name = "native"
    description = "Ekstraksi text layer + tabel via PyMuPDF (tanpa OCR, akurasi penuh)"
    is_neural = False
    does_ocr = False
    whole_document_only = False

    # ------------------------------------------------------------------ #

    def availability(self) -> Availability:
        if not module_installed("pymupdf"):
            return Availability(
                ok=False,
                message="PyMuPDF belum terpasang",
                install_hint="pip install pymupdf",
            )
        return Availability(ok=True, version=version_of("pymupdf"))

    # ------------------------------------------------------------------ #

    def extract(self, pdf_path: str | Path, pages: list[int]) -> dict[int, PageResult]:
        import pymupdf

        log = get_logger()
        out: dict[int, PageResult] = {}

        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as exc:
            raise EngineError(f"gagal membuka {pdf_path}: {exc}") from exc

        try:
            if doc.needs_pass and not doc.authenticate(""):
                raise EngineError("PDF terproteksi kata sandi")

            body_size, heading_sizes = self._font_statistics(doc, pages)
            log.debug(
                "ukuran font badan teks (median berbobot): %.2f pt; "
                "tingkat heading terdeteksi: %s",
                body_size,
                [round(s, 1) for s in heading_sizes],
            )

            for page_no in pages:
                t0 = time.perf_counter()
                pr = PageResult(page=page_no, route=ROUTE_NATIVE, engine=self.name)
                try:
                    page = doc[page_no - 1]
                    tables = self._find_tables(page, pr)
                    blocks = self._page_blocks(
                        page, page_no, tables, body_size, heading_sizes
                    )
                    pr.blocks = blocks
                except Exception as exc:  # pragma: no cover
                    log.warning("hlm %d gagal diekstrak native: %s", page_no, exc)
                    pr.warnings.append(f"ekstraksi native gagal: {exc}")
                pr.elapsed_s = round(time.perf_counter() - t0, 4)
                out[page_no] = pr
        finally:
            doc.close()

        return out

    # ------------------------------------------------------------------ #
    # Statistik font untuk deteksi heading
    # ------------------------------------------------------------------ #

    def _font_statistics(
        self, doc: Any, pages: list[int]
    ) -> tuple[float, list[float]]:
        """Hitung ukuran font badan teks dan daftar ukuran heading.

        Ukuran badan teks diambil sebagai median BERBOBOT jumlah karakter —
        bukan median sederhana — karena halaman judul dengan sedikit teks
        berukuran besar akan menggeser median sederhana ke atas dan membuat
        seluruh dokumen gagal terdeteksi heading-nya.
        """
        sizes: list[float] = []
        # Batasi sampel agar dokumen 500 halaman tidak lambat.
        sample = pages[:60] if len(pages) > 60 else pages
        for page_no in sample:
            try:
                data = doc[page_no - 1].get_text("dict")
            except Exception:  # pragma: no cover
                continue
            for blk in data.get("blocks", []):
                if blk.get("type") != 0:
                    continue
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        size = round(float(span.get("size", 0.0)), 1)
                        if txt and size > 0:
                            sizes.extend([size] * len(txt))

        if not sizes:
            return 10.0, []
        body = float(statistics.median(sizes))

        ratio = self.cfg.native.heading_size_ratio
        distinct = sorted({s for s in sizes if s >= body * ratio}, reverse=True)
        # Maksimal 6 tingkat heading (h1..h6).
        return body, distinct[:6]

    # ------------------------------------------------------------------ #
    # Tabel
    # ------------------------------------------------------------------ #

    def _find_tables(self, page: Any, pr: PageResult) -> list[dict[str, Any]]:
        """Cari tabel pada satu halaman; kembalikan daftar {bbox, rows}."""
        strategy = self.cfg.native.table_strategy
        if strategy == "none":
            return []

        attempts: list[str]
        if strategy == "auto":
            attempts = ["lines_strict", "lines"]
        else:
            attempts = [strategy]

        found: list[dict[str, Any]] = []
        for strat in attempts:
            found = self._try_find_tables(page, strat, pr)
            if found:
                break
        return found

    def _try_find_tables(
        self, page: Any, strategy: str, pr: PageResult
    ) -> list[dict[str, Any]]:
        try:
            finder = page.find_tables(strategy=strategy)
        except TypeError:
            # Versi PyMuPDF lama tidak menerima argumen `strategy`.
            try:
                finder = page.find_tables()
            except Exception as exc:  # pragma: no cover
                pr.warnings.append(f"pencarian tabel gagal: {exc}")
                return []
        except Exception as exc:  # pragma: no cover
            pr.warnings.append(f"pencarian tabel ({strategy}) gagal: {exc}")
            return []

        out: list[dict[str, Any]] = []
        for tbl in getattr(finder, "tables", []):
            try:
                rows = tbl.extract()
            except Exception as exc:  # pragma: no cover
                pr.warnings.append(f"ekstraksi isi tabel gagal: {exc}")
                continue
            n_cells = sum(
                1 for row in rows for cell in row if cell and str(cell).strip()
            )
            if n_cells < self.cfg.native.min_table_cells:
                continue
            out.append(
                {
                    "bbox": tuple(float(v) for v in tbl.bbox),
                    "rows": rows,
                    "strategy": strategy,
                }
            )
        return out

    # ------------------------------------------------------------------ #
    # Blok halaman
    # ------------------------------------------------------------------ #

    def _page_blocks(
        self,
        page: Any,
        page_no: int,
        tables: list[dict[str, Any]],
        body_size: float,
        heading_sizes: list[float],
    ) -> list[Block]:
        ncfg = self.cfg.native
        rect = page.rect
        page_h = float(rect.height) or 1.0
        page_w = float(rect.width) or 1.0

        table_bboxes = [t["bbox"] for t in tables]
        items: list[tuple[tuple[float, float, float, float], Block]] = []

        # --- blok teks ---
        data = page.get_text("dict")
        for blk in data.get("blocks", []):
            if blk.get("type") != 0:  # 1 = gambar
                continue
            bbox = tuple(float(v) for v in blk.get("bbox", (0, 0, 0, 0)))
            if _overlaps_any(bbox, table_bboxes, 0.5):
                continue  # isinya sudah tercakup oleh tabel

            text, max_size, bold_ratio = _block_text(blk, join=ncfg.join_lines)
            if not text.strip():
                continue

            kind, level = self._classify(
                text, max_size, bold_ratio, body_size, heading_sizes, bbox, page_h
            )
            items.append(
                (
                    bbox,
                    Block(
                        kind=kind,
                        text=text,
                        page=page_no,
                        level=level,
                        bbox=bbox,
                        engine=self.name,
                    ),
                )
            )

        # --- blok tabel ---
        consumed_captions: set[int] = set()
        for t_idx, tbl in enumerate(tables, start=1):
            bbox = tbl["bbox"]
            caption, cap_id = _nearest_caption(bbox, items)
            if cap_id is not None:
                # Caption yang dipakai sebagai judul tabel tidak boleh juga
                # dirender sebagai blok terpisah (menghindari duplikasi).
                consumed_captions.add(cap_id)
            table = build_table(
                tbl["rows"],
                page=page_no,
                index=t_idx,
                engine=f"{self.name}:{tbl['strategy']}",
                caption=caption,
                bbox=bbox,
            )
            if table.is_degenerate():
                continue
            md = table_to_markdown(table)
            if not md:
                continue
            items.append(
                (
                    bbox,
                    Block(
                        kind=KIND_TABLE,
                        text=md,
                        page=page_no,
                        bbox=bbox,
                        table=table,
                        engine=self.name,
                    ),
                )
            )

        if consumed_captions:
            items = [
                it for it in items if id(it[1]) not in consumed_captions
            ]

        ordered = _reading_order(items, page_w)
        for i, block in enumerate(ordered):
            block.order = i
        # Nomorkan ulang indeks tabel sesuai urutan baca final.
        t_i = 0
        for block in ordered:
            if block.table is not None:
                t_i += 1
                block.table.index = t_i
        return ordered

    # ------------------------------------------------------------------ #

    def _classify(
        self,
        text: str,
        max_size: float,
        bold_ratio: float,
        body_size: float,
        heading_sizes: list[float],
        bbox: tuple[float, float, float, float],
        page_h: float,
    ) -> tuple[str, int | None]:
        ncfg = self.cfg.native
        first_line = collapse_to_single_line(text.split("\n", 1)[0])

        # Kop surat / footer berdasarkan posisi vertikal.
        if ncfg.mark_header_footer:
            if bbox[3] <= page_h * ncfg.header_zone and len(text) < 200:
                return KIND_HEADER, None
            if bbox[1] >= page_h * (1 - ncfg.footer_zone) and len(text) < 200:
                return KIND_FOOTER, None

        if _CAPTION_PATTERN.match(first_line) and len(text) < 300:
            return KIND_CAPTION, None

        if ncfg.detect_headings and len(text) <= ncfg.heading_max_chars:
            by_pattern = any(p.match(first_line) for p in _HEADING_PATTERNS)
            by_size = max_size >= body_size * ncfg.heading_size_ratio
            by_bold = bold_ratio > 0.85 and len(text) < 90 and not text.rstrip().endswith(".")
            if by_size or by_pattern or by_bold:
                level = _heading_level(max_size, heading_sizes, body_size, by_pattern)
                return KIND_HEADING, level

        if _LIST_PATTERN.match(text):
            return KIND_LIST, None
        return KIND_PARAGRAPH, None


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def _block_text(blk: dict[str, Any], join: bool) -> tuple[str, float, float]:
    """Gabungkan span menjadi teks; kembalikan (teks, ukuran_maks, rasio_bold)."""
    lines: list[str] = []
    max_size = 0.0
    bold_chars = 0
    total_chars = 0
    for line in blk.get("lines", []):
        parts: list[str] = []
        for span in line.get("spans", []):
            txt = span.get("text", "")
            if not txt:
                continue
            parts.append(txt)
            size = float(span.get("size", 0.0))
            max_size = max(max_size, size)
            n = len(txt.strip())
            total_chars += n
            if int(span.get("flags", 0)) & _BOLD_FLAG:
                bold_chars += n
        if parts:
            lines.append("".join(parts))

    raw = "\n".join(lines)
    text = clean_text(raw, join_hyphenated=True)
    if join and text:
        # Satukan hard wrap menjadi paragraf, kecuali baris yang jelas item daftar.
        merged: list[str] = []
        for ln in text.split("\n"):
            if merged and not _LIST_PATTERN.match(ln) and not merged[-1].endswith(("-",)):
                merged[-1] = f"{merged[-1]} {ln}".strip()
            else:
                merged.append(ln)
        text = "\n".join(merged)

    bold_ratio = bold_chars / total_chars if total_chars else 0.0
    return text, max_size, bold_ratio


def _heading_level(
    size: float, heading_sizes: list[float], body_size: float, by_pattern: bool
) -> int:
    for i, s in enumerate(heading_sizes, start=1):
        if size >= s - 0.05:
            return min(i, 6)
    # Heading yang dikenali hanya dari pola penomoran (ukuran font sama dengan
    # badan teks) diberi tingkat menengah.
    return 3 if by_pattern else 4


def _area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _overlaps_any(
    bbox: tuple[float, float, float, float],
    others: list[tuple[float, float, float, float]],
    threshold: float,
) -> bool:
    a = _area(bbox)
    if a <= 0:
        return False
    return any(_intersection_area(bbox, o) / a >= threshold for o in others)


def _nearest_caption(
    table_bbox: tuple[float, float, float, float],
    items: list[tuple[tuple[float, float, float, float], Block]],
) -> tuple[str | None, int | None]:
    """Cari blok keterangan ("Tabel 3.1 ...") tepat di atas/bawah tabel.

    Returns:
        (teks_caption, id_blok). id_blok dipakai pemanggil untuk membuang
        blok caption agar tidak dirender ganda. Keduanya None bila tak ada.
    """
    best: tuple[float, str, int] | None = None
    for bbox, block in items:
        if block.kind != KIND_CAPTION:
            continue
        # Harus tumpang tindih secara horizontal dengan tabel.
        if bbox[2] < table_bbox[0] or bbox[0] > table_bbox[2]:
            continue
        gap_above = table_bbox[1] - bbox[3]
        gap_below = bbox[1] - table_bbox[3]
        gap = gap_above if gap_above >= 0 else (gap_below if gap_below >= 0 else -1)
        if 0 <= gap <= 60 and (best is None or gap < best[0]):
            best = (gap, collapse_to_single_line(block.text), id(block))
    if best is None:
        return None, None
    return best[1], best[2]


def _reading_order(
    items: list[tuple[tuple[float, float, float, float], Block]], page_w: float
) -> list[Block]:
    """Urutkan blok sesuai urutan baca, dengan deteksi dua kolom sederhana.

    Dokumen akademik Indonesia umumnya satu kolom, tetapi jurnal dan beberapa
    panduan memakai dua kolom. Mengurutkan dua kolom murni berdasarkan y akan
    menghasilkan teks yang berganti-ganti kolom dan merusak chunk.
    """
    if not items:
        return []

    mid = page_w / 2.0
    left = [it for it in items if it[0][2] <= mid * 1.05]
    right = [it for it in items if it[0][0] >= mid * 0.95]
    spanning = [
        it for it in items if it[0][0] < mid * 0.95 and it[0][2] > mid * 1.05
    ]

    two_column = (
        len(left) >= 3
        and len(right) >= 3
        and len(spanning) <= max(1, len(items) * 0.15)
    )

    def by_pos(
        seq: list[tuple[tuple[float, float, float, float], Block]]
    ) -> list[tuple[tuple[float, float, float, float], Block]]:
        return sorted(seq, key=lambda it: (round(it[0][1], 1), round(it[0][0], 1)))

    if two_column:
        # Blok yang membentang lebar (judul, tabel lebar) diletakkan sesuai y
        # relatif terhadap kolom: paling aman ditaruh di depan.
        return [b for _, b in by_pos(spanning) + by_pos(left) + by_pos(right)]
    return [b for _, b in by_pos(items)]
