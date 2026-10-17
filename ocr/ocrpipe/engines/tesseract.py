"""Engine `tesseract`: OCR klasik + rekonstruksi tabel berbasis OpenCV.

Peran engine ini dalam pipeline adalah JARING PENGAMAN: ia hanya butuh satu
berkas installer, berjalan CPU-only, muat di RAM 8 GB, dan karena itu selalu
bisa dipakai saat engine neural tidak tersedia (mis. sedang offline dari
Kaggle, atau GPU dipakai proses lain).

Keterbatasan yang harus diakui secara jujur di laporan: Tesseract tidak
memiliki model struktur tabel. Tabel direkonstruksi dari garis yang
terdeteksi secara morfologis, lalu kata hasil OCR dipetakan ke sel
berdasarkan koordinat. Pendekatan ini bekerja baik untuk tabel bergaris
penuh (SK, daftar biaya) dan cukup baik untuk tabel yang hanya bergaris
horizontal (kalender akademik), tetapi akan kalah dari TableFormer/
PP-StructureV3 pada tabel tanpa garis dan sel bergabung yang rumit.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models import (
    KIND_PARAGRAPH,
    KIND_TABLE,
    ROUTE_OCR,
    Block,
    PageResult,
)
from ..preprocess import preprocess_page
from ..tables import build_table, table_to_markdown
from ..utils import clean_text, get_logger
from .base import Availability, Engine, EngineError, version_of


class TesseractEngine(Engine):
    name = "tesseract"
    description = "Tesseract OCR (CPU, ringan) + rekonstruksi tabel via OpenCV"
    is_neural = False
    does_ocr = True
    whole_document_only = False

    # ------------------------------------------------------------------ #

    def availability(self) -> Availability:
        try:
            import pytesseract
        except ImportError:
            return Availability(
                ok=False,
                message="pytesseract belum terpasang",
                install_hint="pip install pytesseract",
            )

        if self.cfg.tesseract.cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cfg.tesseract.cmd
        elif not shutil.which("tesseract"):
            # Lokasi default installer UB-Mannheim di Windows.
            win_default = Path(
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            if win_default.exists():
                pytesseract.pytesseract.tesseract_cmd = str(win_default)
            else:
                return Availability(
                    ok=False,
                    message="biner 'tesseract' tidak ditemukan di PATH",
                    install_hint=(
                        "Windows: pasang installer UB-Mannheim, lalu set "
                        "tesseract.cmd di config atau tambahkan ke PATH. "
                        "Linux: apt install tesseract-ocr tesseract-ocr-ind"
                    ),
                )

        try:
            ver = str(pytesseract.get_tesseract_version())
            langs = set(pytesseract.get_languages(config=""))
        except Exception as exc:
            return Availability(
                ok=False,
                message=f"tesseract tidak dapat dijalankan: {exc}",
                install_hint="periksa instalasi Tesseract",
            )

        wanted = [lg for lg in self.cfg.tesseract.lang.split("+") if lg]
        missing = [lg for lg in wanted if lg not in langs]
        if missing:
            return Availability(
                ok=False,
                version=ver,
                message=(
                    f"data bahasa {missing} belum ada (tersedia: "
                    f"{sorted(langs)[:8]}...)"
                ),
                install_hint=(
                    "unduh ind.traineddata dari github.com/tesseract-ocr/tessdata "
                    "ke folder tessdata, atau: apt install tesseract-ocr-ind"
                ),
            )
        return Availability(ok=True, version=f"{ver} (pytesseract {version_of('pytesseract')})")

    # ------------------------------------------------------------------ #

    def extract(self, pdf_path: str | Path, pages: list[int]) -> dict[int, PageResult]:
        import pytesseract

        avail = self.availability()
        if not avail.ok:
            raise EngineError(f"tesseract tidak siap: {avail.message}")
        if self.cfg.tesseract.cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cfg.tesseract.cmd

        log = get_logger()
        tcfg = self.cfg.tesseract
        config = (
            f"--oem {tcfg.oem} --psm {tcfg.psm} {tcfg.extra_config}"
        ).strip()

        out: dict[int, PageResult] = {}
        for page_no in pages:
            t0 = time.perf_counter()
            pr = PageResult(page=page_no, route=ROUTE_OCR, engine=self.name)
            try:
                prep = preprocess_page(
                    pdf_path, page_no, self.cfg.preprocess, for_neural_engine=False
                )
                pr.warnings.extend(prep.warnings)
                pr.metrics["praproses"] = ", ".join(prep.applied)
                pr.metrics["dpi"] = prep.dpi
                pr.metrics["sudut_deskew"] = prep.deskew_angle

                words = self._ocr_words(
                    pytesseract, prep.image, tcfg.lang, config
                )
                if not words:
                    pr.warnings.append("OCR tidak menghasilkan kata apa pun")
                    pr.blocks = []
                else:
                    pr.blocks = self._build_blocks(words, prep.image, page_no, pr)
            except Exception as exc:
                log.warning("hlm %d gagal di-OCR (tesseract): %s", page_no, exc)
                pr.warnings.append(f"OCR gagal: {exc}")
            pr.elapsed_s = round(time.perf_counter() - t0, 4)
            out[page_no] = pr
        return out

    # ------------------------------------------------------------------ #
    # OCR level kata
    # ------------------------------------------------------------------ #

    def _ocr_words(
        self, pytesseract: Any, img: np.ndarray, lang: str, config: str
    ) -> list[dict[str, Any]]:
        """Jalankan OCR satu kali dan kembalikan kata + bbox + confidence.

        Satu panggilan image_to_data jauh lebih murah daripada meng-OCR tiap
        sel tabel secara terpisah; sel diisi dengan memetakan kata ke grid.
        """
        from pytesseract import Output

        data = pytesseract.image_to_data(
            img, lang=lang, config=config, output_type=Output.DICT
        )
        n = len(data.get("text", []))
        min_conf = self.cfg.tesseract.min_word_conf
        words: list[dict[str, Any]] = []
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf >= 0 and conf < min_conf:
                continue
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            words.append(
                {
                    "text": text,
                    "conf": conf,
                    "bbox": (x, y, x + w, y + h),
                    "cx": x + w / 2.0,
                    "cy": y + h / 2.0,
                    "block": int(data.get("block_num", [0] * n)[i]),
                    "par": int(data.get("par_num", [0] * n)[i]),
                    "line": int(data.get("line_num", [0] * n)[i]),
                }
            )
        return words

    # ------------------------------------------------------------------ #
    # Perakitan blok
    # ------------------------------------------------------------------ #

    def _build_blocks(
        self,
        words: list[dict[str, Any]],
        img: np.ndarray,
        page_no: int,
        pr: PageResult,
    ) -> list[Block]:
        regions: list[tuple[int, int, int, int]] = []
        if self.cfg.tesseract.detect_tables:
            try:
                regions = detect_table_regions(img, self.cfg.tesseract)
            except Exception as exc:  # pragma: no cover
                pr.warnings.append(f"deteksi tabel gagal: {exc}")
                regions = []

        used: set[int] = set()
        items: list[tuple[tuple[float, float, float, float], Block]] = []

        # --- tabel ---
        for t_idx, region in enumerate(regions, start=1):
            inside = [
                i
                for i, w in enumerate(words)
                if _point_in(w["cx"], w["cy"], region)
            ]
            if len(inside) < 4:
                continue

            # Gerbang mutu: tabel pada scan buruk sering menghasilkan sel
            # sampah. Bila confidence rata-rata kata di region tabel rendah,
            # JANGAN sajikan sebagai tabel terstruktur — biarkan kata-katanya
            # jatuh ke alur teks (tidak di-mark used) dan tandai halaman.
            region_confs = [words[i]["conf"] for i in inside if words[i]["conf"] >= 0]
            mean_conf = sum(region_confs) / len(region_confs) if region_confs else 0.0
            if region_confs and mean_conf < self.cfg.tesseract.min_table_conf:
                pr.warnings.append(
                    f"tabel pada region terdeteksi tetapi confidence OCR rendah "
                    f"({mean_conf:.1f} < {self.cfg.tesseract.min_table_conf}); "
                    "tidak disajikan sebagai struktur — OCR ulang halaman ini "
                    "dengan engine neural (docling/paddle) untuk tabelnya"
                )
                continue

            try:
                rows = reconstruct_table(
                    img, region, [words[i] for i in inside], self.cfg.tesseract
                )
            except Exception as exc:  # pragma: no cover
                pr.warnings.append(f"rekonstruksi tabel gagal: {exc}")
                continue
            table = build_table(
                rows,
                page=page_no,
                index=t_idx,
                engine=self.name,
                bbox=tuple(float(v) for v in region),
            )
            if table.is_degenerate():
                continue
            md = table_to_markdown(table)
            if not md:
                continue
            confs = region_confs
            used.update(inside)
            items.append(
                (
                    tuple(float(v) for v in region),
                    Block(
                        kind=KIND_TABLE,
                        text=md,
                        page=page_no,
                        bbox=tuple(float(v) for v in region),
                        table=table,
                        engine=self.name,
                        confidence=round(sum(confs) / len(confs), 2) if confs else None,
                    ),
                )
            )

        # --- paragraf (kata yang belum terpakai, dikelompokkan oleh Tesseract) ---
        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for i, w in enumerate(words):
            if i in used:
                continue
            groups.setdefault((w["block"], w["par"]), []).append(w)

        for key in sorted(groups):
            group = groups[key]
            lines: dict[int, list[dict[str, Any]]] = {}
            for w in group:
                lines.setdefault(w["line"], []).append(w)
            text_lines = [
                " ".join(
                    w["text"] for w in sorted(lines[ln], key=lambda z: z["bbox"][0])
                )
                for ln in sorted(lines)
            ]
            text = clean_text("\n".join(text_lines))
            if not text:
                continue
            xs0 = min(w["bbox"][0] for w in group)
            ys0 = min(w["bbox"][1] for w in group)
            xs1 = max(w["bbox"][2] for w in group)
            ys1 = max(w["bbox"][3] for w in group)
            confs = [w["conf"] for w in group if w["conf"] >= 0]
            items.append(
                (
                    (float(xs0), float(ys0), float(xs1), float(ys1)),
                    Block(
                        kind=KIND_PARAGRAPH,
                        text=text,
                        page=page_no,
                        bbox=(float(xs0), float(ys0), float(xs1), float(ys1)),
                        engine=self.name,
                        confidence=round(sum(confs) / len(confs), 2) if confs else None,
                    ),
                )
            )

        ordered = [
            b
            for _, b in sorted(
                items, key=lambda it: (round(it[0][1] / 12.0), round(it[0][0], 1))
            )
        ]
        for i, block in enumerate(ordered):
            block.order = i
        t_i = 0
        for block in ordered:
            if block.table is not None:
                t_i += 1
                block.table.index = t_i
        return ordered


# --------------------------------------------------------------------------- #
# Deteksi & rekonstruksi tabel (OpenCV)
# --------------------------------------------------------------------------- #


def _to_gray(img: np.ndarray) -> np.ndarray:
    import cv2

    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _line_masks(img: np.ndarray, tcfg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan (mask_garis_horizontal, mask_garis_vertikal)."""
    import cv2

    gray = _to_gray(img)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    h, w = binary.shape[:2]

    h_len = max(25, int(w * max(0.05, tcfg.table_line_frac * 0.6)))
    v_len = max(18, int(h * 0.02))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    return h_lines, v_lines


def detect_table_regions(img: np.ndarray, tcfg: Any) -> list[tuple[int, int, int, int]]:
    """Cari kotak pembatas area tabel berdasarkan kerapatan garis.

    Menerima tabel yang hanya bergaris horizontal (umum pada kalender
    akademik) selama jumlah garis horizontalnya memenuhi ambang.
    """
    import cv2

    h_lines, v_lines = _line_masks(img, tcfg)
    H, W = h_lines.shape[:2]

    combined = cv2.bitwise_or(h_lines, v_lines)
    # Sambungkan garis yang terputus agar satu tabel menjadi satu kontur.
    glue = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    combined = cv2.dilate(combined, glue, iterations=2)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    regions: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < W * 0.18 or h < H * 0.02:
            continue
        if w * h < (W * H) * 0.004:
            continue
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

        n_h = len(_line_positions(h_lines[y0:y1, x0:x1], axis=1, min_frac=0.45))
        # Tabel dengan hanya garis horizontal (mis. kalender akademik) tetap
        # sah, jadi region hanya diseleksi berdasarkan jumlah garis horizontal.
        if n_h < tcfg.table_min_hlines:
            continue
        regions.append((x0, y0, x1, y1))

    # Buang region yang bersarang di region lain.
    regions.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for r in regions:
        if not any(_contains(k, r) for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: (r[1], r[0]))
    return kept


def _line_positions(
    mask: np.ndarray, axis: int, min_frac: float = 0.45, merge_gap: int = 6
) -> list[int]:
    """Posisi garis sepanjang sumbu tertentu.

    axis=1 -> garis horizontal (kembalikan koordinat y)
    axis=0 -> garis vertikal  (kembalikan koordinat x)
    """
    if mask.size == 0:
        return []
    proj = (mask > 0).sum(axis=axis)
    extent = mask.shape[1] if axis == 1 else mask.shape[0]
    if extent == 0:
        return []
    threshold = max(8.0, extent * min_frac)
    hits = np.flatnonzero(proj >= threshold)
    if hits.size == 0:
        return []
    # Gabungkan piksel berdekatan menjadi satu garis.
    groups: list[list[int]] = [[int(hits[0])]]
    for p in hits[1:]:
        if int(p) - groups[-1][-1] <= merge_gap:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    return [int(round(sum(g) / len(g))) for g in groups]


def _cluster_columns(
    words: Sequence[dict[str, Any]], region: tuple[int, int, int, int]
) -> list[int]:
    """Tebak batas kolom dari celah horizontal antar kata (tabel tanpa garis vertikal).

    Membuat profil okupansi horizontal; celah lebar yang konsisten di seluruh
    tinggi region ditafsirkan sebagai pemisah kolom.
    """
    x0, _, x1, _ = region
    width = max(1, x1 - x0)
    occupancy = np.zeros(width, dtype=np.int32)
    for w in words:
        a = max(0, int(w["bbox"][0]) - x0)
        b = min(width, int(w["bbox"][2]) - x0)
        if b > a:
            occupancy[a:b] += 1

    empty = occupancy == 0
    min_gap = max(8, int(width * 0.015))
    boundaries: list[int] = []
    run_start: int | None = None
    for i, is_empty in enumerate(empty):
        if is_empty and run_start is None:
            run_start = i
        elif not is_empty and run_start is not None:
            if i - run_start >= min_gap:
                boundaries.append(x0 + (run_start + i) // 2)
            run_start = None
    return [x0, *boundaries, x1]


def reconstruct_table(
    img: np.ndarray,
    region: tuple[int, int, int, int],
    words: Sequence[dict[str, Any]],
    tcfg: Any,
) -> list[list[str]]:
    """Rakit grid tabel dari garis terdeteksi + kata hasil OCR."""
    x0, y0, x1, y1 = region
    h_lines, v_lines = _line_masks(img, tcfg)
    sub_h = h_lines[y0:y1, x0:x1]
    sub_v = v_lines[y0:y1, x0:x1]

    row_lines = [y0 + p for p in _line_positions(sub_h, axis=1, min_frac=0.40)]
    col_lines = [x0 + p for p in _line_positions(sub_v, axis=0, min_frac=0.35)]

    # Batas baris: pakai garis bila cukup, kalau tidak kelompokkan kata per baris.
    if len(row_lines) >= 2:
        row_bounds = _dedupe_sorted([y0, *row_lines, y1])
    else:
        row_bounds = _rows_from_words(words, y0, y1)

    # Batas kolom: pakai garis vertikal bila cukup, kalau tidak pakai celah.
    if len(col_lines) >= tcfg.table_min_vlines:
        col_bounds = _dedupe_sorted([x0, *col_lines, x1])
    else:
        col_bounds = _dedupe_sorted(_cluster_columns(words, region))

    if len(row_bounds) < 2 or len(col_bounds) < 2:
        return []

    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    if n_rows * n_cols > 4000:  # pagar pengaman terhadap deteksi garis liar
        return []

    cells: list[list[list[dict[str, Any]]]] = [
        [[] for _ in range(n_cols)] for _ in range(n_rows)
    ]
    for w in words:
        r = _bucket(w["cy"], row_bounds)
        c = _bucket(w["cx"], col_bounds)
        if r is None or c is None:
            continue
        cells[r][c].append(w)

    rows: list[list[str]] = []
    for r in range(n_rows):
        row: list[str] = []
        for c in range(n_cols):
            group = sorted(cells[r][c], key=lambda z: (z["bbox"][1], z["bbox"][0]))
            row.append(" ".join(w["text"] for w in group).strip())
        rows.append(row)
    return rows


def _rows_from_words(
    words: Sequence[dict[str, Any]], y0: int, y1: int
) -> list[int]:
    """Bentuk batas baris dari pengelompokan vertikal kata (tanpa garis)."""
    if not words:
        return [y0, y1]
    heights = [w["bbox"][3] - w["bbox"][1] for w in words]
    med_h = float(np.median(heights)) if heights else 12.0
    tol = max(4.0, med_h * 0.6)

    centers = sorted(w["cy"] for w in words)
    clusters: list[list[float]] = [[centers[0]]]
    for cy in centers[1:]:
        if cy - clusters[-1][-1] <= tol:
            clusters[-1].append(cy)
        else:
            clusters.append([cy])
    mids = [sum(c) / len(c) for c in clusters]
    bounds = [float(y0)]
    for a, b in zip(mids, mids[1:]):
        bounds.append((a + b) / 2.0)
    bounds.append(float(y1))
    return _dedupe_sorted([int(round(v)) for v in bounds])


def _dedupe_sorted(values: Sequence[int], min_delta: int = 5) -> list[int]:
    out: list[int] = []
    for v in sorted(int(x) for x in values):
        if not out or v - out[-1] >= min_delta:
            out.append(v)
    return out


def _bucket(value: float, bounds: Sequence[int]) -> int | None:
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return i
    if bounds and value == bounds[-1] and len(bounds) >= 2:
        return len(bounds) - 2
    return None


def _point_in(x: float, y: float, region: tuple[int, int, int, int]) -> bool:
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )
