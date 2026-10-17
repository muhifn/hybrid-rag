"""Klasifikasi halaman: menentukan apakah satu halaman perlu OCR atau tidak.

Inilah inti penanganan PDF "campuran" (sebagian digital, sebagian hasil
scan). Keputusan diambil PER HALAMAN, bukan per dokumen, karena dokumen
akademik nyata sering mencampur keduanya: panduan yang diekspor dari Word
tetapi lampiran SK-nya hasil pindaian bertanda tangan.

Rute keputusan:
  ROUTE_NATIVE : text layer ada, padat, dan tidak rusak -> ekstraksi langsung
                 (akurasi karakter 100%, nol biaya komputasi)
  ROUTE_OCR    : tidak ada text layer, terlalu tipis, atau rusak (CID/mojibake)
  ROUTE_EMPTY  : halaman benar-benar kosong (pemisah bab, halaman belakang)

Mengapa text layer bisa "rusak": PDF yang font-nya tidak tertanam dengan
encoding Identity-H tanpa CIDToGIDMap sering mengembalikan "(cid:42)" atau
U+FFFD. Teks itu tampak ada tetapi tidak dapat dibaca, dan bila lolos ke
vector store akan meracuni indeks. Karena itu halaman seperti ini DIPAKSA
OCR meskipun jumlah karakternya banyak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DetectConfig
from .models import ROUTE_EMPTY, ROUTE_NATIVE, ROUTE_OCR
from .utils import get_logger

# "(cid:123)" dari font Identity-H tanpa pemetaan yang benar.
_CID_RE = re.compile(r"\(cid:\d+\)")
# Karakter pengganti / kotak kosong / area penggunaan pribadi Unicode.
_JUNK_CHARS = re.compile(r"[�■□￾-]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{2,}")


@dataclass
class PageProbe:
    """Hasil pengukuran mentah satu halaman, sebelum keputusan rute."""

    page: int = 0
    width: float = 0.0
    height: float = 0.0
    n_chars: int = 0
    n_words: int = 0
    char_density: float = 0.0  # karakter per 1000 pt^2
    garbled_ratio: float = 0.0
    n_images: int = 0
    image_area_ratio: float = 0.0
    n_drawings: int = 0
    has_full_page_image: bool = False
    rotation: int = 0
    route: str = ROUTE_NATIVE
    reasons: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "halaman": self.page,
            "rute": self.route,
            "karakter": self.n_chars,
            "kata": self.n_words,
            "kepadatan": round(self.char_density, 3),
            "rasio_rusak": round(self.garbled_ratio, 4),
            "jml_gambar": self.n_images,
            "rasio_area_gambar": round(self.image_area_ratio, 3),
            "gambar_penuh": self.has_full_page_image,
            "rotasi": self.rotation,
            "alasan": "; ".join(self.reasons),
        }


@dataclass
class DocProbe:
    path: str = ""
    n_pages: int = 0
    encrypted: bool = False
    needs_password: bool = False
    pdf_metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[PageProbe] = field(default_factory=list)
    error: str | None = None

    @property
    def route_counts(self) -> dict[str, int]:
        counts = {ROUTE_NATIVE: 0, ROUTE_OCR: 0, ROUTE_EMPTY: 0}
        for p in self.pages:
            counts[p.route] = counts.get(p.route, 0) + 1
        return counts

    @property
    def kind(self) -> str:
        """Label ringkas untuk dokumen: digital | scan | campuran | kosong."""
        c = self.route_counts
        n_content = c[ROUTE_NATIVE] + c[ROUTE_OCR]
        if n_content == 0:
            return "kosong"
        if c[ROUTE_OCR] == 0:
            return "digital"
        if c[ROUTE_NATIVE] == 0:
            return "scan"
        return "campuran"

    def pages_by_route(self, route: str) -> list[int]:
        return [p.page for p in self.pages if p.route == route]


# --------------------------------------------------------------------------- #


def _measure_garbled(text: str) -> tuple[float, int]:
    """Kembalikan (rasio karakter rusak, jumlah karakter setelah pembersihan)."""
    if not text:
        return 0.0, 0
    n_cid_chars = sum(len(m.group(0)) for m in _CID_RE.finditer(text))
    n_junk = len(_JUNK_CHARS.findall(text))
    total = len(text)
    bad = n_cid_chars + n_junk
    return (bad / total if total else 0.0), total


def probe_page(page: Any, page_no: int, cfg: DetectConfig) -> PageProbe:
    """Ukur satu halaman PyMuPDF dan tentukan rutenya."""
    rect = page.rect
    area = max(1.0, float(rect.width) * float(rect.height))
    probe = PageProbe(
        page=page_no,
        width=round(float(rect.width), 1),
        height=round(float(rect.height), 1),
        rotation=int(getattr(page, "rotation", 0) or 0),
    )

    raw_text = page.get_text("text") or ""
    probe.garbled_ratio, _ = _measure_garbled(raw_text)
    # Hitung karakter "berguna": buang whitespace dan token rusak.
    usable = _JUNK_CHARS.sub("", _CID_RE.sub("", raw_text))
    probe.n_chars = len("".join(usable.split()))
    probe.n_words = len(_WORD_RE.findall(usable))
    probe.char_density = probe.n_chars / (area / 1000.0)

    # --- citra ---
    try:
        infos = page.get_image_info()
    except Exception:  # pragma: no cover - PDF rusak
        infos = []
    probe.n_images = len(infos)
    img_area = 0.0
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        w = abs(bbox[2] - bbox[0])
        h = abs(bbox[3] - bbox[1])
        a = w * h
        img_area += a
        if a / area >= cfg.full_page_image_ratio:
            probe.has_full_page_image = True
    probe.image_area_ratio = min(img_area / area, 1.0)

    try:
        probe.n_drawings = len(page.get_drawings())
    except Exception:  # pragma: no cover
        probe.n_drawings = 0

    # --- keputusan ---
    probe.route = _decide_route(probe, cfg)
    return probe


def _decide_route(p: PageProbe, cfg: DetectConfig) -> str:
    if cfg.force_route in (ROUTE_NATIVE, ROUTE_OCR):
        p.reasons.append(f"dipaksa oleh konfigurasi force_route={cfg.force_route}")
        return cfg.force_route

    has_visual = p.n_images > 0 or p.n_drawings > 3

    # 1) Text layer rusak -> wajib OCR walaupun karakternya banyak.
    if p.n_chars > 0 and p.garbled_ratio > cfg.max_garbled_ratio:
        p.reasons.append(
            f"text layer rusak (rasio karakter tidak terbaca "
            f"{p.garbled_ratio:.1%} > {cfg.max_garbled_ratio:.1%})"
        )
        return ROUTE_OCR

    # 2) Halaman benar-benar kosong.
    if p.n_chars < cfg.min_chars and not has_visual:
        if cfg.treat_blank_as_empty:
            p.reasons.append("tidak ada teks maupun objek visual — halaman kosong")
            return ROUTE_EMPTY
        p.reasons.append("teks sangat sedikit, tidak ada objek visual")
        return ROUTE_OCR

    # 3) Satu citra menutupi hampir seluruh halaman + teks tipis -> hasil scan.
    if p.has_full_page_image and p.n_chars < cfg.min_chars:
        p.reasons.append(
            f"citra menutupi >={cfg.full_page_image_ratio:.0%} halaman dan hanya "
            f"{p.n_chars} karakter — halaman hasil pindaian"
        )
        return ROUTE_OCR

    # 4) Teks terlalu sedikit / terlalu renggang untuk halaman yang ada isinya.
    if p.n_chars < cfg.min_chars:
        p.reasons.append(
            f"karakter ({p.n_chars}) di bawah ambang {cfg.min_chars} "
            "padahal halaman memuat objek visual"
        )
        return ROUTE_OCR
    if p.char_density < cfg.min_char_density:
        p.reasons.append(
            f"kepadatan teks {p.char_density:.3f} di bawah ambang "
            f"{cfg.min_char_density} — kemungkinan hanya kop/label pada citra"
        )
        return ROUTE_OCR

    p.reasons.append(
        f"text layer sehat ({p.n_chars} karakter, {p.n_words} kata)"
    )
    return ROUTE_NATIVE


def probe_document(
    path: str | Path, cfg: DetectConfig, pages: list[int] | None = None
) -> DocProbe:
    """Ukur seluruh (atau sebagian) halaman satu PDF."""
    import pymupdf

    path = Path(path)
    out = DocProbe(path=str(path))
    log = get_logger()

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        out.error = f"gagal membuka PDF: {exc}"
        log.error("%s: %s", path.name, out.error)
        return out

    try:
        out.n_pages = doc.page_count
        out.encrypted = bool(doc.is_encrypted)
        out.needs_password = bool(doc.needs_pass)
        if out.needs_password:
            # Coba kata sandi kosong — banyak PDF instansi "terenkripsi" tetapi
            # hanya membatasi pencetakan, bukan pembacaan.
            if doc.authenticate(""):
                out.needs_password = False
            else:
                out.error = (
                    "PDF terproteksi kata sandi; buka proteksinya terlebih dahulu "
                    "(mis. qpdf --decrypt) sebelum diproses"
                )
                return out
        out.pdf_metadata = dict(doc.metadata or {})

        targets = pages or range(1, out.n_pages + 1)
        for page_no in targets:
            if not 1 <= page_no <= out.n_pages:
                continue
            try:
                out.pages.append(probe_page(doc[page_no - 1], page_no, cfg))
            except Exception as exc:  # pragma: no cover
                log.warning("%s hlm %d: gagal diukur (%s)", path.name, page_no, exc)
                bad = PageProbe(page=page_no, route=ROUTE_OCR)
                bad.reasons.append(f"gagal diukur ({exc}) — dialihkan ke OCR")
                out.pages.append(bad)
    finally:
        doc.close()

    return out
