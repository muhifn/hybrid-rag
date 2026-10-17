"""Utilitas umum: logging, slug, ID dokumen, subset PDF, teks."""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")

LOGGER_NAME = "ocrpipe"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def setup_logging(verbose: bool = True, logfile: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        p = Path(logfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
        )
        logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# --------------------------------------------------------------------------- #
# Nama & identitas
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 60) -> str:
    """Ubah teks apa pun menjadi slug aman untuk nama berkas lintas-OS."""
    norm = unicodedata.normalize("NFKD", text)
    norm = norm.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_RE.sub("-", norm).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "dokumen"


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def make_doc_id(path: str | Path) -> str:
    """ID dokumen = slug nama berkas + 8 hex pertama sha256 isi berkas.

    Memakai hash isi (bukan hanya nama) berarti dokumen yang diperbarui
    menghasilkan doc_id baru, sehingga versi lama dan baru dapat hidup
    berdampingan di vector store — prasyarat untuk sitasi berbasis versi.
    """
    p = Path(path)
    return f"{slugify(p.stem)}-{file_sha256(p)[:8]}"


# --------------------------------------------------------------------------- #
# Teks
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"[ \t   ]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
# Tanda hubung di akhir baris pada teks Indonesia hasil PDF berkolom.
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str, join_hyphenated: bool = True) -> str:
    """Rapikan whitespace tanpa mengubah isi kata."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")  # soft hyphen
    if join_hyphenated:
        text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return _MULTI_NL_RE.sub("\n\n", text).strip()


def normalize_cell(value: object) -> str:
    """Normalisasi satu sel tabel menjadi string satu-atau-banyak-baris bersih."""
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(_WS_RE.sub(" ", ln).strip() for ln in s.split("\n"))
    s = _MULTI_NL_RE.sub("\n", s)
    return s.strip()


def collapse_to_single_line(value: str) -> str:
    return _WS_RE.sub(" ", value.replace("\n", " ")).strip()


def chunked(seq: Sequence[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


# --------------------------------------------------------------------------- #
# Rentang halaman
# --------------------------------------------------------------------------- #


def parse_page_spec(spec: str | None, n_pages: int) -> list[int] | None:
    """Ubah "1,3,5-8" menjadi [1,3,5,6,7,8] (1-based, sudah difilter & urut).

    Mengembalikan None jika spec kosong (artinya: semua halaman).
    """
    if not spec or not spec.strip():
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                start, end = int(a), int(b)
            except ValueError as exc:
                raise ValueError(f"Rentang halaman tidak valid: '{part}'") from exc
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            try:
                pages.add(int(part))
            except ValueError as exc:
                raise ValueError(f"Nomor halaman tidak valid: '{part}'") from exc
    valid = sorted(p for p in pages if 1 <= p <= n_pages)
    if not valid:
        raise ValueError(
            f"Spesifikasi halaman '{spec}' tidak memilih halaman apa pun "
            f"(dokumen punya {n_pages} halaman)"
        )
    return valid


def make_subset_pdf(
    src: str | Path, pages: Iterable[int], dest: str | Path
) -> dict[int, int]:
    """Buat PDF berisi hanya `pages` (1-based) dari `src`.

    Dipakai untuk engine yang hanya menerima dokumen utuh (Docling, Marker)
    ketika pipeline hanya ingin mengirim sebagian halaman ke OCR.

    Returns:
        Pemetaan {nomor_halaman_di_subset (1-based): nomor_halaman_asli}.
    """
    from pypdf import PdfReader, PdfWriter

    pages = sorted(set(pages))
    reader = PdfReader(str(src))
    writer = PdfWriter()
    mapping: dict[int, int] = {}
    for new_idx, original in enumerate(pages, start=1):
        writer.add_page(reader.pages[original - 1])
        mapping[new_idx] = original
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        writer.write(fh)
    return mapping


def human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"
