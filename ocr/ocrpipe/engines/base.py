"""Kontrak dasar untuk semua engine ekstraksi.

Setiap engine menerima satu PDF + daftar halaman (1-based) dan mengembalikan
`dict[int, PageResult]`. Engine TIDAK bertanggung jawab atas chunking, QA,
maupun penulisan berkas — hanya mengubah halaman menjadi daftar Block.

Semua engine wajib:
  * mengisi `Block.page` dengan nomor halaman ASLI (bukan indeks subset);
  * mengisi `Block.order` sesuai urutan baca dalam halaman;
  * melaporkan ketersediaan dependensi lewat `availability()` TANPA
    mengimpor pustaka berat di tingkat modul (impor berat dilakukan malas
    di dalam metode) supaya `ocrpipe engines` tetap cepat dan tidak crash
    saat pustaka opsional belum terpasang.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import PageResult

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config


@dataclass
class Availability:
    ok: bool
    message: str = ""
    version: str | None = None
    install_hint: str = ""

    def __bool__(self) -> bool:
        return self.ok


class EngineError(RuntimeError):
    """Kegagalan engine yang masih memungkinkan fallback ke engine lain."""


class Engine(ABC):
    #: Nama unik untuk dipakai di CLI dan konfigurasi.
    name: str = "base"
    #: Deskripsi singkat untuk perintah `ocrpipe engines`.
    description: str = ""
    #: True jika engine ini berbasis model neural dan karenanya lebih baik
    #: menerima citra grayscale asli (tanpa binarisasi/denoise agresif).
    is_neural: bool = False
    #: True jika engine melakukan OCR (butuh rasterisasi).
    does_ocr: bool = True
    #: True jika engine hanya bisa menerima dokumen utuh, bukan per halaman.
    #: Pipeline akan membuat PDF subset untuk engine seperti ini.
    whole_document_only: bool = False

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ #

    @abstractmethod
    def availability(self) -> Availability:
        """Periksa apakah engine dapat dijalankan di mesin ini."""

    @abstractmethod
    def extract(
        self, pdf_path: str | Path, pages: list[int]
    ) -> dict[int, PageResult]:
        """Ekstrak halaman `pages` (1-based) dari `pdf_path`."""

    # ------------------------------------------------------------------ #

    def warmup(self) -> None:
        """Muat model lebih awal (opsional).

        Dipanggil sekali sebelum perulangan batch supaya waktu muat model
        tidak tercatat sebagai waktu pemrosesan halaman pertama — penting
        agar pengukuran latensi di skripsi tidak bias.
        """

    def close(self) -> None:
        """Bebaskan sumber daya (model GPU, server inferensi)."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Engine {self.name}>"


def version_of(module_name: str) -> str | None:
    """Ambil versi paket terpasang tanpa mengimpor modulnya."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(module_name)
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover
        return None


def module_installed(module_name: str) -> bool:
    """Cek keberadaan modul tanpa mengeksekusinya (hindari impor berat)."""
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
