"""Registry engine + pemilihan otomatis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .base import Availability, Engine, EngineError

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config

#: Urutan preferensi ketika `ocr_engine = "auto"`.
#: Docling diletakkan pertama karena (a) lisensinya MIT seluruhnya termasuk
#: bobot model, (b) TableFormer mode ACCURATE unggul pada tabel tanpa garis,
#: (c) instalasinya paling mulus di Windows. PP-StructureV3 kedua karena
#: pengenalan strukturnya sangat baik tetapi konfigurasi default-nya berat.
#: Tesseract terakhir sebagai jaring pengaman yang selalu bisa dipasang.
AUTO_PREFERENCE: tuple[str, ...] = ("docling", "paddle", "marker", "tesseract")


def _build_registry() -> dict[str, Callable[..., Engine]]:
    """Impor malas: kelas engine baru dimuat saat benar-benar diminta."""

    def _native(cfg: "Config") -> Engine:
        from .native import NativeEngine

        return NativeEngine(cfg)

    def _tesseract(cfg: "Config") -> Engine:
        from .tesseract import TesseractEngine

        return TesseractEngine(cfg)

    def _paddle(cfg: "Config") -> Engine:
        from .paddle import PaddleStructureEngine

        return PaddleStructureEngine(cfg)

    def _docling(cfg: "Config") -> Engine:
        from .docling_engine import DoclingEngine

        return DoclingEngine(cfg)

    def _marker(cfg: "Config") -> Engine:
        from .marker_engine import MarkerEngine

        return MarkerEngine(cfg)

    return {
        "native": _native,
        "tesseract": _tesseract,
        "paddle": _paddle,
        "docling": _docling,
        "marker": _marker,
    }


REGISTRY = _build_registry()

ENGINE_NAMES: tuple[str, ...] = tuple(REGISTRY)


def get_engine(name: str, cfg: "Config") -> Engine:
    """Buat instance engine berdasarkan nama."""
    try:
        factory = REGISTRY[name]
    except KeyError as exc:
        raise EngineError(
            f"Engine '{name}' tidak dikenal. Pilihan: {', '.join(ENGINE_NAMES)}"
        ) from exc
    return factory(cfg)


def check_all(cfg: "Config") -> dict[str, Availability]:
    """Periksa ketersediaan semua engine (dipakai perintah `engines`)."""
    out: dict[str, Availability] = {}
    for name in ENGINE_NAMES:
        try:
            out[name] = get_engine(name, cfg).availability()
        except Exception as exc:  # pragma: no cover - instansiasi gagal
            out[name] = Availability(
                ok=False, message=f"gagal diperiksa: {exc}", install_hint=""
            )
    return out


def resolve_ocr_engine(cfg: "Config") -> tuple[Engine, list[str]]:
    """Tentukan engine OCR yang akan dipakai, lengkap dengan catatan keputusan.

    Returns:
        (engine_terpilih, catatan). `catatan` berisi riwayat engine yang
        dilewati beserta alasannya — ikut ditulis ke laporan QA supaya
        pemilihan engine dapat dipertanggungjawabkan di sidang.
    """
    notes: list[str] = []
    requested = (cfg.ocr_engine or "auto").strip().lower()

    if requested == "auto":
        candidates = list(AUTO_PREFERENCE)
        notes.append(f"mode auto, urutan preferensi: {' > '.join(candidates)}")
    else:
        candidates = [requested]
        for name in cfg.fallback_chain:
            if name not in candidates:
                candidates.append(name)
        notes.append(
            f"engine diminta: {requested}; rantai fallback: "
            f"{' > '.join(candidates[1:]) or '(kosong)'}"
        )

    last_error = ""
    for name in candidates:
        if name == "native":
            continue  # native bukan engine OCR
        try:
            engine = get_engine(name, cfg)
        except EngineError as exc:
            notes.append(f"{name}: {exc}")
            last_error = str(exc)
            continue
        avail = engine.availability()
        if avail.ok:
            ver = f" v{avail.version}" if avail.version else ""
            notes.append(f"terpilih: {name}{ver}")
            return engine, notes
        notes.append(f"{name}: TIDAK TERSEDIA — {avail.message}")
        if avail.install_hint:
            notes.append(f"  cara pasang: {avail.install_hint}")
        last_error = avail.message

    raise EngineError(
        "Tidak ada engine OCR yang tersedia.\n"
        + "\n".join(f"  - {n}" for n in notes)
        + f"\nKesalahan terakhir: {last_error}\n"
        "Minimal pasang Tesseract: pip install pytesseract + "
        "unduh installer Tesseract (lihat docs/INSTALL-WINDOWS.md)."
    )


__all__ = [
    "AUTO_PREFERENCE",
    "Availability",
    "Engine",
    "EngineError",
    "ENGINE_NAMES",
    "REGISTRY",
    "check_all",
    "get_engine",
    "resolve_ocr_engine",
]
