"""Praproses citra untuk halaman yang diarahkan ke OCR.

Urutan operasi dipilih secara sengaja:
    rasterisasi -> auto-rotate (90/180/270) -> grayscale -> peregangan kontras
    -> deskew (sudut kecil) -> denoise -> binarisasi -> crop bingkai

Rotasi kasar harus dilakukan SEBELUM deskew: estimator sudut berbasis
minAreaRect hanya valid untuk kemiringan kecil (< ~15 derajat); halaman yang
terbalik 90 derajat akan membuatnya menghasilkan sudut yang keliru.

Binarisasi diletakkan paling akhir dan default-nya MATI. Untuk Tesseract
binarisasi membantu, tetapi model neural (PP-StructureV3, TableFormer, Surya)
dilatih pada citra grayscale/berwarna; membinarisasi lebih dulu menghapus
informasi antarmuka sel pada tabel tanpa garis dan menurunkan akurasi
pengenalan struktur tabel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import PreprocessConfig
from .utils import get_logger


@dataclass
class PreprocResult:
    image: np.ndarray  # citra akhir (grayscale 2D atau BGR 3D, uint8)
    dpi: int = 300
    applied: list[str] = field(default_factory=list)
    rotation_applied: int = 0
    deskew_angle: float = 0.0
    warnings: list[str] = field(default_factory=list)
    #: Faktor skala dari titik PDF ke piksel citra (px per pt).
    scale: float = 1.0

    @property
    def is_gray(self) -> bool:
        return self.image.ndim == 2

    def to_bgr(self) -> np.ndarray:
        import cv2

        if self.image.ndim == 2:
            return cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        return self.image

    def to_rgb(self) -> np.ndarray:
        import cv2

        if self.image.ndim == 2:
            return cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def to_pil(self) -> Any:
        from PIL import Image

        return Image.fromarray(self.image if self.is_gray else self.to_rgb())

    def save(self, path: str | Path) -> Path:
        import cv2

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), self.image)
        return p


# --------------------------------------------------------------------------- #
# Rasterisasi
# --------------------------------------------------------------------------- #


def effective_dpi(page_rect: Any, requested_dpi: int, max_pixels: int) -> int:
    """Turunkan DPI bila jumlah piksel melebihi pagar pengaman memori.

    Pada mesin dengan RAM 8 GB, halaman A3 pada 400 DPI sudah dapat memakai
    ratusan MB per citra; dikalikan antrean batch, ini penyebab umum MemoryError.
    """
    w_in = float(page_rect.width) / 72.0
    h_in = float(page_rect.height) / 72.0
    px = w_in * h_in * requested_dpi * requested_dpi
    if px <= max_pixels or w_in <= 0 or h_in <= 0:
        return requested_dpi
    safe = int(math.sqrt(max_pixels / (w_in * h_in)))
    return max(120, safe)


def rasterize_page(
    pdf_path: str | Path, page_no: int, cfg: PreprocessConfig
) -> tuple[np.ndarray, int, float]:
    """Render satu halaman PDF (1-based) menjadi array numpy BGR/grayscale.

    Memakai PyMuPDF agar tidak bergantung pada biner poppler — satu masalah
    instalasi yang lebih sedikit di Windows.

    Returns:
        (citra, dpi_efektif, skala_px_per_pt)
    """
    import cv2
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        if not 1 <= page_no <= doc.page_count:
            raise IndexError(
                f"Halaman {page_no} di luar rentang (dokumen {doc.page_count} halaman)"
            )
        page = doc[page_no - 1]
        dpi = effective_dpi(page.rect, cfg.dpi, cfg.max_pixels)
        pix = page.get_pixmap(dpi=dpi, alpha=False, annots=True)
        buf = np.frombuffer(pix.samples, dtype=np.uint8)
        img = buf.reshape(pix.height, pix.width, pix.n).copy()
        if pix.n == 1:
            img = img[:, :, 0]
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img, dpi, dpi / 72.0
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# Langkah-langkah praproses
# --------------------------------------------------------------------------- #


def detect_osd_rotation(img: np.ndarray) -> int:
    """Deteksi rotasi kasar (0/90/180/270) memakai Tesseract OSD.

    Mengembalikan 0 bila Tesseract tidak tersedia atau tidak yakin — OSD
    membutuhkan cukup banyak teks dan sering gagal pada halaman bergambar.
    """
    try:
        import pytesseract
    except ImportError:
        return 0
    try:
        osd = pytesseract.image_to_osd(img)
    except Exception:
        return 0
    rotate = 0
    conf = 0.0
    for line in osd.splitlines():
        if line.startswith("Rotate:"):
            try:
                rotate = int(line.split(":")[1].strip())
            except ValueError:
                rotate = 0
        elif line.startswith("Orientation confidence:"):
            try:
                conf = float(line.split(":")[1].strip())
            except ValueError:
                conf = 0.0
    # Ambang 1.0 konservatif: OSD melaporkan kepercayaan rendah pada
    # halaman yang didominasi gambar, dan rotasi keliru jauh lebih merusak
    # daripada tidak merotasi sama sekali.
    return rotate if conf >= 1.0 else 0


def rotate_90s(img: np.ndarray, degrees: int) -> np.ndarray:
    import cv2

    deg = degrees % 360
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def estimate_skew(gray: np.ndarray, max_angle: float = 12.0) -> float:
    """Perkirakan sudut kemiringan (derajat) memakai minAreaRect pada piksel teks.

    Positif = teks miring berlawanan arah jarum jam.
    """
    import cv2

    # Perkecil agar cepat dan tahan derau; teks tetap terdeteksi.
    h, w = gray.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    thr = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Satukan huruf menjadi baris supaya orientasi baris yang terukur.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    coords = cv2.findNonZero(closed)
    if coords is None or len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV >= 4.5 mengembalikan sudut pada rentang (0, 90].
    if angle > 45:
        angle -= 90
    if abs(angle) > max_angle:
        return 0.0
    return float(angle)


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    import cv2

    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perluas kanvas agar sudut halaman tidak terpotong.
    cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    mat[0, 2] += new_w / 2.0 - center[0]
    mat[1, 2] += new_h / 2.0 - center[1]
    border = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(
        img,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def stretch_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE ringan: memperbaiki hasil fotokopi yang pudar / tidak rata."""
    import cv2

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    import cv2

    if method == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Adaptif lebih tahan terhadap pencahayaan tidak rata pada hasil pindaian.
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def crop_black_border(img: np.ndarray, tol: int = 40) -> np.ndarray:
    """Buang bingkai gelap di tepi (umum pada hasil fotokopi/pindai buku)."""
    import cv2

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > tol).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    # Jangan memangkas lebih dari 12% tiap sisi — pagar pengaman agar isi
    # halaman tidak ikut terbuang bila deteksi salah.
    H, W = gray.shape[:2]
    if w < W * 0.76 or h < H * 0.76:
        return img
    return img[y : y + h, x : x + w]


# --------------------------------------------------------------------------- #
# Orkestrasi
# --------------------------------------------------------------------------- #


def preprocess_page(
    pdf_path: str | Path,
    page_no: int,
    cfg: PreprocessConfig,
    *,
    for_neural_engine: bool = False,
) -> PreprocResult:
    """Rasterisasi + praproses satu halaman.

    Args:
        for_neural_engine: bila True, binarisasi dan denoise dipaksa mati
            karena model neural berperforma lebih baik pada citra grayscale
            asli. Lihat docstring modul.
    """
    import cv2

    log = get_logger()
    img, dpi, scale = rasterize_page(pdf_path, page_no, cfg)
    res = PreprocResult(image=img, dpi=dpi, scale=scale)
    res.applied.append(f"rasterisasi@{dpi}dpi")
    if dpi != cfg.dpi:
        res.warnings.append(
            f"DPI diturunkan dari {cfg.dpi} ke {dpi} agar tidak melampaui "
            f"batas {cfg.max_pixels:,} piksel"
        )

    # 1) rotasi kasar
    if cfg.auto_rotate_osd:
        deg = detect_osd_rotation(img)
        if deg:
            img = rotate_90s(img, deg)
            res.rotation_applied = deg
            res.applied.append(f"rotasi{deg}")

    # 2) grayscale
    if cfg.grayscale and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res.applied.append("grayscale")

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) kontras
    if cfg.contrast_stretch and img.ndim == 2:
        img = stretch_contrast(img)
        gray = img
        res.applied.append("clahe")

    # 4) deskew
    if cfg.deskew:
        angle = estimate_skew(gray, cfg.deskew_max_angle)
        if abs(angle) >= cfg.deskew_min_angle:
            img = deskew(img, angle)
            res.deskew_angle = round(angle, 3)
            res.applied.append(f"deskew{angle:+.2f}deg")
        elif angle:
            log.debug(
                "hlm %d: kemiringan %.3f derajat di bawah ambang, tidak dirotasi",
                page_no,
                angle,
            )

    # 5) denoise — dilewati untuk engine neural
    if cfg.denoise and not for_neural_engine:
        k = cfg.denoise_ksize if cfg.denoise_ksize % 2 == 1 else cfg.denoise_ksize + 1
        img = cv2.medianBlur(img, k)
        res.applied.append(f"median{k}")
    elif cfg.denoise and for_neural_engine:
        res.applied.append("denoise-dilewati(engine-neural)")

    # 6) binarisasi — dilewati untuk engine neural
    if cfg.binarize and not for_neural_engine:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = binarize(g, cfg.binarize_method)
        res.applied.append(f"binarisasi-{cfg.binarize_method}")
    elif cfg.binarize and for_neural_engine:
        res.applied.append("binarisasi-dilewati(engine-neural)")

    # 7) crop bingkai
    if cfg.crop_border:
        before = img.shape[:2]
        img = crop_black_border(img)
        if img.shape[:2] != before:
            res.applied.append("crop-bingkai")

    res.image = np.ascontiguousarray(img)
    return res
