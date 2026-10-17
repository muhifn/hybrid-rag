"""Pembuat PDF sintetis untuk menguji pipeline (bukan bagian dari produk).

Menghasilkan tiga berkas di folder tujuan:
  1. digital_bertabel.pdf  — teks + heading + tabel, murni text layer
  2. hasil_scan.pdf         — halaman yang dirasterisasi, dimiringkan,
                              diberi derau -> tidak ada text layer (perlu OCR)
  3. campuran.pdf           — gabungan: halaman 1 digital, halaman 2 "scan"

Dipakai oleh test_e2e.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pymupdf


def _page_digital_text(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=595, height=842)  # A4 pt
    page.insert_text((72, 72), "BAB II REGISTRASI MAHASISWA", fontsize=16, fontname="helv")
    page.insert_text(
        (72, 110),
        "Mahasiswa wajib melakukan registrasi pada setiap awal semester "
        "sesuai",
        fontsize=11,
    )
    page.insert_text(
        (72, 126),
        "kalender akademik. Registrasi dilakukan secara daring melalui portal "
        "SIAKAD.",
        fontsize=11,
    )
    page.insert_text((72, 160), "2.1 Syarat Registrasi", fontsize=13, fontname="helv")
    body = [
        "Mahasiswa harus memenuhi persyaratan berikut sebelum registrasi:",
        "a. Telah melunasi biaya pendidikan semester berjalan.",
        "b. Tidak sedang dalam status cuti akademik tanpa izin.",
        "c. Telah mengisi Kartu Rencana Studi (KRS) bersama dosen pembimbing.",
    ]
    y = 185
    for line in body:
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    # footer
    page.insert_text((72, 800), "Panduan Akademik FTII UHAMKA 2026", fontsize=8)
    page.insert_text((520, 800), "hlm. 1", fontsize=8)


def _page_digital_table(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "2.2 Kalender Kegiatan", fontsize=13, fontname="helv")
    page.insert_text((72, 96), "Tabel 2.1 Jadwal kegiatan semester ganjil", fontsize=10)

    # Gambar tabel bergaris: 4 kolom x 5 baris
    x0, y0 = 72, 110
    col_w = [120, 150, 110, 110]
    row_h = 26
    n_rows = 5
    xs = [x0]
    for w in col_w:
        xs.append(xs[-1] + w)
    ys = [y0 + i * row_h for i in range(n_rows + 1)]

    # garis
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), width=0.8)
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), width=0.8)

    data = [
        ["No", "Kegiatan", "Tanggal Mulai", "Tanggal Selesai"],
        ["1", "Registrasi", "1 Sep 2026", "7 Sep 2026"],
        ["2", "Perkuliahan", "8 Sep 2026", "20 Des 2026"],
        ["3", "UTS", "20 Okt 2026", "25 Okt 2026"],
        ["4", "UAS", "5 Jan 2027", "15 Jan 2027"],
    ]
    for r, row in enumerate(data):
        for c, cell in enumerate(row):
            page.insert_text((xs[c] + 5, ys[r] + 17), cell, fontsize=9)

    page.insert_text(
        (72, ys[-1] + 30),
        "Mahasiswa diharapkan memperhatikan tanggal-tanggal penting di atas.",
        fontsize=11,
    )


def _render_page_to_image(page: pymupdf.Page, dpi: int = 150) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    buf = np.frombuffer(pix.samples, dtype=np.uint8)
    img = buf.reshape(pix.height, pix.width, pix.n).copy()
    if pix.n == 4:
        img = img[:, :, :3]
    return img


def _degrade(img: np.ndarray, angle: float = 2.5) -> np.ndarray:
    """Miringkan + beri derau supaya menyerupai hasil scan/fotokopi."""
    import cv2

    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot = cv2.warpAffine(
        img, mat, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    noise = np.random.normal(0, 8, rot.shape).astype(np.int16)
    noisy = np.clip(rot.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def _image_to_pdf_page(doc: pymupdf.Document, img: np.ndarray) -> None:
    """Masukkan citra sebagai SATU halaman penuh (tanpa text layer)."""
    import cv2

    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("gagal encode png")
    h, w = img.shape[:2]
    # 150 dpi -> ukuran titik PDF
    pw, ph = w * 72.0 / 150.0, h * 72.0 / 150.0
    page = doc.new_page(width=pw, height=ph)
    page.insert_image(pymupdf.Rect(0, 0, pw, ph), stream=enc.tobytes())


def build(dirpath: str) -> None:
    out = Path(dirpath)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    # 1) digital murni
    doc = pymupdf.open()
    _page_digital_text(doc)
    _page_digital_table(doc)
    doc.save(str(out / "digital_bertabel.pdf"))
    doc.close()

    # 2) hasil scan (render dua halaman di atas, lalu degrade jadi gambar)
    src = pymupdf.open()
    _page_digital_text(src)
    _page_digital_table(src)
    scan = pymupdf.open()
    for i in range(src.page_count):
        img = _render_page_to_image(src[i], dpi=150)
        img = _degrade(img, angle=2.5 if i == 0 else -1.8)
        _image_to_pdf_page(scan, img)
    scan.save(str(out / "hasil_scan.pdf"))
    scan.close()
    src.close()

    # 3) campuran: halaman 1 digital, halaman 2 scan
    mix = pymupdf.open()
    _page_digital_text(mix)
    tmp = pymupdf.open()
    _page_digital_table(tmp)
    img = _degrade(_render_page_to_image(tmp[0], dpi=150), angle=3.0)
    _image_to_pdf_page(mix, img)
    tmp.close()
    mix.save(str(out / "campuran.pdf"))
    mix.close()

    print(f"PDF uji dibuat di {out}:")
    for p in sorted(out.glob("*.pdf")):
        print(f"  - {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data-uji")
