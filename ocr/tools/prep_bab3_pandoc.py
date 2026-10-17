"""Siapkan Markdown Bab 3 untuk konversi pandoc->docx.

Ganti tiap blok ```mermaid ... ``` dengan sisipan gambar PNG hasil render,
dengan lebar yang sudah dibatasi agar muat di halaman A4.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path("docs/Bab3-Metodologi-Akuisisi-Data.md")
DST = Path("docs/_bab3_pandoc.md")

# Urutan kemunculan diagram mermaid di dokumen -> (berkas gambar, lebar).
# Gambar 3.1 & 3.2 tinggi/portrait -> lebar dibatasi agar tidak melebihi tinggi
# halaman. Gambar 3.3 lebar/landscape -> boleh lebih lebar.
FIGURES = [
    ("docs/gambar/gambar-3-1-tahapan-penelitian.png", "7cm"),
    ("docs/gambar/gambar-3-2-praproses-per-halaman.png", "9cm"),
    ("docs/gambar/gambar-3-3-arsitektur-rag.png", "15cm"),
]

MERMAID_RE = re.compile(r"```mermaid\n.*?```", re.DOTALL)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    blocks = MERMAID_RE.findall(text)
    assert len(blocks) == len(FIGURES), (
        f"jumlah blok mermaid ({len(blocks)}) != jumlah gambar ({len(FIGURES)})"
    )

    idx = 0

    def repl(_m: re.Match) -> str:
        nonlocal idx
        img, width = FIGURES[idx]
        idx += 1
        # Pandoc: lebar gambar + rata tengah via paragraf tersendiri.
        return f"![]({img}){{width={width}}}"

    out = MERMAID_RE.sub(repl, text)

    # Buang baris catatan pemakaian draf (blockquote awal) agar dokumen resmi
    # tidak memuat instruksi internal? -> dipertahankan sebagai catatan, tetapi
    # diubah agar tetap informatif. (Dibiarkan; mudah dihapus manual.)

    DST.write_text(out, encoding="utf-8")
    print(f"ditulis: {DST} ({len(out)} karakter, {idx} gambar disisipkan)")


if __name__ == "__main__":
    main()
