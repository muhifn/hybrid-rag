"""Perbaiki lebar tabel & margin pada docx hasil pandoc.

Masalah: pandoc menandai tabel lebar 100% tetapi tanpa <w:tblLayout fixed>,
sehingga Word memakai mode autofit-to-contents dan kolom lebar (mis. Time Line
Schedule 8 kolom) meluber keluar halaman. Perbaikan: set margin A4 wajar, paksa
tblLayout fixed, dan tulis ulang lebar kolom tiap tabel sesuai profil per jumlah
kolom agar jumlahnya pas lebar area teks.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# A4 = 11906 x 16838 DXA. Margin 2.6 cm (~1474 DXA) kiri-kanan -> area teks:
TEXT_WIDTH = 11906 - 2 * 1474  # = 8958 DXA

# Profil lebar kolom per jumlah kolom (harus menjumlah ~TEXT_WIDTH).
PROFILES = {
    8: [520, 2560, 646, 646, 646, 646, 646, 646],   # Tabel 3.2 Time Line (No, Kegiatan, 6 bulan)
    5: [1350, 1600, 3058, 1150, 1800],               # Tabel 3.1 Peubah
    3: [1200, 3879, 3879],                           # Tabel 3.3 & Lampiran
}


def _fit(profile: list[int]) -> list[int]:
    s = sum(profile)
    scaled = [round(w * TEXT_WIDTH / s) for w in profile]
    scaled[-1] += TEXT_WIDTH - sum(scaled)  # koreksi pembulatan
    return scaled


def fix_tables(xml: str) -> str:
    def repl(m: re.Match) -> str:
        tbl = m.group(0)
        n_cols = len(re.findall(r"<w:gridCol", tbl))
        if n_cols == 0:
            # pandoc kadang menulis <w:tblGrid /> kosong + lebar 0 (penyebab
            # tabel meluber). Hitung kolom dari jumlah sel <w:tc> baris pertama.
            first_row = re.search(r"<w:tr>.*?</w:tr>", tbl, re.DOTALL)
            if first_row:
                n_cols = len(re.findall(r"<w:tc>", first_row.group(0)))
        if n_cols == 0:
            return tbl  # benar-benar tak bisa ditentukan; biarkan apa adanya
        profile = PROFILES.get(n_cols)
        if profile is None:
            # jumlah kolom tak terduga: bagi rata
            profile = [TEXT_WIDTH // n_cols] * n_cols
        widths = _fit(profile)

        # 1) paksa tblLayout fixed + lebar tabel = DXA (bukan pct)
        new_tblpr = (
            "<w:tblPr><w:tblStyle w:val=\"Table\" />"
            f"<w:tblW w:type=\"dxa\" w:w=\"{TEXT_WIDTH}\" />"
            "<w:tblLayout w:type=\"fixed\" />"
            "<w:tblLook w:firstRow=\"1\" /></w:tblPr>"
        )
        tbl = re.sub(r"<w:tblPr>.*?</w:tblPr>", new_tblpr, tbl, count=1, flags=re.DOTALL)

        # 2) tulis ulang tblGrid (tangani bentuk kosong <w:tblGrid /> maupun berisi)
        grid = "<w:tblGrid>" + "".join(f"<w:gridCol w:w=\"{w}\" />" for w in widths) + "</w:tblGrid>"
        if re.search(r"<w:tblGrid\s*/>", tbl):
            tbl = re.sub(r"<w:tblGrid\s*/>", grid, tbl, count=1)
        else:
            tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", grid, tbl, count=1, flags=re.DOTALL)

        # 3) set lebar tiap sel (tcW) mengikuti urutan kolom; ulang per baris
        col = {"i": 0}

        def set_tcw(mm: re.Match) -> str:
            w = widths[col["i"] % n_cols]
            col["i"] += 1
            return f"<w:tcW w:w=\"{w}\" w:type=\"dxa\" />"

        tbl = re.sub(r"<w:tcW[^/]*/>", set_tcw, tbl)
        return tbl

    return re.sub(r"<w:tbl>.*?</w:tbl>", repl, xml, flags=re.DOTALL)


def fix_margins(xml: str) -> str:
    # Sisipkan/overwrite sectPr pgSz (A4) & pgMar.
    pgmar = ('<w:pgMar w:top="1701" w:right="1474" w:bottom="1701" '
             'w:left="1474" w:header="720" w:footer="720" w:gutter="0" />')
    pgsz = '<w:pgSz w:w="11906" w:h="16838" />'
    if "<w:pgSz" in xml:
        xml = re.sub(r"<w:pgSz[^/]*/>", pgsz, xml)
    if "<w:pgMar" in xml:
        xml = re.sub(r"<w:pgMar[^/]*/>", pgmar, xml)
    else:
        # sisipkan pgMar tepat setelah pgSz di dalam sectPr
        xml = xml.replace(pgsz, pgsz + pgmar)
    return xml


def main() -> None:
    unpacked = Path(sys.argv[1])
    doc = unpacked / "word" / "document.xml"
    xml = doc.read_text(encoding="utf-8")
    xml = fix_tables(xml)
    xml = fix_margins(xml)
    doc.write_text(xml, encoding="utf-8")
    print("tabel difixed + margin A4 diterapkan")


if __name__ == "__main__":
    main()
