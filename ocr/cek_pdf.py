#!/usr/bin/env python3
"""cek_pdf.py — survei cepat folder PDF (berdiri sendiri, tanpa paket ocrpipe).

Script ini sengaja dibuat MANDIRI: hanya butuh PyMuPDF (`pip install pymupdf`),
tidak mengimpor paket ocrpipe sama sekali. Tujuannya agar ipank bisa langsung
tahu "PDF saya ini jenisnya apa" sebelum memasang engine OCR yang berat.

Jalankan:

    python cek_pdf.py                 # survei folder saat ini
    python cek_pdf.py data/pdf        # survei folder tertentu
    python cek_pdf.py a.pdf b.pdf     # survei berkas tertentu
    python cek_pdf.py data/pdf --csv laporan.csv

Keluaran: tabel ringkas per berkas (digital/scan/campuran) + saran langkah
berikutnya. Dengan --csv, rincian per halaman ikut disimpan.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ambang keputusan — DISAMAKAN dengan default ocrpipe/config.py (DetectConfig)
# supaya hasil survei di sini konsisten dengan hasil pipeline sebenarnya.
MIN_CHARS = 120
MIN_CHAR_DENSITY = 0.08
MAX_GARBLED_RATIO = 0.02
FULL_PAGE_IMAGE_RATIO = 0.65

import re

_CID_RE = re.compile(r"\(cid:\d+\)")
_JUNK_RE = re.compile(r"[�■□ﾾ-]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{2,}")


def analyze_page(page) -> dict:
    rect = page.rect
    area = max(1.0, float(rect.width) * float(rect.height))
    raw = page.get_text("text") or ""

    n_cid = sum(len(m.group(0)) for m in _CID_RE.finditer(raw))
    n_junk = len(_JUNK_RE.findall(raw))
    garbled = (n_cid + n_junk) / len(raw) if raw else 0.0

    usable = _JUNK_RE.sub("", _CID_RE.sub("", raw))
    n_chars = len("".join(usable.split()))
    n_words = len(_WORD_RE.findall(usable))
    density = n_chars / (area / 1000.0)

    try:
        infos = page.get_image_info()
    except Exception:
        infos = []
    img_area = 0.0
    full = False
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        a = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
        img_area += a
        if a / area >= FULL_PAGE_IMAGE_RATIO:
            full = True
    try:
        n_draw = len(page.get_drawings())
    except Exception:
        n_draw = 0

    has_visual = len(infos) > 0 or n_draw > 3
    if n_chars > 0 and garbled > MAX_GARBLED_RATIO:
        route, why = "scan", f"text layer rusak ({garbled:.0%} karakter tak terbaca)"
    elif n_chars < MIN_CHARS and not has_visual:
        route, why = "kosong", "tidak ada teks maupun objek visual"
    elif full and n_chars < MIN_CHARS:
        route, why = "scan", "citra menutupi hampir seluruh halaman, teks sangat sedikit"
    elif n_chars < MIN_CHARS:
        route, why = "scan", f"hanya {n_chars} karakter padahal ada objek visual"
    elif density < MIN_CHAR_DENSITY:
        route, why = "scan", f"kepadatan teks rendah ({density:.3f})"
    else:
        route, why = "digital", f"text layer sehat ({n_chars} karakter)"

    return {
        "route": route,
        "why": why,
        "n_chars": n_chars,
        "n_words": n_words,
        "density": round(density, 3),
        "garbled": round(garbled, 4),
        "n_images": len(infos),
        "img_ratio": round(min(img_area / area, 1.0), 3),
        "full_image": full,
    }


def analyze_pdf(path: Path) -> dict:
    import pymupdf

    out = {"file": path.name, "pages": [], "error": None, "n_pages": 0}
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        out["error"] = f"gagal dibuka: {exc}"
        return out
    try:
        if doc.needs_pass and not doc.authenticate(""):
            out["error"] = "terproteksi kata sandi"
            out["n_pages"] = doc.page_count
            return out
        out["n_pages"] = doc.page_count
        for i in range(doc.page_count):
            try:
                info = analyze_page(doc[i])
            except Exception as exc:
                info = {"route": "scan", "why": f"gagal diukur ({exc})", "n_chars": 0,
                        "n_words": 0, "density": 0, "garbled": 0, "n_images": 0,
                        "img_ratio": 0, "full_image": False}
            info["page"] = i + 1
            out["pages"].append(info)
    finally:
        doc.close()
    return out


def classify(counts: dict) -> str:
    content = counts.get("digital", 0) + counts.get("scan", 0)
    if content == 0:
        return "kosong"
    if counts.get("scan", 0) == 0:
        return "digital"
    if counts.get("digital", 0) == 0:
        return "scan"
    return "campuran"


def discover(inputs) -> list:
    pdfs = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            pdfs.extend(sorted(p.glob("**/*.pdf")))
            pdfs.extend(sorted(p.glob("**/*.PDF")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        elif not p.exists():
            print(f"  (dilewati, tidak ada: {p})", file=sys.stderr)
    seen, uniq = set(), []
    for p in pdfs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Survei cepat jenis PDF (digital/scan/campuran).")
    ap.add_argument("inputs", nargs="*", default=["."], help="folder atau berkas PDF")
    ap.add_argument("--csv", help="simpan rincian per halaman ke berkas CSV")
    ap.add_argument("--per-page", action="store_true", help="cetak detail setiap halaman")
    args = ap.parse_args(argv)

    import importlib.util

    if importlib.util.find_spec("pymupdf") is None:
        print("PyMuPDF belum terpasang. Jalankan:  pip install pymupdf", file=sys.stderr)
        return 1

    pdfs = discover(args.inputs)
    if not pdfs:
        print("Tidak ada berkas PDF ditemukan.")
        return 1

    print(f"\nMenyurvei {len(pdfs)} berkas PDF\n")
    head = f"{'berkas':<46}{'hlm':>5}{'digital':>9}{'scan':>6}{'kosong':>8}{'jenis':>11}"
    print(head)
    print("-" * len(head))

    totals = {"digital": 0, "scan": 0, "kosong": 0}
    page_rows, doc_rows = [], []

    for pdf in pdfs:
        res = analyze_pdf(pdf)
        if res["error"]:
            print(f"{pdf.name[:46]:<46}{res['n_pages']:>5}   GAGAL: {res['error']}")
            doc_rows.append({"berkas": pdf.name, "galat": res["error"]})
            continue
        counts = {"digital": 0, "scan": 0, "kosong": 0}
        for pg in res["pages"]:
            counts[pg["route"]] = counts.get(pg["route"], 0) + 1
            totals[pg["route"]] += 1
            row = {"berkas": pdf.name, **pg}
            page_rows.append(row)
        kind = classify(counts)
        print(f"{pdf.name[:46]:<46}{res['n_pages']:>5}{counts['digital']:>9}"
              f"{counts['scan']:>6}{counts['kosong']:>8}{kind:>11}")
        doc_rows.append({"berkas": pdf.name, "jml_halaman": res["n_pages"],
                         "digital": counts["digital"], "scan": counts["scan"],
                         "kosong": counts["kosong"], "jenis": kind, "galat": ""})
        if args.per_page:
            for pg in res["pages"]:
                print(f"      hlm {pg['page']:>4}  {pg['route']:<8} {pg['n_chars']:>6} kar  {pg['why']}")

    total = sum(totals.values())
    print("-" * len(head))
    print(f"{'TOTAL':<46}{total:>5}{totals['digital']:>9}{totals['scan']:>6}{totals['kosong']:>8}")

    if total:
        pct = totals["scan"] / total * 100
        print(f"\n{pct:.1f}% halaman memerlukan OCR.\n")
        if totals["scan"] == 0:
            print("Seluruh korpus DIGITAL. Tidak perlu engine OCR berat.")
            print("  ->  python run_ocr.py ingest <folder> --force-native")
        elif pct < 15:
            print("Mayoritas digital, sedikit halaman scan.")
            print("  Tesseract sudah cukup:  python run_ocr.py ingest <folder> --engine tesseract")
        else:
            print("Porsi halaman scan besar — pakai engine neural untuk kualitas tabel.")
            print("  ->  python run_ocr.py ingest <folder> --engine docling")
            print("  (untuk batch besar, jalankan di Kaggle — lihat notebooks/)")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        if page_rows:
            with out.open("w", encoding="utf-8-sig", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(page_rows[0].keys()))
                w.writeheader()
                w.writerows(page_rows)
            print(f"\nRincian per halaman disimpan: {out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
