"""Antarmuka baris perintah.

    python run_ocr.py diagnose data/pdf
    python run_ocr.py ingest   data/pdf -o keluaran
    python run_ocr.py engines
    python run_ocr.py sample   data/pdf -o sampel-anotasi -n 12
    python run_ocr.py eval     sampel-anotasi

Memakai argparse dari pustaka standar, bukan click/typer, agar tidak
menambah dependensi yang harus dipasang di Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import Config
from .detect import probe_document
from .engines import check_all
from .models import ROUTE_EMPTY, ROUTE_NATIVE, ROUTE_OCR
from .pipeline import Pipeline, discover_pdfs, load_metadata
from .qa import aggregate_eval, compute_cer_wer
from .utils import get_logger, parse_page_spec, setup_logging

PROG = "ocrpipe"


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Pipeline OCR dokumen akademik untuk RAG: deteksi halaman "
            "digital/scan per halaman, ekstraksi teks + tabel, keluaran "
            "Markdown + JSONL + CSV/XLSX."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="berkas konfigurasi YAML")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="kurangi keluaran log"
    )
    parser.add_argument("--log-file", help="simpan log ke berkas")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ing = sub.add_parser(
        "ingest", help="proses PDF menjadi Markdown + JSONL + tabel"
    )
    p_ing.add_argument("inputs", nargs="+", help="berkas PDF atau folder")
    p_ing.add_argument(
        "-o", "--out", default="keluaran", help="folder keluaran (default: keluaran)"
    )
    p_ing.add_argument(
        "--engine",
        help="engine OCR: auto|docling|paddle|marker|tesseract (default: auto)",
    )
    p_ing.add_argument("--pages", help='halaman tertentu, mis. "1,3,5-9"')
    p_ing.add_argument("--dpi", type=int, help="DPI rasterisasi halaman scan")
    p_ing.add_argument(
        "--force-ocr",
        action="store_true",
        help="paksa semua halaman lewat OCR (abaikan text layer)",
    )
    p_ing.add_argument(
        "--force-native",
        action="store_true",
        help="paksa semua halaman pakai text layer (tanpa OCR)",
    )
    p_ing.add_argument(
        "--no-recursive", action="store_true", help="jangan telusuri sub-folder"
    )
    p_ing.add_argument("--chunk-size", type=int, help="target karakter per chunk")
    p_ing.add_argument("--overlap", type=int, help="karakter tumpang tindih antar chunk")
    p_ing.add_argument(
        "--device", help="paksa perangkat engine neural: cpu|gpu|gpu:0"
    )
    p_ing.add_argument(
        "--no-tables", action="store_true", help="lewati ekstraksi tabel"
    )

    # --- diagnose ---
    p_diag = sub.add_parser(
        "diagnose",
        help="survei PDF: halaman mana digital, mana scan, berapa tabel (tanpa OCR)",
    )
    p_diag.add_argument("inputs", nargs="+", help="berkas PDF atau folder")
    p_diag.add_argument(
        "-o", "--out", help="simpan laporan ke folder ini (opsional)"
    )
    p_diag.add_argument(
        "--per-page", action="store_true", help="tampilkan detail setiap halaman"
    )

    # --- engines ---
    sub.add_parser("engines", help="tampilkan engine yang tersedia di mesin ini")

    # --- sample ---
    p_sam = sub.add_parser(
        "sample",
        help="siapkan sampel halaman untuk transkripsi manual (pengukuran CER/WER)",
    )
    p_sam.add_argument("inputs", nargs="+", help="berkas PDF atau folder")
    p_sam.add_argument("-o", "--out", default="sampel-anotasi", help="folder keluaran")
    p_sam.add_argument(
        "-n", "--n-samples", type=int, default=10, help="jumlah halaman sampel"
    )
    p_sam.add_argument("--seed", type=int, default=42, help="seed acak (reproduktibel)")
    p_sam.add_argument(
        "--engine", help="engine OCR untuk menghasilkan hipotesis (default: auto)"
    )
    p_sam.add_argument(
        "--image-dpi", type=int, default=150, help="DPI gambar sampel untuk dibaca"
    )

    # --- eval ---
    p_eval = sub.add_parser(
        "eval", help="hitung CER/WER dari pasangan *.ref.txt dan *.hyp.txt"
    )
    p_eval.add_argument("folder", help="folder hasil perintah `sample`")
    p_eval.add_argument("-o", "--out", help="folder laporan (default: folder masukan)")
    p_eval.add_argument(
        "--case-sensitive",
        action="store_true",
        help="jangan ubah ke huruf kecil sebelum membandingkan",
    )
    p_eval.add_argument(
        "--strip-punct",
        action="store_true",
        help="buang tanda baca sebelum membandingkan (laporkan ini di skripsi)",
    )

    return parser


# --------------------------------------------------------------------------- #
# Penyusunan konfigurasi
# --------------------------------------------------------------------------- #


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)

    if getattr(args, "engine", None):
        cfg.ocr_engine = args.engine
    if getattr(args, "dpi", None):
        cfg.preprocess.dpi = args.dpi
    if getattr(args, "chunk_size", None):
        cfg.chunk.target_chars = args.chunk_size
        cfg.chunk.max_chars = max(args.chunk_size + 400, int(args.chunk_size * 1.6))
    if getattr(args, "overlap", None) is not None:
        cfg.chunk.overlap_chars = args.overlap
    if getattr(args, "device", None):
        cfg.paddle.device = args.device
    if getattr(args, "no_tables", False):
        cfg.native.table_strategy = "none"
        cfg.tesseract.detect_tables = False
        cfg.docling.do_table_structure = False
        cfg.paddle.use_table_recognition = False
        cfg.output.csv_tables = False
        cfg.output.xlsx_tables = False
    if getattr(args, "force_ocr", False) and getattr(args, "force_native", False):
        raise SystemExit("--force-ocr dan --force-native tidak bisa dipakai bersamaan")
    if getattr(args, "force_ocr", False):
        cfg.detect.force_route = ROUTE_OCR
    if getattr(args, "force_native", False):
        cfg.detect.force_route = ROUTE_NATIVE

    cfg.verbose = not args.quiet
    return cfg


# --------------------------------------------------------------------------- #
# Perintah
# --------------------------------------------------------------------------- #


def cmd_engines(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    status = check_all(cfg)

    print("\nEngine ekstraksi yang terdeteksi\n" + "=" * 68)
    for name, avail in status.items():
        mark = "TERSEDIA  " if avail.ok else "tidak ada "
        ver = f" (v{avail.version})" if avail.version else ""
        print(f"\n[{mark}] {name}{ver}")
        from .engines import get_engine

        try:
            print(f"            {get_engine(name, cfg).description}")
        except Exception:
            pass
        if not avail.ok:
            print(f"            sebab   : {avail.message}")
            if avail.install_hint:
                print(f"            pasang  : {avail.install_hint}")

    usable = [n for n, a in status.items() if a.ok and n != "native"]
    print("\n" + "=" * 68)
    if usable:
        print(f"Engine OCR siap pakai: {', '.join(usable)}")
    else:
        print(
            "TIDAK ADA engine OCR yang siap. Halaman hasil scan tidak akan "
            "terbaca.\nPaling cepat: pasang Tesseract (lihat "
            "docs/INSTALL-WINDOWS.md)."
        )
    print(
        "\nCatatan: engine `native` bukan OCR — ia membaca text layer PDF dan "
        "selalu\ndipakai untuk halaman digital karena akurasinya penuh.\n"
    )
    return 0 if status.get("native", None) and status["native"].ok else 1


def cmd_diagnose(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    pdfs = discover_pdfs(args.inputs, recursive=True)
    if not pdfs:
        print("Tidak ada PDF ditemukan.")
        return 1

    rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    totals = {ROUTE_NATIVE: 0, ROUTE_OCR: 0, ROUTE_EMPTY: 0}

    print(f"\nMenyurvei {len(pdfs)} berkas PDF (tanpa OCR, hanya pengukuran)\n")
    header = f"{'berkas':<44} {'hlm':>4} {'digital':>8} {'scan':>5} {'kosong':>7} {'jenis':>10}"
    print(header)
    print("-" * len(header))

    for pdf in pdfs:
        probe = probe_document(pdf, cfg.detect, cfg.pages)
        if probe.error:
            print(f"{pdf.name[:44]:<44} {'-':>4}  GAGAL: {probe.error}")
            rows.append({"berkas": pdf.name, "galat": probe.error})
            continue
        counts = probe.route_counts
        for key in totals:
            totals[key] += counts.get(key, 0)
        print(
            f"{pdf.name[:44]:<44} {probe.n_pages:>4} "
            f"{counts[ROUTE_NATIVE]:>8} {counts[ROUTE_OCR]:>5} "
            f"{counts[ROUTE_EMPTY]:>7} {probe.kind:>10}"
        )
        rows.append(
            {
                "berkas": pdf.name,
                "jml_halaman": probe.n_pages,
                "digital": counts[ROUTE_NATIVE],
                "scan": counts[ROUTE_OCR],
                "kosong": counts[ROUTE_EMPTY],
                "jenis": probe.kind,
                "terenkripsi": probe.encrypted,
                "galat": "",
            }
        )
        for pp in probe.pages:
            row = pp.as_row()
            row["berkas"] = pdf.name
            page_rows.append(row)
            if args.per_page:
                print(
                    f"    hlm {pp.page:>4} {pp.route:<7} "
                    f"{pp.n_chars:>6} kar  {'; '.join(pp.reasons)}"
                )

    total_pages = sum(totals.values())
    print("-" * len(header))
    print(
        f"{'TOTAL':<44} {total_pages:>4} {totals[ROUTE_NATIVE]:>8} "
        f"{totals[ROUTE_OCR]:>5} {totals[ROUTE_EMPTY]:>7}"
    )

    if total_pages:
        pct_ocr = totals[ROUTE_OCR] / total_pages * 100
        print(f"\n{pct_ocr:.1f}% halaman memerlukan OCR.")
        if totals[ROUTE_OCR] == 0:
            print(
                "Seluruh korpus digital — engine OCR berat tidak diperlukan. "
                "Jalankan: ingest --force-native (jauh lebih cepat)."
            )
        elif pct_ocr < 15:
            print(
                "Porsi halaman scan kecil. Tesseract sudah memadai; engine "
                "neural hanya perlu bila tabel pada halaman scan itu penting."
            )
        else:
            print(
                "Porsi halaman scan besar. Pakai engine neural (docling/paddle) "
                "untuk kualitas tabel, dan pertimbangkan Kaggle untuk batch besar."
            )

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(out / "diagnosa-dokumen.csv", rows)
        _write_csv(out / "diagnosa-halaman.csv", page_rows)
        print(f"\nLaporan disimpan di {out}")
    print()
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    log = get_logger()

    if args.pages:
        # Butuh jumlah halaman untuk validasi; pakai batas besar lalu
        # pipeline akan memfilter per dokumen.
        cfg.pages = parse_page_spec(args.pages, 100_000)
        log.info("hanya memproses halaman: %s", cfg.pages)

    pipeline = Pipeline(cfg)
    run = pipeline.run(args.inputs, args.out, recursive=not args.no_recursive)

    print("\n" + "=" * 68)
    print("RINGKASAN")
    print("=" * 68)
    print(f"Dokumen diproses      : {len(run.documents)}")
    print(f"Halaman diproses      : {run.total_pages}")
    print(f"Tabel diekstrak       : {run.total_tables}")
    print(f"Chunk dihasilkan      : {run.total_chunks}")
    print(f"Halaman perlu tinjauan: {run.pages_needing_review}")
    print(f"Waktu total           : {run.elapsed_s:.1f} s")
    if run.errors:
        print(f"\n{len(run.errors)} dokumen GAGAL:")
        for err in run.errors:
            print(f"  - {err}")
    print("\nKeluaran:")
    for key, value in run.outputs.items():
        print(f"  {key:<18}: {value}")
    print(f"  {'markdown':<18}: {Path(args.out) / 'markdown'}")
    print(f"  {'tabel (csv/xlsx)':<18}: {Path(args.out) / 'tabel'}")
    if run.pages_needing_review:
        print(
            f"\nLangkah berikutnya: buka {Path(args.out) / 'qa' / 'laporan-qa.md'} "
            "dan tinjau halaman yang ditandai sebelum di-ingest ke vector store."
        )
    print()
    return 1 if run.errors and not run.documents else 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Siapkan sampel halaman untuk transkripsi manual (acuan CER/WER).

    Sampling distratifikasi menurut rute (native/OCR) supaya kedua jalur
    terukur. Mengukur akurasi hanya pada halaman digital akan menghasilkan
    CER mendekati nol dan menyesatkan.
    """
    cfg = config_from_args(args)
    log = get_logger()
    rng = random.Random(args.seed)

    pdfs = discover_pdfs(args.inputs, recursive=True)
    if not pdfs:
        print("Tidak ada PDF ditemukan.")
        return 1

    candidates: list[tuple[Path, int, str]] = []
    for pdf in pdfs:
        probe = probe_document(pdf, cfg.detect, cfg.pages)
        if probe.error:
            continue
        for pp in probe.pages:
            if pp.route != ROUTE_EMPTY:
                candidates.append((pdf, pp.page, pp.route))

    if not candidates:
        print("Tidak ada halaman berisi konten.")
        return 1

    ocr_pool = [c for c in candidates if c[2] == ROUTE_OCR]
    nat_pool = [c for c in candidates if c[2] == ROUTE_NATIVE]
    n = min(args.n_samples, len(candidates))
    # Prioritaskan halaman OCR: di situlah galat sebenarnya terjadi.
    n_ocr = min(len(ocr_pool), max(1, round(n * 0.7))) if ocr_pool else 0
    n_nat = min(len(nat_pool), n - n_ocr)
    if n_ocr + n_nat < n:
        n_ocr = min(len(ocr_pool), n - n_nat)

    chosen = rng.sample(ocr_pool, n_ocr) + rng.sample(nat_pool, n_nat)
    chosen.sort(key=lambda c: (str(c[0]), c[1]))
    log.info(
        "sampel terpilih: %d halaman (%d OCR, %d native) dari %d kandidat",
        len(chosen),
        n_ocr,
        n_nat,
        len(candidates),
    )

    out = Path(args.out)
    (out / "gambar").mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.inputs)
    pipeline = Pipeline(cfg)
    index_rows: list[dict[str, Any]] = []

    for pdf, page_no, route in chosen:
        stem = f"{pdf.stem[:40]}_hlm{page_no:04d}"
        # --- gambar halaman untuk dibaca manusia ---
        try:
            import pymupdf

            doc = pymupdf.open(str(pdf))
            try:
                pix = doc[page_no - 1].get_pixmap(dpi=args.image_dpi, alpha=False)
                img_path = out / "gambar" / f"{stem}.png"
                pix.save(str(img_path))
            finally:
                doc.close()
        except Exception as exc:
            log.warning("gagal merender %s hlm %d: %s", pdf.name, page_no, exc)
            img_path = None

        # --- hipotesis dari pipeline ---
        cfg_single = cfg.override(pages=[page_no])
        single = Pipeline(cfg_single)
        try:
            doc_res = single.process_document(pdf, metadata.get(pdf.name.lower()))
            hyp = "\n\n".join(p.text for p in doc_res.pages)
            engine = doc_res.pages[0].engine if doc_res.pages else "-"
        except Exception as exc:
            log.warning("gagal mengekstrak %s hlm %d: %s", pdf.name, page_no, exc)
            hyp, engine = "", "-"
        finally:
            single.close()

        (out / f"{stem}.hyp.txt").write_text(hyp, encoding="utf-8")
        ref_path = out / f"{stem}.ref.txt"
        if not ref_path.exists():
            ref_path.write_text("", encoding="utf-8")

        index_rows.append(
            {
                "sampel": stem,
                "berkas": pdf.name,
                "halaman": page_no,
                "rute": route,
                "engine": engine,
                "karakter_hipotesis": len(hyp),
                "gambar": str(img_path.name) if img_path else "",
                "acuan_sudah_diisi": False,
            }
        )

    _write_csv(out / "indeks-sampel.csv", index_rows)
    (out / "CARA-ANOTASI.md").write_text(_ANNOTATION_GUIDE, encoding="utf-8")
    pipeline.close()

    print(f"\n{len(chosen)} sampel disiapkan di {out}")
    print(
        "Langkah berikutnya:\n"
        f"  1. Buka gambar di {out / 'gambar'} dan transkripsikan ke berkas "
        "*.ref.txt yang bersesuaian\n"
        f"  2. Baca {out / 'CARA-ANOTASI.md'} agar aturan transkripsi konsisten\n"
        f"  3. Jalankan: {PROG} eval {out}\n"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Bukan folder: {folder}")
        return 1

    pairs: list[tuple[str, Path, Path]] = []
    for ref in sorted(folder.glob("*.ref.txt")):
        stem = ref.name[: -len(".ref.txt")]
        hyp = folder / f"{stem}.hyp.txt"
        if hyp.exists():
            pairs.append((stem, ref, hyp))

    if not pairs:
        print(
            f"Tidak ada pasangan *.ref.txt / *.hyp.txt di {folder}.\n"
            f"Jalankan `{PROG} sample` terlebih dahulu."
        )
        return 1

    results = []
    skipped: list[str] = []
    for stem, ref_path, hyp_path in pairs:
        ref = ref_path.read_text(encoding="utf-8").strip()
        hyp = hyp_path.read_text(encoding="utf-8").strip()
        if not ref:
            skipped.append(stem)
            continue
        results.append(
            compute_cer_wer(
                ref,
                hyp,
                name=stem,
                lowercase=not args.case_sensitive,
                strip_punct=args.strip_punct,
            )
        )

    if not results:
        print(
            f"Semua {len(skipped)} berkas acuan masih kosong. "
            "Transkripsikan dulu secara manual."
        )
        return 1

    agg = aggregate_eval(results)
    norm = (
        f"lowercase={'tidak' if args.case_sensitive else 'ya'}, "
        f"tanda_baca_dibuang={'ya' if args.strip_punct else 'tidak'}"
    )

    print("\nHasil pengukuran akurasi OCR (CER/WER)")
    print("=" * 68)
    print(f"Normalisasi: {norm}")
    print(f"Sampel dinilai: {len(results)}" + (f" (dilewati kosong: {len(skipped)})" if skipped else ""))
    print("-" * 68)
    print(f"{'sampel':<40} {'CER':>8} {'WER':>8} {'kar.acuan':>10}")
    print("-" * 68)
    for r in sorted(results, key=lambda x: x.cer, reverse=True):
        print(f"{r.name[:40]:<40} {r.cer:>8.4f} {r.wer:>8.4f} {r.n_chars_ref:>10}")
    print("-" * 68)
    print(f"{'AGREGAT (mikro)':<40} {agg.cer:>8.4f} {agg.wer:>8.4f} {agg.n_chars_ref:>10}")
    print(
        f"\nAkurasi karakter = {(1 - agg.cer) * 100:.2f}%  |  "
        f"akurasi kata = {(1 - agg.wer) * 100:.2f}%"
    )
    print(
        "\nCatatan pelaporan: agregat dihitung mikro (bobot menurut panjang "
        "acuan),\nbukan rata-rata CER per halaman. Sebutkan hal ini beserta "
        "perlakuan\nnormalisasi di atas saat melaporkan angka di skripsi.\n"
    )

    out = Path(args.out) if args.out else folder
    out.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row() for r in sorted(results, key=lambda x: x.name)]
    rows.append(agg.as_row())
    _write_csv(out / "hasil-cer-wer.csv", rows)
    (out / "hasil-cer-wer.json").write_text(
        json.dumps(
            {
                "normalisasi": norm,
                "jml_sampel": len(results),
                "dilewati_kosong": skipped,
                "agregat": agg.as_row(),
                "per_sampel": [r.as_row() for r in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Laporan disimpan: {out / 'hasil-cer-wer.csv'}\n")
    return 0


# --------------------------------------------------------------------------- #

_ANNOTATION_GUIDE = """# Cara transkripsi manual untuk pengukuran CER/WER

Tujuan berkas `*.ref.txt` adalah menjadi **acuan (ground truth)**: apa yang
benar-benar tertulis pada halaman. Berkas `*.hyp.txt` adalah keluaran mesin —
jangan diedit, karena itulah yang sedang dinilai.

## Aturan agar hasil konsisten

1. Transkripsikan dari **gambar** di folder `gambar/`, bukan dari `*.hyp.txt`.
   Membaca keluaran mesin lebih dahulu menimbulkan bias dan membuat CER
   terlihat lebih baik daripada kenyataan.
2. Tulis teks **apa adanya**, termasuk salah ketik yang ada di dokumen asli.
   Yang dinilai adalah kemampuan mesin membaca halaman, bukan mutu dokumen.
3. Pertahankan urutan baca: kiri ke kanan, atas ke bawah. Untuk dua kolom,
   selesaikan kolom kiri lebih dulu.
4. **Tabel**: tulis satu baris tabel menjadi satu baris teks, antar sel
   dipisahkan satu spasi. Jangan menambahkan karakter `|`. Perlakuan ini
   harus sama untuk semua sampel.
5. **Kop surat, nomor halaman, footer**: sertakan, karena mesin juga
   membacanya.
6. **Gambar/logo/tanda tangan**: jangan dideskripsikan, lewati saja.
7. Satu baris kosong antar paragraf. Jangan memakai tanda hubung di akhir
   baris; tulis kata secara utuh.
8. Jangan memperbaiki singkatan dan jangan menambah tanda baca yang tidak
   ada di halaman.

## Berapa sampel yang cukup

Untuk skripsi, 10-15 halaman yang distratifikasi (sebagian halaman digital,
sebagian halaman hasil scan) sudah memadai untuk melaporkan CER/WER beserta
sebarannya. Yang penting adalah **melaporkan jumlah sampel dan cara
pemilihannya**, bukan jumlahnya besar.

Catatan: halaman digital (`rute = native`) seharusnya memberi CER mendekati
nol. Itu bukan keberhasilan OCR — itu bukti bahwa routing per halaman bekerja
dan halaman tersebut memang tidak perlu di-OCR. Laporkan kedua kelompok
secara terpisah, jangan digabung menjadi satu angka tunggal.

## Setelah selesai

    ocrpipe eval <folder-ini>

Tambahkan `--strip-punct` bila ingin melaporkan CER tanpa tanda baca.
Apa pun pilihannya, sebutkan di laporan — angka dari dua perlakuan yang
berbeda tidak dapat dibandingkan.
"""


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=not args.quiet, logfile=args.log_file)

    handlers = {
        "ingest": cmd_ingest,
        "diagnose": cmd_diagnose,
        "engines": cmd_engines,
        "sample": cmd_sample,
        "eval": cmd_eval,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover
        parser.error(f"perintah tidak dikenal: {args.command}")
        return 2

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nDibatalkan oleh pengguna.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nGalat: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        get_logger().exception("kegagalan tak terduga")
        print(f"\nKegagalan tak terduga: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
