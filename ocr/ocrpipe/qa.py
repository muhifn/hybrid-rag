"""Penjaminan mutu (QA) hasil OCR.

Dua fungsi yang berbeda, keduanya dibutuhkan skripsi:

A. PENYARINGAN OTOMATIS (otomatis, seluruh korpus)
   Metrik murah yang dihitung untuk setiap halaman: confidence engine,
   jumlah karakter, rasio token "sampah", dan rasio token yang dikenali
   sebagai kata Indonesia/Inggris. Halaman yang melewati ambang ditandai
   `needs_review` supaya tidak lolos ke vector store tanpa dilihat manusia.
   Ini BUKAN pengukuran akurasi — ini deteksi anomali.

B. PENGUKURAN AKURASI (manual, sampel)
   CER dan WER terhadap teks acuan (ground truth) yang ditranskripsi
   manual pada sampel halaman. Hanya ini yang sah dilaporkan sebagai
   "akurasi OCR" di Bab 4. Perintah `ocrpipe sample` menyiapkan sampelnya
   dan `ocrpipe eval` menghitung metriknya.

Rasio leksikon memakai daftar kata frekuensi tinggi, bukan kamus lengkap.
Tujuannya mendeteksi keluaran yang rusak total (mojibake, halaman terbalik),
bukan menilai ejaan. Karena itu nilainya pada teks Indonesia normal berkisar
0,3-0,6 dan ambang default sengaja dipasang rendah (0,25).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import QAConfig
from .models import DocResult, PageResult, ROUTE_EMPTY, ROUTE_NATIVE

# --------------------------------------------------------------------------- #
# Leksikon kecil untuk deteksi anomali
# --------------------------------------------------------------------------- #

_ID_WORDS = """
ada adalah akademik akan akhir aktif alamat anak angka antara apabila atas atau
bab bagi bagian bahwa baik banyak baru batas beban bentuk berada berdasarkan
berikut berlaku bersangkutan biaya bidang bila bimbingan bisa buku bukan bulan
cara catatan cuti dalam dan dapat daftar dari dasar data dengan dekan demikian
dengan di dia diberikan diisi dilakukan dimaksud diri dokumen dosen dua dst
fakultas form formulir gelar guna hal halaman harus hasil hari hingga huruf
ijazah ilmu indonesia informasi ini isi itu jadi jadwal jam jenis jika jumlah
juga jurusan kalender kampus kartu karena karya kegiatan kelas keluar kemahasiswaan
kembali kepada keperluan keputusan kerja ketentuan ketua khusus kode kode kuliah
kurikulum lain lama lampiran layanan lebih lembar libur lulus mahasiswa maka
masa masuk mata matakuliah melakukan melalui memberikan memenuhi mendapat
mengajukan mengikuti menjadi menurut merupakan mulai nama negeri nilai nomor
oleh orang pada paling panduan pasal pelaksanaan pembayaran pembimbing pendaftaran
pendidikan penelitian pengajuan pengisian perguruan periode permohonan persen
pertama perubahan prodi program proposal rektor rencana sampai sarjana sebagai
sebagaimana sebelum secara sedang segala selama semester seminar sesuai setelah
setiap sidang sistem skripsi smt studi surat syarat tahap tahun tanggal tata
teknik telah tentang terdaftar terhadap terkait tersebut tidak tingkat total
tugas ujian universitas untuk usulan utama wajib waktu wisuda yaitu yang
""".split()

_EN_WORDS = """
and are as at be by credit date faculty for from has have in is it name no not
number of on or page program semester student study subject the this to total
university with year
""".split()

LEXICON: frozenset[str] = frozenset(_ID_WORDS) | frozenset(_EN_WORDS)

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ]+(?:[-'][A-Za-zÀ-ÿ]+)*")
_ALL_TOKEN_RE = re.compile(r"\S+")
# Token "sampah" khas OCR gagal: campuran huruf-angka tak lazim, simbol
# bertumpuk, atau rentetan konsonan tanpa vokal.
_JUNK_PATTERNS = (
    re.compile(r"^[^\w\s]{2,}$"),  # "|||", "~~#"
    re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{1,}[^\w\s]+[A-Za-z\d]*$"),
    re.compile(r"^[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}$"),
)


# --------------------------------------------------------------------------- #
# Metrik per halaman
# --------------------------------------------------------------------------- #


def text_metrics(text: str) -> dict[str, Any]:
    tokens_all = _ALL_TOKEN_RE.findall(text)
    tokens_alpha = _TOKEN_RE.findall(text)
    n_all = len(tokens_all)
    n_alpha = len(tokens_alpha)

    junk = sum(1 for t in tokens_all if any(p.match(t) for p in _JUNK_PATTERNS))
    known = sum(1 for t in tokens_alpha if t.lower() in LEXICON)
    avg_len = (sum(len(t) for t in tokens_alpha) / n_alpha) if n_alpha else 0.0

    return {
        "jml_karakter": len(text),
        "jml_token": n_all,
        "jml_kata": n_alpha,
        "rasio_sampah": round(junk / n_all, 4) if n_all else 0.0,
        "rasio_leksikon": round(known / n_alpha, 4) if n_alpha else 0.0,
        "panjang_kata_rata2": round(avg_len, 2),
    }


def evaluate_page(page: PageResult, cfg: QAConfig) -> PageResult:
    """Hitung metrik dan isi flag `needs_review` pada satu halaman."""
    text = page.text
    metrics = text_metrics(text)
    metrics["confidence_rata2"] = page.mean_confidence
    metrics["jml_tabel"] = len(page.tables)
    metrics["jml_blok"] = len(page.blocks)
    metrics["detik"] = page.elapsed_s
    page.metrics.update(metrics)

    reasons: list[str] = []

    if page.route == ROUTE_EMPTY:
        page.needs_review = False
        page.review_reasons = []
        return page

    if page.warnings:
        reasons.extend(f"peringatan engine: {w}" for w in page.warnings)

    if metrics["jml_karakter"] < cfg.min_chars_per_page:
        reasons.append(
            f"teks sangat sedikit ({metrics['jml_karakter']} karakter < "
            f"{cfg.min_chars_per_page})"
        )

    conf = page.mean_confidence
    if conf is not None and conf < cfg.min_mean_confidence:
        reasons.append(
            f"confidence rata-rata {conf:.1f} di bawah ambang {cfg.min_mean_confidence}"
        )

    # Metrik tekstual hanya bermakna bila halaman punya cukup kata.
    if metrics["jml_kata"] >= 20:
        if metrics["rasio_sampah"] > cfg.max_garbage_ratio:
            reasons.append(
                f"rasio token tidak wajar {metrics['rasio_sampah']:.1%} > "
                f"{cfg.max_garbage_ratio:.1%}"
            )
        if metrics["rasio_leksikon"] < cfg.min_lexicon_ratio:
            reasons.append(
                f"hanya {metrics['rasio_leksikon']:.1%} token dikenali sebagai "
                f"kata umum (ambang {cfg.min_lexicon_ratio:.1%}) — "
                "kemungkinan keluaran rusak atau bahasa tidak sesuai"
            )

    if cfg.flag_irregular_tables:
        for t in page.tables:
            if t.warnings:
                reasons.append(f"tabel {t.slug}: {'; '.join(t.warnings)}")

    page.review_reasons = reasons
    page.needs_review = bool(reasons)
    return page


def evaluate_document(doc: DocResult, cfg: QAConfig) -> DocResult:
    for page in doc.pages:
        evaluate_page(page, cfg)
    return doc


def document_summary(doc: DocResult) -> dict[str, Any]:
    pages = [p for p in doc.pages if p.route != ROUTE_EMPTY]
    chars = sum(int(p.metrics.get("jml_karakter", 0)) for p in pages)
    confs = [p.mean_confidence for p in pages if p.mean_confidence is not None]
    lex = [
        float(p.metrics.get("rasio_leksikon", 0.0))
        for p in pages
        if int(p.metrics.get("jml_kata", 0)) >= 20
    ]
    return {
        "doc_id": doc.meta.doc_id,
        "judul": doc.meta.title,
        "berkas": Path(doc.meta.source_file).name,
        "versi": doc.meta.version,
        "jml_halaman": doc.n_pages,
        "halaman_diproses": len(doc.pages),
        "rute": doc.route_counts(),
        "engine": doc.engine_counts(),
        "total_karakter": chars,
        "karakter_per_halaman": round(chars / len(pages), 1) if pages else 0,
        "jml_tabel": len(doc.tables),
        "confidence_rata2": round(sum(confs) / len(confs), 2) if confs else None,
        "rasio_leksikon_rata2": round(sum(lex) / len(lex), 4) if lex else None,
        "halaman_perlu_tinjauan": doc.pages_needing_review,
        "jml_perlu_tinjauan": len(doc.pages_needing_review),
        "detik": round(doc.elapsed_s, 2),
        "peringatan_dokumen": doc.doc_warnings,
    }


# --------------------------------------------------------------------------- #
# CER / WER
# --------------------------------------------------------------------------- #


def _levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Jarak edit. Memakai rapidfuzz bila ada (jauh lebih cepat)."""
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore

        return int(Levenshtein.distance(list(a), list(b)))
    except ImportError:
        pass
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,  # hapus
                cur[j - 1] + 1,  # sisip
                prev[j - 1] + (ca != cb),  # ganti
            )
        prev = cur
    return prev[-1]


def normalize_for_eval(text: str, *, lowercase: bool = True, strip_punct: bool = False) -> str:
    """Normalisasi ringan sebelum menghitung CER/WER.

    Normalisasi HARUS dilaporkan di skripsi: CER pada teks yang di-lowercase
    dan dibuang tanda bacanya akan selalu terlihat lebih baik daripada CER
    pada teks mentah, jadi perbandingan antar penelitian hanya sah bila
    perlakuannya disebutkan.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = re.sub(r"[^\w\s]", "", text)
    return text


@dataclass
class EvalResult:
    name: str = ""
    cer: float = 0.0
    wer: float = 0.0
    n_chars_ref: int = 0
    n_words_ref: int = 0
    char_edits: int = 0
    word_edits: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "sampel": self.name,
            "CER": round(self.cer, 4),
            "WER": round(self.wer, 4),
            "karakter_acuan": self.n_chars_ref,
            "kata_acuan": self.n_words_ref,
            "edit_karakter": self.char_edits,
            "edit_kata": self.word_edits,
        }


def compute_cer_wer(
    reference: str,
    hypothesis: str,
    *,
    name: str = "",
    lowercase: bool = True,
    strip_punct: bool = False,
) -> EvalResult:
    ref = normalize_for_eval(reference, lowercase=lowercase, strip_punct=strip_punct)
    hyp = normalize_for_eval(hypothesis, lowercase=lowercase, strip_punct=strip_punct)

    ref_words = ref.split()
    hyp_words = hyp.split()

    char_edits = _levenshtein(ref, hyp)
    word_edits = _levenshtein(ref_words, hyp_words)

    return EvalResult(
        name=name,
        cer=char_edits / len(ref) if ref else (1.0 if hyp else 0.0),
        wer=word_edits / len(ref_words) if ref_words else (1.0 if hyp_words else 0.0),
        n_chars_ref=len(ref),
        n_words_ref=len(ref_words),
        char_edits=char_edits,
        word_edits=word_edits,
    )


def aggregate_eval(results: Sequence[EvalResult]) -> EvalResult:
    """Agregasi mikro (bobot menurut panjang acuan), bukan rata-rata sederhana.

    Rata-rata sederhana atas CER per halaman memberi bobot terlalu besar pada
    halaman pendek; agregasi mikro adalah praktik standar pada pelaporan ASR
    maupun OCR.
    """
    total_chars = sum(r.n_chars_ref for r in results)
    total_words = sum(r.n_words_ref for r in results)
    char_edits = sum(r.char_edits for r in results)
    word_edits = sum(r.word_edits for r in results)
    return EvalResult(
        name=f"AGREGAT ({len(results)} sampel)",
        cer=char_edits / total_chars if total_chars else 0.0,
        wer=word_edits / total_words if total_words else 0.0,
        n_chars_ref=total_chars,
        n_words_ref=total_words,
        char_edits=char_edits,
        word_edits=word_edits,
    )


# --------------------------------------------------------------------------- #
# Laporan
# --------------------------------------------------------------------------- #


def write_qa_report(
    docs: Iterable[DocResult],
    out_dir: str | Path,
    *,
    engine_notes: Sequence[str] = (),
    config_summary: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Tulis laporan QA dalam Markdown + CSV per halaman.

    Returns:
        (path_markdown, path_csv)
    """
    docs = list(docs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "laporan-qa.md"
    csv_path = out_dir / "qa-per-halaman.csv"

    summaries = [document_summary(d) for d in docs]
    total_pages = sum(s["halaman_diproses"] for s in summaries)
    total_review = sum(s["jml_perlu_tinjauan"] for s in summaries)
    total_tables = sum(s["jml_tabel"] for s in summaries)
    total_chars = sum(s["total_karakter"] for s in summaries)

    lines: list[str] = []
    lines.append("# Laporan Penjaminan Mutu Ekstraksi Dokumen\n")
    lines.append(
        "Laporan ini memuat metrik penyaringan otomatis. Metrik di sini "
        "mendeteksi **anomali**, bukan mengukur akurasi. Angka akurasi OCR "
        "(CER/WER) hanya sah diperoleh dari `ocrpipe eval` terhadap transkrip "
        "acuan yang dibuat manual.\n"
    )

    lines.append("## Ringkasan\n")
    lines.append(f"- Dokumen diproses: **{len(docs)}**")
    lines.append(f"- Halaman diproses: **{total_pages}**")
    lines.append(f"- Tabel diekstrak: **{total_tables}**")
    lines.append(f"- Total karakter: **{total_chars:,}**")
    pct = (total_review / total_pages * 100) if total_pages else 0.0
    lines.append(
        f"- Halaman perlu tinjauan manual: **{total_review}** ({pct:.1f}%)\n"
    )

    route_total: dict[str, int] = {}
    engine_total: dict[str, int] = {}
    for s in summaries:
        for k, v in s["rute"].items():
            route_total[k] = route_total.get(k, 0) + v
        for k, v in s["engine"].items():
            engine_total[k] = engine_total.get(k, 0) + v
    lines.append(
        "- Sebaran rute: "
        + ", ".join(f"`{k}`={v}" for k, v in sorted(route_total.items()))
    )
    lines.append(
        "- Sebaran engine: "
        + ", ".join(f"`{k}`={v}" for k, v in sorted(engine_total.items()))
        + "\n"
    )

    if engine_notes:
        lines.append("## Pemilihan engine\n")
        lines.extend(f"- {n}" for n in engine_notes)
        lines.append("")

    if config_summary:
        lines.append("## Konfigurasi efektif (ringkas)\n")
        for k, v in config_summary.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    lines.append("## Per dokumen\n")
    lines.append(
        "| Dokumen | Hlm | Native | OCR | Kosong | Tabel | Karakter/hlm | "
        "Conf. | Leksikon | Perlu tinjauan |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in summaries:
        r = s["rute"]
        conf = f"{s['confidence_rata2']:.1f}" if s["confidence_rata2"] is not None else "-"
        lex = f"{s['rasio_leksikon_rata2']:.2f}" if s["rasio_leksikon_rata2"] is not None else "-"
        lines.append(
            f"| {s['berkas']} | {s['halaman_diproses']} | {r.get(ROUTE_NATIVE, 0)} | "
            f"{r.get('ocr', 0)} | {r.get(ROUTE_EMPTY, 0)} | {s['jml_tabel']} | "
            f"{s['karakter_per_halaman']} | {conf} | {lex} | {s['jml_perlu_tinjauan']} |"
        )
    lines.append("")

    flagged = [(d, p) for d in docs for p in d.pages if p.needs_review]
    lines.append("## Halaman yang perlu ditinjau manual\n")
    if not flagged:
        lines.append("Tidak ada halaman yang melewati ambang. ")
        lines.append(
            "Catatan: ini berarti tidak ada **anomali** yang terdeteksi, bukan "
            "berarti akurasinya sudah terbukti.\n"
        )
    else:
        lines.append(
            "Urutkan penanganan dari atas: halaman di awal daftar punya alasan "
            "paling banyak.\n"
        )
        flagged.sort(key=lambda it: len(it[1].review_reasons), reverse=True)
        for doc, page in flagged[:200]:
            lines.append(
                f"- **{Path(doc.meta.source_file).name} hlm {page.page}** "
                f"(`{page.route}`/`{page.engine}`): "
                + "; ".join(page.review_reasons)
            )
        if len(flagged) > 200:
            lines.append(f"- … dan {len(flagged) - 200} halaman lain (lihat CSV)")
        lines.append("")

    lines.append("## Cara menindaklanjuti\n")
    lines.append(
        "1. Buka `qa-per-halaman.csv`, saring `perlu_tinjauan = True`.\n"
        "2. Untuk halaman hasil scan dengan confidence rendah: naikkan "
        "`preprocess.dpi` ke 400, atau aktifkan `preprocess.binarize` bila "
        "engine-nya Tesseract.\n"
        "3. Untuk halaman dengan `rasio_leksikon` sangat rendah: periksa "
        "orientasi halaman dan bahasa engine (`tesseract.lang`, `paddle.lang`).\n"
        "4. Untuk tabel dengan peringatan kolom tidak konsisten: periksa berkas "
        "XLSX-nya; perbaiki manual di sana, lalu jalankan ulang hanya halaman "
        "tersebut dengan `--pages`.\n"
        "5. Setelah stabil, ambil sampel untuk pengukuran akurasi: "
        "`ocrpipe sample` lalu transkripsi manual dan `ocrpipe eval`.\n"
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # --- CSV per halaman ---
    fieldnames = [
        "doc_id",
        "berkas",
        "halaman",
        "rute",
        "engine",
        "jml_karakter",
        "jml_kata",
        "jml_tabel",
        "jml_blok",
        "confidence_rata2",
        "rasio_sampah",
        "rasio_leksikon",
        "panjang_kata_rata2",
        "detik",
        "perlu_tinjauan",
        "alasan",
        "praproses",
        "dpi",
        "sudut_deskew",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for doc in docs:
            for page in doc.pages:
                row = {
                    "doc_id": doc.meta.doc_id,
                    "berkas": Path(doc.meta.source_file).name,
                    "halaman": page.page,
                    "rute": page.route,
                    "engine": page.engine,
                    "perlu_tinjauan": page.needs_review,
                    "alasan": " | ".join(page.review_reasons),
                }
                row.update({k: v for k, v in page.metrics.items() if k in fieldnames})
                writer.writerow(row)

    return md_path, csv_path
