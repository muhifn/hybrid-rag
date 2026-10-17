# ocrpipe — Pipeline OCR Dokumen Akademik untuk RAG

Alat untuk mengubah berkas **PDF campuran** (sebagian digital, sebagian hasil
pindaian) menjadi data bersih yang siap dimasukkan ke sistem *Retrieval-Augmented
Generation* (RAG): chatbot layanan informasi akademik berbasis LLM lokal.

Dibuat untuk kebutuhan skripsi: korpus dokumen akademik (panduan, kalender,
SK, daftar biaya, kurikulum) yang bentuknya tidak seragam, banyak **tabel**,
dan sebagian hanya tersedia sebagai hasil **scan** bertanda tangan.

---

## Apa yang membuat alat ini berbeda dari "OCR biasa"

1. **Keputusan per halaman, bukan per dokumen.** Setiap halaman diperiksa:
   kalau sudah punya *text layer* yang sehat, teksnya dibaca langsung
   (akurasi 100%, tanpa OCR); hanya halaman hasil scan yang dikirim ke engine
   OCR yang berat. Satu PDF boleh separuh digital separuh scan — ditangani
   otomatis.

2. **Tabel diperlakukan sebagai warga kelas satu.** Tabel diekstrak sebagai
   struktur (bukan teks acak), diekspor ke Markdown, **CSV per tabel**, dan
   **XLSX multi-sheet**. Saat di-chunk untuk RAG, tabel tidak pernah dipotong
   di tengah; tabel panjang dipecah per baris dengan **header diulang**.

3. **Siap sitasi (dokumen, halaman, versi).** Setiap chunk membawa metadata
   lengkap sehingga chatbot bisa menjawab *"menurut Panduan Akademik 2026
   halaman 12…"* tanpa query tambahan.

4. **Dapat dipertanggungjawabkan di sidang.** Ada laporan QA, jejak
   reproduktibilitas (`manifest.json` berisi versi semua pustaka + konfigurasi
   efektif), serta alat ukur akurasi OCR (**CER/WER**) terhadap transkrip
   manual.

---

## Alur kerja yang disarankan

```
                         ┌──────────────────────────┐
   1. Survei dulu  ───►  │  python cek_pdf.py <dir>  │  (hanya butuh pymupdf)
                         └──────────────────────────┘
                                     │  tahu: digital / scan / campuran
                                     ▼
                         ┌──────────────────────────┐
   2. Diagnosa rinci ─►  │ run_ocr.py diagnose <dir> │
                         └──────────────────────────┘
                                     ▼
                         ┌──────────────────────────┐
   3. Ekstraksi      ─►  │ run_ocr.py ingest <dir>   │ ─► markdown/ jsonl/ tabel/ qa/
                         └──────────────────────────┘
                                     ▼
                         ┌──────────────────────────┐
   4. Ukur akurasi   ─►  │ run_ocr.py sample / eval  │ ─► CER / WER
                         └──────────────────────────┘
```

---

## Instalasi singkat

```bash
# 1) dependensi inti (wajib)
pip install -r requirements.txt

# 2) minimal satu engine OCR — pilih sesuai hardware:
pip install -r requirements-tesseract.txt   # ringan, CPU, selalu bisa
pip install -r requirements-docling.txt     # tabel paling rapi, lisensi MIT
pip install -r requirements-paddle.txt      # PP-StructureV3 (baca catatan di dalamnya!)
pip install -r requirements-marker.txt      # Surya VLM (GPU; cek lisensi bobot)

# cek apa yang terdeteksi di mesinmu:
python run_ocr.py engines
```

Mesin Tesseract (biner + bahasa Indonesia) dan PaddlePaddle dipasang
**terpisah** — langkah lengkap Windows ada di
[`docs/INSTALL-WINDOWS.md`](docs/INSTALL-WINDOWS.md). Untuk batch besar di
engine neural, pakai notebook Kaggle/Colab di [`notebooks/`](notebooks/).

> Catatan hardware (RTX 4050 6 GB / RAM 8 GB): RAM adalah bottleneck
> sebenarnya. Default sudah dibuat aman (model mobile, modul berat
> dimatikan, pagar piksel). Bila engine neural kehabisan memori, jalankan
> `--device cpu` atau pindah ke Kaggle.

---

## Contoh penggunaan

```bash
# Survei cepat — jenis PDF apa saja yang saya punya?
python cek_pdf.py "data/pdf"

# Ekstraksi penuh dengan pemilihan engine otomatis
python run_ocr.py ingest "data/pdf" -o keluaran

# Paksa engine tertentu + halaman tertentu + DPI lebih tinggi
python run_ocr.py ingest "data/pdf" -o keluaran --engine docling --pages 1-5,12 --dpi 400

# Korpus seluruhnya digital? lewati OCR, jauh lebih cepat
python run_ocr.py ingest "data/pdf" -o keluaran --force-native

# Pakai berkas konfigurasi
python run_ocr.py ingest "data/pdf" -o keluaran --config config/default.yaml
```

### Dari Python

```python
from ocrpipe import Config, Pipeline

cfg = Config()
cfg.ocr_engine = "docling"
cfg.preprocess.dpi = 400
run = Pipeline(cfg).run(["data/pdf"], "keluaran")
print(run.total_chunks, "chunk;", run.pages_needing_review, "halaman perlu tinjauan")
```

---

## Struktur keluaran

```
keluaran/
├── markdown/<doc_id>.md          ← untuk dibaca & dikoreksi manusia (ada front-matter)
├── jsonl/
│   ├── <doc_id>.jsonl            ← chunk per dokumen
│   └── semua-chunk.jsonl         ← gabungan, SIAP di-ingest ke Qdrant
├── tabel/
│   ├── csv/<doc_id>/p0003_t01.csv
│   └── xlsx/<doc_id>.xlsx        ← multi-sheet + sheet INDEKS
├── qa/
│   ├── laporan-qa.md             ← ringkasan + daftar halaman perlu tinjauan
│   └── qa-per-halaman.csv
├── manifest.json                 ← jejak reproduktibilitas (versi pustaka + config)
├── konfigurasi-efektif.yaml
└── contoh-ingest-qdrant.py       ← skrip contoh ingest ke vector store
```

### Bentuk satu chunk JSONL (ringkas)

```json
{
  "chunk_id": "panduan-akademik-3f2a1b9c::c00007::a1b2c3d4",
  "doc_id": "panduan-akademik-3f2a1b9c",
  "text": "[BAB II > Registrasi]\n\nMahasiswa wajib melakukan registrasi ...",
  "content_type": "text",
  "page_start": 12, "page_end": 12,
  "section_path": ["BAB II", "Registrasi"],
  "doc_title": "Panduan Akademik FTII UHAMKA 2026",
  "doc_version": "2026",
  "doc_date": "2026-08-01",
  "engines": ["native"], "routes": ["native"],
  "mean_confidence": null, "needs_review": false,
  "token_estimate": 240, "checksum": "a1b2c3d4e5f6a7b8"
}
```

---

## Metadata dokumen (opsional tapi disarankan)

Letakkan `metadata.csv` di folder PDF untuk mengisi judul, versi, dan tanggal
berlaku secara akurat (lihat [`contoh-metadata.csv`](contoh-metadata.csv)).
Kolom `berkas` wajib; sisanya opsional. Tanpa berkas ini, metadata ditebak
dari nama berkas.

---

## Mengukur akurasi OCR (untuk Bab 4 skripsi)

Metrik di `laporan-qa.md` adalah **deteksi anomali**, *bukan* akurasi. Angka
akurasi yang sah dilaporkan harus dari perbandingan dengan transkrip manual:

```bash
# 1) siapkan 12 halaman sampel (distratifikasi digital vs scan)
python run_ocr.py sample "data/pdf" -o sampel-anotasi -n 12

# 2) transkripsikan manual tiap *.ref.txt dari gambar di sampel-anotasi/gambar/
#    (ikuti sampel-anotasi/CARA-ANOTASI.md agar konsisten)

# 3) hitung CER/WER
python run_ocr.py eval sampel-anotasi
```

Hasil dilaporkan sebagai agregat **mikro** (bobot menurut panjang acuan) dan
mencantumkan perlakuan normalisasi — keduanya wajib disebut di skripsi agar
angka bisa dibandingkan antar penelitian.

---

## Perintah CLI

| Perintah | Fungsi |
| --- | --- |
| `cek_pdf.py` | Survei mandiri (hanya butuh pymupdf): digital/scan/campuran |
| `ingest` | Proses PDF → Markdown + JSONL + tabel + QA |
| `diagnose` | Laporan routing per halaman tanpa menjalankan OCR |
| `engines` | Daftar engine yang terpasang + cara memasang yang belum |
| `sample` | Siapkan halaman sampel untuk transkripsi manual |
| `eval` | Hitung CER/WER dari pasangan `*.ref.txt` / `*.hyp.txt` |

---

## Pemetaan ke skripsi

Draf **Bab 3** (prosedur akuisisi & praproses data) ada di
[`docs/Bab3-Metodologi-Akuisisi-Data.md`](docs/Bab3-Metodologi-Akuisisi-Data.md):
sudah memuat diagram alir tahapan, lokasi penelitian, peubah yang diukur,
rancangan, teknik pengumpulan & analisis data, prosedur validasi CER/WER, dan
**Jadwal Rencana Kegiatan Penelitian** (Time Line Schedule).

---

## Lisensi & etika

Kode pipeline ini bebas dipakai untuk keperluan skripsi. Perhatikan lisensi
masing-masing engine — khususnya **bobot model Marker** (AI Pubs Open RAIL-M:
bebas untuk riset, terbatas untuk komersial). Docling berlisensi MIT
menyeluruh. Selalu pastikan kamu berhak memproses dokumen institusi yang
dijadikan korpus.
