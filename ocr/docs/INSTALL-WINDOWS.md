# Panduan Instalasi di Windows

Panduan ini untuk menjalankan **ocrpipe** di laptop Windows dengan spesifikasi
seperti milikmu: **AMD Ryzen 5, NVIDIA RTX 4050 (6 GB VRAM), RAM 8 GB**.

> Ringkasan jujur soal hardware: **RAM 8 GB adalah batasan sebenarnya**, bukan
> VRAM. Engine `native` + `tesseract` ringan dan nyaman di laptop ini. Engine
> neural (`docling`, `paddle`, `marker`) **bisa** jalan untuk dokumen kecil,
> tetapi untuk batch besar jauh lebih nyaman di **Kaggle** (lihat
> `notebooks/`). Strategi yang disarankan ada di bagian akhir.

---

## 0. Prasyarat: Python

Pasang **Python 3.10 atau 3.11** (64-bit) dari <https://www.python.org/downloads/>.
Saat instalasi, centang **"Add Python to PATH"**.

Verifikasi di PowerShell:

```powershell
python --version
```

Buat virtual environment (sangat disarankan agar tidak mengacak Python sistem):

```powershell
cd C:\path\ke\ocr-skripsi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell menolak menjalankan skrip aktivasi:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 1. Dependensi inti (WAJIB)

```powershell
pip install -r requirements.txt
```

Ini sudah cukup untuk memproses **PDF digital** dan merekonstruksi tabel dari
text layer. Uji:

```powershell
python cek_pdf.py --help
python run_ocr.py engines
```

`engines` akan menampilkan `native` TERSEDIA dan sisanya "tidak ada" — itu
normal sebelum engine OCR dipasang.

---

## 2. Tesseract (jaring pengaman OCR, CPU — PASANG INI DULU)

Tesseract adalah engine paling mudah dan paling ringan. Ia yang akan membaca
**teks** halaman hasil scan di laptopmu.

### 2a. Pasang mesin Tesseract

1. Unduh installer dari **UB-Mannheim**:
   <https://github.com/UB-Mannheim/tesseract/wiki>
   (pilih berkas `tesseract-ocr-w64-setup-....exe`)
2. Jalankan installer. **PENTING**: pada layar *"Select components"*, buka
   *Additional language data* dan centang **Indonesian** (`ind`). Kalau lupa,
   lihat langkah 2c.
3. Catat folder instalasi, biasanya:
   `C:\Program Files\Tesseract-OCR\`

### 2b. Pasang binding Python

```powershell
pip install -r requirements-tesseract.txt
```

### 2c. Pastikan bahasa Indonesia ada

Jika saat instalasi lupa mencentang Indonesian:

1. Unduh `ind.traineddata` dari
   <https://github.com/tesseract-ocr/tessdata_fast/blob/main/ind.traineddata>
   (klik tombol **Download**)
2. Salin ke `C:\Program Files\Tesseract-OCR\tessdata\`

### 2d. Beri tahu ocrpipe lokasi Tesseract

Jika `tesseract` belum ada di PATH, dua pilihan:

**Pilihan A — set di konfigurasi** (`config/default.yaml`):

```yaml
tesseract:
  cmd: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
  lang: ind+eng
```

**Pilihan B — tambahkan ke PATH Windows**, lalu buka PowerShell baru.

Verifikasi:

```powershell
python run_ocr.py engines
```

`tesseract` harus menjadi **TERSEDIA**. Jika tertulis *"data bahasa ['ind']
belum ada"*, ulangi langkah 2c.

---

## 3. Docling (PREFERENSI untuk tabel — lisensi MIT)

Docling memberi hasil tabel paling rapi dan bebas lisensi penuh. Jalan di CPU
(lebih lambat) maupun GPU.

```powershell
pip install -r requirements-docling.txt
```

Docling memakai Tesseract sebagai mesin OCR default di ocrpipe (demi
reproduktibilitas), jadi langkah 2 tetap perlu. Alternatif tanpa Tesseract:
ubah `config/default.yaml`:

```yaml
docling:
  ocr_engine: easyocr      # lalu: pip install easyocr
  ocr_lang: [id, en]       # EasyOCR memakai kode 2 huruf
```

> Unduhan model pertama kali butuh koneksi internet dan beberapa menit.
> Model TableFormer + layout totalnya beberapa ratus MB.

---

## 4. PaddleOCR PP-StructureV3 (opsional — struktur tabel kuat)

**Urutan pemasangan penting** (sejak PaddleOCR 3.5, library dan mesin
inferensi terpisah):

```powershell
# 1) library + grup dependensi doc-parser
pip install "paddleocr[doc-parser]"

# 2) mesin PaddlePaddle — PILIH SATU:

# 2a. CPU (paling aman untuk RAM 8 GB)
pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 2b. GPU CUDA 12.6 (RTX 4050) — hanya jika driver NVIDIA cukup baru
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Verifikasi:

```powershell
python -c "import paddle; print(paddle.__version__)"
python run_ocr.py engines
```

> **Catatan VRAM**: konfigurasi default PP-StructureV3 dapat memakai belasan GB
> VRAM. ocrpipe sudah mematikan modul formula/chart/seal dan memakai model
> *mobile* agar muat di 6 GB. Jika tetap kehabisan memori GPU, jalankan dengan
> `--device cpu`:
>
> ```powershell
> python run_ocr.py ingest data\pdf -o keluaran --engine paddle --device cpu
> ```

---

## 5. Marker (opsional — perlu GPU, perhatikan lisensi)

```powershell
# PyTorch dulu, sesuaikan dengan CUDA (lihat https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-marker.txt
```

> **Lisensi**: kode Marker Apache-2.0, tetapi **bobot model** memakai AI Pubs
> Open RAIL-M — bebas untuk riset/skripsi, terbatas untuk komersial. Sebutkan
> ini terpisah di laporan.
>
> **Tanpa GPU**: Marker butuh server llama.cpp (tidak ikut pip). Lebih mudah
> pakai Docling di CPU, atau jalankan Marker di Kaggle.

---

## 6. Strategi yang disarankan untuk laptopmu

| Situasi | Lakukan |
| --- | --- |
| Semua dokumen digital | `python run_ocr.py ingest ... --force-native` (tanpa OCR, instan) |
| Sebagian scan, **teksnya** saja penting | `--engine tesseract` (ringan, lokal) |
| Scan dengan **tabel penting**, jumlah sedikit | `--engine docling` di laptop |
| Scan dengan tabel penting, **banyak** | Jalankan di **Kaggle** (`notebooks/`), unduh hasilnya |

Alur praktis:

```powershell
# 1. survei dulu — berapa banyak yang perlu OCR?
python cek_pdf.py "data\pdf"

# 2. kalau mayoritas digital, selesai dalam hitungan detik:
python run_ocr.py ingest "data\pdf" -o keluaran --force-native

# 3. kalau ada scan bertabel, proses halaman itu dengan docling:
python run_ocr.py ingest "data\pdf" -o keluaran --engine docling
```

---

## 7. Masalah umum (troubleshooting)

**`TesseractNotFoundError`** — Tesseract tidak di PATH. Set `tesseract.cmd` di
config (langkah 2d) atau tambahkan ke PATH lalu buka PowerShell baru.

**`data bahasa ['ind'] belum ada`** — salin `ind.traineddata` ke folder
`tessdata` (langkah 2c).

**`MemoryError` / laptop ngelag saat engine neural** — RAM 8 GB kehabisan.
Turunkan beban: proses per sedikit halaman dengan `--pages`, kecilkan `--dpi`
ke 200, atau pindah ke Kaggle. Untuk Paddle, pakai `--device cpu`.

**PaddlePaddle gagal diimpor setelah `pip install paddleocr`** — kamu melewati
langkah 2 bagian 4 (mesin inferensi dipasang terpisah). Jalankan perintah
`paddlepaddle==3.2.0` di atas.

**Karakter Indonesia rusak saat buka CSV di Excel** — ocrpipe menulis CSV
dengan BOM UTF-8, jadi seharusnya benar. Jika masih rusak, buka lewat
*Data > From Text/CSV* dan pilih encoding UTF-8.

**PDF "terproteksi kata sandi"** — buka proteksinya dulu:
`qpdf --decrypt masuk.pdf keluar.pdf` (pasang qpdf), lalu proses `keluar.pdf`.

**Unduhan model Docling/Paddle lambat/gagal** — butuh internet saat pertama
kali. Setelah terunduh, model tersimpan di cache dan tidak perlu internet lagi.

---

## 8. Menjalankan test (opsional, untuk memastikan instalasi sehat)

```powershell
python tests\test_e2e.py
```

Harus menampilkan `6/6 lulus`. Test ini tidak membutuhkan engine OCR eksternal
(memakai jalur `native`), jadi bisa dijalankan segera setelah langkah 1.
