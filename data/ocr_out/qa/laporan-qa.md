# Laporan Penjaminan Mutu Ekstraksi Dokumen

Laporan ini memuat metrik penyaringan otomatis. Metrik di sini mendeteksi **anomali**, bukan mengukur akurasi. Angka akurasi OCR (CER/WER) hanya sah diperoleh dari `ocrpipe eval` terhadap transkrip acuan yang dibuat manual.

## Ringkasan

- Dokumen diproses: **6**
- Halaman diproses: **238**
- Tabel diekstrak: **207**
- Total karakter: **400,552**
- Halaman perlu tinjauan manual: **112** (47.1%)

- Sebaran rute: `empty`=1, `native`=214, `ocr`=23
- Sebaran engine: `-`=1, `native`=237

## Pemilihan engine

- TIDAK ADA ENGINE OCR: Tidak ada engine OCR yang tersedia.
  - mode auto, urutan preferensi: docling > paddle > marker > tesseract
  - docling: TIDAK TERSEDIA — paket 'docling' belum terpasang
  -   cara pasang: pip install docling
  - paddle: TIDAK TERSEDIA — paket 'paddleocr' belum terpasang
  -   cara pasang: pip install "paddleocr[doc-parser]" lalu pasang mesin inferensi: pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ (CPU) atau paddlepaddle-gpu dengan indeks cu126 (GPU)
  - marker: TIDAK TERSEDIA — paket 'marker-pdf' belum terpasang
  -   cara pasang: pip install marker-pdf
  - tesseract: TIDAK TERSEDIA — pytesseract belum terpasang
  -   cara pasang: pip install pytesseract
Kesalahan terakhir: pytesseract belum terpasang
Minimal pasang Tesseract: pip install pytesseract + unduh installer Tesseract (lihat docs/INSTALL-WINDOWS.md).

## Konfigurasi efektif (ringkas)

- `ocr_engine`: auto
- `dpi`: 300
- `deskew`: True
- `binarize`: False
- `detect.min_chars`: 120
- `detect.max_garbled_ratio`: 0.02
- `chunk.target_chars`: 800
- `chunk.overlap_chars`: 120
- `qa.min_mean_confidence`: 72.0

## Per dokumen

| Dokumen | Hlm | Native | OCR | Kosong | Tabel | Karakter/hlm | Conf. | Leksikon | Perlu tinjauan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dokumen-Kurikulum-TRM-2024.docx.pdf | 63 | 47 | 15 | 1 | 35 | 1191.1 | - | 0.23 | 39 |
| Panduan Akademik 2025-2026_Teknik Informatika.pdf | 13 | 12 | 1 | 0 | 15 | 1570.5 | - | 0.32 | 5 |
| Panduan Kurikulum 2023-2024 prodi TE.pdf | 72 | 67 | 5 | 0 | 73 | 1500.6 | - | 0.27 | 45 |
| PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf | 49 | 49 | 0 | 0 | 37 | 2805.9 | - | 0.35 | 9 |
| Panduan-Akademik-2025-2026_STI.pdf | 14 | 12 | 2 | 0 | 19 | 1266.0 | - | 0.30 | 6 |
| TM-Panduan Akademik-2026-2027.pdf | 27 | 27 | 0 | 0 | 28 | 1593.8 | - | 0.35 | 8 |

## Halaman yang perlu ditinjau manual

Urutkan penanganan dari atas: halaman di awal daftar punya alasan paling banyak.

- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 49** (`native`/`native`): hanya 10.4% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0049_t01: 3 kolom kosong dibuang; tabel p0049_t02: 1 kolom kosong dibuang; tabel p0049_t03: 1 kolom kosong dibuang
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 37** (`native`/`native`): rasio token tidak wajar 17.3% > 12.0%; hanya 17.6% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0037_t01: 5 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 34** (`native`/`native`): rasio token tidak wajar 17.6% > 12.0%; hanya 14.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0034_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 48** (`native`/`native`): hanya 12.5% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0048_t01: 3 kolom kosong dibuang; tabel p0048_t02: 1 kolom kosong dibuang
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 9** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (0 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 16** (`native`/`native`): hanya 23.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0016_t02: 1 kolom kosong dibuang
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 19** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (0 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 34** (`native`/`native`): hanya 13.9% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0034_t01: 1 baris kosong dibuang
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 40** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 41** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 42** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 43** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 44** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 45** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 46** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 47** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 48** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (36 karakter < 60)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 53** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (0 karakter < 60)
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 1** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (57 karakter < 60)
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 7** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (0 karakter < 60)
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 19** (`native`/`native`): hanya 23.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0019_t02: 1 baris kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 33** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); tabel p0033_t01: 5 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 35** (`native`/`native`): hanya 19.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0035_t01: 2 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 38** (`native`/`native`): hanya 20.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0038_t01: 5 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 39** (`native`/`native`): hanya 14.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0039_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 43** (`native`/`native`): tabel p0043_t01: 1 kolom kosong dibuang; tabel p0043_t02: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 46** (`native`/`native`): hanya 20.9% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0046_t02: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 47** (`native`/`native`): hanya 12.9% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0047_t02: 2 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 50** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); tabel p0050_t01: 1 baris kosong dibuang; 2 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 60** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); teks sangat sedikit (0 karakter < 60)
- **Panduan-Akademik-2025-2026_STI.pdf hlm 7** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi); tabel p0007_t01: 4 kolom kosong dibuang
- **Panduan-Akademik-2025-2026_STI.pdf hlm 8** (`native`/`native`): hanya 16.1% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai; tabel p0008_t01: 3 kolom kosong dibuang
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 3** (`native`/`native`): hanya 5.8% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 15** (`native`/`native`): hanya 23.8% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 17** (`native`/`native`): hanya 14.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 18** (`native`/`native`): hanya 16.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 20** (`native`/`native`): hanya 23.4% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 22** (`native`/`native`): hanya 1.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 25** (`native`/`native`): hanya 11.4% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 26** (`native`/`native`): hanya 8.2% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 27** (`native`/`native`): hanya 6.6% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 28** (`native`/`native`): hanya 4.5% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 29** (`native`/`native`): hanya 5.8% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 30** (`native`/`native`): hanya 6.1% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 31** (`native`/`native`): hanya 10.1% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 32** (`native`/`native`): hanya 7.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 33** (`native`/`native`): hanya 9.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 35** (`native`/`native`): hanya 1.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 36** (`native`/`native`): hanya 11.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 39** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 52** (`native`/`native`): hanya 20.6% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 54** (`native`/`native`): hanya 22.6% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 55** (`native`/`native`): hanya 21.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 57** (`native`/`native`): hanya 19.1% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 59** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi)
- **Dokumen-Kurikulum-TRM-2024.docx.pdf hlm 63** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi)
- **Panduan Akademik 2025-2026_Teknik Informatika.pdf hlm 1** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi)
- **Panduan Akademik 2025-2026_Teknik Informatika.pdf hlm 3** (`native`/`native`): rasio token tidak wajar 13.5% > 12.0%
- **Panduan Akademik 2025-2026_Teknik Informatika.pdf hlm 5** (`native`/`native`): hanya 14.9% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Akademik 2025-2026_Teknik Informatika.pdf hlm 7** (`native`/`native`): hanya 19.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Akademik 2025-2026_Teknik Informatika.pdf hlm 9** (`native`/`native`): hanya 18.2% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 16** (`native`/`native`): tabel p0016_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 18** (`native`/`native`): hanya 17.8% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 20** (`native`/`native`): hanya 18.1% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 21** (`native`/`native`): hanya 14.1% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 22** (`native`/`native`): hanya 21.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 24** (`native`/`native`): hanya 23.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 25** (`native`/`native`): hanya 19.4% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 26** (`native`/`native`): tabel p0026_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 27** (`native`/`native`): tabel p0027_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 28** (`native`/`native`): tabel p0028_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 29** (`native`/`native`): tabel p0029_t01: 2 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 30** (`native`/`native`): tabel p0030_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 31** (`native`/`native`): hanya 4.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 32** (`native`/`native`): hanya 12.5% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 36** (`native`/`native`): tabel p0036_t01: 2 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 37** (`native`/`native`): tabel p0037_t01: 2 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 40** (`native`/`native`): hanya 9.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 42** (`native`/`native`): tabel p0042_t01: 1 baris kosong dibuang; 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 44** (`native`/`native`): tabel p0044_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 45** (`native`/`native`): tabel p0045_t01: 1 kolom kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 56** (`native`/`native`): tabel p0056_t01: 1 baris kosong dibuang
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 59** (`native`/`native`): hanya 23.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 63** (`native`/`native`): hanya 23.1% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 64** (`native`/`native`): hanya 17.9% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 65** (`native`/`native`): hanya 12.4% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 66** (`native`/`native`): hanya 10.8% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 67** (`native`/`native`): hanya 15.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 68** (`native`/`native`): hanya 13.2% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 69** (`native`/`native`): hanya 15.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan Kurikulum 2023-2024 prodi TE.pdf hlm 70** (`native`/`native`): hanya 15.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 4** (`native`/`native`): rasio token tidak wajar 13.2% > 12.0%
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 36** (`native`/`native`): hanya 20.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 42** (`native`/`native`): hanya 15.5% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 43** (`native`/`native`): hanya 20.2% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 44** (`native`/`native`): hanya 17.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 45** (`native`/`native`): hanya 16.9% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 46** (`native`/`native`): hanya 21.4% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 47** (`native`/`native`): hanya 18.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **PANDUAN MK PENGAYAAN FTII UHAMKA_2025.pdf hlm 48** (`native`/`native`): hanya 13.4% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan-Akademik-2025-2026_STI.pdf hlm 1** (`ocr`/`native`): peringatan engine: tidak ada engine OCR — diekstrak paksa dengan engine native (mode terdegradasi)
- **Panduan-Akademik-2025-2026_STI.pdf hlm 5** (`native`/`native`): hanya 20.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan-Akademik-2025-2026_STI.pdf hlm 6** (`native`/`native`): hanya 9.3% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **Panduan-Akademik-2025-2026_STI.pdf hlm 9** (`native`/`native`): tabel p0009_t01: 4 kolom kosong dibuang
- **TM-Panduan Akademik-2026-2027.pdf hlm 7** (`native`/`native`): hanya 18.5% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **TM-Panduan Akademik-2026-2027.pdf hlm 8** (`native`/`native`): hanya 24.2% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **TM-Panduan Akademik-2026-2027.pdf hlm 9** (`native`/`native`): hanya 24.8% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **TM-Panduan Akademik-2026-2027.pdf hlm 10** (`native`/`native`): hanya 24.7% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **TM-Panduan Akademik-2026-2027.pdf hlm 12** (`native`/`native`): hanya 19.8% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **TM-Panduan Akademik-2026-2027.pdf hlm 13** (`native`/`native`): hanya 22.6% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **TM-Panduan Akademik-2026-2027.pdf hlm 24** (`native`/`native`): hanya 9.0% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai
- **TM-Panduan Akademik-2026-2027.pdf hlm 25** (`native`/`native`): hanya 11.6% token dikenali sebagai kata umum (ambang 25.0%) — kemungkinan keluaran rusak atau bahasa tidak sesuai

## Cara menindaklanjuti

1. Buka `qa-per-halaman.csv`, saring `perlu_tinjauan = True`.
2. Untuk halaman hasil scan dengan confidence rendah: naikkan `preprocess.dpi` ke 400, atau aktifkan `preprocess.binarize` bila engine-nya Tesseract.
3. Untuk halaman dengan `rasio_leksikon` sangat rendah: periksa orientasi halaman dan bahasa engine (`tesseract.lang`, `paddle.lang`).
4. Untuk tabel dengan peringatan kolom tidak konsisten: periksa berkas XLSX-nya; perbaiki manual di sana, lalu jalankan ulang hanya halaman tersebut dengan `--pages`.
5. Setelah stabil, ambil sampel untuk pengukuran akurasi: `ocrpipe sample` lalu transkripsi manual dan `ocrpipe eval`.
