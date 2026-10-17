"""Konfigurasi pipeline: dataclass bersarang + override dari YAML/CLI.

Alasan tidak memakai pydantic: menjaga dependensi tetap ringan supaya
instalasi di Windows (RAM 8 GB) tidak menambah beban, dan supaya file
konfigurasi tetap bisa dibaca manusia untuk dilampirkan di skripsi.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Sub-konfigurasi
# --------------------------------------------------------------------------- #


@dataclass
class DetectConfig:
    """Ambang keputusan "halaman ini punya text layer yang layak dipakai?"."""

    #: Jumlah karakter minimum pada satu halaman agar text layer dianggap ada.
    min_chars: int = 120
    #: Kepadatan karakter minimum (karakter per 1000 pt^2 area halaman).
    min_char_density: float = 0.08
    #: Jika rasio karakter rusak (U+FFFD, "(cid:NN)", kotak kosong) melebihi ini,
    #: text layer ditolak dan halaman dipaksa OCR.
    max_garbled_ratio: float = 0.02
    #: Jika satu gambar menutupi >= rasio ini dari area halaman DAN teks sedikit,
    #: halaman dianggap hasil scan.
    full_page_image_ratio: float = 0.65
    #: Halaman dengan karakter < min_chars dan tanpa gambar berarti -> ROUTE_EMPTY.
    treat_blank_as_empty: bool = True
    #: Paksa semua halaman ke satu rute, mengabaikan deteksi.
    #: Nilai: None | "native" | "ocr"
    force_route: str | None = None


@dataclass
class PreprocessConfig:
    """Praproses citra untuk halaman yang diarahkan ke OCR.

    CATATAN PENTING: profil berbeda untuk engine klasik vs engine neural.
    Tesseract diuntungkan oleh binarisasi; model neural (PP-StructureV5,
    TableFormer, Surya) dilatih pada citra grayscale/berwarna sehingga
    binarisasi agresif justru MENURUNKAN akurasi, terutama pada tabel
    tanpa garis. Karena itu `binarize` default-nya mati untuk engine neural.
    """

    dpi: int = 300
    #: Naikkan ke 400 untuk teks sangat kecil; >400 jarang membantu dan
    #: memperbesar pemakaian RAM secara kuadratik.
    max_pixels: int = 40_000_000  # pagar pengaman agar tidak OOM di RAM 8 GB
    grayscale: bool = True
    deskew: bool = True
    deskew_max_angle: float = 12.0  # derajat; di luar ini dianggap salah deteksi
    deskew_min_angle: float = 0.25  # di bawah ini tidak dirotasi (hindari blur)
    denoise: bool = False  # median blur ringan; merusak garis tabel halus
    denoise_ksize: int = 3
    binarize: bool = False  # lihat catatan di docstring
    binarize_method: str = "adaptive"  # adaptive | otsu
    contrast_stretch: bool = True
    auto_rotate_osd: bool = True  # koreksi 90/180/270 lewat Tesseract OSD
    crop_border: bool = False  # buang bingkai hitam hasil fotokopi
    #: Simpan citra hasil praproses untuk audit/lampiran skripsi.
    save_debug_images: bool = False


@dataclass
class NativeConfig:
    """Ekstraksi halaman yang punya text layer (tanpa OCR sama sekali)."""

    #: Strategi pencarian tabel PyMuPDF.
    #: auto         = coba "lines_strict", bila nol tabel coba "lines"
    #: lines_strict = hanya tabel bergaris penuh (paling sedikit false positive)
    #: lines        = toleran terhadap garis terputus
    #: text         = tabel tanpa garis (BANYAK false positive pada prosa biasa)
    #: none         = jangan cari tabel
    table_strategy: str = "auto"
    #: Minimal jumlah sel agar kandidat dianggap tabel sungguhan.
    min_table_cells: int = 4
    #: Deteksi heading dari ukuran & ketebalan font.
    detect_headings: bool = True
    #: Ambang: ukuran font >= median_dokumen * rasio ini dianggap heading.
    heading_size_ratio: float = 1.18
    #: Panjang maksimum teks yang boleh dianggap heading (karakter).
    heading_max_chars: int = 140
    #: Tandai blok di zona atas/bawah halaman sebagai kop/footer agar bisa
    #: dibuang dari chunk. Nilai = fraksi tinggi halaman.
    mark_header_footer: bool = True
    header_zone: float = 0.055
    footer_zone: float = 0.055
    #: Gabungkan baris dalam satu blok menjadi paragraf (hilangkan hard wrap).
    join_lines: bool = True


@dataclass
class TesseractConfig:
    lang: str = "ind+eng"
    psm: int = 3  # 3 = full auto page segmentation
    oem: int = 3  # 3 = default LSTM
    #: Path executable; biarkan None jika tesseract sudah ada di PATH.
    cmd: str | None = None
    #: Ekstraksi tabel pada halaman scan memakai deteksi garis OpenCV.
    detect_tables: bool = True
    #: Panjang minimum garis (fraksi lebar/tinggi halaman) agar dianggap
    #: garis tabel, bukan garis bawah teks atau noda.
    table_line_frac: float = 0.25
    #: Minimal jumlah garis horizontal & vertikal agar area dianggap tabel.
    table_min_hlines: int = 3
    table_min_vlines: int = 2
    #: Buang kata dengan confidence di bawah ini (0 = jangan buang).
    min_word_conf: float = 30.0
    #: Gerbang mutu tabel: bila confidence rata-rata kata di dalam region
    #: tabel di bawah ini, tabel TIDAK disajikan sebagai struktur (menghindari
    #: tabel sampah dari scan buruk mencemari indeks RAG). Isinya tetap
    #: dipertahankan sebagai teks biasa dan halaman ditandai perlu tinjauan.
    #: Scanned table yang gagal di Tesseract sebaiknya di-OCR ulang dengan
    #: engine neural (docling/paddle).
    min_table_conf: float = 55.0
    extra_config: str = ""


@dataclass
class PaddleConfig:
    """PP-StructureV3 (paddleocr >= 3.5, grup dependensi `doc-parser`).

    Default di sini DIMATIKAN untuk modul yang tidak dibutuhkan dokumen
    akademik, karena konfigurasi default PP-StructureV3 dapat memakai
    >16 GB VRAM. Dengan modul formula/chart/seal dimatikan, pipeline jauh
    lebih ramah untuk GPU 6 GB.
    """

    lang: str = "id"  # resolve ke model Latin PP-OCRv5
    device: str | None = None  # None = auto; "cpu" | "gpu" | "gpu:0"
    use_table_recognition: bool = True
    use_formula_recognition: bool = False
    use_chart_recognition: bool = False
    use_seal_recognition: bool = False
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = False  # berat; nyalakan hanya untuk foto bengkok
    use_textline_orientation: bool = False
    use_region_detection: bool = True
    text_rec_score_thresh: float | None = None
    #: Pakai model mobile (ringan) alih-alih server. Wajib di GPU 6 GB.
    prefer_mobile_models: bool = True
    #: Kirim PDF langsung ke pipeline (hemat I/O) atau per-citra hasil
    #: praproses (lebih terkontrol, bisa deskew dulu). Default per-citra.
    feed_pdf_directly: bool = False


@dataclass
class DoclingConfig:
    """Docling >= 2.x (paket `docling`, kode di `docling-slim`)."""

    do_ocr: bool = True
    do_table_structure: bool = True
    table_mode: str = "accurate"  # accurate | fast
    #: auto | easyocr | tesseract (CLI) | tesserocr | rapidocr
    ocr_engine: str = "tesseract"
    ocr_lang: list[str] = field(default_factory=lambda: ["ind", "eng"])
    do_formula_enrichment: bool = False
    do_code_enrichment: bool = False
    #: Aktifkan pipeline VLM (granite-docling). Jauh lebih berat.
    use_vlm: bool = False
    vlm_preset: str = "granitedocling"
    #: Gunakan teks dari text layer PDF bila ada (hybrid; lebih akurat).
    force_backend_text: bool = False


@dataclass
class MarkerConfig:
    """marker-pdf >= 2.x.

    PERINGATAN LISENSI: kode Marker Apache-2.0, tetapi BOBOT MODEL-nya
    memakai modified AI Pubs Open RAIL-M (gratis untuk riset/personal dan
    organisasi di bawah ambang pendapatan tertentu). Untuk skripsi ini
    termasuk penggunaan riset, tetapi status lisensi bobot harus disebut
    terpisah dari lisensi kode bila dibahas di laporan.
    """

    mode: str | None = None  # None = auto (balanced di CUDA, fast di CPU)
    disable_ocr: bool = False  # True = jalur CPU murni (tanpa server VLM)
    force_ocr: bool = False
    use_llm: bool = False  # butuh API key eksternal; default mati
    output_format: str = "chunks"  # chunks memberi atribusi halaman terbaik
    disable_image_extraction: bool = True


@dataclass
class ChunkConfig:
    target_chars: int = 1000
    max_chars: int = 1600
    min_chars: int = 200  # chunk lebih pendek digabung ke tetangga
    overlap_chars: int = 150
    #: Jangan pernah memotong tabel di tengah baris; jika tabel melebihi
    #: max_chars, tabel dipecah per baris dengan header DIULANG.
    split_large_tables: bool = True
    table_rows_per_shard: int = 25
    #: Sisipkan judul bagian + keterangan kolom di depan chunk tabel supaya
    #: tabel tetap terambil oleh dense retrieval (tabel "telanjang" sering
    #: gagal diambil karena minim kata kunci).
    table_context_prefix: bool = True
    #: Sertakan jejak heading (Bab > Sub-bab) di awal setiap chunk teks.
    prepend_section_path: bool = True
    #: Buang kop surat/footer berulang dari chunk.
    drop_noise_blocks: bool = True
    respect_headings: bool = True


@dataclass
class QAConfig:
    #: Di bawah ambang ini halaman ditandai perlu tinjauan manual.
    min_mean_confidence: float = 72.0
    #: Rasio token "sampah" maksimum (campuran huruf-angka aneh, simbol).
    max_garbage_ratio: float = 0.12
    #: Rasio minimum token yang dikenali sebagai kata Indonesia/Inggris umum.
    min_lexicon_ratio: float = 0.25
    #: Halaman non-kosong dengan karakter di bawah ini dianggap gagal OCR.
    min_chars_per_page: int = 60
    #: Tandai tabel yang jumlah kolomnya tidak konsisten.
    flag_irregular_tables: bool = True


@dataclass
class OutputConfig:
    markdown: bool = True
    jsonl: bool = True
    csv_tables: bool = True
    xlsx_tables: bool = True
    qa_report: bool = True
    #: comment = sisipkan "<!-- halaman: N -->" (tidak terlihat saat dirender)
    #: heading = sisipkan "### [Halaman N]"
    page_marker: str = "comment"
    #: Tulis YAML front matter metadata di awal file Markdown.
    front_matter: bool = True
    #: Satu file JSONL gabungan untuk semua dokumen, selain per-dokumen.
    combined_jsonl: bool = True


@dataclass
class Config:
    #: Daftar engine berurutan sesuai prioritas untuk halaman ROUTE_OCR.
    #: "auto" = pilih engine terbaik yang tersedia di mesin ini.
    ocr_engine: str = "auto"
    #: Engine untuk halaman ROUTE_NATIVE. Hampir selalu "native".
    native_engine: str = "native"
    #: Jika engine OCR pilihan tidak tersedia, turun ke engine berikutnya.
    fallback_chain: list[str] = field(
        default_factory=lambda: ["docling", "paddle", "tesseract"]
    )
    #: Proses hanya halaman ini (1-based), mis. [1, 2, 5]. None = semua.
    pages: list[int] | None = None
    workers: int = 1  # >1 hanya aman untuk engine native/tesseract
    verbose: bool = True
    #: Lanjutkan ke dokumen berikutnya bila satu dokumen gagal total.
    continue_on_error: bool = True

    detect: DetectConfig = field(default_factory=DetectConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    native: NativeConfig = field(default_factory=NativeConfig)
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)
    paddle: PaddleConfig = field(default_factory=PaddleConfig)
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    marker: MarkerConfig = field(default_factory=MarkerConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Config":
        cfg = cls()
        if data:
            _apply_overrides(cfg, data)
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Muat konfigurasi dari YAML. Path None -> konfigurasi default."""
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Berkas konfigurasi tidak ditemukan: {p}")
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML belum terpasang. Jalankan: pip install pyyaml"
            ) from exc
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Isi {p} harus berupa mapping YAML, bukan {type(data)}")
        return cls.from_dict(data)

    def override(self, **kwargs: Any) -> "Config":
        """Salin konfigurasi dengan sebagian nilai diganti (dot-path didukung).

        Contoh: cfg.override(**{"preprocess.dpi": 400, "verbose": False})
        """
        new = copy.deepcopy(self)
        flat = {k: v for k, v in kwargs.items() if v is not None}
        _apply_overrides(new, _unflatten(flat))
        return new

    def dump_yaml(self, path: str | Path) -> None:
        """Simpan konfigurasi efektif — penting untuk reproduktibilitas skripsi."""
        import yaml  # type: ignore

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                self.to_dict(), fh, allow_unicode=True, sort_keys=False, indent=2
            )


# --------------------------------------------------------------------------- #
# Helper override
# --------------------------------------------------------------------------- #


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """{"preprocess.dpi": 400} -> {"preprocess": {"dpi": 400}}"""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _apply_overrides(obj: Any, data: dict[str, Any]) -> None:
    """Terapkan dict bersarang ke dataclass secara rekursif, dengan validasi."""
    valid = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise ValueError(
                f"Kunci konfigurasi tidak dikenal: '{key}' pada "
                f"{type(obj).__name__}. Kunci yang sah: {sorted(valid)}"
            )
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, value)
