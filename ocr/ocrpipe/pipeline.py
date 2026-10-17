"""Orkestrasi pipeline: dari folder PDF menjadi Markdown + JSONL + tabel.

Alur satu dokumen:

    probe (detect.py)            -> rute per halaman
      |-- ROUTE_NATIVE  --> engine `native`      (tanpa OCR)
      |-- ROUTE_OCR     --> engine OCR terpilih  (docling/paddle/marker/tesseract)
      |-- ROUTE_EMPTY   --> dilewati, dicatat
    gabung  -> QA (qa.py) -> chunk (chunker.py) -> tulis (export.py)

Engine OCR diinisialisasi MALAS dan HANYA SEKALI untuk seluruh batch. Ini
penting: memuat model PP-StructureV3 atau Docling bisa memakan puluhan detik,
dan memuatnya ulang per dokumen akan mendominasi total waktu pemrosesan serta
membuat pengukuran latensi di skripsi tidak valid.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chunker import Chunker
from .config import Config
from .detect import DocProbe, probe_document
from .engines import Engine, EngineError, get_engine, resolve_ocr_engine
from .export import (
    append_jsonl,
    write_ingest_example,
    write_jsonl,
    write_manifest,
    write_markdown,
    write_tables,
)
from .models import (
    ROUTE_EMPTY,
    ROUTE_NATIVE,
    ROUTE_OCR,
    Chunk,
    DocMeta,
    DocResult,
    PageResult,
)
from .qa import document_summary, evaluate_document, write_qa_report
from .utils import get_logger, human_size, make_doc_id, slugify

METADATA_FILENAMES = ("metadata.csv", "metadata-dokumen.csv")

#: Kolom metadata.csv yang dikenali (semua opsional kecuali `berkas`).
_META_COLUMNS = {
    "berkas": "source_file",
    "file": "source_file",
    "nama_berkas": "source_file",
    "judul": "title",
    "title": "title",
    "kategori": "category",
    "category": "category",
    "versi": "version",
    "version": "version",
    "tanggal_berlaku": "effective_date",
    "tanggal": "effective_date",
    "unit": "unit",
    "prodi": "unit",
    "url": "url",
    "bahasa": "language",
}


@dataclass
class RunResult:
    documents: list[DocResult] = field(default_factory=list)
    chunks: dict[str, list[Chunk]] = field(default_factory=dict)
    engine_notes: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def total_chunks(self) -> int:
        return sum(len(v) for v in self.chunks.values())

    @property
    def total_pages(self) -> int:
        return sum(len(d.pages) for d in self.documents)

    @property
    def total_tables(self) -> int:
        return sum(len(d.tables) for d in self.documents)

    @property
    def pages_needing_review(self) -> int:
        return sum(len(d.pages_needing_review) for d in self.documents)


# --------------------------------------------------------------------------- #
# Penemuan berkas & metadata
# --------------------------------------------------------------------------- #


def discover_pdfs(paths: Sequence[str | Path], recursive: bool = True) -> list[Path]:
    """Kumpulkan semua PDF dari daftar berkas/folder, urut stabil."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            found.extend(sorted(q for q in p.glob(pattern) if q.is_file()))
            if recursive:
                found.extend(sorted(q for q in p.glob("**/*.PDF") if q.is_file()))
        elif p.is_file():
            if p.suffix.lower() != ".pdf":
                get_logger().warning("dilewati (bukan PDF): %s", p)
                continue
            found.append(p)
        else:
            raise FileNotFoundError(f"Tidak ditemukan: {p}")
    # Buang duplikat sambil menjaga urutan.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def load_metadata(folders: Iterable[str | Path]) -> dict[str, dict[str, str]]:
    """Muat metadata.csv dari folder masukan (opsional).

    Kunci hasil adalah nama berkas huruf kecil, agar pencocokan tidak
    bergantung pada kapitalisasi (penting di Windows).
    """
    log = get_logger()
    out: dict[str, dict[str, str]] = {}
    for folder in folders:
        folder = Path(folder)
        if folder.is_file():
            folder = folder.parent
        for name in METADATA_FILENAMES:
            path = folder / name
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        mapped: dict[str, str] = {}
                        for key, value in row.items():
                            if key is None:
                                continue
                            field_name = _META_COLUMNS.get(key.strip().lower())
                            if field_name and value and value.strip():
                                mapped[field_name] = value.strip()
                        src = mapped.pop("source_file", "")
                        if src:
                            out[Path(src).name.lower()] = mapped
                log.info("metadata dimuat dari %s (%d baris)", path, len(out))
            except Exception as exc:
                log.warning("gagal membaca %s: %s", path, exc)
    return out


def build_doc_meta(
    path: Path, probe: DocProbe, overrides: dict[str, str] | None
) -> DocMeta:
    """Susun metadata dokumen: metadata.csv > metadata PDF > nama berkas."""
    pdf_title = str((probe.pdf_metadata or {}).get("title") or "").strip()
    # Banyak PDF hasil ekspor Word memiliki title berupa nama berkas atau
    # teks pengolah kata; nilai seperti itu tidak lebih informatif daripada
    # nama berkas yang sudah dirapikan.
    if pdf_title.lower().endswith((".doc", ".docx", ".pdf")) or len(pdf_title) < 4:
        pdf_title = ""

    title = pdf_title or path.stem.replace("_", " ").replace("-", " ").strip()
    meta = DocMeta(
        doc_id=make_doc_id(path),
        title=title,
        source_file=str(path),
        category=_guess_category(path.name),
    )
    if overrides:
        for key, value in overrides.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
    if not meta.doc_id:
        meta.doc_id = slugify(path.stem)
    meta.extra.setdefault("ukuran_berkas", human_size(path.stat().st_size))
    meta.extra.setdefault("jenis_dokumen", probe.kind)
    return meta


_CATEGORY_HINTS = (
    ("kalender", "kalender-akademik"),
    ("panduan", "panduan-akademik"),
    ("pedoman", "panduan-akademik"),
    ("buku-pedoman", "panduan-akademik"),
    ("kurikulum", "kurikulum"),
    ("sk-", "surat-keputusan"),
    ("surat-keputusan", "surat-keputusan"),
    ("keputusan", "surat-keputusan"),
    ("biaya", "biaya-kuliah"),
    ("ukt", "biaya-kuliah"),
    ("spp", "biaya-kuliah"),
    ("skripsi", "panduan-skripsi"),
    ("kkn", "panduan-kkn"),
    ("magang", "panduan-magang"),
    ("beasiswa", "beasiswa"),
    ("peraturan", "peraturan"),
    ("statuta", "peraturan"),
    ("sop", "sop-layanan"),
    ("formulir", "formulir"),
    ("form-", "formulir"),
)


def _guess_category(filename: str) -> str:
    name = slugify(filename)
    for hint, category in _CATEGORY_HINTS:
        if hint in name:
            return category
    return "lain-lain"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class Pipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.log = get_logger()
        self._native: Engine | None = None
        self._ocr: Engine | None = None
        self.engine_notes: list[str] = []
        self._ocr_failed: str | None = None

    # ------------------------------------------------------------------ #

    def _native_engine(self) -> Engine:
        if self._native is None:
            self._native = get_engine(self.cfg.native_engine, self.cfg)
            avail = self._native.availability()
            if not avail.ok:
                raise EngineError(
                    f"engine native tidak tersedia: {avail.message}. "
                    f"{avail.install_hint}"
                )
        return self._native

    def _ocr_engine(self) -> Engine | None:
        """Kembalikan engine OCR, atau None bila tidak ada yang tersedia."""
        if self._ocr is not None:
            return self._ocr
        if self._ocr_failed is not None:
            return None
        try:
            engine, notes = resolve_ocr_engine(self.cfg)
            self.engine_notes.extend(notes)
            t0 = time.perf_counter()
            engine.warmup()
            warm = time.perf_counter() - t0
            if warm > 1.0:
                self.engine_notes.append(
                    f"waktu muat model {engine.name}: {warm:.1f} s "
                    "(tidak dihitung sebagai waktu pemrosesan halaman)"
                )
            self._ocr = engine
            return engine
        except EngineError as exc:
            self._ocr_failed = str(exc)
            self.engine_notes.append(f"TIDAK ADA ENGINE OCR: {exc}")
            self.log.error("tidak ada engine OCR yang tersedia:\n%s", exc)
            return None

    def close(self) -> None:
        for engine in (self._ocr, self._native):
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #

    def process_document(
        self, path: str | Path, overrides: dict[str, str] | None = None
    ) -> DocResult:
        path = Path(path)
        t0 = time.perf_counter()
        self.log.info("=== %s ===", path.name)

        probe = probe_document(path, self.cfg.detect, self.cfg.pages)
        meta = build_doc_meta(path, probe, overrides)
        doc = DocResult(meta=meta, n_pages=probe.n_pages)

        if probe.error:
            doc.doc_warnings.append(probe.error)
            doc.elapsed_s = time.perf_counter() - t0
            self.log.error("%s: %s", path.name, probe.error)
            return doc

        native_pages = probe.pages_by_route(ROUTE_NATIVE)
        ocr_pages = probe.pages_by_route(ROUTE_OCR)
        empty_pages = probe.pages_by_route(ROUTE_EMPTY)

        self.log.info(
            "%d halaman: %d native, %d OCR, %d kosong (jenis dokumen: %s)",
            len(probe.pages),
            len(native_pages),
            len(ocr_pages),
            len(empty_pages),
            probe.kind,
        )

        results: dict[int, PageResult] = {}

        # --- halaman dengan text layer ---
        if native_pages:
            try:
                results.update(self._native_engine().extract(path, native_pages))
            except Exception as exc:
                doc.doc_warnings.append(f"ekstraksi native gagal: {exc}")
                self.log.error("ekstraksi native gagal: %s", exc)

        # --- halaman yang perlu OCR ---
        if ocr_pages:
            engine = self._ocr_engine()
            if engine is None:
                doc.doc_warnings.append(
                    f"{len(ocr_pages)} halaman memerlukan OCR tetapi tidak ada "
                    "engine OCR yang tersedia; halaman tersebut dicoba dengan "
                    "engine native (hasil kemungkinan kosong/tidak lengkap). "
                    f"Alasan: {self._ocr_failed}"
                )
                try:
                    degraded = self._native_engine().extract(path, ocr_pages)
                    for page_no, pr in degraded.items():
                        pr.route = ROUTE_OCR
                        pr.warnings.append(
                            "tidak ada engine OCR — diekstrak paksa dengan "
                            "engine native (mode terdegradasi)"
                        )
                        results[page_no] = pr
                except Exception as exc:
                    self.log.error("mode terdegradasi juga gagal: %s", exc)
            else:
                try:
                    results.update(engine.extract(path, ocr_pages))
                except Exception as exc:
                    doc.doc_warnings.append(f"OCR ({engine.name}) gagal: {exc}")
                    self.log.error("OCR (%s) gagal: %s", engine.name, exc)

        # --- halaman kosong ---
        for page_no in empty_pages:
            results[page_no] = PageResult(
                page=page_no, route=ROUTE_EMPTY, engine="-"
            )

        # --- gabungkan + sisipkan alasan keputusan rute ---
        reasons = {p.page: p for p in probe.pages}
        for page_no in sorted(results):
            pr = results[page_no]
            pp = reasons.get(page_no)
            if pp is not None:
                pr.metrics["alasan_rute"] = "; ".join(pp.reasons)
                pr.metrics["rasio_area_gambar"] = round(pp.image_area_ratio, 3)
                pr.metrics["karakter_text_layer"] = pp.n_chars
            doc.pages.append(pr)

        doc.elapsed_s = time.perf_counter() - t0
        evaluate_document(doc, self.cfg.qa)
        self.log.info(
            "selesai dalam %.1f s — %d blok, %d tabel, %d halaman perlu tinjauan",
            doc.elapsed_s,
            sum(len(p.blocks) for p in doc.pages),
            len(doc.tables),
            len(doc.pages_needing_review),
        )
        return doc

    # ------------------------------------------------------------------ #

    def run(
        self,
        inputs: Sequence[str | Path],
        out_dir: str | Path,
        *,
        recursive: bool = True,
    ) -> RunResult:
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pdfs = discover_pdfs(inputs, recursive=recursive)
        if not pdfs:
            raise FileNotFoundError(
                f"Tidak ada berkas PDF pada: {', '.join(str(p) for p in inputs)}"
            )
        self.log.info("menemukan %d berkas PDF", len(pdfs))

        metadata = load_metadata(inputs)
        chunker = Chunker(self.cfg.chunk)
        run = RunResult()

        combined_path = out_dir / "jsonl" / "semua-chunk.jsonl"
        if self.cfg.output.combined_jsonl and combined_path.exists():
            combined_path.unlink()

        doc_summaries: list[dict[str, Any]] = []

        for i, pdf in enumerate(pdfs, start=1):
            self.log.info("[%d/%d] %s", i, len(pdfs), pdf.name)
            try:
                doc = self.process_document(pdf, metadata.get(pdf.name.lower()))
            except Exception as exc:
                msg = f"{pdf.name}: gagal diproses — {exc}"
                run.errors.append(msg)
                self.log.exception("gagal memproses %s", pdf.name)
                if self.cfg.continue_on_error:
                    continue
                raise

            run.documents.append(doc)

            chunks = chunker.chunk_document(doc)
            run.chunks[doc.meta.doc_id] = chunks

            outputs: dict[str, Any] = {}
            if self.cfg.output.markdown:
                outputs["markdown"] = str(
                    write_markdown(doc, out_dir / "markdown", self.cfg.output)
                )
            if self.cfg.output.jsonl:
                outputs["jsonl"] = str(
                    write_jsonl(chunks, out_dir / "jsonl" / f"{doc.meta.doc_id}.jsonl")
                )
                if self.cfg.output.combined_jsonl:
                    append_jsonl(chunks, combined_path)
            if self.cfg.output.csv_tables or self.cfg.output.xlsx_tables:
                outputs["tabel"] = write_tables(doc, out_dir / "tabel", self.cfg.output)

            summary = document_summary(doc)
            summary["keluaran"] = outputs
            summary["jml_chunk"] = len(chunks)
            doc_summaries.append(summary)

        # --- laporan & jejak run ---
        if self.cfg.output.qa_report and run.documents:
            md, csv_path = write_qa_report(
                run.documents,
                out_dir / "qa",
                engine_notes=self.engine_notes,
                config_summary=self._config_summary(),
            )
            run.outputs["qa_markdown"] = str(md)
            run.outputs["qa_csv"] = str(csv_path)

        if self.cfg.output.combined_jsonl and combined_path.exists():
            run.outputs["jsonl_gabungan"] = str(combined_path)

        run.outputs["contoh_ingest"] = str(
            write_ingest_example(out_dir / "contoh-ingest-qdrant.py")
        )
        self.cfg.dump_yaml(out_dir / "konfigurasi-efektif.yaml")
        run.outputs["konfigurasi"] = str(out_dir / "konfigurasi-efektif.yaml")

        run.elapsed_s = time.perf_counter() - t0
        run.engine_notes = list(self.engine_notes)
        finished = datetime.now(timezone.utc)
        run.outputs["manifest"] = str(
            write_manifest(
                out_dir / "manifest.json",
                config=self.cfg,
                documents=doc_summaries,
                engine_notes=self.engine_notes,
                started_at=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                finished_at=finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
                elapsed_s=run.elapsed_s,
            )
        )
        self.close()
        return run

    # ------------------------------------------------------------------ #

    def _config_summary(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "ocr_engine": c.ocr_engine,
            "dpi": c.preprocess.dpi,
            "deskew": c.preprocess.deskew,
            "binarize": c.preprocess.binarize,
            "detect.min_chars": c.detect.min_chars,
            "detect.max_garbled_ratio": c.detect.max_garbled_ratio,
            "chunk.target_chars": c.chunk.target_chars,
            "chunk.overlap_chars": c.chunk.overlap_chars,
            "qa.min_mean_confidence": c.qa.min_mean_confidence,
        }
