"""Engine `docling`: Docling (IBM) dengan TableFormer untuk struktur tabel.

Diverifikasi terhadap API Docling 2.x:
  * `DocumentConverter().convert(path)` -> `ConversionResult`;
  * `res.document.iterate_items()` menghasilkan (item, level) dalam urutan
    baca, dan tiap item punya `prov[0].page_no` untuk atribusi halaman —
    inilah yang dipakai agar sitasi per halaman tetap akurat;
  * `TableItem.export_to_dataframe(doc=...)` dan `export_to_markdown(doc=...)`
    WAJIB diberi argumen `doc=`; tanpa itu Docling mengeluarkan peringatan
    deprecation dan perilakunya akan berubah pada versi berikutnya;
  * OCR default-nya `OcrAutoOptions` (memilih engine terbaik yang tersedia),
    BUKAN EasyOCR seperti yang masih tertulis di banyak tutorial. Karena
    reproduktibilitas penting untuk skripsi, engine OCR di sini diisi
    eksplisit (default: Tesseract CLI dengan bahasa ind+eng).

Dipilih sebagai preferensi pertama pada mode `auto` karena lisensinya MIT
menyeluruh (kode maupun bobot model), sehingga tidak ada batasan pemakaian
yang perlu dibahas khusus di laporan.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from ..models import (
    KIND_CAPTION,
    KIND_FIGURE,
    KIND_FOOTER,
    KIND_FORMULA,
    KIND_HEADER,
    KIND_HEADING,
    KIND_LIST,
    KIND_PARAGRAPH,
    KIND_TABLE,
    ROUTE_OCR,
    Block,
    PageResult,
)
from ..tables import build_table, dataframe_to_rows, table_to_markdown
from ..utils import clean_text, get_logger, make_subset_pdf
from .base import Availability, Engine, EngineError, module_installed, version_of

_LABEL_MAP = {
    "title": KIND_HEADING,
    "section_header": KIND_HEADING,
    "text": KIND_PARAGRAPH,
    "paragraph": KIND_PARAGRAPH,
    "list_item": KIND_LIST,
    "caption": KIND_CAPTION,
    "table": KIND_TABLE,
    "picture": KIND_FIGURE,
    "chart": KIND_FIGURE,
    "formula": KIND_FORMULA,
    "code": KIND_PARAGRAPH,
    "page_header": KIND_HEADER,
    "page_footer": KIND_FOOTER,
    "footnote": KIND_FOOTER,
    "document_index": KIND_PARAGRAPH,
    "reference": KIND_PARAGRAPH,
}


class DoclingEngine(Engine):
    name = "docling"
    description = "Docling + TableFormer (lisensi MIT, tabel paling rapi)"
    is_neural = True
    does_ocr = True
    whole_document_only = True

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._converter: Any | None = None

    # ------------------------------------------------------------------ #

    def availability(self) -> Availability:
        if not module_installed("docling"):
            return Availability(
                ok=False,
                message="paket 'docling' belum terpasang",
                install_hint="pip install docling",
            )
        ver = version_of("docling") or version_of("docling-slim")
        if self.cfg.docling.ocr_engine == "tesseract":
            import shutil

            if not shutil.which("tesseract"):
                win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
                if not win.exists():
                    return Availability(
                        ok=False,
                        version=ver,
                        message=(
                            "docling terpasang tetapi ocr_engine='tesseract' "
                            "dan biner tesseract tidak ditemukan"
                        ),
                        install_hint=(
                            "pasang Tesseract, atau ubah docling.ocr_engine "
                            "menjadi 'easyocr' / 'rapidocr' / 'auto'"
                        ),
                    )
        return Availability(ok=True, version=ver)

    # ------------------------------------------------------------------ #

    def warmup(self) -> None:
        self._get_converter()

    def close(self) -> None:
        self._converter = None

    def _ocr_options(self) -> Any:
        """Bangun objek opsi OCR sesuai konfigurasi."""
        from docling.datamodel import pipeline_options as po  # type: ignore

        d = self.cfg.docling
        engine = (d.ocr_engine or "auto").lower()
        langs = list(d.ocr_lang)

        if engine == "auto":
            return po.OcrAutoOptions(lang=langs) if langs else po.OcrAutoOptions()
        if engine == "tesseract":
            # Tesseract memakai kode ISO 639-2/T: Indonesia = "ind".
            return po.TesseractCliOcrOptions(lang=langs)
        if engine == "tesserocr":
            return po.TesseractOcrOptions(lang=langs)
        if engine == "easyocr":
            # EasyOCR memakai kode 2 huruf: Indonesia = "id".
            mapped = ["id" if lg in ("ind", "id") else "en" if lg in ("eng", "en") else lg for lg in langs]
            return po.EasyOcrOptions(lang=mapped)
        if engine == "rapidocr":
            return po.RapidOcrOptions()
        raise EngineError(
            f"docling.ocr_engine '{engine}' tidak dikenal "
            "(pilihan: auto, tesseract, tesserocr, easyocr, rapidocr)"
        )

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter

        avail = self.availability()
        if not avail.ok:
            raise EngineError(avail.message)

        log = get_logger()
        d = self.cfg.docling
        try:
            from docling.datamodel.base_models import InputFormat  # type: ignore
            from docling.document_converter import (  # type: ignore
                DocumentConverter,
                PdfFormatOption,
            )
        except ImportError as exc:
            raise EngineError(f"gagal mengimpor Docling: {exc}") from exc

        try:
            if d.use_vlm:
                from docling.datamodel import vlm_model_specs  # type: ignore
                from docling.datamodel.pipeline_options import (  # type: ignore
                    VlmPipelineOptions,
                )
                from docling.pipeline.vlm_pipeline import VlmPipeline  # type: ignore

                preset = {
                    "granitedocling": "GRANITEDOCLING_TRANSFORMERS",
                    "smoldocling": "SMOLDOCLING_TRANSFORMERS",
                }.get(d.vlm_preset.lower(), "GRANITEDOCLING_TRANSFORMERS")
                spec = getattr(vlm_model_specs, preset)
                opts = VlmPipelineOptions(vlm_options=spec)
                log.info("memuat Docling pipeline VLM (%s)", preset)
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_cls=VlmPipeline, pipeline_options=opts
                        )
                    }
                )
                return self._converter

            from docling.datamodel.pipeline_options import (  # type: ignore
                PdfPipelineOptions,
                TableFormerMode,
                TableStructureOptions,
            )

            mode = (
                TableFormerMode.FAST
                if d.table_mode.lower() == "fast"
                else TableFormerMode.ACCURATE
            )
            opts = PdfPipelineOptions(
                do_ocr=d.do_ocr,
                do_table_structure=d.do_table_structure,
                table_structure_options=TableStructureOptions(
                    mode=mode, do_cell_matching=True
                ),
                do_formula_enrichment=d.do_formula_enrichment,
                do_code_enrichment=d.do_code_enrichment,
                generate_page_images=False,
            )
            if d.do_ocr:
                opts.ocr_options = self._ocr_options()
            if d.force_backend_text:
                opts.force_backend_text = True

            log.info(
                "memuat Docling (ocr=%s/%s, tabel=%s)",
                d.do_ocr,
                d.ocr_engine,
                d.table_mode,
            )
            # Kunci format_options HARUS anggota enum InputFormat; memakai
            # string "pdf" menyebabkan AttributeError saat konversi.
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"gagal menyiapkan Docling: {exc}") from exc
        return self._converter

    # ------------------------------------------------------------------ #

    def extract(self, pdf_path: str | Path, pages: list[int]) -> dict[int, PageResult]:
        converter = self._get_converter()
        log = get_logger()
        pdf_path = Path(pdf_path)

        total = _page_count(pdf_path)
        full_run = set(pages) == set(range(1, total + 1))

        with tempfile.TemporaryDirectory(prefix="ocrpipe_docling_") as tmpdir:
            if full_run:
                target = pdf_path
                mapping = {i: i for i in pages}
            else:
                target = Path(tmpdir) / "subset.pdf"
                mapping = make_subset_pdf(pdf_path, pages, target)

            out: dict[int, PageResult] = {
                p: PageResult(page=p, route=ROUTE_OCR, engine=self.name) for p in pages
            }
            t0 = time.perf_counter()
            try:
                res = converter.convert(str(target))
                status = getattr(res, "status", None)
                status_name = getattr(status, "name", str(status)) if status else ""
                if status_name and status_name not in ("SUCCESS", "PARTIAL_SUCCESS"):
                    raise EngineError(f"konversi Docling berstatus {status_name}")
                if status_name == "PARTIAL_SUCCESS":
                    for pr in out.values():
                        pr.warnings.append(
                            "Docling melaporkan PARTIAL_SUCCESS — sebagian "
                            "halaman mungkin tidak terkonversi penuh"
                        )
                self._fill_from_result(res, mapping, out)
            except Exception as exc:
                log.warning("konversi Docling gagal: %s", exc)
                for pr in out.values():
                    pr.warnings.append(f"Docling gagal: {exc}")

            elapsed = round((time.perf_counter() - t0) / max(1, len(pages)), 4)
            for pr in out.values():
                pr.elapsed_s = elapsed
            return out

    # ------------------------------------------------------------------ #

    def _fill_from_result(
        self, res: Any, mapping: dict[int, int], out: dict[int, PageResult]
    ) -> None:
        doc = getattr(res, "document", None)
        if doc is None:
            raise EngineError("hasil konversi tidak memiliki atribut 'document'")

        orders: dict[int, int] = {p: 0 for p in out}
        table_counters: dict[int, int] = {p: 0 for p in out}
        used_iterate = False

        try:
            items = list(doc.iterate_items())
            used_iterate = True
        except Exception:
            items = []

        if used_iterate and items:
            for entry in items:
                item, level = entry if isinstance(entry, tuple) else (entry, 0)
                sub_page = _item_page(item)
                if sub_page is None:
                    continue
                page_no = mapping.get(sub_page)
                if page_no is None or page_no not in out:
                    continue
                pr = out[page_no]
                label = _item_label(item)
                kind = _LABEL_MAP.get(label, KIND_PARAGRAPH)

                if kind == KIND_TABLE:
                    table_counters[page_no] += 1
                    block = self._table_block(
                        item, doc, page_no, table_counters[page_no], pr
                    )
                    if block is None:
                        table_counters[page_no] -= 1
                        continue
                else:
                    text = clean_text(str(getattr(item, "text", "") or ""))
                    if not text:
                        continue
                    block = Block(
                        kind=kind,
                        text=text,
                        page=page_no,
                        level=_heading_level(item, label, level),
                        bbox=_item_bbox(item),
                        engine=self.name,
                    )
                block.order = orders[page_no]
                orders[page_no] += 1
                pr.blocks.append(block)
        else:
            # Cadangan: tanpa iterate_items, atribusi halaman tidak mungkin
            # dilakukan dengan tepat. Seluruh Markdown ditaruh di halaman
            # pertama dan ditandai agar tidak dipakai untuk sitasi halaman.
            md = ""
            try:
                md = doc.export_to_markdown()
            except Exception as exc:
                raise EngineError(f"tidak dapat mengekspor Markdown: {exc}") from exc
            first = min(out)
            out[first].blocks.append(
                Block(
                    kind=KIND_PARAGRAPH,
                    text=clean_text(md),
                    page=first,
                    order=0,
                    engine=self.name,
                )
            )
            out[first].warnings.append(
                "iterate_items() tidak tersedia; atribusi halaman tidak akurat "
                "— jangan pakai hasil halaman ini untuk sitasi per halaman"
            )

    def _table_block(
        self, item: Any, doc: Any, page_no: int, index: int, pr: PageResult
    ) -> Block | None:
        rows: list[list[str]] = []
        caption: str | None = None

        try:
            caption_text = item.caption_text(doc)
            caption = clean_text(str(caption_text)) or None
        except Exception:
            caption = None

        # Jalur utama: DataFrame (paling bersih, sel sudah dipetakan).
        try:
            df = item.export_to_dataframe(doc=doc)
            rows = dataframe_to_rows(df)
        except TypeError:
            # Versi lama tidak menerima argumen doc=.
            try:
                df = item.export_to_dataframe()
                rows = dataframe_to_rows(df)
                pr.warnings.append(
                    "export_to_dataframe() dipanggil tanpa doc= (Docling versi lama)"
                )
            except Exception as exc:
                pr.warnings.append(f"ekspor tabel ke DataFrame gagal: {exc}")
        except Exception as exc:
            pr.warnings.append(f"ekspor tabel ke DataFrame gagal: {exc}")

        if not rows:
            # Cadangan: ambil grid langsung dari TableData Docling.
            rows = _rows_from_table_data(item)

        if not rows:
            return None

        table = build_table(
            rows,
            page=page_no,
            index=index,
            engine=self.name,
            caption=caption,
            bbox=_item_bbox(item),
        )
        if table.is_degenerate():
            return None
        md = table_to_markdown(table)
        if not md:
            return None
        return Block(
            kind=KIND_TABLE,
            text=md,
            page=page_no,
            bbox=table.bbox,
            table=table,
            engine=self.name,
        )


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def _page_count(path: Path) -> int:
    try:
        import pymupdf

        doc = pymupdf.open(str(path))
        try:
            return doc.page_count
        finally:
            doc.close()
    except Exception:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)


def _item_label(item: Any) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return "text"
    return str(getattr(label, "value", label)).lower()


def _item_page(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    try:
        page = getattr(prov[0], "page_no", None)
        return int(page) if page is not None else None
    except (IndexError, TypeError, ValueError):
        return None


def _item_bbox(item: Any) -> tuple[float, float, float, float] | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    try:
        bbox = getattr(prov[0], "bbox", None)
        if bbox is None:
            return None
        return (
            float(bbox.l),
            float(bbox.t),
            float(bbox.r),
            float(bbox.b),
        )
    except Exception:
        return None


def _heading_level(item: Any, label: str, tree_level: int) -> int | None:
    if label not in ("title", "section_header"):
        return None
    if label == "title":
        return 1
    lvl = getattr(item, "level", None)
    try:
        return max(1, min(6, int(lvl)))
    except (TypeError, ValueError):
        return max(1, min(6, int(tree_level) or 2))


def _rows_from_table_data(item: Any) -> list[list[str]]:
    """Baca grid langsung dari struktur TableData Docling (cadangan)."""
    data = getattr(item, "data", None)
    if data is None:
        return []
    grid = getattr(data, "grid", None)
    if grid:
        out: list[list[str]] = []
        for row in grid:
            out.append([str(getattr(c, "text", "") or "") for c in row])
        return out
    cells = getattr(data, "table_cells", None)
    if not cells:
        return []
    n_rows = int(getattr(data, "num_rows", 0) or 0)
    n_cols = int(getattr(data, "num_cols", 0) or 0)
    if n_rows <= 0 or n_cols <= 0:
        return []
    out = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for cell in cells:
        try:
            r0 = int(getattr(cell, "start_row_offset_idx", 0))
            r1 = int(getattr(cell, "end_row_offset_idx", r0 + 1))
            c0 = int(getattr(cell, "start_col_offset_idx", 0))
            c1 = int(getattr(cell, "end_col_offset_idx", c0 + 1))
            text = str(getattr(cell, "text", "") or "")
        except (TypeError, ValueError):
            continue
        for r in range(max(0, r0), min(n_rows, r1)):
            for c in range(max(0, c0), min(n_cols, c1)):
                out[r][c] = text
    return out
