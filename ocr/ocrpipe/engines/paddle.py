"""Engine `paddle`: PP-StructureV3 (PaddleOCR >= 3.5).

Diverifikasi terhadap API PaddleOCR 3.x:
  * kelasnya `from paddleocr import PPStructureV3` — kelas lama `PPStructure`
    (PaddleOCR 2.x) SUDAH TIDAK ADA dan akan gagal diimpor;
  * `predict()` mengembalikan list, `predict_iter()` versi generator;
  * hasil memiliki atribut `markdown` (dict), `json`, dan `parsing_res_list`
    (daftar blok dalam urutan baca) serta metode `save_to_*`;
  * `lang="id"` adalah nilai sah dan di-resolve ke model Latin PP-OCRv5.
    Default `lang` adalah "ch" (Tionghoa+Inggris) yang akurasinya buruk
    untuk teks Latin — jadi `lang` WAJIB diisi eksplisit.

Catatan sumber daya: konfigurasi DEFAULT PP-StructureV3 dapat memakai
>16 GB VRAM karena mengaktifkan pengenalan formula dan model berukuran
server. Konfigurasi di `PaddleConfig` karenanya mematikan modul yang tidak
diperlukan dokumen akademik dan memilih model mobile. Dengan itu pipeline
masih berjalan pada GPU 6 GB; bila tetap OOM, jalankan `device="cpu"` atau
pindah ke Kaggle.
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
from ..preprocess import preprocess_page
from ..tables import build_table, html_table_to_rows, table_to_markdown
from ..utils import clean_text, get_logger
from .base import Availability, Engine, EngineError, module_installed, version_of

#: Pemetaan label layout PP-StructureV3 -> jenis blok internal.
_LABEL_MAP = {
    "text": KIND_PARAGRAPH,
    "paragraph": KIND_PARAGRAPH,
    "abstract": KIND_PARAGRAPH,
    "content": KIND_PARAGRAPH,
    "reference": KIND_PARAGRAPH,
    "reference_content": KIND_PARAGRAPH,
    "title": KIND_HEADING,
    "doc_title": KIND_HEADING,
    "paragraph_title": KIND_HEADING,
    "abstract_title": KIND_HEADING,
    "reference_title": KIND_HEADING,
    "content_title": KIND_HEADING,
    "table": KIND_TABLE,
    "table_title": KIND_CAPTION,
    "figure_title": KIND_CAPTION,
    "chart_title": KIND_CAPTION,
    "figure": KIND_FIGURE,
    "image": KIND_FIGURE,
    "chart": KIND_FIGURE,
    "formula": KIND_FORMULA,
    "formula_number": KIND_FORMULA,
    "header": KIND_HEADER,
    "header_image": KIND_HEADER,
    "footer": KIND_FOOTER,
    "footer_image": KIND_FOOTER,
    "footnote": KIND_FOOTER,
    "number": KIND_FOOTER,
    "seal": KIND_FIGURE,
    "aside_text": KIND_PARAGRAPH,
    "algorithm": KIND_PARAGRAPH,
    "list": KIND_LIST,
}

#: Label yang tidak dipetakan ke heading tetapi harus diberi tingkat.
_HEADING_LEVELS = {
    "doc_title": 1,
    "title": 1,
    "abstract_title": 2,
    "content_title": 2,
    "reference_title": 2,
    "paragraph_title": 2,
}


class PaddleStructureEngine(Engine):
    name = "paddle"
    description = "PaddleOCR PP-StructureV3 — layout + struktur tabel (GPU disarankan)"
    is_neural = True
    does_ocr = True
    whole_document_only = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._pipeline: Any | None = None

    # ------------------------------------------------------------------ #

    def availability(self) -> Availability:
        if not module_installed("paddleocr"):
            return Availability(
                ok=False,
                message="paket 'paddleocr' belum terpasang",
                install_hint=(
                    'pip install "paddleocr[doc-parser]" lalu pasang mesin '
                    "inferensi: pip install paddlepaddle==3.2.0 -i "
                    "https://www.paddlepaddle.org.cn/packages/stable/cpu/ "
                    "(CPU) atau paddlepaddle-gpu dengan indeks cu126 (GPU)"
                ),
            )
        if not module_installed("paddle"):
            return Availability(
                ok=False,
                version=version_of("paddleocr"),
                message=(
                    "paddleocr terpasang tetapi mesin inferensi PaddlePaddle "
                    "belum ada (sejak paddleocr 3.5 keduanya dipasang terpisah)"
                ),
                install_hint=(
                    "pip install paddlepaddle==3.2.0 -i "
                    "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
                ),
            )
        return Availability(
            ok=True,
            version=f"paddleocr {version_of('paddleocr')} / paddlepaddle {version_of('paddlepaddle') or version_of('paddlepaddle-gpu')}",
        )

    # ------------------------------------------------------------------ #

    def warmup(self) -> None:
        self._get_pipeline()

    def close(self) -> None:
        self._pipeline = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        avail = self.availability()
        if not avail.ok:
            raise EngineError(avail.message)

        try:
            from paddleocr import PPStructureV3  # type: ignore
        except ImportError as exc:
            raise EngineError(
                "gagal mengimpor PPStructureV3. Pada PaddleOCR 3.x kelas ini "
                "membutuhkan grup dependensi 'doc-parser': "
                'pip install "paddleocr[doc-parser]". '
                f"(detail: {exc})"
            ) from exc

        p = self.cfg.paddle
        kwargs: dict[str, Any] = {
            "lang": p.lang,
            "use_table_recognition": p.use_table_recognition,
            "use_formula_recognition": p.use_formula_recognition,
            "use_chart_recognition": p.use_chart_recognition,
            "use_seal_recognition": p.use_seal_recognition,
            "use_doc_orientation_classify": p.use_doc_orientation_classify,
            "use_doc_unwarping": p.use_doc_unwarping,
            "use_textline_orientation": p.use_textline_orientation,
            "use_region_detection": p.use_region_detection,
        }
        if p.device:
            kwargs["device"] = p.device
        if p.text_rec_score_thresh is not None:
            kwargs["text_rec_score_thresh"] = p.text_rec_score_thresh
        if p.prefer_mobile_models:
            # Model mobile jauh lebih ringan; wajib untuk GPU 6 GB.
            kwargs["layout_detection_model_name"] = "PP-DocLayout-S"

        log = get_logger()
        log.info("memuat PP-StructureV3 (lang=%s, device=%s)", p.lang, p.device or "auto")
        try:
            self._pipeline = PPStructureV3(**kwargs)
        except TypeError as exc:
            # Satu atau lebih kwargs tidak dikenal di versi terpasang:
            # coba lagi dengan set minimal agar tetap bisa berjalan.
            log.warning(
                "sebagian argumen PP-StructureV3 tidak dikenal (%s); "
                "mencoba ulang dengan konfigurasi minimal",
                exc,
            )
            minimal = {"lang": p.lang}
            if p.device:
                minimal["device"] = p.device
            self._pipeline = PPStructureV3(**minimal)
        except Exception as exc:
            raise EngineError(
                f"gagal menginisialisasi PP-StructureV3: {exc}. "
                "Bila pesannya soal memori, matikan modul tambahan atau "
                "jalankan dengan paddle.device='cpu'."
            ) from exc
        return self._pipeline

    # ------------------------------------------------------------------ #

    def extract(self, pdf_path: str | Path, pages: list[int]) -> dict[int, PageResult]:
        pipeline = self._get_pipeline()
        log = get_logger()
        out: dict[int, PageResult] = {}

        if self.cfg.paddle.feed_pdf_directly:
            return self._extract_whole_pdf(pipeline, pdf_path, pages)

        with tempfile.TemporaryDirectory(prefix="ocrpipe_paddle_") as tmpdir:
            for page_no in pages:
                t0 = time.perf_counter()
                pr = PageResult(page=page_no, route=ROUTE_OCR, engine=self.name)
                try:
                    prep = preprocess_page(
                        pdf_path, page_no, self.cfg.preprocess, for_neural_engine=True
                    )
                    pr.warnings.extend(prep.warnings)
                    pr.metrics["praproses"] = ", ".join(prep.applied)
                    pr.metrics["dpi"] = prep.dpi

                    # Menulis ke berkas sementara menghindari ambiguitas urutan
                    # kanal (BGR vs RGB) saat mengirim array numpy.
                    img_path = Path(tmpdir) / f"hlm_{page_no:05d}.png"
                    prep.save(img_path)

                    results = pipeline.predict(str(img_path))
                    results = list(results) if results is not None else []
                    if not results:
                        pr.warnings.append("PP-StructureV3 tidak mengembalikan hasil")
                    else:
                        pr.blocks = self._blocks_from_result(results[0], page_no, pr)
                        conf = _mean_rec_score(results[0])
                        if conf is not None:
                            for b in pr.blocks:
                                if b.confidence is None:
                                    b.confidence = conf
                except Exception as exc:
                    log.warning("hlm %d gagal (paddle): %s", page_no, exc)
                    pr.warnings.append(f"PP-StructureV3 gagal: {exc}")
                pr.elapsed_s = round(time.perf_counter() - t0, 4)
                out[page_no] = pr
        return out

    def _extract_whole_pdf(
        self, pipeline: Any, pdf_path: str | Path, pages: list[int]
    ) -> dict[int, PageResult]:
        """Kirim PDF utuh ke pipeline dan ambil halaman yang diminta.

        Lebih cepat untuk dokumen yang seluruh halamannya perlu OCR, tetapi
        melewatkan tahap praproses (deskew) milik pipeline ini.
        """
        log = get_logger()
        wanted = set(pages)
        out: dict[int, PageResult] = {
            p: PageResult(page=p, route=ROUTE_OCR, engine=self.name) for p in pages
        }
        t0 = time.perf_counter()
        try:
            iterator = (
                pipeline.predict_iter(input=str(pdf_path))
                if hasattr(pipeline, "predict_iter")
                else pipeline.predict(input=str(pdf_path))
            )
            for i, res in enumerate(iterator):
                page_no = _page_index(res)
                page_no = (page_no + 1) if page_no is not None else (i + 1)
                if page_no not in wanted:
                    continue
                pr = out[page_no]
                pr.blocks = self._blocks_from_result(res, page_no, pr)
                conf = _mean_rec_score(res)
                if conf is not None:
                    for b in pr.blocks:
                        if b.confidence is None:
                            b.confidence = conf
        except Exception as exc:
            log.warning("pemrosesan PDF utuh gagal (paddle): %s", exc)
            for pr in out.values():
                pr.warnings.append(f"PP-StructureV3 gagal: {exc}")
        elapsed = round((time.perf_counter() - t0) / max(1, len(pages)), 4)
        for pr in out.values():
            pr.elapsed_s = elapsed
        return out

    # ------------------------------------------------------------------ #

    def _blocks_from_result(
        self, res: Any, page_no: int, pr: PageResult
    ) -> list[Block]:
        parsing = _get_field(res, "parsing_res_list")
        blocks: list[Block] = []

        if parsing:
            t_idx = 0
            for i, item in enumerate(parsing):
                label = str(_get_item(item, "block_label", "text") or "text").lower()
                content = _get_item(item, "block_content", "") or ""
                bbox = _normalize_bbox(_get_item(item, "block_bbox", None))
                kind = _LABEL_MAP.get(label, KIND_PARAGRAPH)

                if kind == KIND_TABLE:
                    t_idx += 1
                    block = self._table_block(
                        str(content), page_no, t_idx, bbox, pr
                    )
                    if block is None:
                        continue
                    blocks.append(block)
                    continue

                text = clean_text(str(content))
                if not text:
                    continue
                blocks.append(
                    Block(
                        kind=kind,
                        text=text,
                        page=page_no,
                        order=i,
                        level=_HEADING_LEVELS.get(label) if kind == KIND_HEADING else None,
                        bbox=bbox,
                        engine=self.name,
                    )
                )
        else:
            # Cadangan: pakai Markdown gabungan bila daftar blok tidak tersedia.
            md = _markdown_text(res)
            if md:
                pr.warnings.append(
                    "parsing_res_list tidak tersedia; memakai keluaran Markdown "
                    "(struktur per-blok dan bbox tidak terekam)"
                )
                blocks.append(
                    Block(
                        kind=KIND_PARAGRAPH,
                        text=clean_text(md),
                        page=page_no,
                        order=0,
                        engine=self.name,
                    )
                )
            else:
                pr.warnings.append("hasil PP-StructureV3 kosong / tidak dikenali")

        for i, b in enumerate(blocks):
            b.order = i
        return blocks

    def _table_block(
        self,
        content: str,
        page_no: int,
        index: int,
        bbox: tuple[float, float, float, float] | None,
        pr: PageResult,
    ) -> Block | None:
        rows: list[list[str]] = []
        if "<" in content and "tr" in content.lower():
            try:
                rows = html_table_to_rows(content)
            except Exception as exc:
                pr.warnings.append(f"parsing HTML tabel gagal: {exc}")
        if not rows:
            # Kadang block_content tabel sudah berupa Markdown.
            rows = _markdown_table_to_rows(content)
        if not rows:
            pr.warnings.append("tabel terdeteksi tetapi isinya tidak dapat diparse")
            return None

        table = build_table(
            rows, page=page_no, index=index, engine=self.name, bbox=bbox
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
            bbox=bbox,
            table=table,
            engine=self.name,
        )


# --------------------------------------------------------------------------- #
# Helper akses hasil (objek hasil PaddleX mendukung atribut maupun indeks)
# --------------------------------------------------------------------------- #


def _get_field(res: Any, key: str, default: Any = None) -> Any:
    for getter in (
        lambda: getattr(res, key),
        lambda: res[key],
        lambda: res.json["res"][key],
        lambda: res.json[key],
    ):
        try:
            value = getter()
            if value is not None:
                return value
        except Exception:
            continue
    return default


def _get_item(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _page_index(res: Any) -> int | None:
    value = _get_field(res, "page_index")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _markdown_text(res: Any) -> str:
    md = _get_field(res, "markdown")
    if isinstance(md, dict):
        return str(md.get("markdown_texts", "") or "")
    return str(md or "")


def _mean_rec_score(res: Any) -> float | None:
    """Rata-rata skor pengenalan teks (0..100) bila dilaporkan."""
    ocr = _get_field(res, "overall_ocr_res")
    if not ocr:
        return None
    scores = _get_item(ocr, "rec_scores", None)
    if not scores:
        return None
    try:
        vals = [float(s) for s in scores]
    except (TypeError, ValueError):
        return None
    if not vals:
        return None
    return round(sum(vals) / len(vals) * 100.0, 2)


def _normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        seq = [float(v) for v in list(value)]
    except (TypeError, ValueError):
        return None
    if len(seq) == 4:
        return (seq[0], seq[1], seq[2], seq[3])
    if len(seq) == 8:  # poligon 4 titik
        xs = seq[0::2]
        ys = seq[1::2]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _markdown_table_to_rows(text: str) -> list[list[str]]:
    """Parse pipe-table Markdown menjadi grid (cadangan)."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(set(c) <= set("-: ") and c for c in cells):
            continue  # baris pemisah header
        rows.append(cells)
    return rows
