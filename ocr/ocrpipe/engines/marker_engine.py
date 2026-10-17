"""Engine `marker`: marker-pdf 2.x (Surya VLM).

Diverifikasi terhadap API Marker 2.x:
  * `PdfConverter(artifact_dict=create_model_dict())`, lalu `converter(path)`;
  * `text_from_rendered(rendered)` mengembalikan **(teks, ekstensi, gambar)** —
    elemen tengah adalah "md"/"html"/"json", BUKAN metadata. Metadata ada di
    `rendered.metadata`;
  * argumen `--languages` SUDAH DIHAPUS pada 2.x (OCR ditangani Surya VLM
    yang multibahasa), sehingga tidak ada kode bahasa yang perlu dikirim;
  * `mode` otomatis: "balanced" di CUDA, "fast" di CPU/MPS;
  * di CPU tanpa `disable_ocr`, Marker membutuhkan server inferensi llama.cpp
    yang TIDAK terpasang lewat pip.

PERINGATAN LISENSI yang harus ditulis apa adanya di laporan: kode Marker
berlisensi Apache-2.0, tetapi BOBOT MODEL-nya memakai modified AI Pubs Open
RAIL-M — bebas untuk riset dan penggunaan pribadi, dengan batasan untuk
penggunaan komersial. Untuk skripsi ini termasuk penggunaan riset, tetapi
lisensi bobot tidak boleh disamakan dengan lisensi kode.

Renderer yang dipakai adalah "chunks" karena hanya renderer itu yang
memberi atribusi halaman per blok secara eksplisit. Karena bentuk objek
keluarannya bisa berubah antar versi minor, seluruh pembacaan hasil di
bawah ini dibuat defensif dengan beberapa lapis cadangan.
"""

from __future__ import annotations

import re
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
from ..tables import build_table, html_table_to_rows, table_to_markdown
from ..utils import clean_text, get_logger, make_subset_pdf
from .base import Availability, Engine, EngineError, module_installed, version_of

_BLOCK_TYPE_MAP = {
    "text": KIND_PARAGRAPH,
    "span": KIND_PARAGRAPH,
    "line": KIND_PARAGRAPH,
    "sectionheader": KIND_HEADING,
    "section_header": KIND_HEADING,
    "title": KIND_HEADING,
    "table": KIND_TABLE,
    "tableofcontents": KIND_PARAGRAPH,
    "listitem": KIND_LIST,
    "listgroup": KIND_LIST,
    "caption": KIND_CAPTION,
    "figure": KIND_FIGURE,
    "picture": KIND_FIGURE,
    "picturegroup": KIND_FIGURE,
    "figuregroup": KIND_FIGURE,
    "equation": KIND_FORMULA,
    "textinlinemath": KIND_PARAGRAPH,
    "pageheader": KIND_HEADER,
    "pagefooter": KIND_FOOTER,
    "footnote": KIND_FOOTER,
    "code": KIND_PARAGRAPH,
    "form": KIND_TABLE,
}

_TAG_RE = re.compile(r"<[^>]+>")
_PAGE_IN_ID_RE = re.compile(r"/page/(\d+)")


class MarkerEngine(Engine):
    name = "marker"
    description = "marker-pdf 2.x / Surya VLM (butuh GPU; lisensi bobot terbatas)"
    is_neural = True
    does_ocr = True
    whole_document_only = True

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._converter: Any | None = None
        self._models: Any | None = None

    # ------------------------------------------------------------------ #

    def availability(self) -> Availability:
        if not module_installed("marker"):
            return Availability(
                ok=False,
                message="paket 'marker-pdf' belum terpasang",
                install_hint="pip install marker-pdf",
            )
        ver = version_of("marker-pdf")
        if not module_installed("torch"):
            return Availability(
                ok=False,
                version=ver,
                message="marker terpasang tetapi PyTorch tidak ada",
                install_hint="pip install torch (sesuaikan dengan versi CUDA)",
            )
        if not self.cfg.marker.disable_ocr and not _cuda_available():
            import shutil

            if not shutil.which("llama-server") and not shutil.which("llama-cli"):
                return Availability(
                    ok=False,
                    version=ver,
                    message=(
                        "tanpa GPU CUDA, Marker memerlukan server inferensi "
                        "llama.cpp untuk menjalankan Surya VLM"
                    ),
                    install_hint=(
                        "pasang biner llama.cpp, ATAU set marker.disable_ocr=true "
                        "(jalur CPU murni, tetapi halaman hasil scan dilewati), "
                        "ATAU pakai engine 'docling'/'tesseract'"
                    ),
                )
        return Availability(ok=True, version=ver)

    # ------------------------------------------------------------------ #

    def warmup(self) -> None:
        self._get_converter()

    def close(self) -> None:
        if self._models is not None:
            try:
                from marker.models import shutdown_models  # type: ignore

                shutdown_models(self._models)
            except Exception:
                pass
        self._converter = None
        self._models = None

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter

        avail = self.availability()
        if not avail.ok:
            raise EngineError(avail.message)

        log = get_logger()
        m = self.cfg.marker
        try:
            from marker.converters.pdf import PdfConverter  # type: ignore
            from marker.models import create_model_dict  # type: ignore
        except ImportError as exc:
            raise EngineError(f"gagal mengimpor Marker: {exc}") from exc

        config: dict[str, Any] = {
            "output_format": m.output_format,
            "disable_image_extraction": m.disable_image_extraction,
            "force_ocr": m.force_ocr,
            "disable_ocr": m.disable_ocr,
            "paginate_output": True,  # dipakai oleh jalur cadangan Markdown
        }
        if m.mode:
            config["mode"] = m.mode

        log.info("memuat Marker (format=%s, mode=%s)", m.output_format, m.mode or "auto")
        try:
            self._models = create_model_dict()
            kwargs: dict[str, Any] = {"artifact_dict": self._models, "config": config}
            try:
                from marker.config.parser import ConfigParser  # type: ignore

                parser = ConfigParser(config)
                kwargs = {
                    "artifact_dict": self._models,
                    "config": parser.generate_config_dict(),
                    "processor_list": parser.get_processors(),
                    "renderer": parser.get_renderer(),
                }
            except Exception as exc:
                log.debug("ConfigParser tidak dipakai (%s); memakai config mentah", exc)
            self._converter = PdfConverter(**kwargs)
        except Exception as exc:
            raise EngineError(f"gagal menyiapkan Marker: {exc}") from exc
        return self._converter

    # ------------------------------------------------------------------ #

    def extract(self, pdf_path: str | Path, pages: list[int]) -> dict[int, PageResult]:
        converter = self._get_converter()
        log = get_logger()
        pdf_path = Path(pdf_path)

        total = _page_count(pdf_path)
        full_run = set(pages) == set(range(1, total + 1))

        with tempfile.TemporaryDirectory(prefix="ocrpipe_marker_") as tmpdir:
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
                rendered = converter(str(target))
                filled = self._fill_from_blocks(rendered, mapping, out)
                if not filled:
                    self._fill_from_markdown(rendered, mapping, out)
            except Exception as exc:
                log.warning("konversi Marker gagal: %s", exc)
                for pr in out.values():
                    pr.warnings.append(f"Marker gagal: {exc}")

            elapsed = round((time.perf_counter() - t0) / max(1, len(pages)), 4)
            for pr in out.values():
                pr.elapsed_s = elapsed
            return out

    # ------------------------------------------------------------------ #

    def _fill_from_blocks(
        self, rendered: Any, mapping: dict[int, int], out: dict[int, PageResult]
    ) -> bool:
        """Jalur utama: renderer 'chunks' yang memuat atribusi halaman."""
        blocks = getattr(rendered, "blocks", None)
        if not blocks:
            return False

        orders: dict[int, int] = {p: 0 for p in out}
        counters: dict[int, int] = {p: 0 for p in out}
        any_block = False

        for raw in blocks:
            sub_page = _block_page(raw)
            if sub_page is None:
                continue
            # Marker memakai indeks halaman 0-based.
            page_no = mapping.get(sub_page + 1)
            if page_no is None or page_no not in out:
                continue
            pr = out[page_no]
            btype = str(_attr(raw, "block_type", "") or "").lower().replace("-", "")
            kind = _BLOCK_TYPE_MAP.get(btype, KIND_PARAGRAPH)
            html = str(_attr(raw, "html", "") or "")
            if not html.strip():
                continue

            if kind == KIND_TABLE or "<table" in html.lower():
                counters[page_no] += 1
                block = self._table_block(html, page_no, counters[page_no], pr)
                if block is None:
                    counters[page_no] -= 1
                    continue
            else:
                text = clean_text(_html_to_text(html))
                if not text:
                    continue
                block = Block(
                    kind=kind,
                    text=text,
                    page=page_no,
                    level=2 if kind == KIND_HEADING else None,
                    engine=self.name,
                )
            block.order = orders[page_no]
            orders[page_no] += 1
            pr.blocks.append(block)
            any_block = True

        return any_block

    def _fill_from_markdown(
        self, rendered: Any, mapping: dict[int, int], out: dict[int, PageResult]
    ) -> None:
        """Jalur cadangan: Markdown berpaginasi, dipecah per penanda halaman."""
        md = str(_attr(rendered, "markdown", "") or "")
        if not md:
            try:
                from marker.output import text_from_rendered  # type: ignore

                # Perhatikan: elemen kedua adalah EKSTENSI, bukan metadata.
                md, _ext, _images = text_from_rendered(rendered)
            except Exception as exc:
                for pr in out.values():
                    pr.warnings.append(f"tidak ada keluaran yang dapat dibaca: {exc}")
                return

        parts = re.split(r"\n*-{20,}\n*", md)
        parts = [p for p in parts if p.strip()]
        sub_pages = sorted(mapping)

        if len(parts) == len(sub_pages):
            note = (
                "memakai jalur cadangan Markdown berpaginasi; struktur blok "
                "dan bbox tidak tersedia"
            )
            for part, sub in zip(parts, sub_pages):
                page_no = mapping[sub]
                pr = out[page_no]
                pr.warnings.append(note)
                pr.blocks.extend(_blocks_from_markdown(part, page_no, self.name))
        else:
            first = min(out)
            out[first].blocks.extend(
                _blocks_from_markdown(md, first, self.name)
            )
            out[first].warnings.append(
                f"jumlah penanda halaman ({len(parts)}) tidak sama dengan jumlah "
                f"halaman yang diminta ({len(sub_pages)}); seluruh teks ditaruh "
                "di halaman pertama — JANGAN dipakai untuk sitasi per halaman"
            )

    def _table_block(
        self, html: str, page_no: int, index: int, pr: PageResult
    ) -> Block | None:
        try:
            rows = html_table_to_rows(html)
        except Exception as exc:
            pr.warnings.append(f"parsing HTML tabel gagal: {exc}")
            return None
        if not rows:
            return None
        table = build_table(rows, page=page_no, index=index, engine=self.name)
        if table.is_degenerate():
            return None
        md = table_to_markdown(table)
        if not md:
            return None
        return Block(
            kind=KIND_TABLE, text=md, page=page_no, table=table, engine=self.name
        )


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


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


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _block_page(raw: Any) -> int | None:
    for key in ("page", "page_id", "page_idx", "page_number"):
        value = _attr(raw, key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    # Cadangan: ambil dari id blok, mis. "/page/3/Text/12".
    bid = _attr(raw, "id")
    if bid:
        m = _PAGE_IN_ID_RE.search(str(bid))
        if m:
            return int(m.group(1))
    return None


def _html_to_text(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    for entity, char in (
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&nbsp;", " "),
    ):
        text = text.replace(entity, char)
    return text


def _blocks_from_markdown(md: str, page_no: int, engine: str) -> list[Block]:
    """Pecah Markdown menjadi blok heading/tabel/paragraf (jalur cadangan)."""
    blocks: list[Block] = []
    order = 0
    t_idx = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal order, buffer
        text = clean_text("\n".join(buffer))
        buffer = []
        if text:
            blocks.append(
                Block(
                    kind=KIND_PARAGRAPH,
                    text=text,
                    page=page_no,
                    order=order,
                    engine=engine,
                )
            )
            order += 1

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(
                Block(
                    kind=KIND_HEADING,
                    text=stripped.lstrip("# ").strip(),
                    page=page_no,
                    order=order,
                    level=max(1, min(6, level)),
                    engine=engine,
                )
            )
            order += 1
            i += 1
            continue
        if stripped.startswith("|"):
            flush()
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [
                [c.strip() for c in ln.strip("|").split("|")]
                for ln in table_lines
                if not all(set(c.strip()) <= set("-: ") and c.strip() for c in ln.strip("|").split("|"))
            ]
            if rows:
                t_idx += 1
                table = build_table(rows, page=page_no, index=t_idx, engine=engine)
                rendered_md = table_to_markdown(table)
                if rendered_md:
                    blocks.append(
                        Block(
                            kind=KIND_TABLE,
                            text=rendered_md,
                            page=page_no,
                            order=order,
                            table=table,
                            engine=engine,
                        )
                    )
                    order += 1
            continue
        buffer.append(line)
        i += 1
    flush()
    return blocks
