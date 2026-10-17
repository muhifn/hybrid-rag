"""Chunking sadar struktur untuk RAG.

Tiga keputusan desain yang membedakan modul ini dari pemotongan naif
"setiap 1000 karakter":

1. TABEL TIDAK PERNAH DIPOTONG DI TENGAH BARIS. Tabel mendapat chunk-nya
   sendiri. Tabel yang terlalu panjang dipecah per baris dengan BARIS HEADER
   DIULANG di setiap pecahan. Tanpa itu, pecahan kedua kehilangan nama kolom
   dan menjadi tak bermakna ketika terambil sendirian oleh retriever.

2. BATAS HEADING DIHORMATI. Chunk tidak melewati batas sub-bab, dan jejak
   heading (Bab > Sub-bab) disisipkan di awal chunk. Ini memperbaiki dua hal
   sekaligus: kualitas embedding (konteks topik ikut ter-embed) dan kualitas
   jawaban (LLM tahu bagian mana yang sedang dikutip).

3. KOP SURAT DAN FOOTER DIBUANG. Pada dokumen resmi, kop dan nomor halaman
   berulang di setiap halaman. Bila dibiarkan, potongan-potongan itu menjadi
   teks paling sering muncul di korpus dan mencemari hasil retrieval.
"""

from __future__ import annotations

import re
from typing import Iterable

from .config import ChunkConfig
from .models import (
    KIND_CAPTION,
    KIND_HEADING,
    NOISE_KINDS,
    Block,
    Chunk,
    DocResult,
    TableData,
)
from .tables import shard_table, table_to_markdown
from .utils import collapse_to_single_line

_SENTENCE_END_RE = re.compile(r"(?<=[.!?:;])\s+")


class Chunker:
    def __init__(self, cfg: ChunkConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ #

    def chunk_document(self, doc: DocResult) -> list[Chunk]:
        cfg = self.cfg
        meta = doc.meta
        page_info = {
            p.page: (p.engine, p.route, p.needs_review, p.mean_confidence)
            for p in doc.pages
        }

        chunks: list[Chunk] = []
        section_stack: list[tuple[int, str]] = []
        buffer: list[Block] = []
        buffer_chars = 0
        carry_over = ""  # teks tumpang-tindih dari chunk sebelumnya
        pending_caption: str | None = None

        def section_path() -> list[str]:
            return [t for _, t in section_stack]

        def flush() -> None:
            nonlocal buffer, buffer_chars, carry_over
            if not buffer:
                return
            body = "\n\n".join(b.text for b in buffer if b.text.strip()).strip()
            if body:
                text = f"{carry_over}\n\n{body}".strip() if carry_over else body
                chunks.append(
                    self._make_chunk(
                        text=text,
                        content_type="text",
                        blocks=buffer,
                        section=section_path(),
                        meta=meta,
                        page_info=page_info,
                        index=len(chunks),
                    )
                )
                carry_over = self._tail(body) if cfg.overlap_chars > 0 else ""
            buffer = []
            buffer_chars = 0

        for block in doc.blocks:
            if cfg.drop_noise_blocks and block.kind in NOISE_KINDS:
                continue
            if not block.text.strip():
                continue

            # --- heading ---
            if block.kind == KIND_HEADING:
                if cfg.respect_headings:
                    flush()
                    carry_over = ""  # jangan bawa konteks melintasi sub-bab
                level = block.level or 3
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, collapse_to_single_line(block.text)))
                buffer.append(block)
                buffer_chars += block.char_count
                continue

            # --- keterangan gambar/tabel: simpan untuk tabel berikutnya ---
            if block.kind == KIND_CAPTION:
                pending_caption = collapse_to_single_line(block.text)
                buffer.append(block)
                buffer_chars += block.char_count
                continue

            # --- tabel ---
            if block.is_table and block.table is not None:
                flush()
                carry_over = ""
                table = block.table
                if table.caption is None and pending_caption:
                    table.caption = pending_caption
                chunks.extend(
                    self._table_chunks(
                        table,
                        block=block,
                        section=section_path(),
                        meta=meta,
                        page_info=page_info,
                        start_index=len(chunks),
                    )
                )
                pending_caption = None
                continue

            pending_caption = None

            # --- teks biasa ---
            if buffer_chars + block.char_count > cfg.max_chars and buffer:
                flush()
            buffer.append(block)
            buffer_chars += block.char_count
            if buffer_chars >= cfg.target_chars:
                flush()

        flush()
        merged = self._merge_small(chunks)
        for i, ch in enumerate(merged):
            ch.chunk_index = i
            ch.chunk_id = ""
            ch.finalize()
        return merged

    # ------------------------------------------------------------------ #

    def _table_chunks(
        self,
        table: TableData,
        *,
        block: Block,
        section: list[str],
        meta,
        page_info: dict,
        start_index: int,
    ) -> list[Chunk]:
        cfg = self.cfg
        shards = (
            shard_table(table, cfg.table_rows_per_shard)
            if cfg.split_large_tables and len(block.text) > cfg.max_chars
            else [table]
        )
        out: list[Chunk] = []
        for i, shard in enumerate(shards):
            md = table_to_markdown(shard) if len(shards) > 1 else block.text
            if not md:
                continue
            text = self._table_prefix(shard, section) + md if cfg.table_context_prefix else md
            ch = self._make_chunk(
                text=text,
                content_type="table",
                blocks=[block],
                section=section,
                meta=meta,
                page_info=page_info,
                index=start_index + i,
            )
            ch.table_refs = [shard.slug]
            out.append(ch)
        return out

    def _table_prefix(self, table: TableData, section: list[str]) -> str:
        """Teks pengantar tabel agar tetap terjangkau dense retrieval."""
        lines: list[str] = []
        if section:
            lines.append(f"Bagian: {' > '.join(section)}")
        if table.caption:
            lines.append(f"Tabel: {table.caption}")
        header = [collapse_to_single_line(h) for h in table.header if h.strip()]
        if header:
            lines.append(f"Kolom tabel: {', '.join(header)}")
        lines.append(f"Sumber: halaman {table.page}")
        return "\n".join(lines) + "\n\n" if lines else ""

    # ------------------------------------------------------------------ #

    def _make_chunk(
        self,
        *,
        text: str,
        content_type: str,
        blocks: list[Block],
        section: list[str],
        meta,
        page_info: dict,
        index: int,
    ) -> Chunk:
        pages = sorted({b.page for b in blocks if b.page})
        p_start = pages[0] if pages else 0
        p_end = pages[-1] if pages else 0

        engines: list[str] = []
        routes: list[str] = []
        needs_review = False
        confs: list[float] = []
        for p in pages:
            info = page_info.get(p)
            if not info:
                continue
            engine, route, review, conf = info
            if engine and engine not in engines:
                engines.append(engine)
            if route and route not in routes:
                routes.append(route)
            needs_review = needs_review or bool(review)
            if conf is not None:
                confs.append(conf)

        body = text
        if self.cfg.prepend_section_path and section and content_type == "text":
            prefix = " > ".join(section)
            if not body.startswith(prefix):
                body = f"[{prefix}]\n\n{body}"

        return Chunk(
            doc_id=meta.doc_id,
            text=body.strip(),
            content_type=content_type,
            page_start=p_start,
            page_end=p_end,
            section_path=list(section),
            heading=section[-1] if section else None,
            chunk_index=index,
            doc_title=meta.title,
            doc_category=meta.category,
            doc_version=meta.version,
            doc_date=meta.effective_date,
            source_file=meta.source_file,
            unit=meta.unit,
            url=meta.url,
            language=meta.language,
            engines=engines,
            routes=routes,
            mean_confidence=round(sum(confs) / len(confs), 2) if confs else None,
            needs_review=needs_review,
        )

    # ------------------------------------------------------------------ #

    def _tail(self, text: str) -> str:
        """Ambil ekor teks sepanjang ~overlap_chars, dipotong di batas kalimat."""
        n = self.cfg.overlap_chars
        if n <= 0 or len(text) <= n:
            return text if n > 0 else ""
        tail = text[-n:]
        parts = _SENTENCE_END_RE.split(tail, maxsplit=1)
        if len(parts) == 2 and len(parts[1]) >= n * 0.3:
            return parts[1].strip()
        # Tidak ada batas kalimat: potong di batas kata.
        space = tail.find(" ")
        return tail[space + 1 :].strip() if space != -1 else tail.strip()

    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """Gabungkan chunk teks yang terlalu pendek dengan tetangganya.

        Chunk tabel tidak pernah digabung: menyatukan dua tabel berbeda dalam
        satu chunk membuat LLM mencampur barisnya saat menjawab.
        """
        cfg = self.cfg
        if not chunks:
            return []
        out: list[Chunk] = []
        for ch in chunks:
            if (
                out
                and ch.content_type == "text"
                and out[-1].content_type == "text"
                and len(ch.text) < cfg.min_chars
                and out[-1].section_path == ch.section_path
                and len(out[-1].text) + len(ch.text) <= cfg.max_chars
            ):
                prev = out[-1]
                prev.text = f"{prev.text}\n\n{ch.text}".strip()
                prev.page_end = max(prev.page_end, ch.page_end)
                prev.page_start = min(prev.page_start or ch.page_start, ch.page_start)
                prev.needs_review = prev.needs_review or ch.needs_review
                for e in ch.engines:
                    if e not in prev.engines:
                        prev.engines.append(e)
                for r in ch.routes:
                    if r not in prev.routes:
                        prev.routes.append(r)
                continue
            out.append(ch)
        return out


def chunk_documents(
    docs: Iterable[DocResult], cfg: ChunkConfig
) -> dict[str, list[Chunk]]:
    chunker = Chunker(cfg)
    return {doc.meta.doc_id: chunker.chunk_document(doc) for doc in docs}
