"""Load source documents and split them into chunks for the vector store.

Two entry points, because the project supports two document sources:

  * `load_documents` + `split_documents` - plain .md/.txt/.pdf files, chunked
    here with a character-based splitter.
  * `load_chunk_jsonl` - chunks already produced by the OCR pipeline in `ocr/`,
    which chunked them with page numbers, section paths, and whole tables
    preserved. Those are better boundaries than a character splitter can find,
    so they are used as-is rather than re-split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}

# Split on the most meaningful boundary available first: Markdown headings,
# then paragraphs, lines, sentences, words.
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a simple `key: value` YAML frontmatter block from the body."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    end = stripped.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in stripped[3:end].splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            meta[key.strip()] = value.strip()
    return meta, stripped[end + 4 :].lstrip("\n")


def _title(meta: dict[str, str], body: str, path: Path) -> str:
    for key in ("title", "judul"):
        if meta.get(key):
            return meta[key]
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def load_documents(documents_dir: Path) -> list[Document]:
    """Load every .md/.txt/.pdf under `documents_dir` (recursively).

    PDFs yield one Document per page with a `page` metadata field (1-based),
    so answers can cite page numbers.
    """
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    files = sorted(
        p for p in documents_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES | PDF_SUFFIXES
    )
    docs: list[Document] = []
    for path in files:
        source = path.relative_to(documents_dir).as_posix()
        try:
            if path.suffix.lower() in PDF_SUFFIXES:
                pages = PyPDFLoader(str(path)).load()
                title = path.stem.replace("-", " ").replace("_", " ").title()
                for page in pages:
                    if not page.page_content.strip():
                        continue
                    docs.append(Document(
                        page_content=page.page_content,
                        metadata={"source": source, "title": title, "doc_type": "pdf",
                                  "page": int(page.metadata.get("page", 0)) + 1},
                    ))
            else:
                meta, body = parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
                if not body.strip():
                    log.warning("Skipping empty document: %s", source)
                    continue
                docs.append(Document(
                    page_content=body,
                    metadata={**meta, "source": source, "title": _title(meta, body, path),
                              "doc_type": path.suffix.lower().lstrip(".")},
                ))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort ingestion
            log.error("Failed to load %s: %s: %s", source, type(exc).__name__, exc)

    log.info("Loaded %d documents from %s", len(docs), documents_dir)
    return docs


def load_chunk_jsonl(path: Path) -> list[Document]:
    """Load chunks produced by the OCR pipeline (`data/ocr_out/jsonl/semua-chunk.jsonl`).

    One JSON object per line. The fields this project uses:

        text          the chunk content (already carries its section path and,
                      for table chunks, the table's column names)
        source_file   original file name          -> `source`, for citations
        doc_title     real title from metadata.csv -> `title`
        page_start    1-based page number          -> `page`, so answers cite pages
        section_path  ["BAB II", "Registrasi"]     -> `section`
        content_type  "text" | "table"             -> lets the UI mark table chunks
        needs_review  QA flag for low-confidence OCR

    Chunks already flagged `needs_review` are still loaded: dropping them would
    silently remove content, and the QA report is the place to deal with them.
    """
    if not path.exists():
        return []

    docs: list[Document] = []
    skipped = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("%s line %d is not valid JSON: %s", path.name, line_no, exc)
            skipped += 1
            continue
        text = (row.get("text") or "").strip()
        if not text:
            skipped += 1
            continue

        section = row.get("section_path") or []
        metadata = {
            "source": row.get("source_file") or row.get("doc_id") or path.name,
            "title": row.get("doc_title") or row.get("doc_id") or "?",
            "doc_type": "ocr",
            "chunk_id": row.get("chunk_id") or f"{path.stem}::{line_no}",
            "chunk_index": row.get("chunk_index", 0),
            "content_type": row.get("content_type", "text"),
            "needs_review": bool(row.get("needs_review", False)),
        }
        if row.get("page_start"):
            metadata["page"] = int(row["page_start"])
        if row.get("page_end") and row["page_end"] != row.get("page_start"):
            metadata["page_end"] = int(row["page_end"])
        if isinstance(section, list) and section:
            metadata["section"] = " > ".join(str(s) for s in section)
        for key in ("doc_version", "doc_date", "doc_category", "unit"):
            if row.get(key):
                metadata[key] = row[key]

        docs.append(Document(page_content=text, metadata=metadata))

    tables = sum(1 for d in docs if d.metadata.get("content_type") == "table")
    review = sum(1 for d in docs if d.metadata.get("needs_review"))
    log.info("Loaded %d OCR chunks from %s (%d table chunks, %d flagged for review%s)",
             len(docs), path.name, tables, review,
             f", {skipped} skipped" if skipped else "")
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Chunk documents and attach a stable `chunk_id` to each chunk.

    Non-first chunks get the document title prepended ("contextual chunking"):
    a chunk from the middle of a document otherwise loses what it is about,
    which hurts both dense and BM25 retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
    )
    chunks: list[Document] = []
    counters: dict[str, int] = {}
    for doc in docs:
        for piece in splitter.split_text(doc.page_content):
            piece = piece.strip()
            if not piece:
                continue
            key = f"{doc.metadata['source']}#p{doc.metadata.get('page', 0)}"
            index = counters.get(key, 0)
            counters[key] = index + 1
            title = doc.metadata.get("title", "")
            text = f"{title}\n\n{piece}" if index > 0 and title else piece
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "chunk_index": index, "chunk_id": f"{key}::{index}"},
            ))
    log.info("Split %d documents into %d chunks (size=%d, overlap=%d)",
             len(docs), len(chunks), chunk_size, chunk_overlap)
    return chunks
