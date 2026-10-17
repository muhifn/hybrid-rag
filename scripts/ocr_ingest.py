"""Turn the source PDFs in `data/pdf/` into RAG-ready chunks and table CSVs.

    python -m scripts.ocr_ingest                 # all PDFs in data/pdf
    python -m scripts.ocr_ingest --force-native  # skip OCR entirely (faster)
    python -m scripts.ocr_ingest --only "Panduan-Akademik-2025-2026_STI.pdf"

This is step 1 of two. It writes to `data/ocr_out/`:

    markdown/<doc>.md        human-readable, for checking the extraction
    jsonl/semua-chunk.jsonl  the chunks the RAG actually ingests
    tabel/csv/<doc>/*.csv    one CSV per detected table  -> step 2 input
    qa/laporan-qa.md         which pages need a human look

Step 2 is `python -m scripts.build_tables`, which normalises those raw table
CSVs into the SQL tables in `data/tables/`.

Why go through OCR at all when the markdown already exists? Three things the
plain markdown cannot give: page numbers (so the bot can cite "hal. 12"),
section paths (so a chunk from the middle of a document still says which
chapter it belongs to), and tables kept intact as tables instead of being cut
in half by a character-based splitter.

The pipeline lives in `ocr/` (vendored, see ocr/README.md). It decides per
PAGE whether the text layer is healthy enough to read directly or the page is
a scan that needs OCR, so a half-digital document is handled correctly.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from app.config import PROJECT_ROOT

OCR_DIR = PROJECT_ROOT / "ocr"
PDF_DIR = PROJECT_ROOT / "data" / "pdf"
OUT_DIR = PROJECT_ROOT / "data" / "ocr_out"
CONFIG = OCR_DIR / "config" / "hybrid-rag.yaml"

log = logging.getLogger("ocr_ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--only", action="append", default=None,
                        help="process just this file name (repeatable)")
    parser.add_argument("--engine", default=None, help="force an OCR engine for scanned pages")
    parser.add_argument("--force-native", action="store_true",
                        help="read every page from the text layer; never OCR")
    parser.add_argument("--clean", action="store_true", help="delete the output folder first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    if not args.pdf_dir.exists():
        print(f"No PDF folder at {args.pdf_dir}. Put the source PDFs there "
              f"(plus an optional metadata.csv for real titles).", file=sys.stderr)
        return 1

    # The pipeline is a standalone tool with its own package root; importing it
    # requires `ocr/` on the path. Done here rather than at module import so the
    # rest of the project never depends on it.
    sys.path.insert(0, str(OCR_DIR))
    try:
        from ocrpipe import Config, Pipeline
    except ImportError as exc:
        print(f"Cannot import the OCR pipeline from {OCR_DIR}: {exc}\n"
              f"Core dependencies: pip install -r {OCR_DIR / 'requirements.txt'}", file=sys.stderr)
        return 1

    inputs: list[str] = []
    if args.only:
        for name in args.only:
            path = args.pdf_dir / name
            if not path.exists():
                print(f"Not found: {path}", file=sys.stderr)
                return 1
            inputs.append(str(path))
    else:
        inputs.append(str(args.pdf_dir))

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
        log.info("Removed %s", args.out)

    cfg = Config.load(str(args.config)) if args.config.exists() else Config()
    if args.engine:
        cfg.ocr_engine = args.engine
    if args.force_native:
        from ocrpipe.models import ROUTE_NATIVE

        cfg.detect.force_route = ROUTE_NATIVE

    run = Pipeline(cfg).run(inputs, str(args.out), recursive=False)

    print()
    print("=" * 70)
    print(f"Documents      : {len(run.documents)}")
    print(f"Pages          : {run.total_pages}")
    print(f"Tables         : {run.total_tables}")
    print(f"Chunks         : {run.total_chunks}")
    print(f"Needs review   : {run.pages_needing_review} pages")
    print(f"Elapsed        : {run.elapsed_s:.1f}s")
    if run.errors:
        print(f"\n{len(run.errors)} document(s) FAILED:")
        for err in run.errors:
            print(f"  - {err}")
    print("=" * 70)
    print(f"\nChunks  : {args.out / 'jsonl' / 'semua-chunk.jsonl'}")
    print(f"Tables  : {args.out / 'tabel' / 'csv'}")
    print(f"QA      : {args.out / 'qa' / 'laporan-qa.md'}")
    print("\nNext:")
    print("  1. python -m scripts.build_tables     # raw table CSVs -> SQL tables")
    print("  2. python -m scripts.ingest           # load into Qdrant + SQLite")
    return 1 if run.errors and not run.documents else 0


if __name__ == "__main__":
    sys.exit(main())
