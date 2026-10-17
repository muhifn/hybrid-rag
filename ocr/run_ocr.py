#!/usr/bin/env python3
"""Titik masuk CLI.

Dijalankan langsung dari folder project tanpa perlu `pip install`:

    python run_ocr.py engines
    python run_ocr.py diagnose data/pdf
    python run_ocr.py ingest data/pdf -o keluaran
"""

from __future__ import annotations

import sys
from pathlib import Path

# Pastikan paket `ocrpipe` di folder ini yang dipakai, bukan versi lain
# yang mungkin ada di site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocrpipe.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
