"""ocrpipe — pipeline OCR dokumen akademik untuk RAG.

Ekstraksi teks DAN tabel dari PDF campuran (sebagian digital, sebagian hasil
pindaian), dengan keluaran siap-ingest ke vector store: Markdown untuk
dibaca manusia, JSONL berisi chunk + metadata sitasi, serta CSV/XLSX per
tabel.

Penggunaan singkat:

    from ocrpipe import Config, Pipeline

    cfg = Config()
    cfg.ocr_engine = "docling"
    run = Pipeline(cfg).run(["data/pdf"], "keluaran")
    print(run.total_chunks, "chunk")

Atau lewat CLI:

    python run_ocr.py ingest data/pdf -o keluaran
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (
    ChunkConfig,
    Config,
    DetectConfig,
    DoclingConfig,
    MarkerConfig,
    NativeConfig,
    OutputConfig,
    PaddleConfig,
    PreprocessConfig,
    QAConfig,
    TesseractConfig,
)
from .models import Block, Chunk, DocMeta, DocResult, PageResult, TableData

__all__ = [
    "__version__",
    "Block",
    "Chunk",
    "ChunkConfig",
    "Config",
    "DetectConfig",
    "DoclingConfig",
    "DocMeta",
    "DocResult",
    "MarkerConfig",
    "NativeConfig",
    "OutputConfig",
    "PaddleConfig",
    "PageResult",
    "Pipeline",
    "PreprocessConfig",
    "QAConfig",
    "TableData",
    "TesseractConfig",
    "probe_document",
]


def __getattr__(name: str):
    """Impor malas untuk simbol yang menarik dependensi berat.

    Membuat `import ocrpipe` tetap murah: PyMuPDF, OpenCV, dan kawan-kawan
    baru dimuat saat Pipeline atau probe_document benar-benar dipakai.
    """
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    if name == "probe_document":
        from .detect import probe_document

        return probe_document
    raise AttributeError(f"module 'ocrpipe' has no attribute '{name}'")
