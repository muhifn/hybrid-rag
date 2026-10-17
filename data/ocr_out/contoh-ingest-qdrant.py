"""Contoh ingest hasil ocrpipe ke Qdrant.

Dibuat otomatis oleh ocrpipe. Sesuaikan nama koleksi dan model embedding.
Skrip ini sengaja TIDAK dijalankan otomatis oleh pipeline: pemilihan model
embedding adalah keputusan penelitian, bukan detail implementasi.
"""

import json
from pathlib import Path

JSONL = Path("keluaran/jsonl/semua-chunk.jsonl")
COLLECTION = "dokumen_akademik"
EMBED_MODEL = "intfloat/multilingual-e5-base"  # ganti sesuai hasil perbandingan

def load_chunks(path=JSONL):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def main():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks()
    print(f"memuat {len(chunks)} chunk")

    # Chunk bertanda needs_review sebaiknya ditahan sampai dikoreksi manual,
    # agar hasil OCR yang meragukan tidak ikut menjawab pertanyaan pengguna.
    chunks = [c for c in chunks if not c.get("needs_review")]
    print(f"{len(chunks)} chunk lolos saringan QA")

    model = SentenceTransformer(EMBED_MODEL)
    # Model keluarga E5 mensyaratkan prefiks "passage: " pada sisi dokumen.
    texts = [f"passage: {c['text']}" for c in chunks]
    vectors = model.encode(texts, batch_size=16, show_progress_bar=True,
                           normalize_embeddings=True)

    client = QdrantClient(path="./qdrant_local")  # atau url="http://localhost:6333"
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
            for i in range(len(chunks))
        ],
    )
    print("selesai. payload sudah memuat doc_title/page_start/doc_version "
          "sehingga sitasi dapat dibangun tanpa query tambahan.")

if __name__ == "__main__":
    main()
