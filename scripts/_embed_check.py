import sys
import traceback

try:
    import sentence_transformers
    import transformers
    print("sentence_transformers", sentence_transformers.__version__)
    print("transformers", transformers.__version__)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-m3", device="cpu")
    v = m.encode(["tes bahasa indonesia"], normalize_embeddings=True)
    print("EMBEDDING OK dim", v.shape[1])
except Exception:
    traceback.print_exc()
    sys.exit(1)
