"""Render 3 diagram Bab 3 (dari Mermaid) ke PNG + SVG memakai Graphviz.

Dipakai untuk menyediakan gambar siap-tempel ke dokumen skripsi dan untuk
ditanamkan ke berkas .docx. Label Bahasa Indonesia.
"""

from __future__ import annotations

from pathlib import Path

from graphviz import Digraph

OUT = Path("docs/gambar")
OUT.mkdir(parents=True, exist_ok=True)

FONT = "Helvetica"


def _base(name: str, rankdir: str = "TB") -> Digraph:
    g = Digraph(name, format="png")
    # splines=polyline (bukan ortho) agar label pada cabang keputusan
    # ("Ya"/"Tidak"/"relevan") tetap tampil. dpi dinaikkan agar tajam di Word.
    g.attr(rankdir=rankdir, splines="polyline", nodesep="0.4", ranksep="0.55",
           bgcolor="white", dpi="200")
    g.attr("node", fontname=FONT, fontsize="11", margin="0.14,0.09")
    g.attr("edge", fontname=FONT, fontsize="10", color="#444444")
    return g


def _render(g: Digraph, stem: str) -> None:
    png = OUT / f"{stem}"
    g.render(filename=str(png), format="png", cleanup=True)
    g.render(filename=str(png), format="svg", cleanup=True)
    print(f"  {stem}.png + {stem}.svg")


# --------------------------------------------------------------------------- #
# Gambar 3.1 — Tahapan penelitian (DSRM)
# --------------------------------------------------------------------------- #


def gambar_3_1() -> None:
    g = _base("tahapan", "TB")
    proc = {"shape": "box", "style": "rounded,filled", "fillcolor": "#E8F0FE"}
    term = {"shape": "ellipse", "style": "filled", "fillcolor": "#D9EAD3"}
    dec = {"shape": "diamond", "style": "filled", "fillcolor": "#FCE5CD"}
    data = {"shape": "box", "style": "filled", "fillcolor": "#FFF2CC"}

    g.node("start", "Mulai", **term)
    g.node("b", "1. Identifikasi Masalah\n(studi pendahuluan &\nkajian pustaka ≥10 jurnal 2021–2026)", **proc)
    g.node("c", "2. Penetapan Tujuan Solusi\n(spesifikasi kebutuhan +\nkriteria keberhasilan terukur)", **proc)
    g.node("d1", "Akuisisi dokumen\nakademik (PDF)", **data)
    g.node("d2", "Wawancara 5 prodi\n(Kaprodi + staf TU)", **data)
    g.node("e1", "Pipeline akuisisi–praproses:\nklasifikasi halaman, OCR,\nekstraksi tabel, chunking,\nmetadata sitasi", **proc)
    g.node("e2", "Pengindeksan ke vector store\n(embedding + Qdrant)", **proc)
    g.node("e3", "Perakitan RAG:\nretrieval + abstensi +\nLLM lokal Qwen3-4B (Ollama)", **proc)
    g.node("f", "5. Demonstrasi & Evaluasi\n(CER/WER; Recall@k, MRR, nDCG;\nkebenaran, sitasi, abstensi, latensi)", **proc)
    g.node("g", "Kriteria\nterpenuhi?", **dec)
    g.node("h", "6. Komunikasi\n(penulisan skripsi & artefak)", **proc)
    g.node("end", "Selesai", **term)

    g.edge("start", "b")
    g.edge("b", "c")
    g.edge("c", "d1", label="3. Pengumpulan data")
    g.edge("c", "d2")
    g.edge("d1", "e1", label="4. Perancangan &\npengembangan")
    g.edge("d2", "e1")
    g.edge("e1", "e2")
    g.edge("e2", "e3")
    g.edge("e3", "f")
    g.edge("f", "g")
    g.edge("g", "e1", label="Tidak (iterasi\nprototyping)")
    g.edge("g", "h", label="Ya")
    g.edge("h", "end")
    _render(g, "gambar-3-1-tahapan-penelitian")


# --------------------------------------------------------------------------- #
# Gambar 3.2 — Sub-proses akuisisi & praproses per halaman
# --------------------------------------------------------------------------- #


def gambar_3_2() -> None:
    g = _base("praproses", "TB")
    proc = {"shape": "box", "style": "rounded,filled", "fillcolor": "#E8F0FE"}
    dec = {"shape": "diamond", "style": "filled", "fillcolor": "#FCE5CD"}
    data = {"shape": "box", "style": "filled", "fillcolor": "#FFF2CC"}
    skip = {"shape": "box", "style": "rounded,filled", "fillcolor": "#F4CCCC"}

    g.node("p", "Berkas PDF sumber", **data)
    g.node("q", "Pemeriksaan tiap halaman\n(text layer, objek visual,\nrasio karakter rusak)", **proc)
    g.node("r", "Jenis halaman?", **dec)
    g.node("s", "Rute NATIVE:\nekstraksi langsung\n(akurasi karakter penuh)", **proc)
    g.node("t1", "Praproses citra:\nrasterisasi, deskew,\nkontras, rotasi otomatis", **proc)
    g.node("t2", "OCR + pengenalan\nstruktur tabel\n(engine neural / klasik)", **proc)
    g.node("u", "Dilewati & dicatat", **skip)
    g.node("v", "Normalisasi blok & tabel", **proc)
    g.node("w", "QA per halaman:\nconfidence, rasio leksikon,\npenandaan perlu tinjauan", **proc)
    g.node("x", "Chunking sadar struktur\n+ metadata sitasi", **proc)
    g.node("y", "Keluaran: Markdown, JSONL,\nCSV/XLSX tabel, laporan QA", **data)

    g.edge("p", "q")
    g.edge("q", "r")
    g.edge("r", "s", label="text layer sehat")
    g.edge("r", "t1", label="scan / rusak")
    g.edge("r", "u", label="kosong")
    g.edge("t1", "t2")
    g.edge("s", "v")
    g.edge("t2", "v")
    g.edge("v", "w")
    g.edge("w", "x")
    g.edge("x", "y")
    _render(g, "gambar-3-2-praproses-per-halaman")


# --------------------------------------------------------------------------- #
# Gambar 3.3 — Arsitektur RAG
# --------------------------------------------------------------------------- #


def gambar_3_3() -> None:
    g = _base("arsitektur", "LR")
    proc = {"shape": "box", "style": "rounded,filled", "fillcolor": "#E8F0FE"}
    data = {"shape": "box", "style": "filled", "fillcolor": "#FFF2CC"}
    store = {"shape": "cylinder", "style": "filled", "fillcolor": "#D9D2E9"}
    dec = {"shape": "diamond", "style": "filled", "fillcolor": "#FCE5CD"}
    ans = {"shape": "box", "style": "rounded,filled", "fillcolor": "#D9EAD3"}
    absn = {"shape": "box", "style": "rounded,filled", "fillcolor": "#F4CCCC"}

    with g.subgraph(name="cluster_kb") as kb:
        kb.attr(label="Lapis Basis Pengetahuan", style="rounded", color="#3C78D8",
                fontname=FONT, fontsize="12")
        kb.node("doc", "Dokumen PDF", **data)
        kb.node("pp", "Pipeline akuisisi–praproses\n(ocrpipe)", **proc)
        kb.node("ch", "Chunk + metadata sitasi\n(dokumen/halaman/versi)", **proc)
        kb.node("em", "Embedding", **proc)
        kb.node("qd", "Vector store\nQdrant", **store)
        kb.edge("doc", "pp"); kb.edge("pp", "ch"); kb.edge("ch", "em"); kb.edge("em", "qd")

    with g.subgraph(name="cluster_rag") as r:
        r.attr(label="Lapis RAG", style="rounded", color="#6AA84F",
               fontname=FONT, fontsize="12")
        r.node("u", "Pertanyaan\npengguna", **data)
        r.node("rt", "Penelusuran\n(dense + opsi BM25/RRF)", **proc)
        r.node("rk", "Pemeriksaan relevansi\n+ abstensi", **dec)
        r.node("llm", "LLM lokal Qwen3-4B\n(Ollama)", **proc)
        r.node("ab", "Jawaban abstain:\n“informasi tidak ditemukan”", **absn)
        r.node("ans", "Jawaban + sitasi", **ans)
        r.edge("u", "rt")
        r.edge("rt", "rk")
        r.edge("rk", "llm", label="relevan")
        r.edge("rk", "ab", label="tidak relevan")
        r.edge("llm", "ans")

    g.edge("qd", "rt", label="top-k", constraint="false")
    _render(g, "gambar-3-3-arsitektur-rag")


if __name__ == "__main__":
    print("Merender diagram Bab 3 ke docs/gambar/ ...")
    gambar_3_1()
    gambar_3_2()
    gambar_3_3()
    print("Selesai.")
