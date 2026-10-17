"""Streamlit chat UI for the Hybrid RAG API — Claude-style interface.

    streamlit run frontend/streamlit_app.py

Talks to the FastAPI backend. Set RAG_API_URL to point at another machine.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Iterator

import httpx
import streamlit as st

DEFAULT_API = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

# Custom CSS: Claude-style left sidebar + clean chat area
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1a1a1a;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }

    /* Main chat area */
    .block-container {
        padding-top: 2rem;
        max-width: 48rem;
        margin: 0 auto;
    }

    /* Tighter message spacing */
    .stChatMessage {
        margin-bottom: 0.75rem;
    }

    /* Hide avatar images for cleaner look */
    .stChatMessage img {
        display: none;
    }

    /* Input styling */
    .stChatInputContainer {
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []
    st.session_state.api_url = DEFAULT_API
    st.session_state.show_sources = True


# ------------------------------------------------------------------ helpers
def stream_chat(base: str, payload: dict, holder: dict[str, Any]) -> Iterator[str]:
    """Yield answer tokens from the SSE endpoint; store meta/done/error in `holder`."""
    with httpx.stream("POST", f"{base}/chat/stream", json=payload, timeout=httpx.Timeout(10, read=600)) as r:
        if r.status_code != 200:
            holder["error"] = f"HTTP {r.status_code}: {r.read().decode(errors='replace')[:500]}"
            return
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
                if event == "token":
                    yield data["text"]
                else:
                    holder[event] = data


def render_sources(details: dict[str, Any]) -> None:
    """Collapsed sources under the answer."""
    if not st.session_state.show_sources:
        return
    if not details.get("answered", True):
        return  # greetings have no sources

    sources = details.get("sources", [])
    graph = details.get("graph")
    if not sources and not graph:
        return

    with st.expander("📎 Lihat sumber", expanded=False):
        for src in sources:
            page = f", hal. {src['page']}" if src.get("page") else ""
            st.markdown(f"**{src['title']}** — `{src['source']}`{page}")
            st.caption(src["snippet"])
            if src != sources[-1]:
                st.divider()
        if graph and graph["ok"] and graph["rows"]:
            st.markdown(f"**📊 Data dari tabel ({graph['language'].upper()})**")
            st.dataframe(graph["rows"], use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("💬 Asisten Akademik")

    # New conversation button at top
    if st.button("➕ Percakapan Baru", use_container_width=True, type="primary"):
        try:
            httpx.delete(f"{st.session_state.api_url}/sessions/{st.session_state.session_id}", timeout=5)
        except httpx.HTTPError:
            pass
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # History placeholder (future: list previous sessions)
    st.subheader("📚 Riwayat")
    st.caption("_Riwayat percakapan akan ditampilkan di sini_")

    st.divider()

    # Settings expander
    with st.expander("⚙️ Pengaturan", expanded=False):
        st.markdown("**Koneksi API**")
        api_url = st.text_input("API URL", st.session_state.api_url, key="api_url_input",
                                help="Backend FastAPI endpoint").rstrip("/")
        if api_url != st.session_state.api_url:
            st.session_state.api_url = api_url
            st.rerun()

        st.divider()
        st.markdown("**Tampilan**")
        st.session_state.show_sources = st.checkbox("Tampilkan sumber", value=st.session_state.show_sources,
                                                     help="Tunjukkan referensi dokumen dan tabel")

        st.divider()
        st.markdown("**Status Sistem**")
        try:
            health = httpx.get(f"{st.session_state.api_url}/health", timeout=5).json()
            ready = health["ready"]
            llm_ok = health["llm"].get("reachable", False)

            st.markdown(f"{'🟢' if ready else '🔴'} **Siap:** {'Ya' if ready else 'Tidak'}")
            st.markdown(f"{'🟢' if llm_ok else '🔴'} **LLM:** `{health['llm'].get('model', '?')}`")
            st.markdown(f"📄 **Dokumen:** {health['vector_store']['chunks']} chunks")

            graph_ok = health["graph"]["available"]
            tables = ', '.join(health['graph']['tables']) if health['graph']['tables'] else '-'
            st.markdown(f"{'🟢' if graph_ok else '🟡'} **Tabel:** {tables}")

            emb = health["embeddings"]
            st.caption(f"Embedding: {emb['model']} (dim {emb['dimension']}, {emb['backend']})")

        except (httpx.HTTPError, KeyError) as exc:
            st.error(f"❌ API tidak terjangkau\n\n{exc}")

        st.divider()
        st.markdown("**Admin**")
        st.caption("Operasi admin untuk rebuild data store")
        admin_key = st.text_input("Admin API Key", type="password",
                                  help="Hanya jika ADMIN_API_KEY diset di .env")
        if st.button("🔄 Re-ingest data/", use_container_width=True):
            with st.spinner("Membangun ulang vector store dan graph database..."):
                try:
                    headers = {"X-Admin-Key": admin_key} if admin_key else {}
                    r = httpx.post(f"{st.session_state.api_url}/admin/ingest",
                                 json={"recreate": True}, timeout=1800, headers=headers)
                    if r.status_code == 200:
                        result = r.json()
                        st.success(f"✅ Selesai: {result.get('documents', 0)} dokumen, "
                                 f"{result.get('chunks', 0)} chunks, {result.get('tables', 0)} tabel")
                    else:
                        st.error(f"❌ HTTP {r.status_code}: {r.text[:300]}")
                except httpx.HTTPError as exc:
                    st.error(f"❌ {exc}")

    # Session ID at bottom
    st.caption(f"Session: `{st.session_state.session_id[:12]}...`")


# --------------------------------------------------------------------- chat
st.title("💬 Tanya Apa Saja")
st.caption("Asisten untuk panduan akademik, kurikulum, mata kuliah, dan data kampus")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("details"):
            render_sources(msg["details"])

# Chat input
if question := st.chat_input("Ketik pertanyaan Anda..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {"question": question, "session_id": st.session_state.session_id}

    with st.chat_message("assistant"):
        holder: dict[str, Any] = {}
        try:
            with st.spinner("🤖 Sedang berpikir..."):
                tokens = stream_chat(st.session_state.api_url, payload, holder)
                first = next(tokens, None)  # spinner covers routing + retrieval

            def replay() -> Iterator[str]:
                if first is not None:
                    yield first
                yield from tokens

            answer = st.write_stream(replay())
        except httpx.HTTPError as exc:
            holder["error"] = f"Tidak bisa menghubungi API: {exc}"
            answer = ""

        if "error" in holder:
            st.error(holder["error"])
            answer = answer or "(error)"

        details = holder.get("done") or holder.get("meta")
        if details:
            render_sources(details)

    st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})
