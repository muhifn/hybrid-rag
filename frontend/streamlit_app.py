"""Streamlit chat UI for the Hybrid RAG API.

    streamlit run frontend/streamlit_app.py

Talks to the FastAPI backend over HTTP only, so it can run on another machine.
Set the backend address with the RAG_API_URL environment variable or in the sidebar.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Iterator

import httpx
import streamlit as st

DEFAULT_API = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
ROUTE_BADGE = {"VECTOR": "📄 VECTOR", "GRAPH": "🗂️ GRAPH", "HYBRID": "🔀 HYBRID"}

st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="💬", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []


# ------------------------------------------------------------------ helpers
def api_get(base: str, path: str, **kwargs) -> httpx.Response:
    return httpx.get(f"{base}{path}", timeout=10, **kwargs)


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


def render_details(details: dict[str, Any]) -> None:
    """Route, sources, table result, and timings under an assistant message."""
    route = details.get("route", "?")
    caption = f"{ROUTE_BADGE.get(route, route)} · decided by {details.get('route_method')}"
    if details.get("standalone_question") and details["standalone_question"] != details.get("question"):
        caption += f" · rewritten as: *{details['standalone_question']}*"
    st.caption(caption)

    with st.expander("Sources and details"):
        st.markdown(f"**Router reason:** {details.get('route_reason', '-')}")
        for src in details.get("sources", []):
            page = f", page {src['page']}" if src.get("page") else ""
            st.markdown(f"**[{src['rank']}] {src['title']}** — `{src['source']}`{page}")
            st.caption(src["snippet"])
        graph = details.get("graph")
        if graph:
            st.markdown(f"**[TABLE]** {graph['language'].upper()} "
                        f"({'ok' if graph['ok'] else 'failed'}, attempt {graph['attempts']})")
            st.code(graph["query"], language="sql" if graph["language"] == "sql" else "cypher")
            if graph["ok"] and graph["rows"]:
                st.dataframe(graph["rows"], use_container_width=True)
            elif graph.get("error"):
                st.error(graph["error"])
        timings = details.get("timings", {})
        if timings:
            st.caption(" · ".join(f"{k} {v:.2f}s" for k, v in timings.items()))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", DEFAULT_API).rstrip("/")
    route_choice = st.selectbox("Route", ["auto", "VECTOR", "GRAPH", "HYBRID"],
                                help="'auto' lets the router decide")
    retrieval_mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"])

    if st.button("New conversation", use_container_width=True):
        try:
            httpx.delete(f"{api_url}/sessions/{st.session_state.session_id}", timeout=5)
        except httpx.HTTPError:
            pass
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("System status")
    try:
        health = api_get(api_url, "/health").json()
        llm = health["llm"]
        st.markdown(f"{'🟢' if health['ready'] else '🔴'} **Ready:** {health['ready']}")
        st.markdown(f"{'🟢' if llm.get('reachable') else '🔴'} **LLM:** `{llm.get('model')}`")
        st.markdown(f"📄 **Chunks:** {health['vector_store']['chunks']} ({health['vector_store']['mode']})")
        graph = health["graph"]
        st.markdown(f"{'🟢' if graph['available'] else '🟡'} **Graph:** {graph['backend']} "
                    f"({', '.join(graph['tables']) or 'unavailable'})")
        st.caption(f"Embeddings: {health['embeddings']['model']} (dim {health['embeddings']['dimension']})")
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        st.error(f"API unreachable at {api_url}\n\n{exc}")

    with st.expander("Admin"):
        admin_key = st.text_input("Admin key", type="password", help="Only if ADMIN_API_KEY is set")
        if st.button("Re-ingest data/", use_container_width=True):
            with st.spinner("Rebuilding vector store and graph..."):
                try:
                    r = httpx.post(f"{api_url}/admin/ingest", json={"recreate": True}, timeout=1800,
                                   headers={"X-Admin-Key": admin_key} if admin_key else {})
                    if r.status_code == 200:
                        st.success(r.json())
                    else:
                        st.error(f"HTTP {r.status_code}: {r.text[:300]}")
                except httpx.HTTPError as exc:
                    st.error(str(exc))

    st.caption(f"Session `{st.session_state.session_id[:8]}`")


# --------------------------------------------------------------------- chat
st.title("💬 Hybrid RAG Chatbot")
st.caption("Vector search over documents + structured queries over tables, answered by a local LLM.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            render_details(message["details"])

if question := st.chat_input("Ask a question (Bahasa Indonesia or English)..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {"question": question, "session_id": st.session_state.session_id,
               "retrieval_mode": retrieval_mode}
    if route_choice != "auto":
        payload["route"] = route_choice

    with st.chat_message("assistant"):
        holder: dict[str, Any] = {}
        try:
            with st.spinner("Routing and retrieving..."):
                tokens = stream_chat(api_url, payload, holder)
                first = next(tokens, None)  # the spinner covers routing + retrieval

            def replay() -> Iterator[str]:
                if first is not None:
                    yield first
                yield from tokens

            answer = st.write_stream(replay())
        except httpx.HTTPError as exc:
            holder["error"] = f"Cannot reach the API: {exc}"
            answer = ""

        if "error" in holder:
            st.error(holder["error"])
            answer = answer or "(error)"
        details = holder.get("done") or holder.get("meta")
        if details:
            render_details(details)
    st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})
