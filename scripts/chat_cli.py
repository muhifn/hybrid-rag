"""Interactive terminal chat against the pipeline (no HTTP server needed).

    python -m scripts.chat_cli

Commands: /reset  /route VECTOR|GRAPH|HYBRID|auto  /mode dense|bm25|hybrid  /quit
"""

from __future__ import annotations

import logging
import uuid

from app.config import get_settings
from app.service import RAGService


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    service = RAGService(get_settings())
    if service.pipeline is None:
        print("No data ingested yet. Run: python -m scripts.ingest")
        return

    session, route, mode = uuid.uuid4().hex, None, "hybrid"
    print(f"Hybrid RAG chat - model: {service.model_name}. Type /quit to exit.\n")
    try:
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/reset":
                service.sessions.clear(session)
                session = uuid.uuid4().hex
                print("(history cleared)\n")
                continue
            if text.startswith("/route"):
                value = text.split(maxsplit=1)[-1].upper()
                route = None if value == "AUTO" else value
                print(f"(route: {route or 'auto'})\n")
                continue
            if text.startswith("/mode"):
                mode = text.split(maxsplit=1)[-1].lower()
                print(f"(retrieval mode: {mode})\n")
                continue

            print("Assistant: ", end="", flush=True)
            final = None
            for kind, payload in service.stream(text, session_id=session, route_override=route,
                                                retrieval_mode=mode):
                if kind == "token":
                    print(payload, end="", flush=True)
                elif kind == "done":
                    final = payload
            print()
            if final is not None:
                if final.standalone_question != final.question:
                    print(f"  (rewritten: {final.standalone_question})")
                print(f"  route={final.route} ({final.route_method})  total={final.timings['total']:.1f}s")
                for n, doc in enumerate(final.documents, start=1):
                    print(f"  [{n}] {doc.metadata.get('source')}")
                if final.graph is not None:
                    print(f"  [TABLE] {final.graph.query}")
            print()
    finally:
        service.close()


if __name__ == "__main__":
    main()
