"""Prompts for question condensation and answer synthesis, plus context formatting.

Design decisions behind the answer prompt (worth citing in the methodology):
  1. Numbered context + mandatory citations ([1], [TABLE]) make every claim
     traceable and discourage fabrication - inventing a fact would also mean
     inventing a citation.
  2. Explicit permission to say "not found". Without it, LLMs fill gaps from
     parametric knowledge, which breaks the claim that answers come from the
     knowledge base.
  3. Table results are marked as exact database computations to be copied, not
     recomputed: LLMs are poor at arithmetic and the database already did it.
  4. Low temperature (0.1): the task is extractive, and reproducible output
     matters for repeatable experiments.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.graph.base import GraphResult

NOT_FOUND = "Informasi tersebut tidak ditemukan dalam basis pengetahuan."

# Greetings and other conversational turns must NOT be routed into retrieval.
# "Halo" has no answer in any document, so the pipeline would search, find
# nothing relevant, and reply with NOT_FOUND - which reads as a broken bot.
# This prompt answers directly instead, and uses the real document titles so
# the bot states what it can actually help with rather than inventing scope.
SMALLTALK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an academic information assistant for Universitas Muhammadiyah Prof. Dr. Hamka "
     "(UHAMKA). You answer questions about the academic documents you have been given.\n\n"
     "The user has just greeted you or asked something conversational rather than asking about "
     "the documents. Reply briefly and warmly (at most 2 short sentences), in the SAME language "
     "as the user (usually Bahasa Indonesia), and invite them to ask about the documents.\n\n"
     "Documents available to you:\n{documents}\n\n"
     "Rules:\n"
     "1. Do NOT invent facts about the documents and do NOT list them all.\n"
     "2. Do NOT mention retrieval, vectors, databases, context, or how you work internally.\n"
     "3. If asked who or what you are, say you are an academic information assistant.\n"
     "4. Keep it short. Do not add citations - there is nothing to cite here."),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's follow-up question as a STANDALONE question that can be "
     "understood without the conversation.\n"
     "Rules:\n"
     "1. Replace pronouns and references (nya, itu, tersebut, dia, it, that) with the noun they refer to.\n"
     "2. Keep the original meaning and language. Do not add new information.\n"
     "3. Do NOT answer the question. Output ONLY the rewritten question.\n"
     "4. If the question is already standalone, output it unchanged."),
    MessagesPlaceholder("history"),
    ("human", "Follow-up question: {question}\n\nStandalone question:"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context provided.\n\n"
     "RULES:\n"
     "1. Answer in the same language as the question (usually Bahasa Indonesia), clearly and concisely.\n"
     "2. Cite the source of every claim with its label, e.g. [1] or [TABLE].\n"
     "3. If the context does not contain the information, say exactly: \"" + NOT_FOUND + "\" "
     "Do not guess and do not use general knowledge.\n"
     "4. Numbers in [TABLE] are exact database results. Copy them as-is; do not recompute or round.\n"
     "5. When the question needs a rule from the documents and data from the table, combine both explicitly.\n"
     "6. Do not mention the words \"context\", \"chunk\", or \"passage\" in the answer.\n"
     "7. If the question asks for a list (\"daftar\", \"apa saja\") and the context contains the matching "
     "table rows, enumerate EVERY row of every matching table; do not summarize, group, or claim the "
     "data is unavailable while rows are present."),
    ("human", "CONTEXT:\n\n{context}\n\n---\n\nQUESTION: {question}"),
])


# The local model is loaded with a 12k-token context window (was 8k). Indonesian
# text runs ~3.5 chars/token, so 20k chars ~= 5.7k tokens; plus ~0.5k of prompt
# and the 3072-token completion budget that leaves ~2.7k tokens of headroom.
MAX_DOC_CHARS = 2800
MAX_CONTEXT_CHARS = 20000


def build_context(documents: list[Document], graph: GraphResult | None) -> str:
    """Number the retrieved chunks and append the table result, if any.

    The local model has an 8k context: oversized table chunks and long top-k
    lists can starve the answer of completion tokens (reasoning models then
    return an empty reply), so cap each chunk and the total context size.
    """
    blocks: list[str] = []
    used = 0
    for n, doc in enumerate(documents, start=1):
        meta = doc.metadata
        page = f" | page {meta['page']}" if meta.get("page") else ""
        body = doc.page_content
        if len(body) > MAX_DOC_CHARS:
            body = body[:MAX_DOC_CHARS].rstrip() + " …[terpotong]"
        block = (f"[{n}] Source: {meta.get('source', '?')}{page} | Title: {meta.get('title', '?')}\n"
                 f"{body}")
        if blocks and used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)

    if graph is not None:
        lang = graph.language.upper()
        if graph.ok and graph.rows:
            blocks.append(f"[TABLE] Result of a {lang} query over the database "
                          f"(exact values computed by the database):\n"
                          f"Query: {graph.query}\n\n{graph.to_markdown()}")
        elif graph.ok:
            blocks.append(f"[TABLE] The {lang} query ran successfully but matched no rows, "
                          f"i.e. no data meets the criteria.\nQuery: {graph.query}")
        else:
            blocks.append(f"[TABLE] The database query failed: {graph.error}")

    return "\n\n---\n\n".join(blocks)
