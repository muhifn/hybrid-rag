"""Natural language -> SQL/Cypher, with execution feedback for self-correction.

Why not `create_sql_query_chain` / `GraphCypherQAChain`? Both generate a query
once and give up on failure. With 7B local models the first attempt fails
often enough that a retry loop is the single biggest accuracy lever, so the
engine is built from the same LangChain primitives (prompt | llm | parser)
with a correction loop around it:

    question -> generate -> validate --fail--+
                   ^                         |
                   +---- error + bad query <-+-- execute --fail--+
                                                 |                |
                                              success             |
                                                 v                |
                                            GraphResult <---------+  (give up after N)

`GraphResult.attempts` records which attempt succeeded, so the evaluation
can report first-try accuracy vs final accuracy - the measured value of the loop.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.graph.base import GraphResult, GraphStore
from app.llm import text_parser
from app.text_utils import strip_code_fences

log = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You translate questions into {language} queries.

{schema}
{examples}
RULES:
1. Output ONLY one {language} query. No explanation, no code fences.
2. Use only tables/labels and columns/properties that exist in the schema above.
3. The query must be read-only. Never create, modify, or delete data.
4. When comparing text, use values exactly as listed in the schema.
5. For totals, averages, or counts, use aggregate functions.
6. Give aggregate result columns descriptive aliases.
7. Questions may be in Indonesian; table and column names are as in the schema."""

FIRST_ATTEMPT = "Question: {question}\n{language}:"

CORRECTION = """Question: {question}

This {language} query FAILED:
{bad_query}

Error: {error}

Write a corrected {language} query. Output only the query."""


class GraphQueryEngine:
    def __init__(self, store: GraphStore, llm: BaseChatModel, max_attempts: int = 3, row_limit: int = 50):
        self.store = store
        self.max_attempts = max_attempts
        self.row_limit = row_limit
        self.language = store.language.upper()

        shots = store.examples()
        examples = ""
        if shots:
            examples = "\nEXAMPLES:\n\n" + "\n\n".join(
                f"Question: {q}\n{self.language}: {query}" for q, query in shots
            ) + "\n"
        # Schema and examples are fixed per store, so bind them once.
        system = SYSTEM_TEMPLATE.format(
            language=self.language, schema=store.schema_description(), examples=examples
        )
        parser = text_parser()
        # Named so a LangSmith trace distinguishes the first attempt from each
        # correction retry; the gap between them is the self-correction metric.
        self._first = (
            ChatPromptTemplate.from_messages([("system", "{system}"), ("human", FIRST_ATTEMPT)])
            .partial(system=system, language=self.language)
            | llm | parser
        ).with_config(run_name=f"text_to_{self.language}")
        self._correct = (
            ChatPromptTemplate.from_messages([("system", "{system}"), ("human", CORRECTION)])
            .partial(system=system, language=self.language)
            | llm | parser
        ).with_config(run_name=f"text_to_{self.language}_correction")

    def generate(self, question: str, failure: tuple[str, str] | None = None) -> str:
        if failure is None:
            raw = self._first.invoke({"question": question})
        else:
            raw = self._correct.invoke({"question": question, "bad_query": failure[0], "error": failure[1]})
        return strip_code_fences(raw)

    def ask(self, question: str) -> GraphResult:
        failure: tuple[str, str] | None = None
        query = ""
        for attempt in range(1, self.max_attempts + 1):
            query = self.generate(question, failure)
            log.info("[%s attempt %d] %s", self.language, attempt, query)

            reason = self.store.validate(query)
            if reason:
                log.info("  rejected by validator: %s", reason)
                failure = (query, f"rejected by validator: {reason}")
                continue
            try:
                columns, rows = self.store.run(query, limit=self.row_limit)
            except Exception as exc:  # noqa: BLE001 - database syntax/semantic errors are the feedback
                message = f"{type(exc).__name__}: {exc}"
                log.info("  execution failed: %s", message[:300])
                failure = (query, message[:1000])
                continue
            return GraphResult(language=self.store.language, query=query,
                               columns=columns, rows=rows, attempts=attempt)

        return GraphResult(
            language=self.store.language, query=query, attempts=self.max_attempts,
            error=f"failed after {self.max_attempts} attempts: {failure[1] if failure else 'unknown'}",
        )
