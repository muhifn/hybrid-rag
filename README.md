# Hybrid RAG Chatbot — LangChain + FastAPI

A retrieval-augmented generation chatbot that combines **two retrieval paths** and answers with a
**local LLM** through an OpenAI-compatible API. Built as a thesis (skripsi) implementation.

```
user question
      │
      ▼
┌─────────────────────┐
│ 1. ROUTER           │  LangChain ChatOpenAI, JSON-schema constrained output
│ VECTOR/GRAPH/HYBRID │  + Indonesian keyword heuristic as fallback
└─────────┬───────────┘
     ┌────┴─────────────────────┐
     ▼                          ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│ 2a. VECTOR RETRIEVAL │  │ 2b. GRAPH QUERY              │
│  bge-m3 embedding    │  │  schema → LLM →              │
│  Qdrant (dense)      │  │  SQL  (SQLite)  or           │
│  + BM25 (sparse)     │  │  Cypher (Neo4j)              │
│  → EnsembleRetriever │  │  + read-only guard           │
│    (weighted RRF)    │  │  + self-correction loop      │
└──────────┬───────────┘  └───────────────┬──────────────┘
           │  numbered chunks             │  table rows
           └──────────────┬───────────────┘
                          ▼
              ┌───────────────────────┐
              │ 3. SYNTHESIZER        │  LLM writes the answer with citations
              │    [1] [2] [TABLE]    │  refuses when context is insufficient
              └───────────┬───────────┘
                          ▼
                   answer + sources
```

| Component | Choice | Why |
|---|---|---|
| Chat LLM | **LM Studio** via `langchain_openai.ChatOpenAI` | Local, free, no token cost, data never leaves the machine |
| Embeddings | **bge-m3**, in-process (`HuggingFaceEmbeddings`) or served by LM Studio (`OpenAIEmbeddings`) — see `EMBEDDING_BACKEND` | Strong multilingual (incl. Bahasa Indonesia), 1024-dim, 8192-token input |
| Vector store | **Qdrant** (Apache-2.0) | Free forever; embedded mode needs no install, same code for production |
| Sparse retrieval | **`BM25Retriever`** + `EnsembleRetriever` | Catches rare exact tokens (document codes, acronyms) that dense search misses |
| Structured data | **SQLite** (text-to-SQL) or **Neo4j** (text-to-Cypher), switchable | SQLite: zero install, reliable with small models. Neo4j: real multi-hop traversal |
| API | **FastAPI** + uvicorn | Typed endpoints, auto docs, SSE streaming |
| Frontend | **Streamlit** | Chat UI showing route, sources, generated query, timings |
| Orchestration | **LangChain 1.x** (LCEL) | Prompts, chains, retrievers, structured output |

---

## 1. Requirements

- **Python 3.10+**
- **LM Studio** with a chat model loaded ([lmstudio.ai](https://lmstudio.ai)), plus an embedding
  model loaded if you use `EMBEDDING_BACKEND=openai`
- ~5 GB free disk for bge-m3 (+4 GB if you also use Neo4j Docker images)
- ~8 GB RAM is enough **only** with `EMBEDDING_BACKEND=openai`; running bge-m3 in-process next to
  LM Studio needs roughly 3 GB more
- Optional: Docker Desktop, only for `QDRANT_MODE=server` or `GRAPH_BACKEND=neo4j`

### Two interpreters on one machine

On this machine the bare `python` on PATH belongs to another tool's venv (you would see
`...\hermes-agent\venv\` in tracebacks) and does **not** have this project's dependencies.
The project ships its own virtualenv in `.venv` that sees the system packages — use it for
every command. From the project folder (`C:\Users\ipank\RAG\hybrid-rag-api`):

```powershell
# Windows (PowerShell) — activate once per terminal, then `python` just works:
.\.venv\Scripts\Activate.ps1

# ...or call it directly without activating (works in cmd too):
.\.venv\Scripts\python.exe -m scripts.ingest
```

```bash
# macOS / Linux:
source .venv/bin/activate
```

To rebuild that venv from scratch:

```powershell
C:\Users\ipank\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # only if system Python lacks deps
```

Symptom of the wrong interpreter: `ModuleNotFoundError: No module named 'langchain_community'`
right after a successful install.

---

## 2. Quick start

```bash
# 1. dependencies
python -m pip install -r requirements.txt

# 2. configuration
cp .env.example .env        # then edit if needed

# 3. LM Studio: load a chat model, then Developer tab -> Start Server (port 1234)

# 4. build both stores from ./data
python -m scripts.ingest

# 5. run the API  (single worker — see §7)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 6. in another terminal: the chat UI
streamlit run frontend/streamlit_app.py
```

- API docs: <http://127.0.0.1:8000/docs>
- Streamlit: <http://localhost:8501>
- Terminal-only chat: `python -m scripts.chat_cli`

With `EMBEDDING_BACKEND=huggingface`, `python -m scripts.ingest` downloads bge-m3 on first run
(about 2.3 GB, cached afterwards). With `EMBEDDING_BACKEND=openai` nothing is downloaded, but
LM Studio must already be serving the embedding model — the API probes it at startup too.

---

## 3. Using your own data

No code changes needed — drop your files in and re-ingest.

| Data | Location | Formats |
|---|---|---|
| Documents (prose) | `data/documents/` | `.md`, `.txt`, `.pdf` |
| Tables (structured) | `data/tables/` | `.csv` |

The bundled files are **sample data**. Delete them and add yours, then:

```bash
python -m scripts.ingest     # or POST /admin/ingest while the API runs
```

### CSV conventions

The schema and its relationships are inferred automatically:

1. **One CSV file = one table** (SQLite) or one node label (Neo4j). `pegawai.csv` → `pegawai` / `:Pegawai`
2. **The first column is the primary key.** `pegawai.csv` → `pegawai_id`
3. **A column ending in `_id` is a foreign key** when it equals, or ends with `_<pk>`, another
   table's primary key:
   - `unit_id` → table `unit` (exact match)
   - `penanggung_jawab_pegawai_id` → table `pegawai` (suffix match)
   - `penanggung_jawab_id` → **not** detected
4. Column types (`INTEGER` / `REAL` / `TEXT`) are inferred from the values.

If your columns don't follow the convention, declare them instead of renaming your data:

```ini
FK_OVERRIDES={"proyek.penanggung_jawab_id": "pegawai"}
```

### Documents

Markdown frontmatter is optional but improves citations:

```markdown
---
judul: Kebijakan Cuti Pegawai
tahun: 2025
---

# Kebijakan Cuti Pegawai
...
```

`judul`/`title` becomes the document title; otherwise the first `# ` heading is used. PDFs yield
one chunk per page with the page number kept in metadata, so answers can cite it.

### Then update the evaluation set

`eval/gold_set.json` holds the test questions and reference queries. **Rewrite it for your own
data** — see §6.

---

## 4. Configuration

Everything lives in `.env` (see `.env.example`). The values that matter most:

| Variable | Default | Notes |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | The `/v1` suffix is required |
| `LLM_MODEL` | *(blank)* | Blank = auto-pick the first loaded chat model |
| `LLM_API_KEY` | `lm-studio` | Enter your real token if LM Studio requires one (see §7) |
| `LLM_MAX_TOKENS` | `1024` | Raise it if answers look empty cut-off (see §8) |
| `ROUTER_STRUCTURED_OUTPUT` | `true` | JSON-schema decoding for the router; `false` if your model rejects it |
| `EMBEDDING_BACKEND` | `huggingface` | `openai` = LM Studio serves the model at `/v1/embeddings` (frees ~2.2 GB RAM on small machines); `huggingface` = in-process |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 1024-dim, multilingual. With `openai` backend, use the LM Studio id (e.g. `text-embedding-bge-m3`) |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` needs a CUDA build of torch; ignored when `EMBEDDING_BACKEND=openai` |
| `QDRANT_MODE` | `embedded` | `server` for Docker or Qdrant Cloud |
| `GRAPH_BACKEND` | `sqlite` | `neo4j` for text-to-Cypher |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Your main chunking experiment variables |
| `TOP_K` | `5` | Chunks passed to the answer LLM |
| `BM25_WEIGHT` | `0.5` | Dense weight is `1 - BM25_WEIGHT` |
| `GRAPH_MAX_ATTEMPTS` | `3` | Self-correction attempts for SQL/Cypher |
| `ADMIN_API_KEY` | *(blank)* | Set to protect `/admin/*` |
| `CORS_ORIGINS` | *(blank)* | e.g. `http://localhost:8501` if the UI runs elsewhere |

Relative paths resolve against the project folder, not the shell's working directory.
`GET /health` reports the effective settings.

---

## 5. HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness, models, chunk count, graph backend, effective config |
| `POST` | `/chat` | One question → answer, route, sources, table result, timings |
| `POST` | `/chat/stream` | Same, streamed as Server-Sent Events (`meta`, `token`*, `done`) |
| `GET` | `/sessions/{id}` | Conversation history for a session |
| `DELETE` | `/sessions/{id}` | Clear a session |
| `POST` | `/admin/ingest` | Rebuild both stores from `data/` (needs `X-Admin-Key` if set) |
| `GET` | `/admin/schema` | The exact schema text given to the text-to-query LLM |

```bash
curl -s http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Proyek mana yang wajib menggunakan tender terbuka?"}' | python -m json.tool
```

Reply shape:

```json
{
  "session_id": "…",
  "question": "Proyek mana yang wajib menggunakan tender terbuka?",
  "standalone_question": "…",
  "answer": "Berdasarkan panduan pengadaan, tiga proyek melewati ambang Rp 200 juta [1] … [TABLE]",
  "route": "HYBRID",
  "route_method": "llm",
  "route_reason": "needs the threshold rule and the project amounts",
  "sources": [{"rank": 1, "source": "panduan-pengadaan.md", "title": "…", "snippet": "…"}],
  "graph": {"language": "sql", "query": "SELECT …", "ok": true, "attempts": 1, "rows": []},
  "timings": {"condense": 0.0, "router": 1.4, "vector": 0.2, "graph": 2.9, "synthesis": 6.4, "total": 10.9}
}
```

Pass `"session_id"` back on the next request to keep conversation memory. Send `"route"` to
override the router, or `"retrieval_mode": "dense" | "bm25" | "hybrid"` to compare retrieval.

---

## 6. Evaluation (Results chapter)

```bash
python -m scripts.evaluate                    # full run
python -m scripts.evaluate --retrieval-only   # dense vs bm25 vs hybrid, no LLM needed
python -m scripts.evaluate --no-judge         # skip LLM-as-judge
```

Outputs to `results/`: `eval_<timestamp>.json` (everything + config, for the appendix),
`eval_<timestamp>.md` (tables for the thesis), `figure1_retrieval.png`, `figure2_latency.png`.

| # | Measures | Metrics |
|---|---|---|
| 1 | Retrieval quality | Hit@k, Recall@k, MRR, nDCG@k |
| 2 | Router correctness | Accuracy (LLM vs heuristic), confusion matrix |
| 3 | Text-to-query | Execution accuracy, first-try accuracy, attempt histogram |
| 4 | End-to-end | Latency per stage, LLM-as-judge faithfulness/relevance |

**Retrieval metrics are computed per document, not per chunk.** The set of relevant chunks changes
whenever `chunk_size` changes, so chunk-level labels would not survive your chunking experiments.
Document labels stay valid across configurations.

**Execution accuracy compares result sets, not query text** (as in Spider, Yu et al., 2018): two
differently written queries returning the same values are both correct. Column and row order are
ignored, and `5`, `5.0`, `"5"` compare equal.

### Experiments worth reporting

Change one variable, re-run §6, compare:

1. **Chunk size** — `CHUNK_SIZE` 400 / 800 / 1200 → effect on nDCG
2. **Retrieval mode** — already measured in Figure 1
3. **Graph backend** — `sqlite` vs `neo4j`, compare execution accuracy (SQL is usually ahead with 7B models)
4. **Self-correction** — set `GRAPH_MAX_ATTEMPTS=1`, measure the drop: that gap is the loop's value
5. **Router on/off** — force everything to HYBRID (`route_override`), compare quality and latency
6. **Model size** — 3B / 7B / 14B at fixed `TOP_K`

### Limitations to state in the Discussion

- The bundled gold set has **15 questions over sample data**. Not enough for an empirical claim —
  aim for 50+ questions over your own data, ideally annotated or verified by someone else.
- The judge is **the same model that wrote the answers** (self-preference bias, Zheng et al., 2023).
  Treat judge scores as a comparison *between configurations*, not absolute quality.

---

## 7. Production notes

```bash
docker compose up -d qdrant    # then QDRANT_MODE=server
docker compose up -d neo4j     # then GRAPH_BACKEND=neo4j  (APOC is enabled — it is required)
```

**Run a single uvicorn worker.** In embedded mode Qdrant locks its folder to one process. For
multiple workers, switch to `QDRANT_MODE=server` and keep session histories in Redis or Postgres
(swap `InMemoryChatMessageHistory` for a persistent `BaseChatMessageHistory` — nothing else changes).

### Why the bundled docker-compose enables APOC

`Neo4jGraph` builds its schema with `apoc.meta.data()`. A plain `neo4j:5` container will fail on
startup. The compose file sets `NEO4J_PLUGINS: '["apoc"]'` for that reason.

### Security posture

- **Read-only SQLite.** Three layers: the validator allows only one `SELECT`/`WITH` statement;
  the engine opens the file with `mode=ro`, so SQLite itself refuses writes; a progress handler
  aborts runaway queries.
- **Read-only Neo4j.** Same validator, plus sessions opened with `default_access_mode="READ"` so
  the server rejects writes independently.
- **Set `ADMIN_API_KEY`** before exposing `/admin/*` beyond localhost.
- **Never expose LM Studio directly to the internet.**

Still missing for multi-user service: authentication on `/chat`, rate limiting, persistent
sessions, incremental re-indexing, and request logging.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `503` "Cannot reach the LLM server" on `/chat` | LM Studio server not started, or `LLM_BASE_URL` missing the `/v1` suffix. LLM connection/auth failures are mapped to a clean 503; `/chat/stream` reports them as an SSE `error` event |
| `401`, "An LM Studio API token is required" | Your server requires a token: create one in LM Studio and set `LLM_API_KEY` in `.env` |
| Empty answers / cut off mid-sentence | Reasoning model spent `LLM_MAX_TOKENS` on hidden reasoning. Raise it, or disable thinking for that model in LM Studio. The API returns an explanatory message instead of an empty string |
| `<think>` tags visible in output | Should not happen — reasoning is stripped in text, streaming, and query generation. If you see it, that model emits a different tag |
| Embedding endpoint error | With `EMBEDDING_BACKEND=huggingface`, bge-m3 runs in-process (torch/transformers issue). With `openai`, it's LM Studio: confirm an embedding model is loaded and the server is on |
| `The paging file is too small` / crash on ingest | Out of memory: in-process bge-m3 needs ~2.2 GB next to LM Studio. On ~8 GB machines set `EMBEDDING_BACKEND=openai` so LM Studio serves it instead |
| `Dimensi tidak cocok` / dimension mismatch | Embedding model changed after the collection was built. Re-ingest with `--recreate` |
| Qdrant "already accessed by another instance" | Embedded mode is single-process. Stop the API before `scripts/ingest.py`, or use `POST /admin/ingest` |
| SQL/Cypher keeps failing | Model too small. Try a 7B+ instruct model; add examples to `eval/gold_set.json`; for small models use `GRAPH_BACKEND=sqlite` |
| Neo4j "cannot open" / schema error | Container not running, or APOC missing — use `docker compose up -d neo4j` |
| Answers look fabricated | Check §6 retrieval metrics first: if the right chunk is never retrieved, generation cannot fix it |

---

## 9. Project layout

```
app/
  config.py           Settings from .env (no secrets in code)
  llm.py              ChatOpenAI (LM Studio) + embeddings (HF or LM Studio) + reasoning stripping
  text_utils.py       Tokenizer, JSON recovery, <think> handling
  ingestion.py        Load .md/.txt/.pdf, frontmatter, recursive chunking
  vectorstore.py      Qdrant + dense/BM25/hybrid retrievers (EnsembleRetriever)
  prompts.py          Condense, answer, and context-formatting prompts
  router.py           VECTOR/GRAPH/HYBRID classification (LLM + heuristic)
  memory.py           Session histories, read-write lock
  pipeline.py         condense → route → retrieve → synthesize (invoke + stream)
  service.py          Owns every component; shared by API, CLI, and evaluation
  metrics.py          Hit@k, Recall@k, MRR, nDCG@k, result-set comparison
  schemas.py          Request/response models
  main.py             FastAPI app factory + endpoints
  graph/
    base.py           GraphStore protocol + GraphResult
    tables.py         CSV loading, type and foreign-key inference
    sqlite_store.py   text-to-SQL backend (read-only, layered guards)
    neo4j_store.py    text-to-Cypher backend (APOC, READ sessions)
    query_engine.py   NL → query with execution-feedback self-correction
scripts/
  ingest.py           Rebuild both stores from data/
  chat_cli.py         Terminal chat
  evaluate.py         Run the four evaluations, write JSON/MD/figures
frontend/
  streamlit_app.py    Chat UI over HTTP
eval/gold_set.json    Test questions + reference queries
data/                 documents/ and tables/  ← replace with your own
tests/                100 tests; fakes.py provides a real BaseChatModel stub
results/              Evaluation output (generated)
```

---

## 10. How the pieces map to LangChain

| Concept | Implementation |
|---|---|
| Chat model | `langchain_openai.ChatOpenAI` → LM Studio's OpenAI-compatible endpoint |
| Embeddings | `langchain_huggingface.HuggingFaceEmbeddings` or `langchain_openai.OpenAIEmbeddings`, selected by `EMBEDDING_BACKEND` |
| Vector store | `langchain_qdrant.QdrantVectorStore` |
| Chunking | `langchain_text_splitters.RecursiveCharacterTextSplitter` |
| Dense retriever | `vectorstore.as_retriever()` |
| Sparse retriever | `langchain_community.retrievers.BM25Retriever` |
| Hybrid fusion | `langchain_classic.retrievers.EnsembleRetriever` (weighted RRF) |
| SQL access | `langchain_community.utilities.SQLDatabase` |
| Graph access | `langchain_neo4j.Neo4jGraph` |
| Structured output | `ChatOpenAI.with_structured_output(RouteChoice, method="json_schema")` |
| Chains | LCEL: `prompt \| llm \| parser`, `MessagesPlaceholder` for history |
| Reasoning models | `<think>` blocks stripped in text, streaming, and query generation |

> **LangChain 1.x note:** `EnsembleRetriever` now lives in `langchain-classic`, not `langchain.retrievers`.
> Tutorials written for 0.x will fail on a fresh install with today's versions.

### Deliberate departures from `create_sql_query_chain` / `GraphCypherQAChain`

Both generate a query once and give up on failure. With 7B local models the first attempt fails
often enough that a retry loop is the single biggest accuracy lever, so the engine is built from the
same primitives (prompt | llm | parser) with correction layered around it. `GraphResult.attempts`
records which attempt succeeded, so §6 can report first-try vs final accuracy — the measured value
of the loop.

Also: `langchain-openai` 1.x serialises `max_tokens` as `max_completion_tokens`, which LM Studio
does not read (generation would run unbounded). For LM Studio the limit is sent via `extra_body`;
for the real OpenAI API the standard field is used.

---

## 11. Tests

```bash
python -m pytest                          # 100 tests, ~1 min; no LM Studio, no model download
RUN_SLOW=1 python -m pytest tests/test_integration.py   # adds in-process bge-m3 (~2.3 GB download, ~3 GB RAM)
docker compose up -d neo4j && python -m pytest tests/test_integration.py   # adds Neo4j
```

`tests/fakes.py` supplies a real LangChain `BaseChatModel`, so tests exercise the actual prompts,
LCEL chains, structured output, and streaming — only text generation is scripted. Anything needing
real infrastructure skips automatically when it is absent.

Covered: type and foreign-key inference, chunk boundaries and stable chunk ids, Qdrant indexing and
the re-ingest regression, BM25 behaviour, RRF deduplication, the SQL validator and read-only
enforcement, runaway-query interruption, the self-correction loop, router fallback, session memory,
condensation, SSE streaming, the metric formulas, reasoning-model handling, and LLM outage → 503
mapping.

---

## 12. Status: verified vs not yet run

**Verified**

- `python -m scripts.ingest` with real bge-m3 (in-process): 3 documents → 9 chunks, 3 tables / 26 rows, SQLite built
- `python -m scripts.evaluate --retrieval-only` end to end, writing `results/*.json|md|png`
- API boot, `/health`, `/admin/ingest` (401 without key, 200 with), `/admin/schema` showing injected
  categorical values and join paths
- Streamlit app loads with no exceptions
- 100 tests pass, 2 skip; pyflakes clean

**Not yet run** (needs a live LLM, so it is yours to check)

- `/chat` and `/chat/stream` against a real chat model
- `POST /admin/ingest` while LM Studio is up, and the full evaluation
- The Neo4j backend (`docker compose up -d neo4j` then the integration test)

Start with: load a chat model in LM Studio → `python -m scripts.ingest` →
`python -m scripts.chat_cli` → ask *"Berapa total anggaran proyek untuk setiap unit?"*
A correct run shows `route=GRAPH` and a table with **Operasional 785.000.000** on top.
