# Unicorn Startups Chatbot

A Python chatbot that answers questions about the **118 Indian unicorn startups (Tracxn snapshot, May 2025)** from the
[Kaggle dataset](https://www.kaggle.com/datasets/srisankargiri/list-of-118-unicorn-startups-in-indiamay-2025).
It keeps conversation context, asks a clarifying question when a request is ambiguous, and ships with input
sanitisation, structured logging and Langfuse tracing.

Stack: **Python 3.10+ · FastAPI · Anthropic Claude (tool use + prompt caching) · Langfuse · pytest**.

```
you> Tell me about fintech unicorns
bot> • Razorpay – Payments – Bengaluru – full-stack payment gateway ... (12 companies)
you> Which of these are based in Bangalore?
bot> Of those, 5 are in Bengaluru: Razorpay, Juspay, PhonePe, ...
you> Which company is best to collaborate with?
bot [clarification]> What kind of collaboration are you after – payments, logistics, or technology/API?
```

---

## 1. Setup

```bash
git clone <this repo> && cd dglobalist
python -m venv .venv && source .venv/bin/activate        # or: uv venv .venv
pip install -r requirements.txt                          # or: uv pip install -r requirements.txt
cp .env.example .env                                     # then put your ANTHROPIC_API_KEY in .env
```

The dataset is committed at `data/unicorns_india_may2025.csv` (the file inside the Kaggle zip, `tracxn.csv`,
unchanged). To refresh it:

```bash
curl -L -o /tmp/unicorns.zip "https://www.kaggle.com/api/v1/datasets/download/srisankargiri/list-of-118-unicorn-startups-in-indiamay-2025"
unzip -p /tmp/unicorns.zip tracxn.csv > data/unicorns_india_may2025.csv
```

Optional: set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST` for self-hosted) to get per-turn traces.
Without them the tracer is a no-op.

## 2. Run

**CLI**

```bash
python -m app.cli                 # interactive; --session <id> to resume, --show-metrics to dump counters on exit
```

**API**

```bash
uvicorn app.api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"What does Razorpay do?"}'
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"Which of these are in Bangalore?","session_id":"<id from previous reply>"}'
curl localhost:8000/metrics
```

| Endpoint | Purpose |
|---|---|
| `POST /chat` `{message, session_id?}` | One turn. Returns `reply`, `kind` (`answer` / `clarification` / `rejected` / `error`), `tool_calls`, token `usage`, `latency_ms`, `trace_id` |
| `GET /sessions/{id}` · `DELETE /sessions/{id}` | Inspect / forget a conversation |
| `GET /metrics` | Counters, latency percentiles, derived rates |
| `GET /health` | Liveness, dataset size, model, whether Langfuse is on |

**Tests** (no network or API key needed - the model is a scripted fake):

```bash
pytest            # 22 tests: sanitiser, data layer, tool executor, orchestration, API
```

**Regenerate the sample transcript** (needs a real key):

```bash
python scripts/sample_session.py   # writes docs/SAMPLE_INTERACTION.md
```

## 3. Architecture

```
            ┌──────────────┐    ┌──────────────┐
 user ────▶ │  api.py      │ or │  cli.py      │
            └──────┬───────┘    └──────┬───────┘
                   └──────┬────────────┘
                          ▼
                 chatbot/engine.py  ───────────────────────────┐
   1. sanitize.py   (normalise · block · flag)                 │  logging_setup.py   (JSON lines)
   2. session.py    (replay condensed history)                 │  observability/metrics.py (counters, latency)
   3. llm.py        (Claude: tools + adaptive thinking + cache)│  observability/tracing.py (Langfuse trace/gen/tool)
   4. tools.py      (get_company · filter · search · clarify)  │
                          │                                     │
                          ▼                                     │
              data/repository.py  (filters, fuzzy name match, BM25)
              data/loader.py      (cp1252 CSV → clean `Company` records)
```

### How a turn works

1. **Sanitise** – NFKC-normalise, strip control chars, collapse whitespace, cap at `MAX_INPUT_CHARS`.
   Empty / meaningless / over-long inputs and *strong* injection patterns (`ignore previous instructions`,
   `<system>` tags, "developer mode"...) are **rejected without calling the model**. *Weak* patterns
   (`pretend you are`, long base64 blobs...) are **flagged**: the text is wrapped in `<user_message>` tags with a note
   telling the model to treat it as data.
2. **Build the prompt** – a static system prompt (behaviour rules + a compact catalogue of all 118 companies:
   name | sector | city | stage | year, ≈3K tokens) with a `cache_control` breakpoint, plus the last
   `MAX_HISTORY_TURNS` user/assistant pairs from the session, plus the new message.
3. **Tool loop** – the model decides which of four strict-schema tools to call:
   * `get_company(name)` – exact → substring → fuzzy (difflib) lookup; returns the full record or the candidate list
   * `filter_companies(sectors, locations, stages, founded_after/before, investor, keywords)` – deterministic filtering
   * `search_companies(query, limit)` – BM25 over name + sector + description for vague, capability-style asks
   * `ask_clarifying_question(question, reason)` – **terminates the turn**; the question is returned to the user

   Results are fed back until the model answers in text (max `MAX_TOOL_ROUNDS` rounds).
4. **Record** – the turn is appended to the session, logged as two JSON events (`user_query`, `bot_response`) with
   timestamp, session id, latency, tool calls and token usage; counters are updated; the Langfuse trace is closed.

### Design decisions

* **Catalogue in the prompt + tools for details, instead of pure RAG.** The dataset is tiny (118 rows). Showing the
  model the whole index lets it resolve names ("byjus" → BYJU'S), map informal sectors ("fintech" → Payments,
  Alternative Lending, Insurance…) and decide *which* tool to call, while the detailed text is fetched on demand.
  The prefix is static, so prompt caching makes it almost free after the first call. A vector store would add a
  dependency and an embedding model without improving recall at this scale; BM25 covers "fuzzy" asks adequately.
* **Clarification is a tool, not free text.** Ending the turn on `ask_clarifying_question` makes clarification an
  observable, testable event (counted in `/metrics`, scored in Langfuse) instead of something inferred from a question
  mark. The system prompt spells out *when* to ask (subjective ask with a missing criterion, ambiguous name,
  dangling "these") and when *not* to (a sensible default exists; never ask twice in a row).
* **Condensed history.** Only user/assistant text is replayed on later turns; intermediate tool calls live in logs and
  traces. Follow-ups like "which of these…" work because the model re-runs the narrower tool call rather than
  trusting its memory. Keeps per-turn cost bounded and the cached prefix dominant.
* **Grounding.** The model is instructed to answer only from tool results, quote amounts exactly as stored and flag
  the May-2025 snapshot date. Numbers are passed through as strings (e.g. `₹1,390Cr as on Mar 31, 2024`) rather than
  parsed into floats, to avoid mixing ₹ crore and USD values.
* **Model settings.** `claude-opus-5-5` with adaptive thinking and `effort=medium` (tunable). Server-side refusal
  fallbacks are on by default (`fallbacks: "default"`), so a safety-classifier refusal is retried on a fallback model
  inside the same call; set `ANTHROPIC_ENABLE_FALLBACKS=false` on Bedrock/Vertex where the parameter is unsupported.
  Tool schemas are `strict`, so arguments always validate.
* **Observability.** `metrics.py` is a thread-safe in-process store (counters with labels + 1000-sample latency
  reservoir → p50/p95). Captured: queries, responses by kind, clarifications, rejected and flagged inputs, LLM calls,
  tool calls per tool, tool errors, token usage (incl. cache reads), refusals, truncations, errors by reason, and
  derived clarification / error / rejection rates. Langfuse gets one `agent` trace per turn (session-grouped) with a
  `generation` per LLM call (model, usage) and a `tool` span per tool execution, plus `clarification_asked` and
  `error` scores for dashboards. The same `Metrics` interface could be backed by Prometheus/OTel without touching
  the engine.
* **Testability.** `llm.py` exposes a 1-method `ChatModel` protocol; tests inject a scripted fake so the whole
  orchestration (tool loop, clarification, history replay, sanitisation, error paths, API) is covered offline.

## 4. Assumptions and limitations

* **Static snapshot.** Facts are as of the May-2025 Tracxn export; 16 "Indian" unicorns are headquartered abroad
  (US/Singapore) and are kept as-is. The CSV is cp1252 with lost ₹ glyphs and stray `\xa0`; `loader.py` repairs those.
* **Sessions are in memory.** Restarting the server forgets conversations. A Redis/Postgres `SessionStore` is the
  obvious swap; the interface is three methods.
* **Single-process metrics.** `/metrics` reflects one worker. Use Langfuse (or export to Prometheus) for fleet-wide views.
* **No embeddings.** Semantic matching relies on the model reasoning over catalogue + BM25 hits. For a much larger
  dataset I would add `sentence-transformers` embeddings behind the same `search_companies` tool.
* **Sanitisation is regex-based.** It stops obvious injection and garbage; the real defence is the system prompt
  treating user text as data and the model having no tools beyond read-only dataset queries.
* **Subjective answers are opinions grounded in descriptions**, not market data; the bot says so.
* **Not done for lack of time:** streaming responses, per-user rate limiting, persistent sessions, an eval set of
  (question, expected-companies) pairs to track answer quality across prompt changes - that eval would be my first
  addition.

## 5. Sample interaction

See [`docs/SAMPLE_INTERACTION.md`](docs/SAMPLE_INTERACTION.md).

## Repository layout

```
app/
  api.py              FastAPI app            app/chatbot/engine.py    turn orchestration
  cli.py              terminal client        app/chatbot/llm.py       Anthropic adapter + ChatModel protocol
  bootstrap.py        wiring                 app/chatbot/tools.py     tool schemas + executor
  config.py           settings (.env)        app/chatbot/prompts.py   system prompt + catalogue
  sanitize.py         input sanitisation     app/chatbot/session.py   in-memory sessions
  logging_setup.py    JSON logging           app/data/loader.py       CSV → Company
  observability/      metrics.py, tracing.py app/data/repository.py   filter / fuzzy lookup / BM25
data/unicorns_india_may2025.csv   scripts/sample_session.py   tests/   docs/SAMPLE_INTERACTION.md
```
