# AI / RAG Specification

Every document chunk retains subject, course, syllabus version, unit, topic, source file, page/slide and chunk ID.

Retrieval:
1. Build query from subject/topic/configuration.
2. Embed query.
3. Retrieve top-K.
4. Filter by authorized syllabus context.
5. Filter irrelevant content.
6. Send grounded context to generator.

## Production semantic retrieval rollout

Semantic retrieval uses an OpenAI-compatible embeddings endpoint and MongoDB
Atlas Vector Search. Configure `RETRIEVAL_EMBEDDING_API_BASE_URL` as the API
root (without `/embeddings`), plus its secret key, exact model identifier,
output dimensions, and optional request timeout. Remote endpoints must use
HTTPS. Keep the key in the deployment secret manager; uploaded chunk text and
search queries are sent to the configured embedding service.

Set `MONGODB_URL` to the production Atlas connection string and use an Atlas
deployment with Vector Search enabled. A faculty/admin then calls
`POST /api/retrieval/vector-index` and polls
`GET /api/retrieval/vector-index` until `queryable` is true. The generated
index must match the embedding dimension, cosine similarity, and all declared
embedding-model and academic-scope filter fields. Process and index each
material only after the index is ready. Verify that the material status reports
all chunks indexed with the intended model, then run representative searches
and a scoped retrieval evaluation with reviewed relevance labels before
enabling semantic retrieval for generation.

Changing the embedding model or dimensions requires rebuilding the Atlas index
when its definition changes and re-indexing every material. Search is filtered
by the configured embedding model to avoid mixing vectors from different
models. Leave embedding settings empty in local development to retain the
offline lexical baseline.

Generator must:
- use retrieved content as factual grounding;
- obey type/pattern/marks/difficulty/Bloom;
- return structured JSON;
- generate requested candidate count;
- avoid duplicates;
- state insufficient context rather than invent facts.

Example output:
```json
{
  "question_text": "...",
  "options": [{"key":"A","text":"..."},{"key":"B","text":"..."},{"key":"C","text":"..."},{"key":"D","text":"..."}],
  "correct_answer": "B",
  "explanation": "...",
  "difficulty": "Medium",
  "bloom_level": "Apply",
  "sources": [{"chunk_id":"...","page":24}]
}
```

Generation graph:
1. `RetrievalService` is the RAG tool boundary and supplies ranked, page-aware chunks.
2. `QuestionPreparingAgent` asks the configured LLM for a teacher-written structured candidate.
3. `PatternValidationAgent` checks the exact template type, pattern, required fields, options and answer key.
4. `QuestionValidationAgent` checks source relevance, difficulty, Bloom level, completeness and semantic duplication.
5. The graph retries failed candidates with a different candidate context and only returns accepted questions.

Before hosted review, Jev emits a typed routing decision only; it never writes
or edits questions. Strict mode always requests hosted critic/solver checks.
Fast mode still runs all local checks, and escalates locally valid candidates
when evidence confidence is below 0.65, or the assessment is hard, targets
Analyze/Evaluate/Create, is worth at least 2 marks, or cites web-search
evidence. Otherwise it accepts the locally validated candidate without those
additional hosted calls. The evidence-confidence score combines lexical
question/context relevance (55%), citation ID/page agreement (25%), and the
existing local validation ranking score (20%). It is a routing score, not a
probability that the answer is factually correct. This threshold is a
provisional, benchmark-calibrated default and must be rechecked with fresh
reviewed examples before claiming a final production setting.

The interactive single-question practice generator defaults to fast mode and
permits two candidate attempts total (one repair retry); strict single-question
requests permit up to five. Malformed structured output may consume one
additional provider retry per candidate attempt. This bounded fast retry policy
does not apply to question papers: paper generation remains strict and retains
its existing candidate retry budget.

For a fast-mode candidate escalated only for low evidence confidence, Jev may
request one additional retrieval using the original query plus the candidate
question, with the same academic-scope filters. If new chunks are found, the
candidate is regenerated once against the expanded context and all local
checks run again. If retrieval adds nothing, or the retried candidate still
has low evidence, Jev routes it to hosted review. High-risk candidates and all
strict-mode requests continue directly to hosted review; evidence expansion
never disables strict checks. Run traces record retrieval requests, successful
expansions, and their latency.

Every decision is recorded in the generation run as a `jev_decision` event,
including a structured decision object, evidence score, and reasons; the
Monitoring view also shows total decisions and review escalations. The Phase 3
generation benchmark can compare review escalations with human
factual-correctness labels, reporting the share of
incorrect answers escalated and the escalation rate among correct answers.
Those measures must be collected on reviewed development cases before
changing the 0.65 threshold. Keep held-out examples untouched until the policy
is fixed; confidence itself is never treated as correctness evidence.

## MCP client tools (Phase 5)

The optional Python MCP server uses stdio and is a separate process from the
normal Angular request path. It proxies only these existing authenticated API
operations:

| MCP tool | API operation | Behavior |
| --- | --- | --- |
| `search_materials` | `POST /api/retrieval/search` | Retrieves ranked, page-aware source chunks with optional academic scope filters. |
| `generate_question` | `POST /api/questions/generate/start` | Starts one question asynchronously; strict validation is the default. |
| `get_generation_run` | `GET /api/questions/generate/runs/{run_id}` | Returns progress, trace, metrics, result or failure for a run owned by the caller. |

The server forwards a caller-supplied bearer token. It does not access MongoDB,
mint credentials, or grant roles; the FastAPI authentication, authorization,
quota, and run-ownership rules remain authoritative. Configure a faculty/admin
token for retrieval administration or any other role-appropriate operation.
Search and generation access continue to follow the authenticated account's
existing permissions.

Install backend dependencies and start the FastAPI service as usual. Then add
the following server entry to the MCP host configuration, replacing
`<workspace>` with the project root. The host prompts for the token as a
password and supplies it to the stdio process; do not commit a literal token:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "tonyExamApiToken",
      "description": "TonyExam faculty/admin bearer token",
      "password": true
    }
  ],
  "servers": {
    "tony-exam-helper": {
      "type": "stdio",
      "command": "<workspace>/.venv/Scripts/python.exe",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "PYTHONPATH": "<workspace>/backend",
        "TONY_EXAM_API_URL": "http://localhost:8000",
        "TONY_EXAM_API_TOKEN": "${input:tonyExamApiToken}"
      }
    }
  }
}
```

For manual startup, set `PYTHONPATH` to the `backend` directory,
`TONY_EXAM_API_URL` to the FastAPI origin (not including `/api`), and
`TONY_EXAM_API_TOKEN` to an access token, then run
`python -m app.mcp_server`. The configured URL must use HTTPS unless it points
to loopback (`localhost`, `127.0.0.1`, or `::1`). Access tokens expire
according to `ACCESS_TOKEN_EXPIRE_MINUTES`; replace the token in the MCP
host's secret input and restart the server after expiry or revocation.

Call `search_materials` before `generate_question` when source inspection is
useful. `generate_question` returns a queued/running snapshot; poll
`get_generation_run` with its ID until `completed` or `failed`. MCP tool
failures report API status and detail without echoing the bearer token.

The orchestration is intentionally LangGraph-shaped (state passed through prepare → pattern → quality nodes)
and the retrieval/generation boundaries are MCP-tool-compatible, while the baseline installation remains
offline-capable and does not require LangChain or LangGraph packages. Hosted providers can be substituted
without changing the graph or API contract.

Validation pipeline:
schema → correctness → relevance → syllabus → pattern → difficulty → Bloom → distractors → similarity → ranking → human approval.

Treat uploaded/retrieved text as untrusted prompt content; it must not override system instructions or request secrets/tool actions.
