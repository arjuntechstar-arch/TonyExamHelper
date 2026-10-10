# AI Examination Studio

AI Examination Studio is a full-stack question-generation and assessment platform for academic institutions. The project combines a FastAPI backend, a MongoDB-backed document store, and an Angular frontend to support syllabus-aware question generation, approval workflows, student practice, analytics, and deployment-ready service setup.

## What is implemented

The backend currently covers the core academic and assessment workflow:

- authentication and RBAC
- public email/password registration for student and faculty accounts
- subject, course, syllabus, and topic management
- study material upload, validation, extraction, and chunking
- retrieval and indexing with stop-word-filtered BM25/keyword ranking, quoted exact-phrase matching, and diversity-aware result selection; optional embedding-provider and MongoDB Atlas Vector Search fusion
- deterministic retrieval evaluation with Precision@k, Recall@k, MRR@k, and nDCG@k benchmark metrics
- template-driven question generation
- question review, approval, and bank creation
- model-paper generation and publication
- student practice and scoring
- student analytics and weak-topic summaries
- research evaluation metrics
- optional stdio MCP tools for authenticated material search, asynchronous question generation, and run status
- security headers and rate limiting
- protected administrator user management (create, list, activate/deactivate)
- authenticated student practice player with scoring and explanations

## Architecture

- Frontend: Angular application in the frontend folder
- Backend: FastAPI service under backend/app
- Persistence: MongoDB with PyMongo indexes and mongomock-based test coverage
- API prefix: /api
- Interactive API docs: /docs and /redoc
- API contract summary: [docs/04-API-CONTRACTS.md](docs/04-API-CONTRACTS.md)

## Local development

### Prerequisites

- Python 3.13
- Node.js v20.19.0+, v22.12.0+, or v24+ (required by Angular CLI 21)
- MongoDB instance or MongoDB-compatible local service

### 1) Create and activate the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

### 2) Configure environment variables

Copy the example file and adjust values as needed:

```powershell
Copy-Item .env.example .env
```

The default configuration uses:

- MONGODB_URL=mongodb://localhost:27017/
- MONGODB_DATABASE=ai_examination_studio
- JWT_SECRET_KEY must be set for authenticated endpoints
- LLM_PROVIDER defaults to `deterministic`; set it to `nvidia` and provide `NVIDIA_API_KEY` to use NVIDIA NIM with `moonshotai/kimi-k3`
- To use Ollama hosted from Kaggle, set `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL` to the current HTTPS ngrok URL, `OLLAMA_MODEL=qwen2.5:32b`, and adjust `OLLAMA_TIMEOUT_SECONDS` (default 300). The tunnel must forward to Ollama port 11434 and the Kaggle runtime must stay alive. The app calls Ollama's OpenAI-compatible JSON endpoint and does not use streaming because each question is validated as one structured JSON result. Ollama mode is selected directly and does not fall back to OpenRouter.
- Set `TAVILY_API_KEY` to enable Tavily Search when open-domain question generation has no indexed material matches. In that fallback, the topic is sent to Tavily and result URLs are included as citations.
- Semantic retrieval is opt-in. Set `RETRIEVAL_EMBEDDING_API_BASE_URL` (the API root, without `/embeddings`), `RETRIEVAL_EMBEDDING_API_KEY`, `RETRIEVAL_EMBEDDING_MODEL`, and the exact `RETRIEVAL_EMBEDDING_DIMENSIONS` returned by the model. Remote embedding endpoints must use HTTPS; `RETRIEVAL_EMBEDDING_TIMEOUT_SECONDS` controls the request timeout (default 30 seconds). Material chunk text and retrieval queries are sent to that configured provider. Leave the URL and key empty to retain local lexical retrieval.
- Semantic search requires MongoDB Atlas Vector Search (or a MongoDB deployment that supports the same stage). After configuring semantic retrieval, a faculty/admin calls `POST /api/retrieval/vector-index`, waits until `GET /api/retrieval/vector-index` reports `queryable: true`, then re-indexes processed materials through `POST /api/retrieval/materials/{material_id}/index`. The index filters include embedding model and academic scope metadata.

For production rollout, configure Atlas and the embedding provider in the deployment's secret store, create the vector index, and wait for it to become queryable before indexing material. Verify each material reports the expected `indexed_chunk_count`, total `chunk_count`, and embedding model from `GET /api/materials/{material_id}/status`; then exercise representative searches and the scoped retrieval benchmark before routing generation traffic through semantic retrieval. Re-index existing materials after changing embedding model or dimensions. The app does not enable semantic mode just because credentials exist: index creation and material indexing are explicit steps.

ChatGPT Plus is a consumer subscription and does not provide API access automatically. OpenAI API usage requires a separate API key and billing account. The local deterministic provider is therefore the default and requires no external key.

### 3) Start the backend

```powershell
uvicorn app.main:app --app-dir backend --reload
```

The API will be available at:

- health: http://localhost:8000/api/health
- docs: http://localhost:8000/docs
- redoc: http://localhost:8000/redoc

During local question-paper generation, the frontend starts a background run and
polls its progress. The generation panel shows retrieval, model, pattern
validation, question validation, retries, and the exact rejection message.
Structured backend logs are printed in the Uvicorn terminal as JSON. The
development status endpoints are:

- `POST /api/questions/generate/paper/start`
- `GET /api/questions/generate/runs/{run_id}`

### 4) Start the frontend

```powershell
Set-Location frontend
npm install
npm start
```

The frontend runs at http://localhost:4200.

Compose intentionally runs the backend and MongoDB only. The Angular
development server remains a separate local process; no production web-server
or cloud provider is assumed by this repository.

### Deployment checklist

Before using the scaffold outside local development:

1. Set a unique `JWT_SECRET_KEY` (at least 32 characters).
2. Set `ENVIRONMENT=production`, explicit `ALLOWED_ORIGINS`, and a managed
   `MONGODB_URL`/`MONGODB_DATABASE`.
3. Use a persistent, access-controlled `MATERIAL_STORAGE_PATH` and back it up.
4. Keep `LLM_PROVIDER=deterministic` unless the selected provider credentials
   and billing have been configured.
5. Put TLS and network access controls in the institution's chosen ingress or
   platform; this repository does not prescribe one.
6. Verify `GET /api/health`, `/docs`, and the backend/frontend checks after
   deployment.

## Docker deployment

The repository includes Docker support for a quick deployment workflow.

### Build and run with Docker Compose

```powershell
Copy-Item .env.example .env
# Set JWT_SECRET_KEY to a new secret of at least 32 characters in .env
docker compose up --build
```

This starts:

- the FastAPI backend on port 8000
- MongoDB on port 27017

### Individual backend container

```powershell
docker build -f backend\Dockerfile -t ai-exam-backend .
docker run --rm -p 8000:8000 --env-file .env ai-exam-backend
```

## Testing

Run the backend validation suite:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q
```

The backend suite covers authentication, retrieval, generation, evaluation,
and MCP integration.

Run the deterministic Phase 13 research benchmark from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_research_evaluation.py
```

Run the Phase 3 generation-quality benchmark against recorded outputs (the
runner never makes model calls and requires measured per-output durations):

```powershell
.\.venv\Scripts\python.exe scripts\run_generation_benchmark.py --results path\to\recorded-results.json
```

### MCP client integration

An optional local stdio MCP server exposes authenticated `search_materials`,
`generate_question`, and `get_generation_run` tools. It proxies the existing
FastAPI endpoints, so normal authentication, role checks, generation quotas,
and run ownership rules continue to apply. Start the backend and configure a
client using the steps in [docs/05-AI-RAG.md](docs/05-AI-RAG.md). Supply a
faculty/admin bearer token through the MCP host's secret environment/input
mechanism; do not save the token in repository files. Remote API URLs must use
HTTPS (plain HTTP is permitted only for loopback development).

Run the frontend build and unit tests:

```powershell
Set-Location frontend
npm run build
npm test -- --watch=false
```

The frontend tests run with Angular's configured Vitest/jsdom builder. Browser-provider flags such as `--browsers=ChromeHeadless` are not required for this project.

## Repository notes

- The app follows the roadmap in the project specification documents.
- The API uses request correlation IDs and structured JSON logging.
- The Monitoring view reports retrieval and model-call durations, call counts, validation retries, and model failures for new generation runs.
- Generated content is treated as untrusted and validation remains explicit.
- Security controls include RBAC, validation, headers, and rate limiting.
- Pull requests and pushes to `main`/`master` run the repository CI workflow
  (`.github/workflows/ci.yml`) for backend tests and frontend build/tests.

## Production integration boundaries

The repository is complete as a deterministic local MVP and deployment scaffold. These institution-specific integrations remain intentionally behind existing abstractions:

- connect a real LLM provider behind the generation abstraction
- replace hashing embeddings with a managed vector index for production workloads
- configure MongoDB, JWT secrets, CORS origins, and storage paths per environment
- choose and configure CI/CD, ingress, backups, monitoring, and an
  institutional deployment policy for staging and production
