# AI Examination Studio

AI Examination Studio is a full-stack question-generation and assessment platform for academic institutions. The project combines a FastAPI backend, a MongoDB-backed document store, and an Angular frontend to support syllabus-aware question generation, approval workflows, student practice, analytics, and deployment-ready service setup.

## What is implemented

The backend currently covers the core academic and assessment workflow:

- authentication and RBAC
- subject, course, syllabus, and topic management
- study material upload, validation, extraction, and chunking
- retrieval and indexing support
- template-driven question generation
- question review, approval, and bank creation
- model-paper generation and publication
- student practice and scoring
- student analytics and weak-topic summaries
- research evaluation metrics
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
- Node.js 18+
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

The current backend test suite passes and covers the implemented phases.

Run the deterministic Phase 13 research benchmark from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_research_evaluation.py
```

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
