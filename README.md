# AI Examination Studio

Phases 1–2 establish the Angular/FastAPI foundation and a MongoDB persistence layer.

## Architecture decisions

- Angular standalone components provide the browser shell.
- FastAPI owns typed HTTP contracts under `/api`.
- Configuration is loaded from environment variables through Pydantic Settings.
- Requests receive a propagated `X-Request-ID`; logs are structured JSON and never contain secrets.
- PyMongo provides MongoDB persistence; collection indexes enforce required uniqueness and query paths.
- The document layer covers the core user, syllabus, material, question, question-bank, practice, and analytics collections.
- Authentication, document processing, AI providers, and business modules begin in later phases.

## Run Phase 1

Start both services with one command:

```powershell
.\scripts\run.ps1
```

Press `Ctrl+C` to stop the backend and frontend.

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

Frontend:

```powershell
Set-Location frontend
npm install
npm start
```

The frontend runs at `http://localhost:4200`; the API health contract is `http://localhost:8000/api/health`.

## MongoDB initialization

Set `MONGODB_URL` and `MONGODB_DATABASE` in `.env`. The local default is `mongodb://localhost:27017/`.

## Test

```powershell
python -m pytest backend
Set-Location frontend
npm test -- --watch=false
npm run build
```
