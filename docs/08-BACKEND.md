# FastAPI Backend Structure

```text
backend/app/
├── main.py
├── core/
├── api/routes/
│   ├── auth.py
│   ├── subjects.py
│   ├── syllabus.py
│   ├── materials.py
│   ├── templates.py
│   ├── questions.py
│   ├── question_bank.py
│   ├── practice.py
│   └── analytics.py
├── models/
├── schemas/
├── repositories/
├── services/
│   ├── document_processing/
│   ├── embeddings/
│   ├── retrieval/
│   ├── generation/
│   ├── validation/
│   ├── similarity/
│   └── ranking/
└── tests/
```

Use provider interfaces:
LLMProvider.generate_structured(...)
EmbeddingProvider.embed(...)
VectorStore.search(...)

Keep business logic independent of a specific AI provider.

## Kaggle-hosted Ollama

To route generation directly to an Ollama instance exposed from a Kaggle
notebook, set `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL` to the active HTTPS
ngrok origin, `OLLAMA_MODEL=qwen2.5:32b`, and optionally
`OLLAMA_TIMEOUT_SECONDS` (default 300). The tunnel must forward to Ollama port
11434. Optional `OLLAMA_API_KEY` is forwarded as a bearer token when a
protecting proxy is configured; Ollama itself normally does not authenticate
requests. Do not expose an unauthenticated public tunnel to untrusted users.

The backend calls `/v1/chat/completions` with JSON response mode and
`stream=false`, then applies the same local validation and generation-run
tracking as other providers. It sends `ngrok-skip-browser-warning: true` so
ngrok's browser interstitial is not mistaken for an API response. OpenRouter
keys may remain configured, but selecting `LLM_PROVIDER=ollama` uses the
Ollama provider directly and will not silently route prompts to OpenRouter.
Kaggle runtime shutdown or ngrok URL rotation requires updating
`OLLAMA_BASE_URL` and restarting the backend.
