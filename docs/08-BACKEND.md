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
