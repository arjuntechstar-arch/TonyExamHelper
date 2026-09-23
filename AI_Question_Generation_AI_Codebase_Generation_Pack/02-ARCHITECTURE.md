# Architecture

```text
Angular
  ↓
FastAPI REST API
  ├── SQL Server
  └── AI Orchestrator
       ├── Document Processing
       ├── Retriever → Vector Database
       ├── Template Engine
       ├── LLM Provider
       ├── Validation
       ├── Similarity
       └── Ranking
```

RAG:
Document → extraction → cleaning → section detection → chunking → metadata → embeddings → vector store.

Query:
configuration → query embedding → top-K retrieval → context filtering → prompt → structured output.

Generation:
configuration + retrieved context + template → N candidates → validation → similarity → ranking → faculty approval → question bank.
