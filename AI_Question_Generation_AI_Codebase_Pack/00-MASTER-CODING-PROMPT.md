# Master Coding Prompt

You are the lead software architect and senior full-stack AI engineer.

Build the complete application described by all specification files in this repository.

Project: **A Retrieval-Augmented Generative AI Framework for Pattern-Aware and Difficulty-Controlled Examination Question Generation**

Stack:
- Angular frontend
- Python + FastAPI backend
- Microsoft SQL Server
- Configurable LLM provider
- Configurable embedding provider
- Configurable vector database

Core flow:
Syllabus + Study Material + Exam Pattern + Constraints
→ Processing → Embeddings → Vector DB → RAG
→ Template + Constraints → LLM → Multiple Candidates
→ Validation → Semantic Similarity → Ranking
→ Human Approval → Question Bank
→ Student Practice / Institutional Assessment

Rules:
1. Read every specification file before coding.
2. Implement phase by phase; do not create a giant unverified implementation.
3. Do not replace real functionality with fake/mock functionality.
4. Keep LLM, embeddings and vector-store providers behind interfaces.
5. Never hard-code secrets; provide `.env.example`.
6. Use typed API contracts and consistent error handling.
7. Implement authentication and RBAC.
8. Validate uploads and treat uploaded/retrieved text as untrusted content.
9. Preserve source/page/chunk metadata for generated questions.
10. Add tests for each completed backend module.
11. Keep Angular contracts synchronized with FastAPI schemas.
12. After every phase: build, test, fix failures and update CHANGELOG.md.
13. Similarity thresholds and ranking weights must be configurable.
14. Do not mark a feature complete until it is implemented and tested.

Implementation order:
1. Foundation
2. Database/migrations
3. Authentication/RBAC
4. Subjects/courses/syllabus
5. Document processing
6. Embeddings/vector retrieval
7. Template engine
8. LLM generation
9. Validation/similarity/ranking
10. Faculty question bank
11. Student practice/mock tests
12. Analytics
13. Evaluation
14. Security/testing
15. Documentation/deployment

Before coding Phase 1, inspect all docs and output the final repository tree, architecture decisions and dependencies.
