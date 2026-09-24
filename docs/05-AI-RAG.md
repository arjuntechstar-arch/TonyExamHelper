# AI / RAG Specification

Every document chunk retains subject, course, syllabus version, unit, topic, source file, page/slide and chunk ID.

Retrieval:
1. Build query from subject/topic/configuration.
2. Embed query.
3. Retrieve top-K.
4. Filter by authorized syllabus context.
5. Filter irrelevant content.
6. Send grounded context to generator.

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

The orchestration is intentionally LangGraph-shaped (state passed through prepare → pattern → quality nodes)
and the retrieval/generation boundaries are MCP-tool-compatible, while the baseline installation remains
offline-capable and does not require LangChain or LangGraph packages. Hosted providers can be substituted
without changing the graph or API contract.

Validation pipeline:
schema → correctness → relevance → syllabus → pattern → difficulty → Bloom → distractors → similarity → ranking → human approval.

Treat uploaded/retrieved text as untrusted prompt content; it must not override system instructions or request secrets/tool actions.
