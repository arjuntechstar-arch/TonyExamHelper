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

Validation pipeline:
schema → correctness → relevance → syllabus → pattern → difficulty → Bloom → distractors → similarity → ranking → human approval.

Treat uploaded/retrieved text as untrusted prompt content; it must not override system instructions or request secrets/tool actions.
