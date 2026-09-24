from app.api.evaluation import evaluate_question_set, run_research_evaluation


def test_research_evaluation_computes_core_quality_metrics() -> None:
    questions = [
        {
            "question_text": "Explain the binary search algorithm.",
            "difficulty": "Medium",
            "bloom_level": "Apply",
            "pattern": "Direct Concept",
            "options": [{"key": "A", "text": "Left"}, {"key": "B", "text": "Right"}],
            "correct_answer": "A",
            "explanation": "Binary search halves the search space.",
            "sources": [{"chunk_id": "chunk-1", "page": 1}],
        },
        {
            "question_text": "Explain the binary search algorithm.",
            "difficulty": "Medium",
            "bloom_level": "Apply",
            "pattern": "Direct Concept",
            "options": [{"key": "A", "text": "Left"}, {"key": "B", "text": "Right"}],
            "correct_answer": "A",
            "explanation": "Binary search halves the search space.",
            "sources": [{"chunk_id": "chunk-1", "page": 1}],
        },
        {
            "question_text": "Describe the time complexity of merge sort.",
            "difficulty": "Hard",
            "bloom_level": "Analyze",
            "pattern": "Scenario Based",
            "options": [{"key": "A", "text": "O(n)"}, {"key": "B", "text": "O(log n)"}],
            "correct_answer": "B",
            "explanation": "Merge sort is O(n log n).",
            "sources": [{"chunk_id": "chunk-2", "page": 3}],
        },
    ]

    result = evaluate_question_set(questions)
    route_result = run_research_evaluation(questions)

    assert result["total_questions"] == 3
    assert result["duplicate_rate"] == 0.33
    assert result["coverage"] >= 0.5
    assert route_result["metrics"]["duplicate_rate"] == result["duplicate_rate"]
