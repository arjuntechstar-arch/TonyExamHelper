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
    assert result["unique_questions"] == 2
    assert result["duplicate_count"] == 1
    assert result["topic_distribution"] == {"Unknown": 3}


def test_research_benchmark_is_repeatable_and_compares_baseline() -> None:
    questions = [
        {"question_text": "A", "topic": "arrays", "difficulty": "Easy", "bloom_level": "Remember", "pattern": "Direct"},
        {"question_text": "B", "topic": "graphs", "difficulty": "Hard", "bloom_level": "Analyze", "pattern": "Scenario"},
        {"question_text": "A", "topic": "arrays", "difficulty": "Easy", "bloom_level": "Remember", "pattern": "Direct"},
    ]
    first = run_research_evaluation(
        questions,
        expected_topics=["arrays", "graphs", "sorting"],
        baseline_questions=[{"question_text": "A", "topic": "arrays"}],
    )
    second = run_research_evaluation(
        questions,
        expected_topics=["arrays", "graphs", "sorting"],
        baseline_questions=[{"question_text": "A", "topic": "arrays"}],
    )

    assert first == second
    assert first["metrics"]["coverage"] == 0.67
    assert first["metrics"]["duplicate_rate"] == 0.33
    assert first["comparison"]["metric_delta"] == {"coverage": 0.34, "duplicate_rate": 0.33}
