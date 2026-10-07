import pytest

from app.models import QuestionTemplateDocument
from app.services.generation import GeneratedQuestion, QuestionOption, QuestionSource
from app.services.quality import QualityConfig, QuestionQualityService, context_relevance, lexical_similarity, validate_paper


def template() -> QuestionTemplateDocument:
    return QuestionTemplateDocument(
        name="Direct Concept",
        question_type="MCQ",
        pattern="Direct Concept",
        required_fields=["question_text", "options", "correct_answer"],
        supported_difficulties=["Medium"],
        supported_bloom_levels=["Apply"],
        version="1.0",
    )


def question(text: str = "Where are smaller values placed in a binary search tree?") -> GeneratedQuestion:
    return GeneratedQuestion(
        question_text=text,
        options=[
            QuestionOption(key="A", text="Left subtree"),
            QuestionOption(key="B", text="Right subtree"),
            QuestionOption(key="C", text="Root node"),
            QuestionOption(key="D", text="A separate graph"),
        ],
        correct_answer="A",
        explanation="Smaller values are placed in the left subtree.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[QuestionSource(chunk_id="chunk-1", page=2)],
    )


def test_valid_question_gets_high_quality_score() -> None:
    result = QuestionQualityService().validate(
        question(),
        template=template(),
        context=["A binary search tree places smaller values in the left subtree."],
    )

    assert result.valid is True
    assert result.issues == []
    assert result.ranking_score > 0.5


def test_quality_flags_bad_correct_answer_and_irrelevance() -> None:
    invalid = question("What is the color of the sky?")
    invalid.correct_answer = "E"

    result = QuestionQualityService(QualityConfig(relevance_threshold=0.2)).validate(
        invalid,
        template=template(),
        context=["A binary search tree places smaller values in the left subtree."],
    )

    assert result.valid is False
    assert {issue.code for issue in result.issues} == {"irrelevant", "correct_answer"}


def test_quality_flags_semantic_duplicate_using_similarity_threshold() -> None:
    result = QuestionQualityService().validate(
        question(),
        template=template(),
        context=["Binary search tree values."],
        existing_questions=[question("Where are smaller values placed in a binary search tree?")],
    )

    assert result.valid is False
    assert any(issue.code == "duplicate" for issue in result.issues)
    assert lexical_similarity("binary tree", "binary tree") == pytest.approx(1.0)


def test_relevance_uses_best_chunk_for_long_retrieved_context() -> None:
    long_context = [
        "A binary search tree places smaller values in the left subtree. "
        + "Additional unrelated textbook content. " * 80,
        "A separate chapter discusses graph traversal and hashing.",
    ]

    result = QuestionQualityService().validate(
        question(),
        template=template(),
        context=long_context,
    )

    assert context_relevance(question(), long_context) >= 0.08
    assert result.valid is True


def test_quality_rejects_passage_copy_and_source_noise() -> None:
    copied = question("Which statement is correct about model evaluation?")
    copied.options[0].text = "Perplexity measures how well a probability model predicts a sample and lower values indicate better prediction quality."
    copied.options[1].text = "Accuracy"
    copied.options[2].text = "Recall"
    copied.options[3].text = "Precision"
    copied.explanation = "The answer follows from the model-evaluation concept."

    result = QuestionQualityService().validate(
        copied,
        template=template(),
        context=["Perplexity measures how well a probability model predicts a sample and lower values indicate better prediction quality."],
    )

    assert {issue.code for issue in result.issues} >= {"source_copy"}

    noisy = question()
    noisy.options[1].text = "arjunrajagopal97@gmail.com"
    result = QuestionQualityService().validate(noisy, template=template(), context=["A binary search tree places smaller values in the left subtree."])
    assert any(issue.code == "source_noise" for issue in result.issues)


def test_quality_rejects_template_shell_and_generic_distractors() -> None:
    candidate = question("Which conclusion is best supported by the material about N-grams don't do well at?")
    candidate.options[1].text = "It represents the opposite relationship."
    candidate.options[2].text = "It is unrelated to the stated concept."
    candidate.options[3].text = "It applies only when the evidence is absent."

    result = QuestionQualityService().validate(
        candidate, template=template(), context=["N-grams have difficulty generalizing to unseen sequences."],
    )

    assert {issue.code for issue in result.issues} >= {"template_shell", "generic_distractor"}


def test_paper_validation_catches_cross_format_semantic_duplicates() -> None:
    mcq = GeneratedQuestion(
        question_text="Why do N-gram language models give low probabilities to unseen combinations of synonymous words?",
        options=[
            QuestionOption(key="A", text="They rely on observed token sequences and cannot generalize unseen combinations."),
            QuestionOption(key="B", text="They always assign equal probability to every word."),
            QuestionOption(key="C", text="They ignore the words that come before a token."),
            QuestionOption(key="D", text="They only evaluate grammar rules."),
        ],
        correct_answer="A",
        explanation="N-grams estimate probability from observed local sequences.",
        difficulty="Medium",
        bloom_level="Understand",
        sources=[QuestionSource(chunk_id="chunk-1", page=2)],
        question_type="MCQ",
        pattern="Direct Concept",
        marks=1,
    )
    short_answer = GeneratedQuestion(
        question_text="Explain the N-gram limitation for unseen synonym word combinations.",
        options=[],
        correct_answer=None,
        expected_answer="N-gram models rely on observed sequences, so unseen synonymous combinations receive low probability.",
        explanation="The model cannot infer an unobserved local sequence from synonymy alone.",
        difficulty="Medium",
        bloom_level="Analyze",
        sources=[QuestionSource(chunk_id="chunk-1", page=2)],
        question_type="Short Answer",
        pattern="Explain",
        marks=2,
    )

    issues = validate_paper(
        [mcq, short_answer],
        sections=[
            {"question_type": "MCQ", "pattern": "Direct Concept", "count": 1, "marks": 1},
            {"question_type": "Short Answer", "pattern": "Explain", "count": 1, "marks": 2},
        ],
    )

    assert any(issue.code == "semantic_duplicate" for issue in issues)


def test_paper_validation_requires_a_real_expected_answer_for_written_responses() -> None:
    written = GeneratedQuestion(
        question_text="Explain the limitation of an N-gram model.",
        options=[QuestionOption(key="A", text="A placeholder MCQ option")],
        correct_answer="A",
        expected_answer="Key Answer: Verified",
        explanation="The model has a limited context window.",
        difficulty="Medium",
        bloom_level="Analyze",
        sources=[QuestionSource(chunk_id="chunk-1", page=2)],
        question_type="Short Answer",
        pattern="Explain",
        marks=2,
    )

    issues = validate_paper(
        [written],
        sections=[{"question_type": "Short Answer", "pattern": "Explain", "count": 1, "marks": 2}],
    )

    assert {issue.code for issue in issues} >= {
        "descriptive_options", "descriptive_correct_answer", "placeholder_expected_answer",
    }
