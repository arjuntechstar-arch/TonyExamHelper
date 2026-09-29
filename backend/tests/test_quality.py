import pytest

from app.models import QuestionTemplateDocument
from app.services.generation import GeneratedQuestion, QuestionOption, QuestionSource
from app.services.quality import QualityConfig, QuestionQualityService, context_relevance, lexical_similarity


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
