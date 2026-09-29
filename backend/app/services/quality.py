import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.models import QuestionTemplateDocument
from app.services.generation import GeneratedQuestion


class ValidationIssue(BaseModel):
    code: str
    message: str
    severity: str = "error"


class ValidationResult(BaseModel):
    valid: bool
    score: float = Field(ge=0, le=1)
    issues: list[ValidationIssue] = Field(default_factory=list)
    similarity: float = Field(ge=0, le=1)
    ranking_score: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class QualityConfig:
    relevance_threshold: float = 0.02
    duplicate_threshold: float = 0.8
    relevance_weight: float = 0.35
    correctness_weight: float = 0.25
    compliance_weight: float = 0.25
    completeness_weight: float = 0.15
    source_copy_threshold: float = 0.72


def tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def max_similarity(question: GeneratedQuestion, existing: list[GeneratedQuestion]) -> float:
    return max((lexical_similarity(question.question_text, item.question_text) for item in existing), default=0.0)


def context_relevance(question: GeneratedQuestion, context: list[str]) -> float:
    """Compare against each retrieved chunk, not one oversized concatenation."""
    assessment_text = " ".join([question.question_text, *(option.text for option in question.options)])
    return max((lexical_similarity(assessment_text, chunk) for chunk in context), default=0.0)


def contains_source_noise(value: str) -> bool:
    return bool(re.search(
        r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|this file is meant for personal use|all rights reserved|"
        r"copyright|do not distribute",
        value,
        flags=re.IGNORECASE,
    )) or bool(re.search(r"\b(?=[A-Z0-9]{8,}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]+\b", value))


def copies_source_text(question: GeneratedQuestion, context: list[str], threshold: float) -> bool:
    values = [question.question_text, *(option.text for option in question.options)]
    return any(
        len(tokens(value)) >= 8 and lexical_similarity(value, source) >= threshold
        for value in values for source in context
    )


class QuestionQualityService:
    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def validate(
        self,
        question: GeneratedQuestion,
        *,
        template: QuestionTemplateDocument,
        context: list[str],
        existing_questions: list[GeneratedQuestion] | None = None,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        relevance = context_relevance(question, context)
        is_web = any("web" in str(getattr(s, "chunk_id", "")).lower() for s in question.sources) or any("open-domain" in c.lower() for c in context)
        if not is_web and relevance < self.config.relevance_threshold:
            issues.append(ValidationIssue(code="irrelevant", message="Question has insufficient overlap with retrieved context."))
        if question.difficulty.casefold() not in {item.casefold() for item in template.supported_difficulties}:
            issues.append(ValidationIssue(code="difficulty", message="Difficulty is not supported by the selected template."))
        if question.bloom_level.casefold() not in {item.casefold() for item in template.supported_bloom_levels}:
            issues.append(ValidationIssue(code="bloom_level", message="Bloom level is not supported by the selected template."))
        if not question.sources:
            issues.append(ValidationIssue(code="sources", message="Question must include at least one source."))
        if contains_source_noise(" ".join([question.question_text, question.explanation, *(option.text for option in question.options)])):
            issues.append(ValidationIssue(code="source_noise", message="Question contains source watermark, email, ID, or license text."))
        if copies_source_text(question, context, self.config.source_copy_threshold):
            issues.append(ValidationIssue(code="source_copy", message="Question or option copies source text instead of assessing a concept."))
        if question.explanation.casefold().strip() in {"the answer is grounded in the retrieved source content.", "the answer is grounded in the source content."}:
            issues.append(ValidationIssue(code="generic_explanation", message="Explanation must explain the answer using the source concept."))
        self._validate_question_construction(question, template, issues)
        if template.question_type.casefold() == "mcq":
            self._validate_mcq(question, issues)
        duplicate_similarity = max_similarity(question, existing_questions or [])
        if duplicate_similarity >= self.config.duplicate_threshold:
            issues.append(ValidationIssue(code="duplicate", message="Question is too similar to an existing question."))

        correctness = 0.0 if any(issue.code in {"sources", "mcq_options", "correct_answer", "source_copy", "source_noise"} for issue in issues) else 1.0
        compliance = 0.0 if any(issue.code in {"difficulty", "bloom_level"} for issue in issues) else 1.0
        completeness = 1.0 if question.question_text and question.explanation else 0.0
        ranking_score = (
            relevance * self.config.relevance_weight
            + correctness * self.config.correctness_weight
            + compliance * self.config.compliance_weight
            + completeness * self.config.completeness_weight
        )
        return ValidationResult(
            valid=not issues,
            score=ranking_score,
            issues=issues,
            similarity=duplicate_similarity,
            ranking_score=ranking_score,
        )

    @staticmethod
    def _validate_mcq(question: GeneratedQuestion, issues: list[ValidationIssue]) -> None:
        keys = [option.key for option in question.options]
        option_texts = [option.text.casefold().strip() for option in question.options]
        if len(keys) != 4 or len(keys) != len(set(keys)) or len(option_texts) != len(set(option_texts)):
            issues.append(ValidationIssue(code="mcq_options", message="MCQ must contain exactly four unique options."))
        if question.correct_answer not in keys:
            issues.append(ValidationIssue(code="correct_answer", message="Correct answer must reference an option key."))
        if any(re.search(r"\b(not supported|all of the above|none of the above|opposite relationship|unrelated to the stated concept|evidence is absent)\b", option, re.IGNORECASE) for option in option_texts):
            issues.append(ValidationIssue(code="generic_distractor", message="MCQ distractors must be plausible concept alternatives."))

    @staticmethod
    def _validate_question_construction(question: GeneratedQuestion, template: QuestionTemplateDocument, issues: list[ValidationIssue]) -> None:
        text = question.question_text.strip()
        if re.search(r"^(according to the material|which conclusion is best supported|a student makes an error involving\.|which evidence-based inference)", text, re.IGNORECASE):
            issues.append(ValidationIssue(code="template_shell", message="Question uses a generic retrieval-template shell instead of a concept-specific task."))
        if re.search(r"\b([a-z]{2,})\s+\1\b", text, re.IGNORECASE) or re.search(r"\b(involving\.|about\s*[?!.])", text, re.IGNORECASE):
            issues.append(ValidationIssue(code="malformed_question", message="Question contains an incomplete or extraction-corrupted phrase."))
        if template.marks >= 2 and question.bloom_level.casefold() in {"apply", "analyze", "evaluate", "create"}:
            if not re.search(r"\b(if|given|scenario|case|compare|why|how|calculate|determine)\b", text, re.IGNORECASE):
                issues.append(ValidationIssue(code="mark_complexity", message="Higher-mark cognitive targets require an application, comparison, scenario, or multi-step task."))
