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
    return max((lexical_similarity(question.question_text, chunk) for chunk in context), default=0.0)


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
        if template.question_type.casefold() == "mcq":
            self._validate_mcq(question, issues)
        duplicate_similarity = max_similarity(question, existing_questions or [])
        if duplicate_similarity >= self.config.duplicate_threshold:
            issues.append(ValidationIssue(code="duplicate", message="Question is too similar to an existing question."))

        correctness = 0.0 if any(issue.code in {"sources", "mcq_options", "correct_answer"} for issue in issues) else 1.0
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
        if len(keys) < 2 or len(keys) != len(set(keys)):
            issues.append(ValidationIssue(code="mcq_options", message="MCQ must contain at least two unique options."))
        if question.correct_answer not in keys:
            issues.append(ValidationIssue(code="correct_answer", message="Correct answer must reference an option key."))
