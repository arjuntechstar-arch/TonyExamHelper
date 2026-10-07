import re
from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.models import QuestionTemplateDocument
from app.services.assessment_schema import blueprint_section_key, is_mcq_question_type, section_format_error
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
    duplicate_threshold: float = 0.72
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


_DUPLICATE_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "cannot", "do", "does", "for", "from", "give", "how", "in", "is", "it", "of", "on", "or", "should", "that", "the", "their", "this", "to", "using", "what", "when", "which", "why", "with", "would",
    "answer", "answers", "concept", "correct", "describe", "explain", "following", "material", "model", "models", "question", "response", "source", "student", "statement", "terms",
}
_SEMANTIC_EQUIVALENTS = {
    "combinations": "combination", "combining": "combination", "contexts": "context", "encountering": "encounter", "generated": "generate", "generating": "generate", "limitations": "limitation", "limited": "limitation", "probabilities": "probability", "predicting": "predict", "prediction": "predict", "sequences": "sequence", "synonymous": "synonym", "unobserved": "unseen", "unknown": "unseen", "unseen": "unseen", "words": "word",
}
_CONTRASTING_CONCEPTS = (
    {"left", "right"}, {"smaller", "larger"}, {"lower", "higher"}, {"increase", "decrease"}, {"true", "false"}, {"precision", "recall"},
)


def semantic_tokens(value: str) -> set[str]:
    normalized = re.sub(r"\bn[\s-]?grams?\b", "ngram", value.casefold())
    normalized = normalized.replace("language models", "language model")
    words = re.findall(r"[a-z0-9]+", normalized)
    canonical = [_SEMANTIC_EQUIVALENTS.get(word, word) for word in words]
    return {word for word in canonical if word not in _DUPLICATE_STOP_WORDS and len(word) > 1}


def _response_text(question: GeneratedQuestion) -> str:
    if question.correct_answer:
        option = next((item.text for item in question.options if item.key == question.correct_answer), "")
        if option:
            return option
    return question.expected_answer or question.explanation


def _set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    jaccard = overlap / len(left | right)
    containment = overlap / min(len(left), len(right))
    return 0.25 * jaccard + 0.75 * containment


def _contains_contrast(left: set[str], right: set[str]) -> bool:
    return any((left & pair) and (right & pair) and (left & pair) != (right & pair) for pair in _CONTRASTING_CONCEPTS)


def semantic_similarity(left: GeneratedQuestion, right: GeneratedQuestion) -> float:
    """Compare the assessed idea and expected response, not just stem wording.

    This lightweight semantic fingerprint normalizes common inflections and
    domain phrasing such as ``N-gram``/``N-grams``. It deliberately includes
    the keyed or expected answer so an MCQ and short-answer version of the same
    concept are caught as duplicates.
    """
    left_stem = semantic_tokens(left.question_text)
    right_stem = semantic_tokens(right.question_text)
    left_response = semantic_tokens(_response_text(left))
    right_response = semantic_tokens(_response_text(right))
    stem_score = _set_similarity(left_stem, right_stem)
    response_score = _set_similarity(left_response, right_response)
    signature_score = _set_similarity(left_stem | left_response, right_stem | right_response)
    score = max(stem_score, response_score, signature_score)
    if _contains_contrast(left_stem | left_response, right_stem | right_response):
        score *= 0.5
    return score


def max_similarity(question: GeneratedQuestion, existing: list[GeneratedQuestion]) -> float:
    return max((semantic_similarity(question, item) for item in existing), default=0.0)


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
        format_error = section_format_error(template.question_type, template.pattern, template.marks)
        if format_error:
            issues.append(ValidationIssue(code="blueprint_format", message=format_error))
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
        if is_mcq_question_type(template.question_type):
            self._validate_mcq(question, issues)
        else:
            self._validate_descriptive(question, issues)
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
        if question.expected_answer and question.expected_answer.strip():
            issues.append(ValidationIssue(code="mcq_expected_answer", message="MCQ must use an option key rather than a written expected answer."))

    @staticmethod
    def _validate_descriptive(question: GeneratedQuestion, issues: list[ValidationIssue]) -> None:
        if question.options:
            issues.append(ValidationIssue(code="descriptive_options", message="Written-response questions cannot include MCQ options."))
        if question.correct_answer is not None:
            issues.append(ValidationIssue(code="descriptive_correct_answer", message="Written-response questions cannot use an option-key answer."))
        expected_answer = (question.expected_answer or "").strip()
        if not expected_answer:
            issues.append(ValidationIssue(code="expected_answer", message="Written-response questions require a concrete expected answer."))
        elif re.search(r"\b(?:key\s*answer\s*[:\-]?\s*)?verified\b|\bsee\s+(?:the\s+)?rationale\b", expected_answer, re.IGNORECASE):
            issues.append(ValidationIssue(code="placeholder_expected_answer", message="Expected answer must contain assessable response points, not a verification placeholder."))

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


def validate_paper(
    questions: list[GeneratedQuestion],
    *,
    sections: list[dict[str, object]] | None = None,
    config: QualityConfig | None = None,
) -> list[ValidationIssue]:
    """Run non-negotiable acceptance checks over the whole assembled paper."""
    issues: list[ValidationIssue] = []
    policy = config or QualityConfig()
    if not questions:
        return [ValidationIssue(code="paper_empty", message="A generated paper must contain at least one question.")]

    if sections:
        expected: Counter[tuple[str, str, int]] = Counter()
        for section in sections:
            key = blueprint_section_key(str(section["question_type"]), str(section["pattern"]), int(section["marks"]))
            expected[key] += int(section["count"])
        actual = Counter(blueprint_section_key(item.question_type, item.pattern, item.marks) for item in questions)
        if actual != expected:
            issues.append(ValidationIssue(
                code="blueprint_mismatch",
                message="Generated questions do not exactly match the blueprint's type, pattern, marks, and count.",
            ))
        for section in sections:
            error = section_format_error(str(section["question_type"]), str(section["pattern"]), int(section["marks"]))
            if error:
                issues.append(ValidationIssue(code="blueprint_format", message=error))

    for index, question in enumerate(questions, start=1):
        prefix = f"Question {index}: "
        if is_mcq_question_type(question.question_type):
            local: list[ValidationIssue] = []
            QuestionQualityService._validate_mcq(question, local)
        else:
            local = []
            QuestionQualityService._validate_descriptive(question, local)
        issues.extend(
            ValidationIssue(code=item.code, message=prefix + item.message, severity=item.severity)
            for item in local
        )

    for left_index, left in enumerate(questions):
        for right_index in range(left_index + 1, len(questions)):
            similarity = semantic_similarity(left, questions[right_index])
            if similarity >= policy.duplicate_threshold:
                issues.append(ValidationIssue(
                    code="semantic_duplicate",
                    message=(
                        f"Questions {left_index + 1} and {right_index + 1} assess the same idea "
                        f"(semantic similarity {similarity:.2f})."
                    ),
                ))
    return issues
