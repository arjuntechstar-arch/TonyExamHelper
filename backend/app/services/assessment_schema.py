"""Shared structural rules for assessment blueprints and generated questions."""

import re


_MCQ_PATTERNS = {
    "mcq", "multiple choice", "direct concept", "scenario based", "statement based",
    "assertion and reason", "case based", "application based",
}
_WRITTEN_RESPONSE_PATTERNS = {
    "short answer", "long answer", "descriptive", "essay", "define", "explain",
    "compare", "differentiate", "apply", "analyze", "case study", "design", "problem solving",
}


def normalized_question_type(question_type: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", question_type.casefold()))


def is_mcq_question_type(question_type: str) -> bool:
    normalized = normalized_question_type(question_type)
    return normalized in {"mcq", "multiple choice", "multiple choice question"}


def is_descriptive_question_type(question_type: str) -> bool:
    """This product currently models every non-MCQ format as a written response."""
    return not is_mcq_question_type(question_type)


def section_format_error(question_type: str, pattern: str, marks: int) -> str | None:
    """Return a human-readable blueprint incompatibility, if one exists.

    A two-mark MCQ is valid when a blueprint deliberately calls for one. The
    important rule is that the declared type and pattern agree, and that a
    written-response section has enough marks to assess a substantive answer.
    """
    normalized_pattern = " ".join(re.findall(r"[a-z0-9]+", pattern.casefold()))
    is_mcq = is_mcq_question_type(question_type)
    descriptive_pattern = normalized_pattern in _WRITTEN_RESPONSE_PATTERNS
    mcq_pattern = normalized_pattern in _MCQ_PATTERNS

    if is_mcq and descriptive_pattern:
        return "MCQ sections cannot use a written-response pattern. Choose an MCQ pattern or change the question type."
    if not is_mcq and mcq_pattern:
        return "Written-response sections cannot use an MCQ pattern. Choose a written-response pattern or change the question type."
    if not is_mcq and marks < 2:
        return "Written-response sections must be worth at least 2 marks."
    return None


def blueprint_section_key(question_type: str, pattern: str, marks: int) -> tuple[str, str, int]:
    return (
        normalized_question_type(question_type),
        " ".join(re.findall(r"[a-z0-9]+", pattern.casefold())),
        marks,
    )
