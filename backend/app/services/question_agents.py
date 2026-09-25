"""Multi-agent question generation orchestration.

The agents are deliberately small and deterministic at their boundaries so the
pipeline can later be backed by LangGraph/LangChain or exposed as MCP tools
without changing the API contract.
"""

from dataclasses import dataclass

from app.models import QuestionTemplateDocument
from app.services.generation import MAX_CONTEXT_CHUNKS, GeneratedQuestion, GenerationError, GenerationService, LLMProvider, ProviderRateLimitError
from app.services.quality import QuestionQualityService, ValidationIssue


@dataclass(frozen=True)
class AgentState:
    candidates: list[GeneratedQuestion]
    accepted: list[GeneratedQuestion]
    rejected: list[str]


class QuestionPreparingAgent:
    def __init__(self, generator: GenerationService) -> None:
        self.generator = generator

    def prepare(
        self,
        *,
        template: QuestionTemplateDocument,
        chunks: list[dict],
        difficulty: str,
        bloom_level: str,
        candidate_index: int,
        guidance: str | None = None,
    ) -> GeneratedQuestion:
        selected_chunks = select_context_window(chunks, candidate_index)
        return self.generator.generate(
            template=template,
            chunks=selected_chunks,
            difficulty=difficulty,
            bloom_level=bloom_level,
            candidate_count=1,
            candidate_index=candidate_index,
            guidance=guidance,
        )[0]


def select_context_window(chunks: list[dict], candidate_index: int, max_chunks: int = MAX_CONTEXT_CHUNKS) -> list[dict]:
    """Bound each model call while rotating source coverage across a paper.

    A paper may have hundreds of indexed chunks. Sending all of them in every
    prompt exceeds hosted-model context limits and was the direct cause of paper
    generation failures. A rotated window keeps each question grounded while
    distributing candidates across the uploaded material.
    """
    if len(chunks) <= max_chunks:
        return chunks
    start = (candidate_index * max_chunks) % len(chunks)
    return [chunks[(start + offset) % len(chunks)] for offset in range(max_chunks)]


class PatternValidationAgent:
    def validate(self, question: GeneratedQuestion, template: QuestionTemplateDocument) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if question.question_type.casefold() != template.question_type.casefold():
            issues.append(ValidationIssue(code="question_type", message="Question type does not match the template."))
        if question.pattern.casefold() != template.pattern.casefold():
            issues.append(ValidationIssue(code="pattern", message="Question pattern does not match the template."))
        if "question_text" in template.required_fields and not question.question_text.strip():
            issues.append(ValidationIssue(code="question_text", message="Question text is required."))
        if "explanation" in template.required_fields and not question.explanation.strip():
            issues.append(ValidationIssue(code="explanation", message="Explanation is required."))
        if template.question_type.casefold() == "mcq":
            keys = [option.key for option in question.options]
            if len(keys) < 2 or len(keys) != len(set(keys)):
                issues.append(ValidationIssue(code="mcq_options", message="MCQ must contain at least two unique options."))
            if question.correct_answer not in keys:
                issues.append(ValidationIssue(code="correct_answer", message="Correct answer must reference an option key."))
        return issues


class QuestionValidationAgent:
    def __init__(self, quality: QuestionQualityService | None = None) -> None:
        self.quality = quality or QuestionQualityService()

    def validate(
        self,
        question: GeneratedQuestion,
        *,
        template: QuestionTemplateDocument,
        context: list[str],
        existing_questions: list[GeneratedQuestion],
    ) -> list[ValidationIssue]:
        result = self.quality.validate(
            question,
            template=template,
            context=context,
            existing_questions=existing_questions,
        )
        return result.issues


class QuestionGenerationGraph:
    """LangGraph-style prepare -> pattern-check -> quality-check loop."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        max_retries: int = 1,
        max_attempts_per_question: int = 2,
    ) -> None:
        self.generator = GenerationService(provider=provider, max_retries=max_retries)
        self.preparer = QuestionPreparingAgent(self.generator)
        self.pattern_validator = PatternValidationAgent()
        self.question_validator = QuestionValidationAgent()
        self.max_attempts_per_question = max_attempts_per_question

    def generate(
        self,
        *,
        template: QuestionTemplateDocument,
        chunks: list[dict],
        difficulty: str,
        bloom_level: str,
        candidate_count: int = 1,
        existing_questions: list[GeneratedQuestion] | None = None,
        trace=None,
        guidance: str | None = None,
    ) -> list[GeneratedQuestion]:
        if candidate_count < 1 or candidate_count > 20:
            raise GenerationError("candidate_count must be between 1 and 20.")
        context = [result["chunk"].content for result in chunks]
        accepted = list(existing_questions or [])
        rejected: list[str] = []
        for candidate_index in range(candidate_count):
            if trace:
                trace(f"Preparing candidate {candidate_index + 1}/{candidate_count}.", stage="preparing")
            question: GeneratedQuestion | None = None
            for attempt in range(self.max_attempts_per_question):
                if trace:
                    trace(f"Calling model for candidate {candidate_index + 1}, attempt {attempt + 1}.", stage="model")
                try:
                    question = self.preparer.prepare(
                        template=template,
                        chunks=chunks,
                        difficulty=difficulty,
                        bloom_level=bloom_level,
                        candidate_index=candidate_index + attempt,
                        guidance=guidance,
                    )
                except ProviderRateLimitError as error:
                    if trace:
                        trace("Hosted model rate limit reached; switching this paper section to the grounded fallback.", stage="model_rate_limited")
                    raise error
                except GenerationError as error:
                    rejected.append(str(error))
                    continue
                issues = self.pattern_validator.validate(question, template)
                if trace:
                    trace(
                        f"Pattern validation: {'passed' if not issues else ', '.join(issue.code for issue in issues)}.",
                        stage="pattern_validation",
                    )
                if not issues:
                    issues = self.question_validator.validate(
                        question,
                        template=template,
                        context=context,
                        existing_questions=accepted,
                    )
                    if trace:
                        trace(
                            f"Question validation: {'passed' if not issues else '; '.join(f'{issue.code}: {issue.message}' for issue in issues)}.",
                            stage="question_validation",
                        )
                if not issues:
                    accepted.append(question)
                    break
                rejected.extend(issue.code for issue in issues)
            if question is None or not accepted or accepted[-1] is not question:
                continue
        generated = accepted[len(existing_questions or []):]
        if len(generated) != candidate_count:
            detail = "; ".join(rejected[-5:]) or "No candidate passed validation."
            raise GenerationError(
                f"Question generation produced {len(generated)} of {candidate_count} required candidates: {detail}"
            )
        return generated
