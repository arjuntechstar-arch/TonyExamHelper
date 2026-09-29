"""Multi-agent question generation orchestration.

The agents are deliberately small and deterministic at their boundaries so the
pipeline can later be backed by LangGraph/LangChain or exposed as MCP tools
without changing the API contract.
"""

from dataclasses import dataclass
import re

from pydantic import BaseModel, Field, field_validator

from app.models import QuestionTemplateDocument
from app.services.generation import DeterministicLLMProvider, MAX_CONTEXT_CHUNKS, GeneratedQuestion, GenerationError, GenerationService, LLMProvider, ProviderRateLimitError, source_evidence
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
        avoid_questions: list[str] | None = None,
    ) -> GeneratedQuestion:
        selected_chunks = select_context_window(chunks, candidate_index)
        facts = ConceptExtractionAgent().extract(selected_chunks)
        if not facts:
            raise GenerationError("Retrieved material did not contain complete, assessment-ready concept facts.")
        return self.generator.generate(
            template=template,
            chunks=selected_chunks,
            difficulty=difficulty,
            bloom_level=bloom_level,
            candidate_count=1,
            candidate_index=candidate_index,
            guidance=guidance,
            avoid_questions=avoid_questions,
            concept_facts=facts,
        )[0]


class ConceptExtractionAgent:
    """Turns clean evidence into bounded fact records before question construction."""

    def extract(self, chunks: list[dict]) -> list[dict]:
        facts: list[dict] = []
        for result in chunks:
            chunk = result["chunk"]
            evidence = source_evidence(chunk.content)
            for sentence in re.split(r"(?<=[.!?])\s+", evidence):
                fact = sentence.strip()
                if not self._usable(fact):
                    continue
                facts.append({"chunk_id": chunk.id, "page": chunk.page_number, "fact": fact})
                if len(facts) >= 8:
                    return facts
        return facts

    @staticmethod
    def _usable(fact: str) -> bool:
        words = re.findall(r"[A-Za-z][A-Za-z-]*", fact)
        return (
            len(words) >= 5
            and fact.endswith((".", "!", "?"))
            and not re.search(r"\b([a-z]{2,})\s+\1\b", fact, re.IGNORECASE)
            and not re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|personal use|copyright", fact, re.IGNORECASE)
        )


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
            if len(keys) != 4 or len(keys) != len(set(keys)):
                issues.append(ValidationIssue(code="mcq_options", message="MCQ must contain exactly four unique options."))
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


class CriticVerdict(BaseModel):
    grounded: bool
    answerable: bool
    single_correct_answer: bool
    question_complete: bool
    distractors_plausible: bool
    contains_source_noise: bool
    bloom_match: bool
    difficulty_match: bool
    mark_match: bool
    quality_score: float = Field(ge=0, le=1)
    problems: list[str] = Field(default_factory=list)

    @field_validator("quality_score", mode="before")
    @classmethod
    def normalize_ten_point_score(cls, value: object) -> object:
        """Accept a common 0–10 critic score while storing one canonical 0–1 value."""
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
        return numeric / 10 if 1 < numeric <= 10 else numeric


class SolverVerdict(BaseModel):
    answer: str
    grounded: bool
    rationale: str = ""


class LLMQuestionCriticAgent:
    """A separate model call that critiques a candidate before it is accepted."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    @property
    def enabled(self) -> bool:
        return not isinstance(self.provider, DeterministicLLMProvider)

    def validate(self, question: GeneratedQuestion, *, context: list[str], template: QuestionTemplateDocument) -> list[ValidationIssue]:
        if not self.enabled:
            return []
        evidence = "\n".join(f"EVIDENCE|{source_evidence(item)}" for item in context if source_evidence(item))
        prompt = "\n".join([
            "ROLE|QUESTION_CRITIC",
            "Return only one JSON object. Critique the candidate without rewriting it. quality_score must be a decimal from 0.0 to 1.0, never a 0–10 grade.",
            "JSON_SCHEMA|{\"grounded\":true,\"answerable\":true,\"single_correct_answer\":true,\"question_complete\":true,\"distractors_plausible\":true,\"contains_source_noise\":false,\"bloom_match\":true,\"difficulty_match\":true,\"mark_match\":true,\"quality_score\":0.0,\"problems\":[\"string\"]}",
            f"TYPE|{template.question_type}", f"PATTERN|{template.pattern}", f"MARKS|{template.marks}",
            f"DIFFICULTY|{question.difficulty}", f"BLOOM|{question.bloom_level}",
            f"QUESTION|{question.question_text}",
            *(f"OPTION|{option.key}|{option.text}" for option in question.options),
            f"EXPLANATION|{question.explanation}", evidence,
        ])
        verdict: CriticVerdict | None = None
        last_error: Exception | None = None
        for _ in range(2):
            try:
                verdict = CriticVerdict.model_validate(self.provider.generate_structured(prompt))
                break
            except ProviderRateLimitError:
                raise
            except Exception as error:
                last_error = error
        if verdict is None:
            return [ValidationIssue(code="critic_unavailable", message=f"Question critic did not return a valid verdict after 2 attempts: {last_error}")]
        failed = [
            name for name in ("grounded", "answerable", "single_correct_answer", "question_complete", "distractors_plausible", "bloom_match", "difficulty_match", "mark_match")
            if not getattr(verdict, name)
        ]
        if verdict.contains_source_noise:
            failed.append("contains_source_noise")
        if verdict.quality_score < 0.80:
            failed.append("quality_score")
        return [ValidationIssue(code="llm_critic", message="; ".join(verdict.problems or failed or ["Critic rejected candidate."]))] if failed else []


class IndependentAnswerSolverAgent:
    """Solves from evidence without receiving the generator's answer key."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    @property
    def enabled(self) -> bool:
        return not isinstance(self.provider, DeterministicLLMProvider)

    def validate(self, question: GeneratedQuestion, *, context: list[str]) -> list[ValidationIssue]:
        if not self.enabled or question.question_type.casefold() != "mcq":
            return []
        evidence = "\n".join(f"EVIDENCE|{source_evidence(item)}" for item in context if source_evidence(item))
        prompt = "\n".join([
            "ROLE|INDEPENDENT_ANSWER_SOLVER",
            "Return only one JSON object. Solve exclusively from EVIDENCE. The generator answer key is intentionally withheld.",
            "JSON_SCHEMA|{\"answer\":\"A\",\"grounded\":true,\"rationale\":\"string\"}",
            f"QUESTION|{question.question_text}", *(f"OPTION|{option.key}|{option.text}" for option in question.options), evidence,
        ])
        verdict: SolverVerdict | None = None
        last_error: Exception | None = None
        for _ in range(2):
            try:
                verdict = SolverVerdict.model_validate(self.provider.generate_structured(prompt))
                break
            except ProviderRateLimitError:
                raise
            except Exception as error:
                last_error = error
        if verdict is None:
            return [ValidationIssue(code="solver_unavailable", message=f"Independent answer solver did not return a valid verdict after 2 attempts: {last_error}")]
        if not verdict.grounded or verdict.answer != question.correct_answer:
            return [ValidationIssue(code="solver_disagreement", message="Independent answer solver could not verify the generated answer from evidence.")]
        return []


class QuestionGenerationGraph:
    """LangGraph-style prepare -> pattern-check -> quality-check loop."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        critic_provider: LLMProvider | None = None,
        max_retries: int = 1,
        max_attempts_per_question: int = 5,
    ) -> None:
        self.generator = GenerationService(provider=provider, max_retries=max_retries)
        self.preparer = QuestionPreparingAgent(self.generator)
        self.pattern_validator = PatternValidationAgent()
        self.question_validator = QuestionValidationAgent()
        verification_provider = critic_provider or self.generator.provider
        self.critic = LLMQuestionCriticAgent(verification_provider)
        self.solver = IndependentAnswerSolverAgent(verification_provider)
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
        allow_partial: bool = False,
        trace=None,
        guidance: str | None = None,
    ) -> list[GeneratedQuestion]:
        if candidate_count < 1 or candidate_count > 20:
            raise GenerationError("candidate_count must be between 1 and 20.")
        context = [result["chunk"].content for result in chunks]
        accepted = list(existing_questions or [])
        rejected: list[str] = []
        generation_offset = len(accepted)
        for candidate_index in range(candidate_count):
            if trace:
                trace(f"Preparing candidate {candidate_index + 1}/{candidate_count}.", stage="preparing")
            question: GeneratedQuestion | None = None
            for attempt in range(self.max_attempts_per_question):
                if trace:
                    trace(f"Calling model for candidate {candidate_index + 1}, attempt {attempt + 1}.", stage="model")
                try:
                    # A new angle is used for every retry and for later requests
                    # against the same template. This prevents a duplicate from
                    # being retried with an effectively identical prompt.
                    attempt_index = generation_offset + candidate_index + (attempt * candidate_count)
                    question = self.preparer.prepare(
                        template=template,
                        chunks=chunks,
                        difficulty=difficulty,
                        bloom_level=bloom_level,
                        candidate_index=attempt_index,
                        guidance=guidance,
                        avoid_questions=[item.question_text for item in accepted],
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
                    issues = self.critic.validate(question, context=context, template=template)
                    if trace:
                        trace(f"LLM critic: {'passed' if not issues else issues[0].message}", stage="llm_critic")
                if not issues:
                    issues = self.solver.validate(question, context=context)
                    if trace:
                        trace(f"Independent solver: {'passed' if not issues else issues[0].message}", stage="independent_solver")
                if not issues:
                    accepted.append(question)
                    break
                rejected.extend(issue.code for issue in issues)
            if question is None or not accepted or accepted[-1] is not question:
                continue
        generated = accepted[len(existing_questions or []):]
        if len(generated) != candidate_count:
            detail = "; ".join(rejected[-5:]) or "No candidate passed validation."
            if allow_partial and generated:
                if trace:
                    trace(
                        f"Returned {len(generated)} of {candidate_count} validated questions; {candidate_count - len(generated)} could not be generated: {detail}",
                        stage="partial",
                    )
                return generated
            raise GenerationError(
                f"Question generation produced {len(generated)} of {candidate_count} required candidates: {detail}"
            )
        return generated
