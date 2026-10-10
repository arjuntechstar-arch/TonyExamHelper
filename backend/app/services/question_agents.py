"""Multi-agent question generation orchestration.

The agents are deliberately small and deterministic at their boundaries so the
pipeline can later be backed by LangGraph/LangChain or exposed as MCP tools
without changing the API contract.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re
from time import perf_counter
from typing import Callable

import httpx

from pydantic import BaseModel, Field, field_validator

from app.models import QuestionTemplateDocument
from app.services.assessment_schema import is_mcq_question_type, section_format_error
from app.services.generation import DeterministicLLMProvider, MAX_CONTEXT_CHUNKS, GeneratedQuestion, GenerationError, GenerationService, LLMProvider, ProviderRateLimitError, source_evidence
from app.services.generation_decision import JevDecisionAgent
from app.services.quality import QuestionQualityService, ValidationIssue, ValidationResult


@dataclass(frozen=True)
class AgentState:
    candidates: list[GeneratedQuestion]
    accepted: list[GeneratedQuestion]
    rejected: list[str]


class TimedLLMProvider:
    def __init__(self, provider: LLMProvider, trace: Callable[..., None]) -> None:
        self.provider = provider
        self.trace = trace

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    def generate_structured(self, prompt: str) -> dict:
        if prompt.startswith("ROLE|QUESTION_CRITIC\n"):
            role = "critic"
        elif prompt.startswith("ROLE|INDEPENDENT_ANSWER_SOLVER\n"):
            role = "solver"
        else:
            role = "generation"
        started = perf_counter()
        failed = False
        try:
            return self.provider.generate_structured(prompt)
        except Exception:
            failed = True
            raise
        finally:
            duration_ms = (perf_counter() - started) * 1000
            metric_deltas: dict[str, int | float] = {
                "model_calls": 1,
                f"{role}_calls": 1,
                "model_duration_ms": duration_ms,
                f"{role}_duration_ms": duration_ms,
            }
            if failed:
                metric_deltas["model_failures"] = 1
            self.trace(
                f"{role.capitalize()} model call took {duration_ms / 1000:.2f}s.",
                stage=f"{role}_model",
                duration_ms=duration_ms,
                level="error" if failed else "info",
                metric_deltas=metric_deltas,
            )


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
        format_error = section_format_error(template.question_type, template.pattern, template.marks)
        if format_error:
            issues.append(ValidationIssue(code="blueprint_format", message=format_error))
        if question.question_type.casefold() != template.question_type.casefold():
            issues.append(ValidationIssue(code="question_type", message="Question type does not match the template."))
        if question.pattern.casefold() != template.pattern.casefold():
            issues.append(ValidationIssue(code="pattern", message="Question pattern does not match the template."))
        if "question_text" in template.required_fields and not question.question_text.strip():
            issues.append(ValidationIssue(code="question_text", message="Question text is required."))
        if "explanation" in template.required_fields and not question.explanation.strip():
            issues.append(ValidationIssue(code="explanation", message="Explanation is required."))
        if is_mcq_question_type(template.question_type):
            keys = [option.key for option in question.options]
            if len(keys) != 4 or len(keys) != len(set(keys)):
                issues.append(ValidationIssue(code="mcq_options", message="MCQ must contain exactly four unique options."))
            if question.correct_answer not in keys:
                issues.append(ValidationIssue(code="correct_answer", message="Correct answer must reference an option key."))
            if question.expected_answer and question.expected_answer.strip():
                issues.append(ValidationIssue(code="mcq_expected_answer", message="MCQ must use an option key rather than a written expected answer."))
        else:
            if question.options:
                issues.append(ValidationIssue(code="descriptive_options", message="Written-response questions cannot include MCQ options."))
            if question.correct_answer is not None:
                issues.append(ValidationIssue(code="descriptive_correct_answer", message="Written-response questions cannot use an option-key answer."))
            expected_answer = (question.expected_answer or "").strip()
            if not expected_answer:
                issues.append(ValidationIssue(code="expected_answer", message="Written-response questions require a concrete expected answer."))
            elif re.fullmatch(r"(?:key\s*answer\s*[:\-]?\s*)?verified|see\s+(?:the\s+)?rationale", expected_answer, re.IGNORECASE):
                issues.append(ValidationIssue(code="placeholder_expected_answer", message="Expected answer must contain assessable response points, not a verification placeholder."))
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
        return self.evaluate(
            question,
            template=template,
            context=context,
            existing_questions=existing_questions,
        ).issues

    def evaluate(
        self,
        question: GeneratedQuestion,
        *,
        template: QuestionTemplateDocument,
        context: list[str],
        existing_questions: list[GeneratedQuestion],
    ) -> ValidationResult:
        return self.quality.validate(
            question,
            template=template,
            context=context,
            existing_questions=existing_questions,
        )


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
    semantic_duplicate: bool = False
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
        provider = getattr(self.provider, "provider", self.provider)
        return not isinstance(provider, DeterministicLLMProvider)

    def validate(
        self,
        question: GeneratedQuestion,
        *,
        context: list[str],
        template: QuestionTemplateDocument,
        existing_questions: list[GeneratedQuestion] | None = None,
    ) -> list[ValidationIssue]:
        if not self.enabled:
            return []
        evidence = "\n".join(f"EVIDENCE|{source_evidence(item)}" for item in context if source_evidence(item))
        comparison_set = (existing_questions or [])[-12:]
        prompt = "\n".join([
            "ROLE|QUESTION_CRITIC",
            "Compare the candidate's assessed concept and answer criterion against every EXISTING_QUESTION, not merely identical wording. Set semantic_duplicate true if it tests substantially the same knowledge or expected response.",
            "For an MCQ, single_correct_answer means exactly one option is correct. For a written response, set it true when EXPECTED_ANSWER provides concrete, assessable response points; false if it is missing or only a placeholder.",
            "Return only one JSON object. Critique the candidate without rewriting it. quality_score must be a decimal from 0.0 to 1.0, never a 0–10 grade.",
            "JSON_SCHEMA|{\"grounded\":true,\"answerable\":true,\"single_correct_answer\":true,\"question_complete\":true,\"distractors_plausible\":true,\"contains_source_noise\":false,\"bloom_match\":true,\"difficulty_match\":true,\"mark_match\":true,\"semantic_duplicate\":false,\"quality_score\":0.0,\"problems\":[\"string\"]}",
            f"TYPE|{template.question_type}", f"PATTERN|{template.pattern}", f"MARKS|{template.marks}",
            f"DIFFICULTY|{question.difficulty}", f"BLOOM|{question.bloom_level}",
            f"QUESTION|{question.question_text}",
            *(f"OPTION|{option.key}|{option.text}" for option in question.options),
            f"EXPECTED_ANSWER|{question.expected_answer or ''}",
            "SEMANTIC_DUPLICATE_FIELD|Include semantic_duplicate as true or false in the JSON verdict.",
            *(
                f"EXISTING_QUESTION|{index}|{item.question_type}|{item.question_text}|{item.correct_answer or item.expected_answer or item.explanation}"
                for index, item in enumerate(comparison_set, start=1)
            ),
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
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                return [ValidationIssue(code="critic_unavailable", message="Question critic service is unavailable; automatic network retries were stopped.")]
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
        if verdict.semantic_duplicate:
            failed.append("semantic_duplicate")
        if verdict.quality_score < 0.80:
            failed.append("quality_score")
        return [ValidationIssue(code="llm_critic", message="; ".join(verdict.problems or failed or ["Critic rejected candidate."]))] if failed else []


class IndependentAnswerSolverAgent:
    """Solves from evidence without receiving the generator's answer key."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    @property
    def enabled(self) -> bool:
        provider = getattr(self.provider, "provider", self.provider)
        return not isinstance(provider, DeterministicLLMProvider)

    def validate(self, question: GeneratedQuestion, *, context: list[str]) -> list[ValidationIssue]:
        if not self.enabled or not is_mcq_question_type(question.question_type):
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
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                return [ValidationIssue(code="solver_unavailable", message="Independent answer solver service is unavailable; automatic network retries were stopped.")]
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
        self.decision_agent = JevDecisionAgent()
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
        question_offset: int = 0,
        existing_questions: list[GeneratedQuestion] | None = None,
        allow_partial: bool = False,
        trace=None,
        guidance: str | None = None,
        run_llm_review: bool = True,
        retrieve_additional_evidence: Callable[[GeneratedQuestion], list[dict]] | None = None,
    ) -> list[GeneratedQuestion]:
        if candidate_count < 1 or candidate_count > 20:
            raise GenerationError("candidate_count must be between 1 and 20.")
        candidate_index = 0
        if trace:
            original_trace = trace
            def trace(message, **kwargs):
                details = dict(kwargs.pop("details", None) or {})
                details["question_index"] = question_offset + candidate_index + 1
                original_trace(message, details=details, **kwargs)
        if trace:
            self.generator.provider = TimedLLMProvider(self.generator.provider, trace)
            self.preparer.generator.provider = self.generator.provider
            self.critic.provider = TimedLLMProvider(self.critic.provider, trace)
            self.solver.provider = self.critic.provider
        accepted = list(existing_questions or [])
        rejected: list[str] = []
        generation_offset = len(accepted)
        for candidate_index in range(candidate_count):
            candidate_chunks = list(chunks)
            candidate_context = [result["chunk"].content for result in candidate_chunks]
            retrieval_expansion_used = False
            if trace:
                trace(f"Preparing candidate {candidate_index + 1}/{candidate_count}.", stage="preparing")
            question: GeneratedQuestion | None = None
            for attempt in range(self.max_attempts_per_question):
                if attempt and trace:
                    trace(
                        f"Retrying candidate {candidate_index + 1} after validation failure (attempt {attempt + 1}).",
                        stage="candidate_retry",
                        metric_deltas={"candidate_retries": 1},
                    )
                if trace:
                    trace(f"Calling model for candidate {candidate_index + 1}, attempt {attempt + 1}.", stage="model")
                try:
                    # A new angle is used for every retry and for later requests
                    # against the same template. This prevents a duplicate from
                    # being retried with an effectively identical prompt.
                    attempt_index = generation_offset + candidate_index + (attempt * candidate_count)
                    question = self.preparer.prepare(
                        template=template,
                        chunks=candidate_chunks,
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
                quality_result = None
                if not issues:
                    quality_result = self.question_validator.evaluate(
                        question,
                        template=template,
                        context=candidate_context,
                        existing_questions=accepted,
                    )
                    issues = quality_result.issues
                    if trace:
                        trace(
                            f"Question validation: {'passed' if not issues else '; '.join(f'{issue.code}: {issue.message}' for issue in issues)}.",
                            stage="question_validation",
                        )
                decision = None
                if not issues and quality_result is not None:
                    decision = self.decision_agent.decide(
                        question,
                        template=template,
                        context_chunks=candidate_chunks,
                        validation=quality_result,
                        strict_review_requested=run_llm_review,
                        allow_retrieval_expansion=(
                            retrieve_additional_evidence is not None
                            and not retrieval_expansion_used
                            and attempt + 1 < self.max_attempts_per_question
                        ),
                    )
                    if trace:
                        trace(
                            (
                                f"Jev decision: {decision.action} "
                                f"(evidence confidence {decision.evidence_confidence:.2f}; "
                                f"reasons: {', '.join(decision.reasons) or 'none'})."
                            ),
                            stage="jev_decision",
                            metric_deltas={
                                "jev_decisions": 1,
                                "jev_review_escalations": int(decision.review_required),
                                "jev_retrieval_requests": int(decision.action == "retrieve_more_evidence"),
                                "jev_evidence_confidence_total": decision.evidence_confidence,
                            },
                            details={"decision": decision.model_dump()},
                        )
                if (
                    not issues
                    and decision is not None
                    and decision.action == "retrieve_more_evidence"
                    and retrieve_additional_evidence is not None
                    and question is not None
                ):
                    retrieval_started = perf_counter()
                    additional_chunks = retrieve_additional_evidence(question)
                    known_chunk_ids = {str(item["chunk"].id) for item in candidate_chunks}
                    new_chunks = [
                        item for item in additional_chunks
                        if str(item["chunk"].id) not in known_chunk_ids
                    ]
                    retrieval_duration_ms = (perf_counter() - retrieval_started) * 1000
                    retrieval_expansion_used = True
                    if new_chunks:
                        candidate_chunks.extend(new_chunks)
                        candidate_context = [
                            result["chunk"].content for result in candidate_chunks
                        ]
                        if trace:
                            trace(
                                f"Jev added {len(new_chunks)} new source chunks; retrying once with expanded evidence.",
                                stage="jev_retrieval",
                                duration_ms=retrieval_duration_ms,
                                metric_deltas={
                                    "retrieval_duration_ms": retrieval_duration_ms,
                                    "jev_retrieval_expansions": 1,
                                },
                            )
                        continue

                    decision = decision.model_copy(update={
                        "action": "review_with_hosted_checks",
                        "review_required": True,
                        "reasons": [*decision.reasons, "no_additional_evidence"],
                    })
                    if trace:
                        trace(
                            "Jev found no new source chunks; escalating to hosted review.",
                            stage="jev_retrieval",
                            duration_ms=retrieval_duration_ms,
                            metric_deltas={
                                "retrieval_duration_ms": retrieval_duration_ms,
                                "jev_review_escalations": 1,
                            },
                        )
                if not issues and decision is not None and decision.review_required:
                    if self.critic.enabled and self.solver.enabled:
                        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="question-review") as reviewers:
                            critic_result = reviewers.submit(
                                self.critic.validate,
                                question,
                                context=candidate_context,
                                template=template,
                                existing_questions=accepted,
                            )
                            solver_result = reviewers.submit(
                                self.solver.validate,
                                question,
                                context=candidate_context,
                            )
                            critic_issues = critic_result.result()
                            solver_issues = solver_result.result()
                    else:
                        critic_issues = self.critic.validate(
                            question,
                            context=candidate_context,
                            template=template,
                            existing_questions=accepted,
                        )
                        solver_issues = self.solver.validate(question, context=candidate_context)
                    if trace:
                        trace(f"LLM critic: {'passed' if not critic_issues else critic_issues[0].message}", stage="llm_critic")
                        trace(f"Independent solver: {'passed' if not solver_issues else solver_issues[0].message}", stage="independent_solver")
                    issues = critic_issues + solver_issues
                    if any(issue.code in {"critic_unavailable", "solver_unavailable"} for issue in issues):
                        if trace:
                            trace("Review service unavailable; stopping candidate retries.", stage="review_unavailable", level="error")
                        raise GenerationError("Required review service is unavailable; candidate retries stopped.")
                elif not issues and decision is not None and trace:
                    trace(
                        "Jev accepted the locally validated question; hosted critic and independent solver were skipped.",
                        stage="llm_review_skipped",
                        metric_deltas={"llm_reviews_skipped": 1},
                    )
                if not issues:
                    accepted.append(question)
                    if trace:
                        trace("Question passed all required checks.", stage="question_accepted")
                    break
                rejected.extend(issue.code for issue in issues)
                if trace:
                    trace(
                        f"Candidate rejected by validation: {', '.join(issue.code for issue in issues)}.",
                        stage="candidate_rejected",
                        metric_deltas={"validation_rejections": 1},
                    )
            if question is None or not accepted or accepted[-1] is not question:
                if trace:
                    trace("Question could not pass validation.", stage="question_failed", level="warning")
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
