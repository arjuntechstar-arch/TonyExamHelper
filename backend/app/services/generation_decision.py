from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.models import QuestionTemplateDocument
from app.services.generation import GeneratedQuestion
from app.services.quality import ValidationResult, context_relevance


class JevDecision(BaseModel):
    action: Literal["accept_locally", "retrieve_more_evidence", "review_with_hosted_checks"]
    evidence_confidence: float = Field(ge=0, le=1)
    review_required: bool
    reasons: list[str]

    @model_validator(mode="after")
    def action_matches_review_flag(self) -> "JevDecision":
        if self.review_required != (self.action == "review_with_hosted_checks"):
            raise ValueError("action must agree with review_required.")
        return self


@dataclass(frozen=True)
class JevDecisionPolicy:
    # Provisional default for the fast-vs-strict policy. This is intentionally
    # calibrated conservatively for the current benchmark, while retaining a
    # measured path to revisit the threshold with more reviewed examples.
    minimum_evidence_confidence: float = 0.65
    high_risk_difficulty: frozenset[str] = frozenset({"hard"})
    high_risk_bloom_levels: frozenset[str] = frozenset({"analyze", "evaluate", "create"})
    high_risk_marks: int = 2


class JevDecisionAgent:
    """Routes an already locally validated question; never writes or rewrites it."""

    def __init__(self, policy: JevDecisionPolicy | None = None) -> None:
        self.policy = policy or JevDecisionPolicy()

    def decide(
        self,
        question: GeneratedQuestion,
        *,
        template: QuestionTemplateDocument,
        context_chunks: list[dict],
        validation: ValidationResult,
        strict_review_requested: bool,
        allow_retrieval_expansion: bool = False,
    ) -> JevDecision:
        context = [item["chunk"].content for item in context_chunks]
        relevance_support = min(context_relevance(question, context) / 0.20, 1.0)
        available_pages = {
            (item["chunk"].id, item["chunk"].page_number)
            for item in context_chunks
        }
        citation_support = float(
            bool(question.sources)
            and all((source.chunk_id, source.page) in available_pages for source in question.sources)
        )
        evidence_confidence = round(
            0.55 * relevance_support
            + 0.25 * citation_support
            + 0.20 * validation.ranking_score,
            4,
        )

        reasons: list[str] = []
        if strict_review_requested:
            reasons.append("strict_review_requested")
        if evidence_confidence < self.policy.minimum_evidence_confidence:
            reasons.append("low_evidence_confidence")
        if question.difficulty.casefold() in self.policy.high_risk_difficulty:
            reasons.append("high_difficulty")
        if question.bloom_level.casefold() in self.policy.high_risk_bloom_levels:
            reasons.append("advanced_bloom_level")
        if template.marks >= self.policy.high_risk_marks:
            reasons.append("multi_mark_question")
        if any(item["chunk"].metadata.get("source") == "web_search" for item in context_chunks):
            reasons.append("external_web_evidence")

        can_expand_evidence = (
            allow_retrieval_expansion
            and not strict_review_requested
            and reasons == ["low_evidence_confidence"]
        )
        if can_expand_evidence:
            return JevDecision(
                action="retrieve_more_evidence",
                evidence_confidence=evidence_confidence,
                review_required=False,
                reasons=reasons,
            )

        review_required = bool(reasons)
        return JevDecision(
            action="review_with_hosted_checks" if review_required else "accept_locally",
            evidence_confidence=evidence_confidence,
            review_required=review_required,
            reasons=reasons,
        )
