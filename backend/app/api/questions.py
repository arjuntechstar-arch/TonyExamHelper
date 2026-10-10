import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.api.auth import get_current_user
from app.core.database import get_database
from app.core.config import Settings, get_settings
from app.models import QuestionDocument, QuestionTemplateDocument, UserDocument
from app.services.assessment_schema import section_format_error
from app.services.generation import FailoverProvider, GeneratedQuestion, GenerationError, NvidiaProvider, OllamaProvider, OpenAICompatibleProvider, OpenRouterProvider, ProviderRateLimitError
from app.services.question_agents import QuestionGenerationGraph
from app.services.generation_runs import generation_runs, GenerationRun
from app.services.quality import QuestionQualityService, ValidationResult, validate_paper
from app.services.retrieval import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    RetrievalService,
    VectorSearchError,
    embedding_provider_from_settings,
)
from app.services.web_search import TavilySearchProvider, WebSearchError

router = APIRouter(prefix="/questions", tags=["questions"])
QuestionUser = Depends(get_current_user)
logger = logging.getLogger(__name__)


def _persist_run(database: Database, run: GenerationRun) -> None:
    document = run.snapshot()
    document["_id"] = run.id
    database.generation_runs.replace_one({"_id": run.id}, document, upsert=True)


class GenerateRequest(BaseModel):
    template_id: str
    query: str = Field(min_length=1, max_length=2_000)
    difficulty: str = Field(min_length=1, max_length=50)
    bloom_level: str = Field(min_length=1, max_length=50)
    # Current product flow: exactly one validated question per request.
    # Multi-question selection will return as an explicit UI feature.
    candidate_count: int = Field(default=1, ge=1, le=1)
    top_k: int = Field(default=5, ge=1, le=50)
    validation_mode: Literal["fast", "strict"] = "strict"
    subject_id: str | None = None
    course_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None
    allow_web_knowledge: bool = False


class BatchGenerateRequest(BaseModel):
    requests: list[GenerateRequest] = Field(min_length=1, max_length=20)


class PaperGenerateRequest(BaseModel):
    validation_mode: Literal["fast", "strict"] = "strict"
    template_id: str
    difficulty: str = Field(default="Medium", min_length=1, max_length=50)
    bloom_level: str = Field(default="Understand", min_length=1, max_length=50)
    query: str | None = Field(default=None, min_length=1, max_length=2_000)
    question_count: int = Field(default=1, ge=1, le=20)
    top_k: int = Field(default=5, ge=1, le=50)
    subject_id: str | None = None
    material_id: str | None = None
    course_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None
    allow_web_knowledge: bool = False


class ValidateRequest(BaseModel):
    template_id: str
    question: GeneratedQuestion
    context: list[str] = Field(min_length=1, max_length=50)
    existing_questions: list[GeneratedQuestion] = Field(default_factory=list, max_length=100)


class CreateQuestionRequest(BaseModel):
    question: GeneratedQuestion
    question_type: str = Field(min_length=1, max_length=30)
    pattern: str = Field(min_length=1, max_length=100)
    template_id: str | None = None
    subject_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None
    marks: int = Field(default=1, ge=1, le=100)


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    improvement_area: str | None = Field(default=None, min_length=2, max_length=80)
    comment: str | None = Field(default=None, max_length=1_000)


DAILY_QUESTION_LIMIT = 50


def _resolve_generation_template(
    database: Database,
    template_id: str,
    *,
    allow_web_knowledge: bool,
) -> QuestionTemplateDocument:
    """Load a configured blueprint or provide a safe web-workshop fallback."""
    template_data = database.question_templates.find_one({"_id": template_id, "status": "active"})
    if template_data:
        return QuestionTemplateDocument.model_validate(template_data)
    if not allow_web_knowledge:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found.")

    fallback = database.question_templates.find_one({"status": "active"})
    if fallback:
        return QuestionTemplateDocument.model_validate(fallback)
    return QuestionTemplateDocument(
        id=template_id or "default-web-template",
        name="Calibrated Standard Blueprint",
        question_type="MCQ",
        pattern="Direct Concept",
        required_fields=["question_text", "options", "explanation"],
        supported_difficulties=["Easy", "Medium", "Hard"],
        supported_bloom_levels=["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
        version="1.0",
        marks=1,
        total_marks=1,
        status="active",
    )


def _web_search_chunks(query: str, settings: Settings) -> list[dict]:
    if not settings.tavily_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search is not configured. Set TAVILY_API_KEY, or upload and index study material first.",
        )
    try:
        return TavilySearchProvider(settings.tavily_api_key).search(
            query,
            max_results=settings.tavily_max_results,
        )
    except WebSearchError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Web search failed: {error}",
        ) from error


def _attach_web_citations(
    questions: list[GeneratedQuestion],
    chunks: list[dict],
) -> list[GeneratedQuestion]:
    sources = {
        item["chunk"].id: item["chunk"].metadata
        for item in chunks
        if item["chunk"].metadata.get("source") == "web_search"
    }
    if not sources:
        return questions
    enriched: list[GeneratedQuestion] = []
    for question in questions:
        citations = [
            source.model_copy(update={
                "title": sources[source.chunk_id]["title"],
                "url": sources[source.chunk_id]["url"],
            })
            if source.chunk_id in sources else source
            for source in question.sources
        ]
        enriched.append(question.model_copy(update={"sources": citations}))
    return enriched


def _reserve_daily_quota(database: Database, user_id: str, requested: int) -> dict:
    """Atomically reserve generation capacity before a model call."""
    day = datetime.now(UTC).date().isoformat()
    try:
        usage = database.generation_usage.find_one_and_update(
            {"user_id": user_id, "day": day, "count": {"$lte": DAILY_QUESTION_LIMIT - requested}},
            {"$inc": {"count": requested}, "$setOnInsert": {"user_id": user_id, "day": day}},
            upsert=True,
            return_document=True,
        )
    except DuplicateKeyError:
        usage = None
    if usage is None:
        current = database.generation_usage.find_one({"user_id": user_id, "day": day}) or {"count": DAILY_QUESTION_LIMIT}
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"Daily generation limit reached. {max(0, DAILY_QUESTION_LIMIT - current['count'])} questions remain today.")
    return usage


def _personal_guidance(database: Database, user_id: str) -> str | None:
    areas = list(database.question_feedback.aggregate([
        {"$match": {"user_id": user_id, "rating": {"$lt": 3}, "improvement_area": {"$ne": None}}},
        {"$group": {"_id": "$improvement_area", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}, {"$limit": 2},
    ]))
    personal = ", ".join(str(area["_id"]) for area in areas)
    global_areas = [rule["improvement_area"] for rule in database.generation_feedback_rules.find({"active": True}, {"improvement_area": 1})]
    parts = []
    if personal:
        parts.append("Prioritize this learner's feedback: improve " + personal + ".")
    if global_areas:
        parts.append("Apply validated studio-wide improvements: " + ", ".join(global_areas) + ".")
    return " ".join(parts) or None


def _retrieval_service(database: Database) -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        database,
        embedding_provider=embedding_provider_from_settings(settings),
        vector_index_name=settings.retrieval_vector_index_name,
    )


def generate_questions(payload: GenerateRequest, database: Database, user_id: str | None = None, trace=None) -> list[GeneratedQuestion]:
    template = _resolve_generation_template(
        database,
        payload.template_id,
        allow_web_knowledge=payload.allow_web_knowledge,
    )
    settings = get_settings()
    try:
        retrieval_service = _retrieval_service(database)
        retrieval_started = perf_counter()
        chunks = retrieval_service.retrieve(
            payload.query,
            top_k=payload.top_k,
            subject_id=payload.subject_id,
            course_id=payload.course_id,
            syllabus_id=payload.syllabus_id,
            topic_id=payload.topic_id,
        )
    except (EmbeddingConfigurationError, EmbeddingProviderError, VectorSearchError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    retrieval_duration_ms = (perf_counter() - retrieval_started) * 1000
    if trace:
        trace(
            f"Retrieved {len(chunks)} ranked source chunks in {retrieval_duration_ms / 1000:.2f}s for query '{payload.query[:120]}'.",
            stage="retrieval",
            duration_ms=retrieval_duration_ms,
            metric_deltas={"retrieval_duration_ms": retrieval_duration_ms},
        )
    if not chunks:
        if payload.allow_web_knowledge:
            web_search_started = perf_counter()
            chunks = _web_search_chunks(payload.query, get_settings())
            web_search_duration_ms = (perf_counter() - web_search_started) * 1000
            if trace:
                trace(
                    f"Retrieved {len(chunks)} real web sources in {web_search_duration_ms / 1000:.2f}s for the topic.",
                    stage="web_search",
                    duration_ms=web_search_duration_ms,
                    metric_deltas={
                        "web_search_duration_ms": web_search_duration_ms,
                        "web_sources": len(chunks),
                    },
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No indexed source chunks are available for this query. Process and index study material first.",
            )
    try:
        provider = _configured_provider(settings)
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No AI generation model is available. Configure a supported provider API key and model, then try again.",
            )
        _attach_failover_trace(provider, trace)
        critic_provider = (
            _configured_provider(settings, settings.llm_critic_model)
            if payload.validation_mode == "strict" and settings.llm_critic_model
            else provider
        )
        if critic_provider is not provider:
            _attach_failover_trace(critic_provider, trace)
        existing_questions = _existing_questions(
            database,
            subject_id=payload.subject_id,
            syllabus_id=payload.syllabus_id,
            template_id=payload.template_id,
        )
        def retrieve_additional_evidence(question: GeneratedQuestion) -> list[dict]:
            return retrieval_service.retrieve(
                f"{payload.query}\n{question.question_text}",
                top_k=min(payload.top_k + 5, 50),
                subject_id=payload.subject_id,
                course_id=payload.course_id,
                syllabus_id=payload.syllabus_id,
                topic_id=payload.topic_id,
            )

        try:
            graph = QuestionGenerationGraph(
                provider=provider,
                critic_provider=critic_provider,
                max_retries=1,
                max_attempts_per_question=2 if payload.validation_mode == "fast" else 5,
            )
            generated = graph.generate(
                template=template,
                chunks=chunks,
                difficulty=payload.difficulty,
                bloom_level=payload.bloom_level,
                candidate_count=payload.candidate_count,
                existing_questions=existing_questions, guidance=_personal_guidance(database, user_id) if user_id else None, trace=trace,
                run_llm_review=payload.validation_mode == "strict",
                retrieve_additional_evidence=retrieve_additional_evidence,
            )
            return _attach_web_citations(generated, chunks)
        except ProviderRateLimitError:
            # The deterministic baseline cannot produce production-quality
            # distractors or concept synthesis. Never silently publish it when
            # a hosted model is temporarily unavailable.
            raise
        except GenerationError:
            if provider is None:
                raise
            logger.warning("Configured LLM provider failed; using deterministic fallback.", exc_info=True)
            generated = QuestionGenerationGraph(
                max_retries=1,
                max_attempts_per_question=2 if payload.validation_mode == "fast" else 5,
            ).generate(
                template=template,
                chunks=chunks,
                difficulty=payload.difficulty,
                bloom_level=payload.bloom_level,
                candidate_count=payload.candidate_count,
                existing_questions=existing_questions, guidance=_personal_guidance(database, user_id) if user_id else None, trace=trace,
                run_llm_review=payload.validation_mode == "strict",
                retrieve_additional_evidence=retrieve_additional_evidence,
            )
            return _attach_web_citations(generated, chunks)
    except ProviderRateLimitError as error:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="The configured question model is rate-limited. No fallback paper was created; retry after the provider cooldown.") from error
    except (EmbeddingConfigurationError, EmbeddingProviderError, VectorSearchError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except GenerationError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


@router.post("/generate", response_model=list[GeneratedQuestion])
def generate(
    payload: GenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> list[GeneratedQuestion]:
    _reserve_daily_quota(database, user.id, payload.candidate_count)
    run = generation_runs.begin(request_type="single", user_id=user.id, on_update=lambda item: _persist_run(database, item))
    run.log("Single-question generation worker started.", stage="started")
    try:
        questions = generate_questions(payload, database, user.id, trace=run.log)
        generation_runs.complete(run, [question.model_dump() for question in questions])
        return questions
    except Exception as error:
        generation_runs.fail(run, error)
        raise


@router.post("/generate/start")
def start_single_question_generation(
    payload: GenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> dict:
    _reserve_daily_quota(database, user.id, payload.candidate_count)

    def worker(run: GenerationRun) -> list[dict]:
        run.log("Starting single-question generation.", stage="started")
        questions = generate_questions(payload, database, user.id, trace=run.log)
        return [question.model_dump() for question in questions]

    run = generation_runs.create(
        worker,
        request_type="single",
        user_id=user.id,
        on_update=lambda item: _persist_run(database, item),
    )
    return run.snapshot()


@router.post("/generate/batch", response_model=list[list[GeneratedQuestion]])
def generate_batch(
    payload: BatchGenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> list[list[GeneratedQuestion]]:
    _reserve_daily_quota(database, user.id, sum(request.candidate_count for request in payload.requests))
    run = generation_runs.begin(request_type="batch", user_id=user.id, on_update=lambda item: _persist_run(database, item))
    try:
        results = []
        for index, request in enumerate(payload.requests, start=1):
            run.log(f"Starting batch item {index}/{len(payload.requests)}.", stage="batch")
            results.append(generate_questions(request, database, user.id, trace=run.log))
        generation_runs.complete(run, [question.model_dump() for group in results for question in group])
        return results
    except Exception as error:
        generation_runs.fail(run, error)
        raise


@router.post("/generate/paper", response_model=list[GeneratedQuestion])
def generate_paper(
    payload: PaperGenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
    trace=None,
) -> list[GeneratedQuestion]:
    template = _resolve_generation_template(
        database,
        payload.template_id,
        allow_web_knowledge=payload.allow_web_knowledge,
    )
    try:
        retrieval = _retrieval_service(database)
        retrieval_started = perf_counter()
        if payload.material_id:
            chunks = retrieval.retrieve_all(
                subject_id=payload.subject_id,
                study_material_id=payload.material_id,
                course_id=payload.course_id,
                syllabus_id=payload.syllabus_id,
                topic_id=payload.topic_id,
            )
        elif payload.query:
            chunks = retrieval.retrieve(
                payload.query,
                top_k=payload.top_k,
                subject_id=payload.subject_id,
                course_id=payload.course_id,
                syllabus_id=payload.syllabus_id,
                topic_id=payload.topic_id,
            )
        else:
            chunks = retrieval.retrieve_all(
                subject_id=payload.subject_id,
                course_id=payload.course_id,
                syllabus_id=payload.syllabus_id,
                topic_id=payload.topic_id,
            )
        retrieval_duration_ms = (perf_counter() - retrieval_started) * 1000
        if trace:
            trace(
                f"Retrieved {len(chunks)} source chunks in {retrieval_duration_ms / 1000:.2f}s for paper generation.",
                stage="retrieval",
                duration_ms=retrieval_duration_ms,
                metric_deltas={"retrieval_duration_ms": retrieval_duration_ms},
            )
        if not chunks:
            if payload.allow_web_knowledge and payload.query:
                web_search_started = perf_counter()
                chunks = _web_search_chunks(payload.query, get_settings())
                web_search_duration_ms = (perf_counter() - web_search_started) * 1000
                if trace:
                    trace(
                        f"No indexed source chunks matched; retrieved {len(chunks)} real web sources in {web_search_duration_ms / 1000:.2f}s.",
                        stage="web_search",
                        duration_ms=web_search_duration_ms,
                        metric_deltas={
                            "web_search_duration_ms": web_search_duration_ms,
                            "web_sources": len(chunks),
                        },
                    )
            else:
                raise HTTPException(status_code=422, detail="No indexed source chunks are available. Upload material or provide a competitive-exam topic.")
        generated: list[GeneratedQuestion] = []
        existing_questions = _existing_questions(
            database,
            subject_id=payload.subject_id,
            syllabus_id=payload.syllabus_id,
            template_id=payload.template_id,
        )
        settings = get_settings()
        provider = _configured_provider(settings)
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No AI generation model is available. Configure a supported provider API key and model, then try again.",
            )
        _attach_failover_trace(provider, trace)
        critic_provider = (
            _configured_provider(settings, settings.llm_critic_model)
            if settings.llm_critic_model
            else provider
        )
        if critic_provider is not provider:
            _attach_failover_trace(critic_provider, trace)
        sections = template.sections or [{
            "question_type": template.question_type,
            "pattern": template.pattern,
            "count": payload.question_count,
            "marks": template.marks,
        }]
        if trace:
            trace("Question blueprint ready.", stage="blueprint", details={"question_total": sum(int(section["count"]) for section in sections)})
        for section in sections:
            format_error = section_format_error(str(section["question_type"]), str(section["pattern"]), int(section["marks"]))
            if format_error:
                raise GenerationError(format_error)
        # A paper request must honour the complete selected blueprint. Each
        # section contributes its configured number of questions to one paper.
        for section_index, section in enumerate(sections, start=1):
            section_difficulties = [str(value) for value in section.get("supported_difficulties", [])] or [payload.difficulty]
            section_blooms = [str(value) for value in section.get("supported_bloom_levels", [])] or [payload.bloom_level]
            section_template = template.model_copy(update={
                "question_type": str(section["question_type"]),
                "pattern": str(section["pattern"]),
                "marks": int(section["marks"]),
                "supported_difficulties": section_difficulties,
                "supported_bloom_levels": section_blooms,
            })
            if trace:
                trace(
                    f"Blueprint section {section_index}: {section['count']} {section['question_type']} questions.",
                    stage="blueprint",
                )
            allocations: dict[tuple[str, str], int] = {}
            for candidate_index in range(int(section["count"])):
                key = (
                    section_difficulties[candidate_index % len(section_difficulties)],
                    section_blooms[candidate_index % len(section_blooms)],
                )
                allocations[key] = allocations.get(key, 0) + 1
            for (difficulty, bloom_level), candidate_count in allocations.items():
                try:
                    generated.extend(
                        QuestionGenerationGraph(provider=provider, critic_provider=critic_provider).generate(
                            template=section_template,
                            chunks=chunks,
                            difficulty=difficulty,
                            bloom_level=bloom_level,
                                candidate_count=candidate_count,
                                question_offset=len(generated),
                                existing_questions=existing_questions + generated,
                                allow_partial=True,
                                trace=trace,
                                run_llm_review=payload.validation_mode == "strict", guidance=_personal_guidance(database, getattr(user, "id", "")) if getattr(user, "id", None) else None,
                        )
                    )
                except ProviderRateLimitError:
                    raise
                except GenerationError:
                    if provider is None:
                        raise
                    logger.warning("Configured LLM provider failed during paper generation; using deterministic fallback.", exc_info=True)
                    if trace:
                        trace("Configured model failed validation; retrying with the local grounded fallback.", stage="model_fallback")
                    generated.extend(
                        QuestionGenerationGraph().generate(
                            template=section_template,
                            chunks=chunks,
                            difficulty=difficulty,
                            bloom_level=bloom_level,
                                candidate_count=candidate_count,
                                question_offset=len(generated),
                                existing_questions=existing_questions + generated,
                                allow_partial=True,
                                trace=trace,
                                run_llm_review=payload.validation_mode == "strict", guidance=_personal_guidance(database, getattr(user, "id", "")) if getattr(user, "id", None) else None,
                        )
                    )
        paper_issues = validate_paper(generated, sections=sections)
        if paper_issues:
            detail = "; ".join(issue.message for issue in paper_issues[:3])
            if trace:
                trace(f"Final paper validation rejected the draft: {detail}", stage="paper_validation")
            raise GenerationError(detail)
        if trace:
            trace("Final paper validation passed: blueprint, response schema, and semantic uniqueness verified.", stage="paper_validation")
        return _attach_web_citations(generated, chunks)
    except ProviderRateLimitError as error:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="The configured question model is rate-limited. No fallback paper was created; retry after the provider cooldown.") from error
    except (EmbeddingConfigurationError, EmbeddingProviderError, VectorSearchError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except (GenerationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/generate/paper/start")
def start_generate_paper(
    payload: PaperGenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> dict:
    template = _resolve_generation_template(
        database,
        payload.template_id,
        allow_web_knowledge=payload.allow_web_knowledge,
    )
    sections = template.sections or [{"count": payload.question_count}]
    _reserve_daily_quota(database, user.id, sum(int(section["count"]) for section in sections))
    def worker(run: GenerationRun) -> list[dict]:
        run.log("Retrieving indexed source context.", stage="retrieval")
        questions = generate_paper(payload, database, trace=run.log)
        return [question.model_dump() for question in questions]

    run = generation_runs.create(worker, request_type="paper", user_id=user.id, on_update=lambda item: _persist_run(database, item))
    return run.snapshot()


@router.get("/generate/runs")
def list_generation_runs(
    limit: int = 50,
    user: UserDocument = QuestionUser,
    database: Database = Depends(get_database),
) -> list[dict]:
    bounded_limit = max(1, min(limit, 100))
    persisted = list(database.generation_runs.find({"user_id": user.id}, {"_id": 0}).sort("started_at", -1).limit(bounded_limit))
    live = {run.id: run.snapshot() for run in generation_runs.list(user_id=user.id, limit=bounded_limit)}
    merged = {str(item["id"]): item for item in persisted}
    merged.update(live)
    return sorted(merged.values(), key=lambda item: item.get("started_at", ""), reverse=True)[:bounded_limit]


@router.get("/usage")
def generation_usage(database: Database = Depends(get_database), user: UserDocument = QuestionUser) -> dict:
    day = datetime.now(UTC).date().isoformat()
    usage = database.generation_usage.find_one({"user_id": user.id, "day": day}) or {"count": 0}
    totals = list(database.generation_usage.aggregate([{"$match": {"user_id": user.id}}, {"$group": {"_id": None, "count": {"$sum": "$count"}}}]))
    created = totals[0]["count"] if totals else 0
    return {"daily_limit": DAILY_QUESTION_LIMIT, "used_today": usage["count"], "remaining_today": max(0, DAILY_QUESTION_LIMIT - usage["count"]), "questions_created": created}


@router.get("/generate/runs/{run_id}")
def get_generation_run(
    run_id: str,
    _: UserDocument = QuestionUser,
    database: Database = Depends(get_database),
) -> dict:
    run = generation_runs.get(run_id)
    if run is not None and run.user_id is not None and run.user_id != _.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found.")
    if run is not None:
        return run.snapshot()
    persisted = database.generation_runs.find_one({"_id": run_id, "user_id": _.id}, {"_id": 0})
    if persisted is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found.")
    return persisted


def _configured_provider(settings: Settings, model_override: str | None = None):
    if settings.llm_provider.casefold() == "ollama":
        if not settings.ollama_base_url:
            return None
        try:
            return OllamaProvider(
                settings.ollama_base_url,
                model_override or settings.ollama_model,
                settings.ollama_timeout_seconds,
                settings.ollama_api_key,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ollama provider configuration is invalid: {error}",
            ) from error

    # OpenRouter keys are used in order. We advance only on HTTP 429; NVIDIA
    # is intentionally the final hosted fallback for this deployment.
    openrouter_configs = (
        (settings.openrouter_api_key, settings.openrouter_model, settings.openrouter_app_name, settings.openrouter_timeout_seconds),
        (settings.openrouter2_api_key, settings.openrouter2_model or settings.openrouter_model, settings.openrouter2_app_name or settings.openrouter_app_name, settings.openrouter2_timeout_seconds or settings.openrouter_timeout_seconds),
        (settings.openrouter3_api_key, settings.openrouter3_model or settings.openrouter_model, settings.openrouter3_app_name or settings.openrouter_app_name, settings.openrouter3_timeout_seconds or settings.openrouter_timeout_seconds),
    )
    providers = [
        OpenRouterProvider(api_key, model_override or model, app_name, timeout)
        for api_key, model, app_name, timeout in openrouter_configs
        if api_key
    ]
    if settings.nvidia_api_key:
        providers.append(NvidiaProvider(settings.nvidia_api_key, model_override or settings.nvidia_model, settings.nvidia_base_url, settings.nvidia_timeout_seconds))
    if providers:
        return FailoverProvider(providers)
    if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
        return OpenAICompatibleProvider(settings.openai_api_key, model_override or settings.openai_model)
    return None


def _attach_failover_trace(provider: object, trace) -> None:
    """Record route rotation without exposing API keys in request traces."""
    if not trace:
        return
    if isinstance(provider, FailoverProvider):
        provider.on_failover = lambda message: trace(message, stage="model_failover", level="warning")
        provider.on_route = lambda message: trace(f"Successful model route: {message}.", stage="model_route")
    else:
        provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
        model = getattr(provider, "model", None)
        model_label = f", model={model}" if isinstance(model, str) and model else ""
        trace(f"Configured model route: {provider_name}{model_label}.", stage="model_route")


def _existing_questions(
    database: Database,
    *,
    subject_id: str | None,
    syllabus_id: str | None,
    template_id: str | None,
) -> list[GeneratedQuestion]:
    query: dict[str, object] = {"status": {"$in": ["active", "approved", "draft"]}}
    for key, value in {
        "subject_id": subject_id,
        "syllabus_id": syllabus_id,
        "template_id": template_id,
    }.items():
        if value is not None:
            query[key] = value
    return [
        GeneratedQuestion.model_validate(item)
        for item in database.questions.find(query, {"_id": 0})
        if item.get("question_text")
    ]


@router.post("", response_model=QuestionDocument, status_code=status.HTTP_201_CREATED)
def create_question(
    payload: CreateQuestionRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> QuestionDocument:
    subject_id = payload.subject_id
    if subject_id is None and payload.question.sources:
        source_ids = [source.chunk_id for source in payload.question.sources]
        source_chunk = database.document_chunks.find_one(
            {"_id": {"$in": source_ids}, "metadata.subject_id": {"$exists": True}},
            {"metadata.subject_id": 1},
        )
        if source_chunk:
            subject_id = source_chunk.get("metadata", {}).get("subject_id")
    question = QuestionDocument(
        question_type=payload.question_type,
        pattern=payload.pattern,
        question_text=payload.question.question_text,
        options=[option.model_dump() for option in payload.question.options],
        correct_answer=payload.question.correct_answer,
        expected_answer=payload.question.expected_answer,
        explanation=payload.question.explanation,
        difficulty=payload.question.difficulty,
        bloom_level=payload.question.bloom_level,
        sources=[source.model_dump(exclude_none=True) for source in payload.question.sources],
        template_id=payload.template_id,
        subject_id=subject_id,
        syllabus_id=payload.syllabus_id,
        topic_id=payload.topic_id,
        marks=payload.marks,
        created_by_id=user.id,
    )
    database.questions.insert_one(question.model_dump(by_alias=True))
    return question


@router.post("/{question_id}/feedback")
def rate_question(
    question_id: str,
    payload: FeedbackRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> dict:
    if payload.rating < 3 and not payload.improvement_area:
        raise HTTPException(status_code=422, detail="Choose an improvement area for ratings below 3.")
    document = {
        "question_id": question_id, "user_id": user.id, "rating": payload.rating,
        "improvement_area": payload.improvement_area, "comment": payload.comment,
        "created_at": datetime.now(UTC), "updated_at": datetime.now(UTC),
    }
    database.question_feedback.update_one({"question_id": question_id, "user_id": user.id}, {"$set": document}, upsert=True)
    global_matches = 0
    if payload.improvement_area:
        global_matches = database.question_feedback.count_documents({"rating": {"$lt": 3}, "improvement_area": payload.improvement_area})
        if global_matches >= 10:
            database.generation_feedback_rules.update_one(
                {"improvement_area": payload.improvement_area},
                {"$set": {"improvement_area": payload.improvement_area, "active": True, "evidence_count": global_matches, "updated_at": datetime.now(UTC)}}, upsert=True,
            )
    return {"saved": True, "personalized": payload.rating < 3, "global_rule_active": global_matches >= 10}


@router.get("/review", response_model=list[QuestionDocument])
def review_questions(
    review_status: str = "draft",
    database: Database = Depends(get_database),
    _: UserDocument = QuestionUser,
) -> list[QuestionDocument]:
    return [QuestionDocument.model_validate(item) for item in database.questions.find({"review_status": review_status}).sort("created_at", -1)]


@router.post("/{question_id}/approve", response_model=QuestionDocument)
def approve_question(
    question_id: str,
    database: Database = Depends(get_database),
    _: UserDocument = QuestionUser,
) -> QuestionDocument:
    question = database.questions.find_one({"_id": question_id})
    if not question:
        raise HTTPException(status_code=404, detail="Question not found.")
    database.questions.update_one({"_id": question_id}, {"$set": {"review_status": "approved", "status": "approved"}})
    return QuestionDocument.model_validate(database.questions.find_one({"_id": question_id}))


@router.post("/{question_id}/reject", response_model=QuestionDocument)
def reject_question(
    question_id: str,
    note: str = "",
    database: Database = Depends(get_database),
    _: UserDocument = QuestionUser,
) -> QuestionDocument:
    question = database.questions.find_one({"_id": question_id})
    if not question:
        raise HTTPException(status_code=404, detail="Question not found.")
    database.questions.update_one({"_id": question_id}, {"$set": {"review_status": "rejected", "status": "rejected", "review_note": note}})
    return QuestionDocument.model_validate(database.questions.find_one({"_id": question_id}))


@router.post("/{question_id}/validate", response_model=ValidationResult)
def validate_question(
    question_id: str,
    payload: ValidateRequest,
    database: Database = Depends(get_database),
    _: UserDocument = QuestionUser,
) -> ValidationResult:
    template_data = database.question_templates.find_one({"_id": payload.template_id, "status": "active"})
    if not template_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found.")
    result = QuestionQualityService().validate(
        payload.question,
        template=QuestionTemplateDocument.model_validate(template_data),
        context=payload.context,
        existing_questions=payload.existing_questions,
    )
    return result
