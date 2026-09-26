import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.api.auth import get_current_user
from app.core.database import get_database
from app.core.config import Settings, get_settings
from app.models import QuestionDocument, QuestionTemplateDocument, UserDocument
from app.services.generation import GeneratedQuestion, GenerationError, NvidiaProvider, OpenAICompatibleProvider, OpenRouterProvider
from app.services.question_agents import QuestionGenerationGraph
from app.services.generation_runs import generation_runs, GenerationRun
from app.services.quality import QuestionQualityService, ValidationResult
from app.services.retrieval import RetrievalService

router = APIRouter(prefix="/questions", tags=["questions"])
QuestionUser = Depends(get_current_user)
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    template_id: str
    query: str = Field(min_length=1, max_length=2_000)
    difficulty: str = Field(min_length=1, max_length=50)
    bloom_level: str = Field(min_length=1, max_length=50)
    candidate_count: int = Field(default=1, ge=1, le=20)
    top_k: int = Field(default=5, ge=1, le=50)
    subject_id: str | None = None
    course_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None
    allow_web_knowledge: bool = False


class BatchGenerateRequest(BaseModel):
    requests: list[GenerateRequest] = Field(min_length=1, max_length=20)


class PaperGenerateRequest(BaseModel):
    template_id: str
    difficulty: str = Field(default="Medium", min_length=1, max_length=50)
    bloom_level: str = Field(default="Understand", min_length=1, max_length=50)
    top_k: int = Field(default=5, ge=1, le=50)
    subject_id: str | None = None
    material_id: str | None = None
    course_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None


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


def generate_questions(payload: GenerateRequest, database: Database, user_id: str | None = None) -> list[GeneratedQuestion]:
    template_data = database.question_templates.find_one({"_id": payload.template_id, "status": "active"})
    if not template_data:
        if payload.allow_web_knowledge:
            template_data = database.question_templates.find_one({"status": "active"})
            if not template_data:
                template = QuestionTemplateDocument(
                    id=payload.template_id or "default-web-template",
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
            else:
                template = QuestionTemplateDocument.model_validate(template_data)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found.")
    else:
        template = QuestionTemplateDocument.model_validate(template_data)
    chunks = RetrievalService(database).retrieve(
        payload.query,
        top_k=payload.top_k,
        subject_id=payload.subject_id,
        course_id=payload.course_id,
        syllabus_id=payload.syllabus_id,
        topic_id=payload.topic_id,
    )
    if not chunks:
        if payload.allow_web_knowledge:
            from app.models import DocumentChunkDocument
            synthetic_chunk = DocumentChunkDocument(
                id="web-knowledge-chunk-1",
                study_material_id="open-web-knowledge",
                chunk_index=0,
                page_number=1,
                content=(
                    f"Comprehensive open-domain web reference on {payload.query}. "
                    f"Covers fundamental concepts, theoretical models, practical applications, analysis, evaluation, "
                    f"and core engineering principles regarding {payload.query}."
                ),
                metadata={"source": "open_domain_web", "topic": payload.query},
            )
            chunks = [{"chunk": synthetic_chunk, "score": 1.0}]
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No indexed source chunks are available for this query. Process and index study material first.",
            )
    try:
        settings = get_settings()
        provider = _configured_provider(settings)
        existing_questions = _existing_questions(
            database,
            subject_id=payload.subject_id,
            syllabus_id=payload.syllabus_id,
            template_id=payload.template_id,
        )
        try:
            return QuestionGenerationGraph(provider=provider).generate(
                template=template,
                chunks=chunks,
                difficulty=payload.difficulty,
                bloom_level=payload.bloom_level,
                candidate_count=payload.candidate_count,
                existing_questions=existing_questions, guidance=_personal_guidance(database, user_id) if user_id else None,
            )
        except GenerationError:
            if provider is None:
                raise
            logger.warning("Configured LLM provider failed; using deterministic fallback.", exc_info=True)
            return QuestionGenerationGraph().generate(
                template=template,
                chunks=chunks,
                difficulty=payload.difficulty,
                bloom_level=payload.bloom_level,
                candidate_count=payload.candidate_count,
                existing_questions=existing_questions, guidance=_personal_guidance(database, user_id) if user_id else None,
            )
    except GenerationError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


@router.post("/generate", response_model=list[GeneratedQuestion])
def generate(
    payload: GenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> list[GeneratedQuestion]:
    _reserve_daily_quota(database, user.id, payload.candidate_count)
    return generate_questions(payload, database, user.id)


@router.post("/generate/batch", response_model=list[list[GeneratedQuestion]])
def generate_batch(
    payload: BatchGenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> list[list[GeneratedQuestion]]:
    _reserve_daily_quota(database, user.id, sum(request.candidate_count for request in payload.requests))
    return [generate_questions(request, database, user.id) for request in payload.requests]


@router.post("/generate/paper", response_model=list[GeneratedQuestion])
def generate_paper(
    payload: PaperGenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
    trace=None,
) -> list[GeneratedQuestion]:
    template_data = database.question_templates.find_one({"_id": payload.template_id, "status": "active"})
    if not template_data:
        raise HTTPException(status_code=404, detail="Template not found.")
    template = QuestionTemplateDocument.model_validate(template_data)
    try:
        chunks = RetrievalService(database).retrieve_all(
            subject_id=payload.subject_id,
            study_material_id=payload.material_id,
            course_id=payload.course_id,
            syllabus_id=payload.syllabus_id,
            topic_id=payload.topic_id,
        )
        if trace:
            trace(f"Retrieved {len(chunks)} indexed source chunks.", stage="retrieval")
        if not chunks:
            raise HTTPException(status_code=422, detail="No indexed source chunks are available for this material. Upload the material and try again.")
        generated: list[GeneratedQuestion] = []
        existing_questions = _existing_questions(
            database,
            subject_id=payload.subject_id,
            syllabus_id=payload.syllabus_id,
            template_id=payload.template_id,
        )
        settings = get_settings()
        provider = _configured_provider(settings)
        sections = template.sections or [{
            "question_type": template.question_type,
            "pattern": template.pattern,
            "count": 1,
            "marks": template.marks,
        }]
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
                        QuestionGenerationGraph(provider=provider).generate(
                            template=section_template,
                            chunks=chunks,
                            difficulty=difficulty,
                            bloom_level=bloom_level,
                            candidate_count=candidate_count,
                            existing_questions=existing_questions + generated,
                            trace=trace, guidance=_personal_guidance(database, getattr(user, "id", "")) if getattr(user, "id", None) else None,
                        )
                    )
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
                            existing_questions=existing_questions + generated,
                            trace=trace, guidance=_personal_guidance(database, getattr(user, "id", "")) if getattr(user, "id", None) else None,
                        )
                    )
        return generated
    except (GenerationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/generate/paper/start")
def start_generate_paper(
    payload: PaperGenerateRequest,
    database: Database = Depends(get_database),
    user: UserDocument = QuestionUser,
) -> dict:
    template = database.question_templates.find_one({"_id": payload.template_id, "status": "active"})
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")
    sections = template.get("sections") or [{"count": 1}]
    _reserve_daily_quota(database, user.id, sum(int(section["count"]) for section in sections))
    def worker(run: GenerationRun) -> list[dict]:
        run.log("Retrieving indexed source context.", stage="retrieval")
        questions = generate_paper(payload, database, trace=run.log)
        return [question.model_dump() for question in questions]

    run = generation_runs.create(worker)
    return run.snapshot()


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
) -> dict:
    run = generation_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found.")
    return run.snapshot()


def _configured_provider(settings: Settings):
    if settings.llm_provider.lower() == "openrouter" and settings.openrouter_api_key:
        return OpenRouterProvider(
            settings.openrouter_api_key,
            settings.openrouter_model,
            settings.openrouter_app_name,
            settings.openrouter_timeout_seconds,
        )
    if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
        return OpenAICompatibleProvider(settings.openai_api_key, settings.openai_model)
    if settings.llm_provider.lower() in {"nvidia", "nvidia-nim", "kimi-k3"} and settings.nvidia_api_key:
        return NvidiaProvider(
            settings.nvidia_api_key,
            settings.nvidia_model,
            settings.nvidia_base_url,
            settings.nvidia_timeout_seconds,
        )
    return None


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
        explanation=payload.question.explanation,
        difficulty=payload.question.difficulty,
        bloom_level=payload.question.bloom_level,
        sources=[source.model_dump() for source in payload.question.sources],
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
