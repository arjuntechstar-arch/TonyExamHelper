from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database

from app.api.auth import require_roles
from app.core.database import get_database
from app.core.config import get_settings
from app.models import QuestionDocument, QuestionTemplateDocument, UserDocument
from app.services.generation import GeneratedQuestion, GenerationError, GenerationService, OpenAICompatibleProvider, OpenRouterProvider
from app.services.quality import QuestionQualityService, ValidationResult
from app.services.retrieval import RetrievalService

router = APIRouter(prefix="/questions", tags=["questions"])
QuestionUser = Depends(require_roles("admin", "faculty"))


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


class BatchGenerateRequest(BaseModel):
    requests: list[GenerateRequest] = Field(min_length=1, max_length=20)


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


def generate_questions(payload: GenerateRequest, database: Database) -> list[GeneratedQuestion]:
    template_data = database.question_templates.find_one({"_id": payload.template_id, "status": "active"})
    if not template_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found.")
    template = QuestionTemplateDocument.model_validate(template_data)
    chunks = RetrievalService(database).retrieve(
        payload.query,
        top_k=payload.top_k,
        subject_id=payload.subject_id,
        course_id=payload.course_id,
        syllabus_id=payload.syllabus_id,
        topic_id=payload.topic_id,
    )
    try:
        settings = get_settings()
        provider = None
        if settings.llm_provider.lower() == "openrouter" and settings.openrouter_api_key:
            provider = OpenRouterProvider(settings.openrouter_api_key, settings.openrouter_model, settings.openrouter_app_name, settings.openrouter_timeout_seconds)
        elif settings.llm_provider.lower() == "openai" and settings.openai_api_key:
            provider = OpenAICompatibleProvider(settings.openai_api_key, settings.openai_model)
        return GenerationService(provider=provider).generate(
            template=template,
            chunks=chunks,
            difficulty=payload.difficulty,
            bloom_level=payload.bloom_level,
            candidate_count=payload.candidate_count,
        )
    except GenerationError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


@router.post("/generate", response_model=list[GeneratedQuestion])
def generate(
    payload: GenerateRequest,
    database: Database = Depends(get_database),
    _: UserDocument = QuestionUser,
) -> list[GeneratedQuestion]:
    return generate_questions(payload, database)


@router.post("/generate/batch", response_model=list[list[GeneratedQuestion]])
def generate_batch(
    payload: BatchGenerateRequest,
    database: Database = Depends(get_database),
    _: UserDocument = QuestionUser,
) -> list[list[GeneratedQuestion]]:
    return [generate_questions(request, database) for request in payload.requests]


@router.post("", response_model=QuestionDocument, status_code=status.HTTP_201_CREATED)
def create_question(
    payload: CreateQuestionRequest,
    database: Database = Depends(get_database),
    _: UserDocument = QuestionUser,
) -> QuestionDocument:
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
        subject_id=payload.subject_id,
        syllabus_id=payload.syllabus_id,
        topic_id=payload.topic_id,
    )
    database.questions.insert_one(question.model_dump(by_alias=True))
    return question


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