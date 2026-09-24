from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import QuestionTemplateDocument, UserDocument
from app.services.generation import GeneratedQuestion, GenerationError, GenerationService
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
        return GenerationService().generate(
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