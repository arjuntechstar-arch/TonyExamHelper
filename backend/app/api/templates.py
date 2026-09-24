from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import QuestionTemplateDocument, UserDocument

router = APIRouter(prefix="/templates", tags=["templates"])
TemplateUser = Depends(require_roles("admin", "faculty"))


class TemplatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    question_type: str = Field(min_length=1, max_length=30)
    pattern: str = Field(min_length=1, max_length=100)
    required_fields: list[str] = Field(min_length=1)
    supported_difficulties: list[str] = Field(min_length=1)
    supported_bloom_levels: list[str] = Field(min_length=1)
    version: str = Field(min_length=1, max_length=30)


class TemplateUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=150)
    question_type: str | None = Field(default=None, min_length=1, max_length=30)
    pattern: str | None = Field(default=None, min_length=1, max_length=100)
    required_fields: list[str] | None = Field(default=None, min_length=1)
    supported_difficulties: list[str] | None = Field(default=None, min_length=1)
    supported_bloom_levels: list[str] | None = Field(default=None, min_length=1)
    version: str | None = Field(default=None, min_length=1, max_length=30)


@router.get("", response_model=list[QuestionTemplateDocument])
def list_templates(
    question_type: str | None = None,
    pattern: str | None = None,
    difficulty: str | None = None,
    bloom_level: str | None = None,
    database: Database = Depends(get_database),
    _: UserDocument = TemplateUser,
) -> list[QuestionTemplateDocument]:
    query: dict = {"status": "active"}
    if question_type:
        query["question_type"] = question_type
    if pattern:
        query["pattern"] = pattern
    if difficulty:
        query["supported_difficulties"] = difficulty
    if bloom_level:
        query["supported_bloom_levels"] = bloom_level
    return [QuestionTemplateDocument.model_validate(item) for item in database.question_templates.find(query).sort("name")]


@router.post("", response_model=QuestionTemplateDocument, status_code=status.HTTP_201_CREATED)
def create_template(
    payload: TemplatePayload,
    database: Database = Depends(get_database),
    _: UserDocument = TemplateUser,
) -> QuestionTemplateDocument:
    template = QuestionTemplateDocument(**payload.model_dump())
    try:
        database.question_templates.insert_one(template.model_dump(by_alias=True))
    except DuplicateKeyError as error:
        raise HTTPException(status_code=409, detail="A template with this name and version already exists.") from error
    return template


@router.put("/{template_id}", response_model=QuestionTemplateDocument)
def update_template(
    template_id: str,
    payload: TemplateUpdate,
    database: Database = Depends(get_database),
    _: UserDocument = TemplateUser,
) -> QuestionTemplateDocument:
    current = database.question_templates.find_one({"_id": template_id, "status": "active"})
    if not current:
        raise HTTPException(status_code=404, detail="Template not found.")
    changes = payload.model_dump(exclude_unset=True)
    if changes:
        try:
            database.question_templates.update_one({"_id": template_id}, {"$set": changes})
        except DuplicateKeyError as error:
            raise HTTPException(status_code=409, detail="A template with this name and version already exists.") from error
    updated = database.question_templates.find_one({"_id": template_id})
    return QuestionTemplateDocument.model_validate(updated)