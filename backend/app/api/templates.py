from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import QuestionTemplateDocument, UserDocument

router = APIRouter(prefix="/templates", tags=["templates"])
TemplateUser = Depends(require_roles("admin", "faculty"))


class PatternSection(BaseModel):
    question_type: str = Field(min_length=1, max_length=30)
    pattern: str = Field(min_length=1, max_length=100)
    count: int = Field(ge=1, le=100)
    marks: int = Field(ge=1, le=100)


class TemplatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    question_type: str = Field(min_length=1, max_length=30)
    pattern: str = Field(min_length=1, max_length=100)
    required_fields: list[str] = Field(min_length=1)
    supported_difficulties: list[str] = Field(min_length=1)
    supported_bloom_levels: list[str] = Field(min_length=1)
    version: str = Field(min_length=1, max_length=30)
    marks: int = Field(default=1, ge=1, le=100)
    sections: list["PatternSection"] = Field(default_factory=list)
    total_marks: int = Field(default=1, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_blueprint(self) -> "TemplatePayload":
        if self.sections:
            expected = sum(section.count * section.marks for section in self.sections)
            if expected != self.total_marks:
                raise ValueError("total_marks must equal the sum of section count × marks.")
        return self


class TemplateUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=150)
    question_type: str | None = Field(default=None, min_length=1, max_length=30)
    pattern: str | None = Field(default=None, min_length=1, max_length=100)
    required_fields: list[str] | None = Field(default=None, min_length=1)
    supported_difficulties: list[str] | None = Field(default=None, min_length=1)
    supported_bloom_levels: list[str] | None = Field(default=None, min_length=1)
    version: str | None = Field(default=None, min_length=1, max_length=30)
    marks: int | None = Field(default=None, ge=1, le=100)
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
    values = payload.model_dump()
    if not values["sections"]:
        values["sections"] = [{
            "question_type": values["question_type"],
            "pattern": values["pattern"],
            "count": 1,
            "marks": values["marks"],
        }]
        values["total_marks"] = values["marks"]
    template = QuestionTemplateDocument(**values)
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