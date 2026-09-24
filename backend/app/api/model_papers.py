from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import ModelPaperDocument, UserDocument
from app.services.model_papers import ModelPaperError, ModelPaperService

router = APIRouter(prefix="/model-papers", tags=["model-papers"])
PaperUser = Depends(require_roles("admin", "faculty"))


class ModelPaperCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    subject_id: str
    question_bank_id: str
    question_count: int = Field(ge=1, le=200)
    question_type_counts: dict[str, int] = Field(default_factory=dict)
    difficulty_counts: dict[str, int] = Field(default_factory=dict)
    bloom_level_counts: dict[str, int] = Field(default_factory=dict)


def paper_error(error: ModelPaperError) -> HTTPException:
    message = str(error)
    code = status.HTTP_404_NOT_FOUND if message == "Model paper not found." else status.HTTP_409_CONFLICT
    return HTTPException(status_code=code, detail=message)


@router.post("", response_model=ModelPaperDocument, status_code=status.HTTP_201_CREATED)
def create_model_paper(
    payload: ModelPaperCreate,
    database: Database = Depends(get_database),
    _: UserDocument = PaperUser,
) -> ModelPaperDocument:
    try:
        return ModelPaperService(database).create(**payload.model_dump())
    except ModelPaperError as error:
        raise paper_error(error) from error


@router.get("/{paper_id}", response_model=ModelPaperDocument)
def get_model_paper(
    paper_id: str,
    database: Database = Depends(get_database),
    _: UserDocument = PaperUser,
) -> ModelPaperDocument:
    paper = database.model_papers.find_one({"_id": paper_id})
    if not paper:
        raise HTTPException(status_code=404, detail="Model paper not found.")
    return ModelPaperDocument.model_validate(paper)


@router.post("/{paper_id}/publish", response_model=ModelPaperDocument)
def publish_model_paper(
    paper_id: str,
    database: Database = Depends(get_database),
    _: UserDocument = PaperUser,
) -> ModelPaperDocument:
    try:
        return ModelPaperService(database).publish(paper_id)
    except ModelPaperError as error:
        raise paper_error(error) from error