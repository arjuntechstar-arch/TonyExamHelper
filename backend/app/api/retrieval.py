from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from pymongo.database import Database

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import UserDocument
from app.services.retrieval import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
RetrievalUser = Depends(require_roles("admin", "faculty"))


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=50)
    subject_id: str | None = None
    course_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None


def get_retrieval_service(database: Database = Depends(get_database)) -> RetrievalService:
    return RetrievalService(database)


@router.post("/materials/{material_id}/index")
def index_material(
    material_id: str,
    service: RetrievalService = Depends(get_retrieval_service),
    _: UserDocument = RetrievalUser,
) -> dict[str, int | str]:
    try:
        count = service.index_material(material_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return {"material_id": material_id, "indexed_chunks": count, "embedding_model": service.embedding_provider.model_name}


@router.post("/search")
def search(
    payload: RetrievalRequest,
    service: RetrievalService = Depends(get_retrieval_service),
    _: UserDocument = RetrievalUser,
) -> dict[str, list[dict]]:
    results = service.retrieve(
        payload.query,
        top_k=payload.top_k,
        subject_id=payload.subject_id,
        course_id=payload.course_id,
        syllabus_id=payload.syllabus_id,
        topic_id=payload.topic_id,
    )
    return {
        "results": [
            {
                "score": result["score"],
                "chunk": result["chunk"],
            }
            for result in results
        ]
    }