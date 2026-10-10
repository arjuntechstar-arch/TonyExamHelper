from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from pymongo.database import Database

from app.api.auth import get_current_user, require_roles
from app.core.config import get_settings
from app.core.database import get_database
from app.models import UserDocument
from app.services.retrieval import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    RetrievalService,
    VectorSearchError,
    embedding_provider_from_settings,
)
from app.services.retrieval_evaluation import evaluate_retrieval_benchmark

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
RetrievalUser = Depends(get_current_user)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=50)
    subject_id: str | None = None
    course_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None


class RetrievalBenchmarkCase(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    relevant_chunk_ids: list[str] = Field(min_length=1, max_length=50)


class RetrievalBenchmarkRequest(BaseModel):
    cases: list[RetrievalBenchmarkCase] = Field(min_length=1, max_length=20)
    k: int = Field(default=5, ge=1, le=20)
    subject_id: str | None = None
    study_material_id: str | None = None
    course_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None

    @model_validator(mode="after")
    def require_scope_filter(self) -> "RetrievalBenchmarkRequest":
        if not any((
            self.subject_id,
            self.study_material_id,
            self.course_id,
            self.syllabus_id,
            self.topic_id,
        )):
            raise ValueError("Provide at least one scope filter to limit benchmark retrieval.")
        return self


def get_retrieval_service(database: Database = Depends(get_database)) -> RetrievalService:
    settings = get_settings()
    try:
        return RetrievalService(
            database,
            embedding_provider=embedding_provider_from_settings(settings),
            vector_index_name=settings.retrieval_vector_index_name,
        )
    except EmbeddingConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error


def _vector_search_unavailable(error: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(error),
    )


@router.post("/vector-index")
def setup_vector_index(
    service: RetrievalService = Depends(get_retrieval_service),
    _: UserDocument = Depends(require_roles("admin", "faculty")),
) -> dict[str, str | bool]:
    try:
        return service.setup_vector_index()
    except (EmbeddingConfigurationError, VectorSearchError) as error:
        raise _vector_search_unavailable(error) from error


@router.get("/vector-index")
def vector_index_status(
    service: RetrievalService = Depends(get_retrieval_service),
    _: UserDocument = Depends(require_roles("admin", "faculty")),
) -> dict[str, str | bool]:
    try:
        return service.vector_index_status()
    except (EmbeddingConfigurationError, VectorSearchError) as error:
        raise _vector_search_unavailable(error) from error


@router.post("/materials/{material_id}/index")
def index_material(
    material_id: str,
    service: RetrievalService = Depends(get_retrieval_service),
    _: UserDocument = RetrievalUser,
) -> dict[str, int | str]:
    try:
        count = service.index_material(material_id)
    except (EmbeddingConfigurationError, EmbeddingProviderError, VectorSearchError) as error:
        raise _vector_search_unavailable(error) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return {"material_id": material_id, "indexed_chunks": count, "embedding_model": service.embedding_provider.model_name}


@router.post("/search")
def search(
    payload: RetrievalRequest,
    service: RetrievalService = Depends(get_retrieval_service),
    _: UserDocument = RetrievalUser,
) -> dict[str, Any]:
    try:
        results = service.retrieve(
            payload.query,
            top_k=payload.top_k,
            subject_id=payload.subject_id,
            course_id=payload.course_id,
            syllabus_id=payload.syllabus_id,
            topic_id=payload.topic_id,
        )
    except (EmbeddingConfigurationError, EmbeddingProviderError, VectorSearchError) as error:
        raise _vector_search_unavailable(error) from error
    return {
        "strategy": "atlas_vector_hybrid_bm25_keyword_mmr" if service.semantic_enabled else "hybrid_bm25_keyword_mmr",
        "results": [
            {
                "score": result["score"],
                "semantic_score": result.get("semantic_score"),
                "lexical_score": result.get("lexical_score"),
                "phrase_score": result.get("phrase_score"),
                "hybrid_score": result.get("hybrid_score"),
                "chunk": result["chunk"],
            }
            for result in results
        ]
    }


@router.post("/evaluate")
def evaluate_retrieval(
    payload: RetrievalBenchmarkRequest,
    service: RetrievalService = Depends(get_retrieval_service),
    _: UserDocument = Depends(require_roles("admin", "faculty")),
) -> dict[str, Any]:
    try:
        benchmark = evaluate_retrieval_benchmark(
            [case.model_dump() for case in payload.cases],
            lambda query, k: service.retrieve(
                query,
                top_k=k,
                subject_id=payload.subject_id,
                study_material_id=payload.study_material_id,
                course_id=payload.course_id,
                syllabus_id=payload.syllabus_id,
                topic_id=payload.topic_id,
            ),
            k=payload.k,
        )
    except (EmbeddingConfigurationError, EmbeddingProviderError, VectorSearchError) as error:
        raise _vector_search_unavailable(error) from error
    return {
        "strategy": "atlas_vector_hybrid_bm25_keyword_mmr" if service.semantic_enabled else "hybrid_bm25_keyword_mmr",
        "scope": {
            key: value
            for key, value in {
                "subject_id": payload.subject_id,
                "study_material_id": payload.study_material_id,
                "course_id": payload.course_id,
                "syllabus_id": payload.syllabus_id,
                "topic_id": payload.topic_id,
            }.items()
            if value is not None
        },
        **benchmark,
    }
