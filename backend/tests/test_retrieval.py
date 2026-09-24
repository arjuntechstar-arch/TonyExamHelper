import mongomock
import pytest

from app.models import DocumentChunkDocument, StudyMaterialDocument
from app.services.retrieval import HashEmbeddingProvider, RetrievalService, cosine_similarity


@pytest.fixture
def retrieval_service(tmp_path) -> RetrievalService:
    database = mongomock.MongoClient().test
    material = StudyMaterialDocument(
        subject_id="subject-1",
        filename="algorithms.txt",
        storage_key="algorithms.txt",
        content_type="text/plain",
        size_bytes=10,
        status="processed",
    )
    database.study_materials.insert_one(material.model_dump(by_alias=True))
    chunks = [
        DocumentChunkDocument(
            study_material_id=material.id,
            chunk_index=0,
            page_number=1,
            content="Binary trees have nodes and child pointers.",
            metadata={"source_file": "algorithms.txt", "subject_id": "subject-1"},
        ),
        DocumentChunkDocument(
            study_material_id=material.id,
            chunk_index=1,
            page_number=2,
            content="Relational databases organize data into tables.",
            metadata={"source_file": "algorithms.txt", "subject_id": "subject-1"},
        ),
        DocumentChunkDocument(
            study_material_id=material.id,
            chunk_index=2,
            page_number=3,
            content="Binary trees are also used in another subject.",
            metadata={"source_file": "other.txt", "subject_id": "subject-2"},
        ),
    ]
    database.document_chunks.insert_many([chunk.model_dump(by_alias=True) for chunk in chunks])
    return RetrievalService(database)


def test_hash_embeddings_are_normalized_and_deterministic() -> None:
    provider = HashEmbeddingProvider(dimensions=32)

    first = provider.embed("Binary trees")
    second = provider.embed("Binary trees")

    assert first == second
    assert cosine_similarity(first, first) == pytest.approx(1.0)


def test_indexing_persists_embeddings_and_retrieval_ranks_matching_content(retrieval_service: RetrievalService) -> None:
    count = retrieval_service.index_material(next(retrieval_service.database.study_materials.find()) ["_id"])

    results = retrieval_service.retrieve("binary trees", top_k=2, subject_id="subject-1")

    assert count == 3
    assert results[0]["chunk"].content.startswith("Binary trees")
    assert len(results) == 2
    assert all(result["chunk"].embedding_model == "hash-embedding-v1" for result in results)


def test_retrieval_filters_out_chunks_from_other_subjects(retrieval_service: RetrievalService) -> None:
    material_id = next(retrieval_service.database.study_materials.find())["_id"]
    retrieval_service.index_material(material_id)

    results = retrieval_service.retrieve("binary trees", top_k=10, subject_id="subject-1")

    assert len(results) == 2
    assert all(result["chunk"].metadata["subject_id"] == "subject-1" for result in results)