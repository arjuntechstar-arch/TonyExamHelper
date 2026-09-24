import mongomock
import pytest

from app.services.document_processing import DocumentProcessingError, DocumentProcessingService, chunk_pages


@pytest.fixture
def document_service(tmp_path):
    database = mongomock.MongoClient().test
    database.subjects.insert_one({"_id": "subject-1", "code": "CS101"})
    return DocumentProcessingService(database, tmp_path, max_upload_bytes=1000)


def test_upload_processes_text_and_preserves_chunk_metadata(document_service: DocumentProcessingService) -> None:
    material = document_service.upload(
        subject_id="subject-1",
        course_id="course-1",
        syllabus_id="syllabus-1",
        topic_id="topic-1",
        filename="notes.txt",
        content_type="text/plain",
        content=b"First line\nSecond line",
    )

    processed, chunk_count = document_service.process(material.id)
    status = document_service.get_status(material.id)
    chunk = document_service.database.document_chunks.find_one({"study_material_id": material.id})

    assert processed.status == "processed"
    assert chunk_count == 1
    assert status["chunk_count"] == 1
    assert chunk["page_number"] == 1
    assert chunk["metadata"] == {
        "source_file": "notes.txt",
        "subject_id": "subject-1",
        "course_id": "course-1",
        "syllabus_id": "syllabus-1",
        "topic_id": "topic-1",
    }
    assert chunk["content"] == "First line Second line"


def test_upload_rejects_mismatched_content_type(document_service: DocumentProcessingService) -> None:
    with pytest.raises(DocumentProcessingError, match="does not match"):
        document_service.upload(
            subject_id="subject-1",
            filename="notes.pdf",
            content_type="text/plain",
            content=b"not a pdf",
        )


def test_upload_resolves_subject_code_to_subject_id(document_service: DocumentProcessingService) -> None:
    material = document_service.upload(
        subject_id="cs101",
        filename="notes.txt",
        content_type="text/plain",
        content=b"notes",
    )

    assert material.subject_id == "subject-1"


def test_upload_rejects_files_over_configured_limit(document_service: DocumentProcessingService) -> None:
    with pytest.raises(DocumentProcessingError, match="too large"):
        document_service.upload(
            subject_id="subject-1",
            filename="notes.txt",
            content_type="text/plain",
            content=b"x" * 1001,
        )


def test_chunk_pages_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages([(1, "content")], chunk_size=100, overlap=100)


def test_status_reports_indexed_chunks_and_embedding_model(document_service: DocumentProcessingService) -> None:
    material = document_service.upload(
        subject_id="subject-1",
        filename="notes.txt",
        content_type="text/plain",
        content=b"notes",
    )
    document_service.process(material.id)
    document_service.database.document_chunks.update_many(
        {"study_material_id": material.id},
        {"$set": {"embedding": [1.0, 0.0], "embedding_model": "hash-embedding-v1"}},
    )

    result = document_service.get_status(material.id)

    assert result["chunk_count"] == 1
    assert result["indexed_chunk_count"] == 1
    assert result["embedding_model"] == "hash-embedding-v1"