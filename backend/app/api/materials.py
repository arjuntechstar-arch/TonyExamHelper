from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pymongo.database import Database

from app.api.auth import get_current_user
from app.core.config import get_settings
from app.core.database import get_database
from app.models import StudyMaterialDocument, UserDocument
from app.services.document_processing import DocumentProcessingError, DocumentProcessingService

router = APIRouter(prefix="/materials", tags=["materials"])
MaterialUser = Depends(get_current_user)


def get_document_service(database: Database = Depends(get_database)) -> DocumentProcessingService:
    settings = get_settings()
    return DocumentProcessingService(database, settings.material_storage_path, settings.max_upload_bytes)


def processing_error(error: DocumentProcessingError) -> HTTPException:
    code = status.HTTP_404_NOT_FOUND if str(error) in {"Material not found.", "Subject not found."} else status.HTTP_400_BAD_REQUEST
    return HTTPException(status_code=code, detail=str(error))


@router.post("/upload", response_model=StudyMaterialDocument, status_code=status.HTTP_201_CREATED)
async def upload_material(
    file: Annotated[UploadFile, File()],
    subject_id: Annotated[str | None, Form()] = None,
    course_id: Annotated[str | None, Form()] = None,
    syllabus_id: Annotated[str | None, Form()] = None,
    topic_id: Annotated[str | None, Form()] = None,
    service: DocumentProcessingService = Depends(get_document_service),
    _: UserDocument = MaterialUser,
) -> StudyMaterialDocument:
    try:
        content = await file.read(service.max_upload_bytes + 1)
        return service.upload(
            subject_id=subject_id,
            course_id=course_id,
            syllabus_id=syllabus_id,
            topic_id=topic_id,
            filename=file.filename or "",
            content_type=file.content_type or "application/octet-stream",
            content=content,
        )
    except DocumentProcessingError as error:
        raise processing_error(error) from error


@router.post("/{material_id}/process")
def process_material(
    material_id: str,
    service: DocumentProcessingService = Depends(get_document_service),
    _: UserDocument = MaterialUser,
) -> dict:
    try:
        material, chunk_count = service.process(material_id)
    except DocumentProcessingError as error:
        raise processing_error(error) from error
    return {"material": material, "chunk_count": chunk_count}


@router.get("/{material_id}/status")
def material_status(
    material_id: str,
    service: DocumentProcessingService = Depends(get_document_service),
    _: UserDocument = MaterialUser,
) -> dict:
    try:
        return service.get_status(material_id)
    except DocumentProcessingError as error:
        raise processing_error(error) from error
