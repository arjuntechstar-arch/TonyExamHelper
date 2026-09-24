from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import PracticeTestDocument, UserDocument
from app.services.practice import PracticeError, PracticeService

router = APIRouter(prefix="/practice", tags=["practice"])
PracticeUser = Depends(require_roles("student", "admin", "faculty"))


class PracticeStartRequest(BaseModel):
    subject_id: str
    topic_id: str | None = None
    question_count: int = Field(default=5, ge=1, le=50)
    difficulty: str | None = Field(default=None, max_length=50)


class PracticeAnswerSubmission(BaseModel):
    answers: dict[str, str] = Field(default_factory=dict)


@router.post("/start", response_model=PracticeTestDocument, status_code=status.HTTP_201_CREATED)
def start_practice(
    payload: PracticeStartRequest,
    database: Database = Depends(get_database),
    _: UserDocument = PracticeUser,
) -> PracticeTestDocument:
    try:
        return PracticeService(database).start_test(
            student_id=getattr(_, "id", "student"),
            subject_id=payload.subject_id,
            question_count=payload.question_count,
            difficulty=payload.difficulty,
            topic_id=payload.topic_id,
        )
    except PracticeError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{practice_id}/submit")
def submit_practice_answers(
    practice_id: str,
    payload: PracticeAnswerSubmission,
    database: Database = Depends(get_database),
    user: UserDocument = PracticeUser,
) -> dict:
    try:
        result = PracticeService(database).submit_answers(practice_id, payload.answers)
    except PracticeError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return result


@router.get("/{practice_id}")
def get_practice(
    practice_id: str,
    database: Database = Depends(get_database),
    user: UserDocument = PracticeUser,
) -> dict:
    try:
        return PracticeService(database).get_test(
            practice_id,
            requester_id=user.id,
            requester_roles=user.roles,
        )
    except PracticeError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.get("/{practice_id}/result")
def get_practice_result(
    practice_id: str,
    database: Database = Depends(get_database),
    _: UserDocument = PracticeUser,
) -> dict:
    try:
        return PracticeService(database).get_result(practice_id)
    except PracticeError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
