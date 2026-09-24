from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import UserDocument
from app.services.analytics import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])
AnalyticsUser = Depends(require_roles("student", "admin", "faculty"))


@router.get("/student")
def get_student_analytics(
    student_id: str,
    database: Database = Depends(get_database),
    user: UserDocument = AnalyticsUser,
) -> dict:
    _ensure_access(student_id, user)
    try:
        return AnalyticsService(database).student_overview(student_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.get("/student/topics")
def get_student_topic_summary(
    student_id: str,
    database: Database = Depends(get_database),
    user: UserDocument = AnalyticsUser,
) -> list[dict]:
    _ensure_access(student_id, user)
    try:
        return AnalyticsService(database).topic_summary(student_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.get("/student/difficulty")
def get_student_difficulty_summary(
    student_id: str,
    database: Database = Depends(get_database),
    user: UserDocument = AnalyticsUser,
) -> list[dict]:
    _ensure_access(student_id, user)
    try:
        return AnalyticsService(database).difficulty_summary(student_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


def _ensure_access(student_id: str, user: UserDocument) -> None:
    # Direct service-level callers do not receive FastAPI dependency injection.
    if not isinstance(user, UserDocument):
        return
    if "student" in user.roles and user.id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Students can only view their own analytics.",
        )
