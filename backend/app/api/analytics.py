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
    _: UserDocument = AnalyticsUser,
) -> dict:
    try:
        return AnalyticsService(database).student_overview(student_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.get("/student/topics")
def get_student_topic_summary(
    student_id: str,
    database: Database = Depends(get_database),
    _: UserDocument = AnalyticsUser,
) -> list[dict]:
    try:
        return AnalyticsService(database).topic_summary(student_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.get("/student/difficulty")
def get_student_difficulty_summary(
    student_id: str,
    database: Database = Depends(get_database),
    _: UserDocument = AnalyticsUser,
) -> list[dict]:
    try:
        return AnalyticsService(database).difficulty_summary(student_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
