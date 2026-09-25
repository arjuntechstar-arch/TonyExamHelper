from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database

from app.api.auth import get_current_user
from app.core.database import get_database
from app.models import CourseDocument, SubjectDocument, SyllabusDocument, SyllabusTopicDocument

router = APIRouter(tags=["academics"])
AcademicUser = Depends(get_current_user)
AcademicReadUser = Depends(get_current_user)


class SubjectCreate(BaseModel):
    code: str = Field(min_length=2, max_length=50)
    name: str = Field(min_length=2, max_length=200)


class CourseCreate(SubjectCreate):
    subject_id: str


class SyllabusCreate(BaseModel):
    course_id: str
    version: str = Field(min_length=1, max_length=50)


class TopicCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    unit_number: int | None = Field(default=None, ge=1)
    parent_id: str | None = None
    learning_objective: str | None = None


@router.post("/subjects", response_model=SubjectDocument, status_code=status.HTTP_201_CREATED)
def create_subject(payload: SubjectCreate, database: Database = Depends(get_database), _: object = AcademicUser) -> SubjectDocument:
    if database.subjects.find_one({"code": payload.code.upper()}):
        raise HTTPException(status_code=409, detail="Subject code already exists.")
    subject = SubjectDocument(code=payload.code.upper(), name=payload.name)
    database.subjects.insert_one(subject.model_dump(by_alias=True))
    return subject


@router.get("/subjects", response_model=list[SubjectDocument])
def list_subjects(database: Database = Depends(get_database), _: object = AcademicReadUser) -> list[SubjectDocument]:
    return [SubjectDocument.model_validate(item) for item in database.subjects.find().sort("code")]


@router.post("/courses", response_model=CourseDocument, status_code=status.HTTP_201_CREATED)
def create_course(payload: CourseCreate, database: Database = Depends(get_database), _: object = AcademicUser) -> CourseDocument:
    if not database.subjects.find_one({"_id": payload.subject_id}):
        raise HTTPException(status_code=404, detail="Subject not found.")
    course = CourseDocument(subject_id=payload.subject_id, code=payload.code.upper(), name=payload.name)
    database.courses.insert_one(course.model_dump(by_alias=True))
    return course


@router.post("/syllabus", response_model=SyllabusDocument, status_code=status.HTTP_201_CREATED)
def create_syllabus(payload: SyllabusCreate, database: Database = Depends(get_database), _: object = AcademicUser) -> SyllabusDocument:
    if not database.courses.find_one({"_id": payload.course_id}):
        raise HTTPException(status_code=404, detail="Course not found.")
    syllabus = SyllabusDocument(course_id=payload.course_id, version=payload.version, status="draft")
    database.syllabi.insert_one(syllabus.model_dump(by_alias=True))
    return syllabus


@router.get("/syllabus/{syllabus_id}")
def get_syllabus(syllabus_id: str, database: Database = Depends(get_database), _: object = AcademicUser) -> dict:
    syllabus = database.syllabi.find_one({"_id": syllabus_id})
    if not syllabus:
        raise HTTPException(status_code=404, detail="Syllabus not found.")
    return {"syllabus": SyllabusDocument.model_validate(syllabus), "topics": [SyllabusTopicDocument.model_validate(item) for item in database.syllabus_topics.find({"syllabus_id": syllabus_id})]}


@router.post("/syllabus/{syllabus_id}/topics", response_model=SyllabusTopicDocument, status_code=status.HTTP_201_CREATED)
def create_topic(syllabus_id: str, payload: TopicCreate, database: Database = Depends(get_database), _: object = AcademicUser) -> SyllabusTopicDocument:
    if not database.syllabi.find_one({"_id": syllabus_id}):
        raise HTTPException(status_code=404, detail="Syllabus not found.")
    topic = SyllabusTopicDocument(syllabus_id=syllabus_id, **payload.model_dump())
    database.syllabus_topics.insert_one(topic.model_dump(by_alias=True))
    return topic
