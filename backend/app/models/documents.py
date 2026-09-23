from datetime import UTC, datetime
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field


class AuditDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str = Field(default_factory=lambda: str(uuid4()), alias="_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by_id: str | None = None
    updated_by_id: str | None = None
    status: str = "active"


class SubjectDocument(AuditDocument):
    code: str
    name: str


class UserDocument(AuditDocument):
    email: str
    display_name: str
    password_hash: str
    roles: list[str] = Field(default_factory=lambda: ["student"])
    is_active: bool = True


class CourseDocument(AuditDocument):
    subject_id: str
    code: str
    name: str


class SyllabusDocument(AuditDocument):
    course_id: str
    version: str


class SyllabusTopicDocument(AuditDocument):
    syllabus_id: str
    parent_id: str | None = None
    unit_number: int | None = Field(default=None, ge=1)
    title: str
    learning_objective: str | None = None


class StudyMaterialDocument(AuditDocument):
    subject_id: str
    filename: str
    storage_key: str
    content_type: str


class QuestionDocument(AuditDocument):
    question_type: str
    question_text: str
    difficulty: str
    bloom_level: str


class PracticeTestDocument(AuditDocument):
    student_id: str
    subject_id: str
    status: str = "started"
