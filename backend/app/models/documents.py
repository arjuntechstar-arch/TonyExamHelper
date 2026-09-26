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
    bio: str | None = None
    institution: str | None = None


class QuestionFeedbackDocument(AuditDocument):
    question_id: str
    user_id: str
    rating: int = Field(ge=1, le=5)
    improvement_area: str | None = None
    comment: str | None = None


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
    subject_id: str | None = None
    course_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None
    filename: str
    storage_key: str
    content_type: str
    size_bytes: int = Field(ge=0)
    status: str = "uploaded"


class DocumentChunkDocument(AuditDocument):
    study_material_id: str
    chunk_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    content: str
    metadata: dict[str, str | int] = Field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_model: str | None = None


class QuestionTemplateDocument(AuditDocument):
    name: str = Field(min_length=1, max_length=150)
    question_type: str = Field(min_length=1, max_length=30)
    pattern: str = Field(min_length=1, max_length=100)
    required_fields: list[str] = Field(min_length=1)
    supported_difficulties: list[str] = Field(min_length=1)
    supported_bloom_levels: list[str] = Field(min_length=1)
    version: str = Field(min_length=1, max_length=30)
    marks: int = Field(default=1, ge=1, le=100)
    sections: list[dict[str, object]] = Field(default_factory=list)
    total_marks: int = Field(default=1, ge=1, le=1_000)


class QuestionDocument(AuditDocument):
    question_type: str
    pattern: str
    question_text: str
    options: list[dict[str, str]] = Field(default_factory=list)
    correct_answer: str | None = None
    explanation: str
    difficulty: str
    bloom_level: str
    sources: list[dict[str, str | int]] = Field(default_factory=list)
    template_id: str | None = None
    subject_id: str | None = None
    syllabus_id: str | None = None
    topic_id: str | None = None
    marks: int = Field(default=1, ge=1, le=100)
    review_status: str = "draft"
    review_note: str | None = None


class QuestionBankDocument(AuditDocument):
    name: str = Field(min_length=1, max_length=150)
    description: str | None = None
    subject_id: str
    question_ids: list[str] = Field(default_factory=list)
    approval_status: str = "draft"


class ModelPaperDocument(AuditDocument):
    name: str = Field(min_length=1, max_length=150)
    subject_id: str
    question_bank_id: str
    question_ids: list[str] = Field(min_length=1)
    question_count: int = Field(ge=1)
    blueprint: dict[str, dict[str, int]] = Field(default_factory=dict)
    publication_status: str = "draft"


class PracticeTestDocument(AuditDocument):
    student_id: str
    subject_id: str
    topic_id: str | None = None
    question_ids: list[str] = Field(default_factory=list)
    question_count: int = Field(ge=1)
    duration_minutes: int = Field(default=30, ge=1)
    difficulty: str | None = None
    status: str = "started"
    score: float = 0.0
    correct_count: int = 0
    total_questions: int = 0
    percentage: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None


class StudentAnswerDocument(AuditDocument):
    practice_test_id: str
    question_id: str
    selected_answer: str | None = None
    is_correct: bool = False
    score: int = 0
    explanation: str | None = None
    submitted_at: datetime | None = None
