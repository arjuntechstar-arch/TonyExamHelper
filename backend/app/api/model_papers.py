from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from pymongo.database import Database

from app.api.auth import get_current_user
from app.core.database import get_database
from app.models import ModelPaperDocument, QuestionDocument, UserDocument
from app.services.generation import GeneratedQuestion
from app.services.model_papers import ModelPaperError, ModelPaperService

router = APIRouter(prefix="/model-papers", tags=["model-papers"])
PaperUser = Depends(get_current_user)


class ModelPaperCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    subject_id: str
    question_bank_id: str
    question_count: int = Field(ge=1, le=200)
    question_type_counts: dict[str, int] = Field(default_factory=dict)
    difficulty_counts: dict[str, int] = Field(default_factory=dict)
    bloom_level_counts: dict[str, int] = Field(default_factory=dict)


class GeneratedPaperCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    subject_id: str = Field(min_length=1)
    template_id: str | None = None
    questions: list[GeneratedQuestion] = Field(min_length=1, max_length=200)


class ModelPaperPreview(BaseModel):
    paper: ModelPaperDocument
    questions: list[QuestionDocument]


def paper_error(error: ModelPaperError) -> HTTPException:
    message = str(error)
    code = status.HTTP_404_NOT_FOUND if message == "Model paper not found." else status.HTTP_409_CONFLICT
    return HTTPException(status_code=code, detail=message)


@router.post("", response_model=ModelPaperDocument, status_code=status.HTTP_201_CREATED)
def create_model_paper(
    payload: ModelPaperCreate,
    database: Database = Depends(get_database),
    _: UserDocument = PaperUser,
) -> ModelPaperDocument:
    try:
        return ModelPaperService(database).create(**payload.model_dump())
    except ModelPaperError as error:
        raise paper_error(error) from error


@router.post("/generated", response_model=ModelPaperDocument, status_code=status.HTTP_201_CREATED)
def save_generated_paper(
    payload: GeneratedPaperCreate,
    database: Database = Depends(get_database),
    user: UserDocument = PaperUser,
) -> ModelPaperDocument:
    """Persist a generated paper as reviewable drafts without bypassing approval."""
    questions = [
        QuestionDocument(
            question_type=question.question_type,
            pattern=question.pattern,
            question_text=question.question_text,
            options=[option.model_dump() for option in question.options],
            correct_answer=question.correct_answer,
            explanation=question.explanation,
            difficulty=question.difficulty,
            bloom_level=question.bloom_level,
            sources=[source.model_dump() for source in question.sources],
            template_id=payload.template_id,
            subject_id=payload.subject_id,
            marks=question.marks,
            created_by_id=user.id,
        )
        for question in payload.questions
    ]
    database.questions.insert_many([question.model_dump(by_alias=True) for question in questions])
    paper = ModelPaperDocument(
        name=payload.name.strip(),
        subject_id=payload.subject_id,
        question_bank_id="generated-drafts",
        question_ids=[question.id for question in questions],
        question_count=len(questions),
        blueprint={"source": {"generated": len(questions)}},
        created_by_id=user.id,
    )
    database.model_papers.insert_one(paper.model_dump(by_alias=True))
    return paper


@router.get("", response_model=list[ModelPaperDocument])
def list_model_papers(
    subject_id: str | None = None,
    database: Database = Depends(get_database),
    user: UserDocument = PaperUser,
) -> list[ModelPaperDocument]:
    query: dict[str, str] = {"created_by_id": user.id}
    if subject_id:
        query["subject_id"] = subject_id
    return [ModelPaperDocument.model_validate(item) for item in database.model_papers.find(query).sort("created_at", -1)]


@router.get("/{paper_id}", response_model=ModelPaperDocument)
def get_model_paper(
    paper_id: str,
    database: Database = Depends(get_database),
    user: UserDocument = PaperUser,
) -> ModelPaperDocument:
    paper = database.model_papers.find_one({"_id": paper_id, "created_by_id": user.id})
    if not paper:
        raise HTTPException(status_code=404, detail="Model paper not found.")
    return ModelPaperDocument.model_validate(paper)


@router.get("/{paper_id}/preview", response_model=ModelPaperPreview)
def preview_model_paper(
    paper_id: str,
    database: Database = Depends(get_database),
    user: UserDocument = PaperUser,
) -> ModelPaperPreview:
    paper_data = database.model_papers.find_one({"_id": paper_id, "created_by_id": user.id})
    if not paper_data:
        raise HTTPException(status_code=404, detail="Model paper not found.")
    paper = ModelPaperDocument.model_validate(paper_data)
    question_by_id = {item["_id"]: item for item in database.questions.find({"_id": {"$in": paper.question_ids}})}
    questions = [QuestionDocument.model_validate(question_by_id[question_id]) for question_id in paper.question_ids if question_id in question_by_id]
    return ModelPaperPreview(paper=paper, questions=questions)


@router.get("/{paper_id}/download")
def download_model_paper(
    paper_id: str,
    database: Database = Depends(get_database),
    user: UserDocument = PaperUser,
) -> Response:
    preview = preview_model_paper(paper_id, database, user)
    filename = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in preview.paper.name).strip("_") or "question-paper"
    return Response(
        content=build_paper_pdf(preview.paper, preview.questions),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
    )


def build_paper_pdf(paper: ModelPaperDocument, questions: list[QuestionDocument]) -> bytes:
    """Create a lightweight, dependency-free printable PDF for a question paper."""
    lines = [paper.name, f"Questions: {paper.question_count}", ""]
    for index, question in enumerate(questions, start=1):
        lines.extend(wrap_pdf_text(f"{index}. {question.question_text}"))
        for option in question.options:
            lines.extend(wrap_pdf_text(f"   {option.get('key', '')}. {option.get('text', '')}"))
        lines.append("")
    content_lines = ["BT", "/F1 12 Tf", "50 790 Td", "15 TL"]
    for line in lines:
        content_lines.append(f"({escape_pdf_text(line)}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, item in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(item)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(output)


def escape_pdf_text(value: str) -> str:
    return value.encode("latin-1", "replace").decode("latin-1").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_pdf_text(value: str, width: int = 88) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + ([current] if current else [""])


@router.post("/{paper_id}/publish", response_model=ModelPaperDocument)
def publish_model_paper(
    paper_id: str,
    database: Database = Depends(get_database),
    _: UserDocument = PaperUser,
) -> ModelPaperDocument:
    try:
        return ModelPaperService(database).publish(paper_id)
    except ModelPaperError as error:
        raise paper_error(error) from error
