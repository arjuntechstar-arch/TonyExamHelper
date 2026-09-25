import mongomock
import pytest

from app.database import ensure_indexes
from app.models import ModelPaperDocument, QuestionBankDocument, QuestionDocument
from app.api.model_papers import GeneratedPaperCreate, download_model_paper, preview_model_paper, save_generated_paper
from app.services.generation import GeneratedQuestion
from app.services.model_papers import ModelPaperError, ModelPaperService


@pytest.fixture
def database():
    database = mongomock.MongoClient().test
    ensure_indexes(database)
    return database


def add_question(database, question_type: str, difficulty: str, bloom_level: str) -> QuestionDocument:
    question = QuestionDocument(
        question_type=question_type,
        pattern="Direct Concept",
        question_text=f"Question {question_type} {difficulty} {bloom_level}",
        options=[],
        explanation="Explanation",
        difficulty=difficulty,
        bloom_level=bloom_level,
        sources=[{"chunk_id": "chunk-1", "page": 1}],
        subject_id="subject-1",
        review_status="approved",
    )
    database.questions.insert_one(question.model_dump(by_alias=True))
    return question


def setup_bank(database, questions: list[QuestionDocument]) -> QuestionBankDocument:
    bank = QuestionBankDocument(
        name="Approved Bank",
        subject_id="subject-1",
        question_ids=[question.id for question in questions],
        approval_status="approved",
    )
    database.question_banks.insert_one(bank.model_dump(by_alias=True))
    return bank


def test_model_paper_selects_requested_blueprint_and_publishes(database) -> None:
    questions = [
        add_question(database, "MCQ", "Easy", "Remember"),
        add_question(database, "Short", "Medium", "Understand"),
        add_question(database, "Long", "Hard", "Apply"),
    ]
    bank = setup_bank(database, questions)

    service = ModelPaperService(database)
    paper = service.create(
        name="Midterm",
        subject_id="subject-1",
        question_bank_id=bank.id,
        question_count=3,
        question_type_counts={"MCQ": 1, "Short": 1, "Long": 1},
        difficulty_counts={"Easy": 1, "Medium": 1, "Hard": 1},
    )
    published = service.publish(paper.id)

    assert set(paper.question_ids) == {question.id for question in questions}
    assert published.publication_status == "published"


def test_model_paper_requires_approved_bank(database) -> None:
    question = add_question(database, "MCQ", "Easy", "Remember")
    bank = QuestionBankDocument(name="Draft Bank", subject_id="subject-1", question_ids=[question.id])
    database.question_banks.insert_one(bank.model_dump(by_alias=True))

    with pytest.raises(ModelPaperError, match="approved question bank"):
        ModelPaperService(database).create(
            name="Paper",
            subject_id="subject-1",
            question_bank_id=bank.id,
            question_count=1,
        )


def test_model_paper_rejects_unsatisfied_blueprint(database) -> None:
    question = add_question(database, "MCQ", "Easy", "Remember")
    bank = setup_bank(database, [question])

    with pytest.raises(ModelPaperError, match="enough questions"):
        ModelPaperService(database).create(
            name="Paper",
            subject_id="subject-1",
            question_bank_id=bank.id,
            question_count=2,
            question_type_counts={"MCQ": 2},
        )


def test_generated_paper_can_be_previewed_and_downloaded_as_pdf(database) -> None:
    generated = GeneratedQuestion(
        question_text="Which side contains smaller binary-search-tree values?",
        options=[{"key": "A", "text": "Left"}, {"key": "B", "text": "Right"}],
        correct_answer="A",
        explanation="Smaller values are stored on the left.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[{"chunk_id": "chunk-1", "page": 1}],
    )
    user = type("User", (), {"id": "faculty-1"})()
    paper = save_generated_paper(
        GeneratedPaperCreate(name="Unit 1 Test", subject_id="subject-1", questions=[generated]),
        database,
        user,
    )

    preview = preview_model_paper(paper.id, database, user)
    download = download_model_paper(paper.id, database, user)

    assert preview.paper.name == "Unit 1 Test"
    assert preview.questions[0].question_text == generated.question_text
    assert download.media_type == "application/pdf"
    assert download.body.startswith(b"%PDF-1.4")
