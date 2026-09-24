import mongomock
import pytest
from fastapi import HTTPException

from app.api.question_bank import QuestionBankCreate, approve_question_bank, create_question_bank
from app.api.questions import reject_question, review_questions
from app.database import ensure_indexes
from app.models import QuestionDocument


def make_question(subject_id: str = "subject-1") -> QuestionDocument:
    return QuestionDocument(
        question_type="MCQ",
        pattern="Direct Concept",
        question_text="Where are smaller values placed?",
        options=[{"key": "A", "text": "Left"}, {"key": "B", "text": "Right"}],
        correct_answer="A",
        explanation="Smaller values go left.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[{"chunk_id": "chunk-1", "page": 1}],
        subject_id=subject_id,
    )


@pytest.fixture
def database():
    database = mongomock.MongoClient().test
    ensure_indexes(database)
    return database


def test_review_rejects_question_and_preserves_note(database) -> None:
    question = make_question()
    database.questions.insert_one(question.model_dump(by_alias=True))

    rejected = reject_question(question.id, "Needs a clearer distractor.", database, object())
    review = review_questions("rejected", database, object())

    assert rejected.review_status == "rejected"
    assert rejected.review_note == "Needs a clearer distractor."
    assert [item.id for item in review] == [question.id]


def test_question_bank_requires_approved_questions_and_can_be_approved(database) -> None:
    question = make_question()
    database.questions.insert_one(question.model_dump(by_alias=True))
    with pytest.raises(HTTPException) as draft_error:
        create_question_bank(QuestionBankCreate(name="Bank", subject_id="subject-1", question_ids=[question.id]), database, object())
    assert draft_error.value.status_code == 409

    database.questions.update_one({"_id": question.id}, {"$set": {"review_status": "approved"}})
    bank = create_question_bank(QuestionBankCreate(name="Bank", subject_id="subject-1", question_ids=[question.id]), database, object())
    approved = approve_question_bank(bank.id, database, object())

    assert approved.approval_status == "approved"
    assert approved.question_ids == [question.id]


def test_question_bank_rejects_mixed_subjects(database) -> None:
    first = make_question("subject-1")
    second = make_question("subject-2")
    first.review_status = second.review_status = "approved"
    database.questions.insert_many([first.model_dump(by_alias=True), second.model_dump(by_alias=True)])

    with pytest.raises(HTTPException) as error:
        create_question_bank(QuestionBankCreate(name="Mixed", subject_id="subject-1", question_ids=[first.id, second.id]), database, object())

    assert error.value.status_code == 409