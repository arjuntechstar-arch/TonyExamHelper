import mongomock
import pytest

from app.api.practice import PracticeAnswerSubmission, submit_practice_answers
from app.database import ensure_indexes
from app.models import QuestionDocument, PracticeTestDocument
from app.services.practice import PracticeError, PracticeService


def make_question(subject_id: str = "subject-1", topic_id: str | None = None, difficulty: str = "Medium") -> QuestionDocument:
    return QuestionDocument(
        question_type="MCQ",
        pattern="Direct Concept",
        question_text="Where are smaller values placed?",
        options=[
            {"key": "A", "text": "Left"},
            {"key": "B", "text": "Right"},
            {"key": "C", "text": "Above"},
            {"key": "D", "text": "Below"},
        ],
        correct_answer="A",
        explanation="Smaller values go left.",
        difficulty=difficulty,
        bloom_level="Apply",
        sources=[{"chunk_id": "chunk-1", "page": 1}],
        subject_id=subject_id,
        topic_id=topic_id,
        review_status="approved",
    )


def test_practice_service_starts_and_scores_a_test() -> None:
    database = mongomock.MongoClient().test
    ensure_indexes(database)

    question_one = make_question()
    question_two = make_question(difficulty="Easy")
    database.questions.insert_many([question_one.model_dump(by_alias=True), question_two.model_dump(by_alias=True)])

    service = PracticeService(database)
    test = service.start_test(
        student_id="student-1",
        subject_id="subject-1",
        question_count=2,
        difficulty="Medium",
    )

    assert isinstance(test, PracticeTestDocument)
    assert test.status == "started"
    assert len(test.question_ids) == 2

    result = service.submit_answers(
        test.id,
        {test.question_ids[0]: "A", test.question_ids[1]: "B"},
    )

    assert result["score"] == 1
    assert result["total_questions"] == 2
    assert result["correct_count"] == 1
    assert result["percentage"] == 50.0
    assert result["question_results"][0]["is_correct"] in {True, False}

    practice_payload = service.get_test(test.id, requester_id="student-1", requester_roles=["student"])
    assert practice_payload["practice_test_id"] == test.id
    assert len(practice_payload["questions"]) == 2
    assert "correct_answer" not in practice_payload["questions"][0]


def test_practice_service_rejects_access_to_another_students_test() -> None:
    database = mongomock.MongoClient().test
    question = QuestionDocument(
        question_type="mcq", pattern="direct", question_text="Which?", options=[{"key": "A", "text": "Yes"}],
        correct_answer="A", explanation="Because.", difficulty="easy", bloom_level="remember", subject_id="subject-1", review_status="approved",
    )
    database.questions.insert_one(question.model_dump(by_alias=True))
    practice = PracticeTestDocument(student_id="student-1", subject_id="subject-1", question_ids=[question.id], question_count=1, total_questions=1)
    database.practice_tests.insert_one(practice.model_dump(by_alias=True))

    with pytest.raises(PracticeError, match="not allowed"):
        PracticeService(database).get_test(practice.id, requester_id="student-2", requester_roles=["student"])


def test_submit_answer_route_accepts_student_answers() -> None:
    database = mongomock.MongoClient().test
    ensure_indexes(database)

    question = make_question()
    database.questions.insert_one(question.model_dump(by_alias=True))

    practice = PracticeTestDocument(
        student_id="student-2",
        subject_id="subject-1",
        question_ids=[question.id],
        question_count=1,
        duration_minutes=15,
    )
    database.practice_tests.insert_one(practice.model_dump(by_alias=True))

    result = submit_practice_answers(
        practice_id=practice.id,
        payload=PracticeAnswerSubmission(answers={question.id: "A"}),
        database=database,
        user=object(),
    )

    assert result["status"] == "completed"
    assert result["score"] == 1
