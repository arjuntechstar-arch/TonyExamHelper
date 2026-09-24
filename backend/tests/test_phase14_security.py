import mongomock
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.core.database import get_database
from app.database import ensure_indexes
from app.main import app
from app.models import PracticeTestDocument, QuestionDocument, UserDocument
from app.services.practice import PracticeService


def make_question() -> QuestionDocument:
    return QuestionDocument(
        question_type="MCQ",
        pattern="Direct Concept",
        question_text="Where are smaller values placed?",
        options=[{"key": "A", "text": "Left"}, {"key": "B", "text": "Right"}],
        correct_answer="A",
        explanation="Smaller values go left.",
        difficulty="Medium",
        bloom_level="Apply",
        subject_id="subject-1",
        review_status="approved",
    )


def test_practice_service_protects_submission_and_result_ownership() -> None:
    database = mongomock.MongoClient().test
    ensure_indexes(database)
    question = make_question()
    database.questions.insert_one(question.model_dump(by_alias=True))
    practice = PracticeService(database).start_test(
        student_id="student-1", subject_id="subject-1", question_count=1
    )
    service = PracticeService(database)

    for operation in (
        lambda: service.submit_answers(
            practice.id, {question.id: "A"}, requester_id="student-2", requester_roles=["student"]
        ),
        lambda: service.get_result(
            practice.id, requester_id="student-2", requester_roles=["student"]
        ),
    ):
        try:
            operation()
        except ValueError as error:
            assert str(error) == "You are not allowed to access this practice test."
        else:
            raise AssertionError("Expected cross-student practice access to be denied")


def test_practice_integration_start_submit_and_result() -> None:
    database = mongomock.MongoClient().test
    ensure_indexes(database)
    question = make_question()
    database.questions.insert_one(question.model_dump(by_alias=True))
    student = UserDocument(
        email="student@example.edu",
        display_name="Student",
        password_hash="not-used",
        roles=["student"],
    )

    app.dependency_overrides[get_database] = lambda: database
    app.dependency_overrides[get_current_user] = lambda: student
    try:
        with TestClient(app) as client:
            started = client.post(
                "/api/practice/start",
                json={"subject_id": "subject-1", "question_count": 1},
            )
            assert started.status_code == 201
            practice_id = started.json()["_id"]

            viewed = client.get(f"/api/practice/{practice_id}")
            assert viewed.status_code == 200
            question_id = viewed.json()["questions"][0]["question_id"]
            assert "correct_answer" not in viewed.json()["questions"][0]

            submitted = client.post(
                f"/api/practice/{practice_id}/submit",
                json={"answers": {question_id: "A"}},
            )
            assert submitted.status_code == 200
            assert submitted.json()["percentage"] == 100.0

            result = client.get(f"/api/practice/{practice_id}/result")
            assert result.status_code == 200
            assert result.json()["correct_count"] == 1
    finally:
        app.dependency_overrides.clear()


def test_protected_practice_endpoint_has_stable_auth_error_contract() -> None:
    response = TestClient(app).get("/api/practice/missing")

    assert response.status_code == 401
    body = response.json()
    assert body["error"] == "http_error"
    assert body["message"] == "Authentication is required."
    assert body["request_id"] == response.headers["X-Request-ID"]
