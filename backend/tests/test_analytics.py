import mongomock

from fastapi import HTTPException

from app.api.analytics import get_student_analytics, get_student_difficulty_summary, get_student_topic_summary
from app.database import ensure_indexes
from app.models import PracticeTestDocument, QuestionDocument, UserDocument


def make_question(subject_id: str = "subject-1", topic_id: str = "topic-1", difficulty: str = "Medium") -> QuestionDocument:
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


def test_student_analytics_rolls_up_attempts_by_topic_and_difficulty() -> None:
    database = mongomock.MongoClient().test
    ensure_indexes(database)

    q1 = make_question(topic_id="topic-1", difficulty="Medium")
    q2 = make_question(topic_id="topic-1", difficulty="Easy")
    q3 = make_question(topic_id="topic-2", difficulty="Medium")
    database.questions.insert_many([q1.model_dump(by_alias=True), q2.model_dump(by_alias=True), q3.model_dump(by_alias=True)])

    t1 = PracticeTestDocument(student_id="student-1", subject_id="subject-1", question_ids=[q1.id, q2.id], question_count=2, total_questions=2)
    t2 = PracticeTestDocument(student_id="student-1", subject_id="subject-1", question_ids=[q3.id], question_count=1, total_questions=1)
    database.practice_tests.insert_many([t1.model_dump(by_alias=True), t2.model_dump(by_alias=True)])

    database.student_answers.insert_many([
        {"_id": "sa-1", "practice_test_id": t1.id, "question_id": q1.id, "selected_answer": "A", "is_correct": True, "score": 1, "explanation": "Smaller values go left."},
        {"_id": "sa-2", "practice_test_id": t1.id, "question_id": q2.id, "selected_answer": "B", "is_correct": False, "score": 0, "explanation": "Smaller values go left."},
        {"_id": "sa-3", "practice_test_id": t2.id, "question_id": q3.id, "selected_answer": "A", "is_correct": True, "score": 1, "explanation": "Smaller values go left."},
    ])

    overview = get_student_analytics("student-1", database)
    topics = get_student_topic_summary("student-1", database)
    difficulties = get_student_difficulty_summary("student-1", database)

    assert overview["total_attempts"] == 2
    assert overview["average_percentage"] == 66.67
    assert overview["correct_answers"] == 2
    assert overview["weak_topics"][0]["topic_id"] == "topic-1"
    assert topics[0]["topic_id"] == "topic-1"
    assert difficulties[0]["difficulty"] in {"Easy", "Medium"}


def test_students_cannot_view_another_students_analytics() -> None:
    database = mongomock.MongoClient().test
    user = UserDocument(
        email="student@example.com",
        display_name="Student",
        password_hash="hash",
        roles=["student"],
    )

    try:
        get_student_analytics("another-student", database, user)
    except HTTPException as error:
        assert error.status_code == 403
    else:
        raise AssertionError("Expected analytics access to be denied")
