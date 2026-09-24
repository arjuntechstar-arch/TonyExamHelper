from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any

from pymongo.database import Database

from app.models import PracticeTestDocument, StudentAnswerDocument


class PracticeError(ValueError):
    pass


class PracticeService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def start_test(
        self,
        *,
        student_id: str,
        subject_id: str,
        question_count: int = 5,
        difficulty: str | None = None,
        topic_id: str | None = None,
    ) -> PracticeTestDocument:
        if question_count < 1:
            raise PracticeError("question_count must be at least 1.")

        base_query: dict[str, Any] = {"subject_id": subject_id, "review_status": "approved"}
        if topic_id:
            base_query["topic_id"] = topic_id

        preferred = list(self.database.questions.find({**base_query, **({"difficulty": difficulty} if difficulty else {})}).sort("created_at", -1))
        fallback = list(self.database.questions.find(base_query).sort("created_at", -1))
        if not preferred and not fallback:
            raise PracticeError("No approved questions are available for this practice set.")

        if len(preferred) >= question_count:
            selected = random.sample(preferred, question_count)
        else:
            selected = list(preferred)
            seen_ids = {question["_id"] for question in selected}
            available_rest = [question for question in fallback if question["_id"] not in seen_ids]
            remaining = question_count - len(selected)
            if remaining > 0:
                selected.extend(random.sample(available_rest, min(remaining, len(available_rest))))

        test = PracticeTestDocument(
            student_id=student_id,
            subject_id=subject_id,
            topic_id=topic_id,
            question_ids=[question["_id"] for question in selected],
            question_count=len(selected),
            duration_minutes=30,
            difficulty=difficulty,
            status="started",
            score=0.0,
            correct_count=0,
            total_questions=len(selected),
            started_at=datetime.now(UTC),
        )
        self.database.practice_tests.insert_one(test.model_dump(by_alias=True))
        return test

    def submit_answers(
        self,
        practice_test_id: str,
        answers: dict[str, str],
        *,
        requester_id: str | None = None,
        requester_roles: list[str] | None = None,
    ) -> dict[str, Any]:
        test = self.database.practice_tests.find_one({"_id": practice_test_id})
        if not test:
            raise PracticeError("Practice test not found.")
        self._ensure_access(test, requester_id, requester_roles or [])

        question_ids = test.get("question_ids", [])
        if not question_ids:
            raise PracticeError("This practice test has no questions assigned.")

        questions = list(self.database.questions.find({"_id": {"$in": question_ids}}))
        by_id = {question["_id"]: question for question in questions}

        question_results: list[dict[str, Any]] = []
        correct_count = 0
        for question_id in question_ids:
            question = by_id.get(question_id)
            if question is None:
                continue
            selected_answer = answers.get(question_id)
            is_correct = selected_answer == question.get("correct_answer")
            if is_correct:
                correct_count += 1

            result = {
                "question_id": question_id,
                "question_text": question.get("question_text"),
                "selected_answer": selected_answer,
                "correct_answer": question.get("correct_answer"),
                "is_correct": is_correct,
                "explanation": question.get("explanation"),
            }
            question_results.append(result)

            self.database.student_answers.update_one(
                {"practice_test_id": practice_test_id, "question_id": question_id},
                {
                    "$set": {
                        "selected_answer": selected_answer,
                        "is_correct": is_correct,
                        "score": int(is_correct),
                        "explanation": question.get("explanation"),
                        "submitted_at": datetime.now(UTC),
                    }
                },
                upsert=True,
            )

        total_questions = len(question_results)
        percentage = round((correct_count / total_questions) * 100, 2) if total_questions else 0.0
        self.database.practice_tests.update_one(
            {"_id": practice_test_id},
            {
                "$set": {
                    "status": "completed",
                    "score": float(correct_count),
                    "correct_count": correct_count,
                    "total_questions": total_questions,
                    "percentage": percentage,
                    "completed_at": datetime.now(UTC),
                }
            },
        )

        return {
            "practice_test_id": practice_test_id,
            "status": "completed",
            "score": correct_count,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "percentage": percentage,
            "question_results": question_results,
        }

    def get_test(self, practice_test_id: str, *, requester_id: str, requester_roles: list[str]) -> dict[str, Any]:
        test = self.database.practice_tests.find_one({"_id": practice_test_id})
        if not test:
            raise PracticeError("Practice test not found.")
        if test.get("student_id") != requester_id and not set(requester_roles).intersection({"admin", "faculty"}):
            raise PracticeError("You are not allowed to access this practice test.")

        questions = list(self.database.questions.find({"_id": {"$in": test.get("question_ids", [])}}))
        by_id = {question["_id"]: question for question in questions}
        safe_questions = []
        for question_id in test.get("question_ids", []):
            question = by_id.get(question_id)
            if not question:
                continue
            safe_questions.append({
                "question_id": question_id,
                "question_text": question.get("question_text"),
                "options": question.get("options", []),
                "difficulty": question.get("difficulty"),
                "topic_id": question.get("topic_id"),
            })

        return {
            "practice_test_id": practice_test_id,
            "status": test.get("status", "started"),
            "duration_minutes": test.get("duration_minutes", 30),
            "questions": safe_questions,
        }

    def get_result(
        self,
        practice_test_id: str,
        *,
        requester_id: str | None = None,
        requester_roles: list[str] | None = None,
    ) -> dict[str, Any]:
        test = self.database.practice_tests.find_one({"_id": practice_test_id})
        if not test:
            raise PracticeError("Practice test not found.")
        self._ensure_access(test, requester_id, requester_roles or [])

        answers = list(self.database.student_answers.find({"practice_test_id": practice_test_id}).sort("submitted_at", -1))
        question_results = [
            {
                "question_id": answer["question_id"],
                "selected_answer": answer.get("selected_answer"),
                "is_correct": answer.get("is_correct"),
                "explanation": answer.get("explanation"),
            }
            for answer in answers
        ]
        return {
            "practice_test_id": practice_test_id,
            "status": test.get("status", "started"),
            "score": test.get("score", 0),
            "total_questions": test.get("total_questions", len(test.get("question_ids", []))),
            "correct_count": test.get("correct_count", 0),
            "percentage": test.get("percentage", 0.0),
            "question_results": question_results,
        }

    @staticmethod
    def _ensure_access(test: dict[str, Any], requester_id: str | None, requester_roles: list[str]) -> None:
        if requester_id is None:
            return
        if test.get("student_id") != requester_id and not set(requester_roles).intersection({"admin", "faculty"}):
            raise PracticeError("You are not allowed to access this practice test.")
