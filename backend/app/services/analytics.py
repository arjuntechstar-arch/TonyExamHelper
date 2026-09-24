from __future__ import annotations

from collections import defaultdict
from typing import Any

from pymongo.database import Database


class AnalyticsService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def student_overview(self, student_id: str) -> dict[str, Any]:
        tests = list(self.database.practice_tests.find({"student_id": student_id}))
        if not tests:
            return {
                "student_id": student_id,
                "total_attempts": 0,
                "total_questions": 0,
                "correct_answers": 0,
                "average_percentage": 0.0,
                "weak_topics": [],
            }

        answers = list(self.database.student_answers.find({"practice_test_id": {"$in": [test["_id"] for test in tests]}}))
        total_questions = len(answers)
        correct_answers = sum(1 for answer in answers if answer.get("is_correct") is True)
        average_percentage = round((correct_answers / total_questions) * 100, 2) if total_questions else 0.0

        topic_stats = defaultdict(lambda: {"score": 0, "attempts": 0, "correct": 0})
        for answer in answers:
            question = self.database.questions.find_one({"_id": answer["question_id"]})
            if not question:
                continue
            topic_id = question.get("topic_id") or "unassigned"
            topic_stats[topic_id]["attempts"] += 1
            topic_stats[topic_id]["correct"] += int(answer.get("is_correct") is True)
            topic_stats[topic_id]["score"] = round((topic_stats[topic_id]["correct"] / topic_stats[topic_id]["attempts"]) * 100, 2)

        weak_topics = [
            {"topic_id": topic_id, "accuracy": stats["score"], "attempts": stats["attempts"], "correct_answers": stats["correct"]}
            for topic_id, stats in sorted(topic_stats.items(), key=lambda item: (item[1]["score"], item[1]["attempts"]))
        ]

        return {
            "student_id": student_id,
            "total_attempts": len(tests),
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "average_percentage": average_percentage,
            "weak_topics": weak_topics,
        }

    def topic_summary(self, student_id: str) -> list[dict[str, Any]]:
        tests = list(self.database.practice_tests.find({"student_id": student_id}))
        if not tests:
            return []

        topic_stats = defaultdict(lambda: {"attempts": 0, "correct": 0})
        for test in tests:
            answers = list(self.database.student_answers.find({"practice_test_id": test["_id"]}))
            for answer in answers:
                question = self.database.questions.find_one({"_id": answer["question_id"]})
                if not question:
                    continue
                topic_id = question.get("topic_id") or "unassigned"
                topic_stats[topic_id]["attempts"] += 1
                topic_stats[topic_id]["correct"] += int(answer.get("is_correct") is True)

        return [
            {
                "topic_id": topic_id,
                "attempts": stats["attempts"],
                "correct_answers": stats["correct"],
                "accuracy": round((stats["correct"] / stats["attempts"]) * 100, 2) if stats["attempts"] else 0.0,
            }
            for topic_id, stats in sorted(
                topic_stats.items(),
                key=lambda item: ((item[1]["correct"] / item[1]["attempts"]) if item[1]["attempts"] else 1.0, item[1]["attempts"]),
            )
        ]

    def difficulty_summary(self, student_id: str) -> list[dict[str, Any]]:
        tests = list(self.database.practice_tests.find({"student_id": student_id}))
        if not tests:
            return []

        difficulty_stats = defaultdict(lambda: {"attempts": 0, "correct": 0})
        for test in tests:
            answers = list(self.database.student_answers.find({"practice_test_id": test["_id"]}))
            for answer in answers:
                question = self.database.questions.find_one({"_id": answer["question_id"]})
                if not question:
                    continue
                difficulty = question.get("difficulty") or "Unknown"
                difficulty_stats[difficulty]["attempts"] += 1
                difficulty_stats[difficulty]["correct"] += int(answer.get("is_correct") is True)

        return [
            {
                "difficulty": difficulty,
                "attempts": stats["attempts"],
                "correct_answers": stats["correct"],
                "accuracy": round((stats["correct"] / stats["attempts"]) * 100, 2) if stats["attempts"] else 0.0,
            }
            for difficulty, stats in sorted(
                difficulty_stats.items(),
                key=lambda item: ((item[1]["correct"] / item[1]["attempts"]) if item[1]["attempts"] else 1.0, item[1]["attempts"]),
            )
        ]


def get_student_analytics(student_id: str, database: Database) -> dict[str, Any]:
    return AnalyticsService(database).student_overview(student_id)


def get_student_topic_summary(student_id: str, database: Database) -> list[dict[str, Any]]:
    return AnalyticsService(database).topic_summary(student_id)


def get_student_difficulty_summary(student_id: str, database: Database) -> list[dict[str, Any]]:
    return AnalyticsService(database).difficulty_summary(student_id)
