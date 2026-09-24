from collections import Counter

from pymongo.database import Database

from app.models import ModelPaperDocument


class ModelPaperError(ValueError):
    pass


class ModelPaperService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(
        self,
        *,
        name: str,
        subject_id: str,
        question_bank_id: str,
        question_count: int,
        question_type_counts: dict[str, int] | None = None,
        difficulty_counts: dict[str, int] | None = None,
        bloom_level_counts: dict[str, int] | None = None,
    ) -> ModelPaperDocument:
        if question_count < 1:
            raise ModelPaperError("question_count must be at least 1.")
        distributions = {
            "question_type": question_type_counts or {},
            "difficulty": difficulty_counts or {},
            "bloom_level": bloom_level_counts or {},
        }
        for dimension, values in distributions.items():
            if values and sum(values.values()) != question_count:
                raise ModelPaperError(f"{dimension} distribution must sum to question_count.")
            if any(value < 0 for value in values.values()):
                raise ModelPaperError(f"{dimension} distribution cannot contain negative counts.")

        bank = self.database.question_banks.find_one(
            {"_id": question_bank_id, "subject_id": subject_id, "approval_status": "approved"}
        )
        if not bank:
            raise ModelPaperError("An approved question bank for this subject is required.")
        questions = list(
            self.database.questions.find(
                {"_id": {"$in": bank["question_ids"]}, "subject_id": subject_id, "review_status": "approved"}
            ).sort("_id")
        )
        selected = self._select(questions, question_count, distributions)
        paper = ModelPaperDocument(
            name=name,
            subject_id=subject_id,
            question_bank_id=question_bank_id,
            question_ids=[question["_id"] for question in selected],
            question_count=question_count,
            blueprint=distributions,
        )
        self.database.model_papers.insert_one(paper.model_dump(by_alias=True))
        return paper

    def publish(self, paper_id: str) -> ModelPaperDocument:
        paper = self.database.model_papers.find_one({"_id": paper_id})
        if not paper:
            raise ModelPaperError("Model paper not found.")
        self.database.model_papers.update_one(
            {"_id": paper_id}, {"$set": {"publication_status": "published", "status": "published"}}
        )
        return ModelPaperDocument.model_validate(self.database.model_papers.find_one({"_id": paper_id}))

    @staticmethod
    def _select(questions: list[dict], count: int, distributions: dict[str, dict[str, int]]) -> list[dict]:
        if len(questions) < count:
            raise ModelPaperError("The approved question bank does not contain enough questions.")
        remaining = {dimension: Counter(values) for dimension, values in distributions.items()}
        selected: list[dict] = []
        available = list(questions)
        if not any(remaining.values()):
            return available[:count]
        while available and len(selected) < count:
            ranked = sorted(
                available,
                key=lambda question: sum(
                    1 for dimension, targets in remaining.items() if targets.get(question.get(dimension), 0) > 0
                ),
                reverse=True,
            )
            question = ranked[0]
            if any(remaining[dimension].get(question.get(dimension), 0) > 0 for dimension in remaining):
                selected.append(question)
                available.remove(question)
                for dimension in remaining:
                    value = question.get(dimension)
                    if remaining[dimension].get(value, 0) > 0:
                        remaining[dimension][value] -= 1
            else:
                break
        if len(selected) != count or any(value for values in remaining.values() for value in values.values()):
            raise ModelPaperError("The approved question bank cannot satisfy the requested blueprint.")
        return selected