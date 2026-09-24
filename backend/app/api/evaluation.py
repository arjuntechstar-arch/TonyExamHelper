from typing import Any

from fastapi import APIRouter, Depends

from app.api.auth import require_roles
from app.models import UserDocument
from app.services.evaluation import evaluate_question_set, run_research_evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
EvaluationUser = Depends(require_roles("admin", "faculty"))


@router.post("/questions")
def evaluate_questions(
    questions: list[dict[str, Any]],
    _: UserDocument = EvaluationUser,
) -> dict[str, Any]:
    return evaluate_question_set(questions)


@router.post("/research")
def research_evaluation(
    questions: list[dict[str, Any]],
    _: UserDocument = EvaluationUser,
) -> dict[str, Any]:
    return run_research_evaluation(questions)
