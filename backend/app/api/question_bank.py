from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from pymongo.database import Database

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import QuestionBankDocument, UserDocument

router = APIRouter(prefix="/question-bank", tags=["question-bank"])
BankUser = Depends(require_roles("admin", "faculty"))


class QuestionBankCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: str | None = None
    subject_id: str
    question_ids: list[str] = Field(min_length=1)


@router.get("", response_model=list[QuestionBankDocument])
def list_question_banks(
    subject_id: str | None = None,
    database: Database = Depends(get_database),
    _: UserDocument = BankUser,
) -> list[QuestionBankDocument]:
    query = {"subject_id": subject_id} if subject_id else {}
    return [QuestionBankDocument.model_validate(item) for item in database.question_banks.find(query).sort("name")]


@router.post("", response_model=QuestionBankDocument, status_code=status.HTTP_201_CREATED)
def create_question_bank(
    payload: QuestionBankCreate,
    database: Database = Depends(get_database),
    _: UserDocument = BankUser,
) -> QuestionBankDocument:
    questions = list(database.questions.find({"_id": {"$in": payload.question_ids}}))
    if len(questions) != len(set(payload.question_ids)):
        raise HTTPException(status_code=404, detail="One or more questions were not found.")
    if any(question.get("review_status") != "approved" for question in questions):
        raise HTTPException(status_code=409, detail="Only approved questions can be added to a question bank.")
    if any(question.get("subject_id") != payload.subject_id for question in questions):
        raise HTTPException(status_code=409, detail="All questions must belong to the selected subject.")
    bank = QuestionBankDocument(**payload.model_dump())
    database.question_banks.insert_one(bank.model_dump(by_alias=True))
    return bank


@router.post("/{bank_id}/approve", response_model=QuestionBankDocument)
def approve_question_bank(
    bank_id: str,
    database: Database = Depends(get_database),
    _: UserDocument = BankUser,
) -> QuestionBankDocument:
    bank = database.question_banks.find_one({"_id": bank_id})
    if not bank:
        raise HTTPException(status_code=404, detail="Question bank not found.")
    questions = list(database.questions.find({"_id": {"$in": bank["question_ids"]}}))
    if len(questions) != len(bank["question_ids"]) or any(question.get("review_status") != "approved" for question in questions):
        raise HTTPException(status_code=409, detail="Every question in the bank must be approved first.")
    database.question_banks.update_one({"_id": bank_id}, {"$set": {"approval_status": "approved", "status": "approved"}})
    return QuestionBankDocument.model_validate(database.question_banks.find_one({"_id": bank_id}))