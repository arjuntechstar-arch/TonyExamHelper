from pymongo.database import Database
from app.models import SubjectDocument
from app.repositories.base import MongoRepository


class SubjectRepository(MongoRepository):
    def __init__(self, database: Database) -> None:
        super().__init__(database.subjects)

    def add_subject(self, subject: SubjectDocument) -> SubjectDocument:
        self.add(subject.model_dump(by_alias=True))
        return subject

    def get_by_code(self, code: str) -> SubjectDocument | None:
        document = self.collection.find_one({"code": code})
        return SubjectDocument.model_validate(document) if document else None
