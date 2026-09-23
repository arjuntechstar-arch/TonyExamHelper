from collections.abc import Sequence
from typing import Any
from pymongo.collection import Collection


class MongoRepository:
    def __init__(self, collection: Collection) -> None:
        self.collection = collection

    def get(self, document_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"_id": document_id})

    def list(self, *, offset: int = 0, limit: int = 100) -> Sequence[dict[str, Any]]:
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset must be non-negative and limit must be between 1 and 100")
        return list(self.collection.find().skip(offset).limit(limit))

    def add(self, document: dict[str, Any]) -> dict[str, Any]:
        self.collection.insert_one(document)
        return document

    def delete(self, document_id: str) -> bool:
        return self.collection.delete_one({"_id": document_id}).deleted_count == 1
