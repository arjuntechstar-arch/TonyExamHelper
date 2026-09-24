import hashlib
import math
import re
from collections.abc import Iterable
from typing import Protocol

from pymongo.database import Database

from app.models import DocumentChunkDocument


class EmbeddingProvider(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Offline, deterministic baseline embedding for development and tests."""

    model_name = "hash-embedding-v1"

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("Embedding dimensions must be at least 8.")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class VectorStore(Protocol):
    def upsert(self, chunk: DocumentChunkDocument) -> None: ...

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[dict]: ...


class MongoVectorStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(self, chunk: DocumentChunkDocument) -> None:
        self.database.document_chunks.replace_one(
            {"_id": chunk.id},
            chunk.model_dump(by_alias=True),
            upsert=True,
        )

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query = {"embedding": {"$exists": True, "$ne": None}}
        for key, value in (filters or {}).items():
            query[f"metadata.{key}"] = value
        matches = []
        for item in self.database.document_chunks.find(query):
            score = cosine_similarity(query_embedding, item["embedding"])
            matches.append({"chunk": DocumentChunkDocument.model_validate(item), "score": score})
        return sorted(matches, key=lambda item: item["score"], reverse=True)[:top_k]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("Embedding dimensions must match.")
    left_magnitude = math.sqrt(sum(value * value for value in left_values))
    right_magnitude = math.sqrt(sum(value * value for value in right_values))
    if not left_magnitude or not right_magnitude:
        return 0.0
    return sum(a * b for a, b in zip(left_values, right_values, strict=True)) / (left_magnitude * right_magnitude)


class RetrievalService:
    def __init__(self, database: Database, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.database = database
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.vector_store = MongoVectorStore(database)

    def index_material(self, material_id: str) -> int:
        if not self.database.study_materials.find_one({"_id": material_id}):
            raise ValueError("Material not found.")
        chunks = self.database.document_chunks.find({"study_material_id": material_id})
        count = 0
        for item in chunks:
            chunk = DocumentChunkDocument.model_validate(item)
            chunk.embedding = self.embedding_provider.embed(chunk.content)
            chunk.embedding_model = self.embedding_provider.model_name
            self.vector_store.upsert(chunk)
            count += 1
        if count:
            self.database.study_materials.update_one({"_id": material_id}, {"$set": {"status": "indexed"}})
        return count

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject_id: str | None = None,
        course_id: str | None = None,
        syllabus_id: str | None = None,
        topic_id: str | None = None,
    ) -> list[dict]:
        filters = {
            key: value
            for key, value in {
                "subject_id": subject_id,
                "course_id": course_id,
                "syllabus_id": syllabus_id,
                "topic_id": topic_id,
            }.items()
            if value is not None
        }
        query_embedding = self.embedding_provider.embed(query)
        return self.vector_store.search(query_embedding, top_k=top_k, filters=filters)