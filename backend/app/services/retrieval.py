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


def lexical_similarity(left: str, right: str) -> float:
    """A transparent sparse-search signal used alongside vector similarity."""
    left_tokens = set(re.findall(r"[a-z0-9]+", left.casefold()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def rerank_hybrid(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Combine dense and sparse relevance, then avoid near-identical chunks.

    The lightweight MMR pass expands coverage for question generation without
    requiring a hosted search engine.  Each returned item retains its component
    scores so the UI and evaluation endpoints can explain retrieval quality.
    """
    scored = []
    for candidate in candidates:
        chunk = candidate["chunk"]
        semantic_score = float(candidate["score"])
        lexical_score = lexical_similarity(query, chunk.content)
        scored.append({
            **candidate,
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "hybrid_score": (0.75 * semantic_score) + (0.25 * lexical_score),
        })
    remaining = sorted(scored, key=lambda item: item["hybrid_score"], reverse=True)
    selected: list[dict] = []
    while remaining and len(selected) < top_k:
        def mmr_score(item: dict) -> float:
            redundancy = max(
                (lexical_similarity(item["chunk"].content, chosen["chunk"].content) for chosen in selected),
                default=0.0,
            )
            return (0.85 * item["hybrid_score"]) - (0.15 * redundancy)

        best = max(remaining, key=mmr_score)
        best["score"] = round(mmr_score(best), 6)
        selected.append(best)
        remaining.remove(best)
    return selected


class RetrievalService:
    def __init__(self, database: Database, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.database = database
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.vector_store = MongoVectorStore(database)

    def index_material(self, material_id: str) -> int:
        material = self.database.study_materials.find_one({"_id": material_id})
        if material is None:
            material = self.database.study_materials.find_one({"id": material_id})
        if material is None:
            raise ValueError("Material not found.")
        resolved_material_id = str(material.get("id", material.get("_id", material_id)))
        chunks = self.database.document_chunks.find({"study_material_id": resolved_material_id})
        count = 0
        for item in chunks:
            chunk = DocumentChunkDocument.model_validate(item)
            chunk.embedding = self.embedding_provider.embed(chunk.content)
            chunk.embedding_model = self.embedding_provider.model_name
            self.vector_store.upsert(chunk)
            count += 1
        if count:
            material_key = "id" if material.get("id") else "_id"
            self.database.study_materials.update_one(
                {material_key: material.get(material_key, material_id)},
                {"$set": {"status": "indexed"}},
            )
        return count

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject_id: str | None = None,
        study_material_id: str | None = None,
        course_id: str | None = None,
        syllabus_id: str | None = None,
        topic_id: str | None = None,
    ) -> list[dict]:
        filters = {
            key: value
            for key, value in {
                "subject_id": subject_id,
                "study_material_id": study_material_id,
                "course_id": course_id,
                "syllabus_id": syllabus_id,
                "topic_id": topic_id,
            }.items()
            if value is not None
        }
        query_embedding = self.embedding_provider.embed(query)
        # Retrieve a broader dense candidate pool before hybrid re-ranking.
        candidates = self.vector_store.search(
            query_embedding,
            top_k=min(150, max(top_k * 4, top_k)),
            filters=filters,
        )
        return rerank_hybrid(query, candidates, top_k)

    def retrieve_all(
        self,
        *,
        subject_id: str | None = None,
        study_material_id: str | None = None,
        course_id: str | None = None,
        syllabus_id: str | None = None,
        topic_id: str | None = None,
    ) -> list[dict]:
        filters = {
            key: value
            for key, value in {
                "subject_id": subject_id,
                "study_material_id": study_material_id,
                "course_id": course_id,
                "syllabus_id": syllabus_id,
                "topic_id": topic_id,
            }.items()
            if value is not None
        }
        query = {"embedding": {"$exists": True, "$ne": None}}
        query.update({f"metadata.{key}": value for key, value in filters.items()})
        return [
            {"chunk": DocumentChunkDocument.model_validate(item), "score": 1.0}
            for item in self.database.document_chunks.find(query).sort("chunk_index", 1)
        ]
