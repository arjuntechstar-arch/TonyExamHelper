import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from pymongo.errors import PyMongoError
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from app.models import DocumentChunkDocument


_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
})


def _tokens(text: str, *, remove_stop_words: bool = False) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    if remove_stop_words:
        return [token for token in tokens if token not in _STOP_WORDS]
    return tokens


def _quoted_phrases(query: str) -> list[list[str]]:
    return [
        tokens
        for phrase in re.findall(r'"([^"]+)"', query)
        if (tokens := _tokens(phrase))
    ]


def _contains_phrase(tokens: list[str], phrase: list[str]) -> bool:
    phrase_length = len(phrase)
    return any(
        tokens[index:index + phrase_length] == phrase
        for index in range(len(tokens) - phrase_length + 1)
    )


class EmbeddingProvider(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingConfigurationError(ValueError):
    pass


class VectorSearchError(RuntimeError):
    pass


class VectorIndexNotFoundError(VectorSearchError):
    pass


class OpenAICompatibleEmbeddingProvider:
    """Calls a configured OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str,
        model_name: str,
        dimensions: int,
        timeout_seconds: float = 30,
    ) -> None:
        if not api_base_url.strip() or not api_key.strip() or not model_name.strip():
            raise EmbeddingConfigurationError("Embedding API URL, key, and model must all be configured.")
        if dimensions < 1 or dimensions > 4096:
            raise EmbeddingConfigurationError("Embedding dimensions must be between 1 and 4096.")
        try:
            parsed_url = urlsplit(api_base_url.strip())
            hostname = parsed_url.hostname
        except ValueError as error:
            raise EmbeddingConfigurationError("The embedding API base URL is invalid.") from error
        if (
            parsed_url.scheme not in {"https", "http"}
            or not hostname
            or parsed_url.username
            or parsed_url.password
            or parsed_url.query
            or parsed_url.fragment
            or parsed_url.path.rstrip("/").lower().endswith("/embeddings")
        ):
            raise EmbeddingConfigurationError(
                "Configure an embedding API base URL without credentials, query parameters, fragments, or an /embeddings suffix."
            )
        if parsed_url.scheme != "https" and hostname.casefold() not in {"localhost", "127.0.0.1", "::1"}:
            raise EmbeddingConfigurationError(
                "The embedding API must use HTTPS unless it is running on this machine."
            )
        base_path = parsed_url.path.rstrip("/")
        self.endpoint = urlunsplit((
            parsed_url.scheme,
            parsed_url.netloc,
            f"{base_path}/embeddings",
            "",
            "",
        ))
        self.api_key = api_key.strip()
        self.model_name = model_name.strip()
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model_name, "input": texts},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            status_detail = f" (HTTP {status_code})" if status_code is not None else ""
            raise EmbeddingProviderError(
                f"The configured embedding service request failed{status_detail}."
            ) from error

        try:
            payload = response.json()
            data = payload["data"]
            if not isinstance(data, list) or len(data) != len(texts):
                raise ValueError("The embedding response count does not match the input count.")
            ordered = sorted(data, key=lambda item: item["index"])
            if [item["index"] for item in ordered] != list(range(len(texts))):
                raise ValueError("The embedding response indices are missing or duplicated.")
            vectors = [item["embedding"] for item in ordered]
            if any(not isinstance(vector, list) or len(vector) != self.dimensions for vector in vectors):
                raise ValueError("An embedding vector has an unexpected size.")
            values = [[float(value) for value in vector] for vector in vectors]
            if not all(math.isfinite(value) for vector in values for value in vector):
                raise ValueError("An embedding vector contains non-finite values.")
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise EmbeddingProviderError(
                f"The embedding service returned invalid vectors for model '{self.model_name}'."
            ) from error
        if any(not any(vector) for vector in values):
            raise EmbeddingProviderError("The embedding service returned a zero vector.")
        return values


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

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class VectorStore(Protocol):
    def upsert(self, chunk: DocumentChunkDocument) -> None: ...

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
        embedding_model: str,
    ) -> list[dict]: ...


class MongoVectorStore:
    _FILTER_PATHS = (
        "embedding_model",
        "metadata.subject_id",
        "metadata.study_material_id",
        "metadata.course_id",
        "metadata.syllabus_id",
        "metadata.topic_id",
    )

    def __init__(self, database: Database, *, vector_index_name: str = "document_chunks_vector_v1") -> None:
        self.database = database
        self.vector_index_name = vector_index_name

    def upsert(self, chunk: DocumentChunkDocument) -> None:
        self.database.document_chunks.replace_one(
            {"_id": chunk.id},
            chunk.model_dump(by_alias=True),
            upsert=True,
        )

    def candidates(self, *, filters: dict[str, str] | None = None) -> list[DocumentChunkDocument]:
        query = {f"metadata.{key}": value for key, value in (filters or {}).items()}
        return [
            DocumentChunkDocument.model_validate(item)
            for item in self.database.document_chunks.find(query)
        ]

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
        embedding_model: str,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        vector_filter: dict[str, object] = {"embedding_model": {"$eq": embedding_model}}
        vector_filter.update({
            f"metadata.{key}": {"$eq": value}
            for key, value in (filters or {}).items()
        })
        candidate_limit = min(150, max(top_k * 4, top_k))
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(candidate_limit * 20, candidate_limit),
                    "limit": candidate_limit,
                    "filter": vector_filter,
                },
            },
            {"$addFields": {"vector_score": {"$meta": "vectorSearchScore"}}},
        ]
        try:
            matches = self.database.document_chunks.aggregate(pipeline)
            return [
                {
                    "chunk": DocumentChunkDocument.model_validate(item),
                    "score": float(item["vector_score"]),
                    "semantic_score": float(item["vector_score"]),
                }
                for item in matches
            ]
        except PyMongoError as error:
            raise VectorSearchError(
                f"MongoDB vector search failed. Ensure Atlas Vector Search index "
                f"'{self.vector_index_name}' is ready and matches the configured embedding dimensions."
            ) from error

    def vector_index_status(self, *, dimensions: int) -> dict[str, str | bool]:
        collection = self.database.document_chunks
        try:
            indexes = list(collection.list_search_indexes())
            existing = next((item for item in indexes if item.get("name") == self.vector_index_name), None)
            if existing is None:
                raise VectorIndexNotFoundError(
                    f"Atlas Vector Search index '{self.vector_index_name}' does not exist. Create it before indexing materials."
                )
            if existing.get("type") not in (None, "vectorSearch"):
                raise VectorSearchError(
                    f"Atlas index '{self.vector_index_name}' exists but is not a vectorSearch index."
                )
            definition = existing.get("latestDefinition") or existing.get("definition") or {}
            vector_field = next(
                (
                    field for field in definition.get("fields", [])
                    if field.get("type") == "vector" and field.get("path") == "embedding"
                ),
                None,
            )
            indexed_filter_paths = {
                field.get("path")
                for field in definition.get("fields", [])
                if field.get("type") == "filter"
            }
            if (
                vector_field is None
                or vector_field.get("numDimensions") != dimensions
                or vector_field.get("similarity") != "cosine"
                or not set(self._FILTER_PATHS).issubset(indexed_filter_paths)
            ):
                raise VectorSearchError(
                    f"Atlas Vector Search index '{self.vector_index_name}' exists with an incompatible vector or filter definition."
                )
            return {
                "name": self.vector_index_name,
                "status": str(existing.get("status", "BUILDING")),
                "queryable": bool(existing.get("queryable", False)),
            }
        except PyMongoError as error:
            raise VectorSearchError(
                "Could not create or inspect the Atlas Vector Search index. "
                "Semantic retrieval requires MongoDB Atlas or a MongoDB deployment with Vector Search enabled."
            ) from error

    def setup_vector_index(self, *, dimensions: int) -> dict[str, str | bool]:
        try:
            return self.vector_index_status(dimensions=dimensions)
        except VectorIndexNotFoundError:
            pass
        definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                },
                *({"type": "filter", "path": path} for path in self._FILTER_PATHS),
            ],
        }
        try:
            self.database.document_chunks.create_search_index(
                SearchIndexModel(
                    definition=definition,
                    name=self.vector_index_name,
                    type="vectorSearch",
                )
            )
        except PyMongoError as error:
            raise VectorSearchError(
                "Could not create the Atlas Vector Search index. "
                "Semantic retrieval requires MongoDB Atlas or a MongoDB deployment with Vector Search enabled."
            ) from error
        return {"name": self.vector_index_name, "status": "BUILDING", "queryable": False}

    def require_vector_index_ready(self, *, dimensions: int) -> None:
        try:
            state = self.vector_index_status(dimensions=dimensions)
        except VectorIndexNotFoundError as error:
            raise VectorSearchError(
                f"{error} Use POST /api/retrieval/vector-index to create the index."
            ) from error
        if not state["queryable"]:
            raise VectorSearchError(
                f"Atlas Vector Search index '{self.vector_index_name}' is {state['status']}; wait for it to become READY before indexing or searching."
            )


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
    """A stop-word-filtered keyword-overlap signal."""
    left_tokens = set(_tokens(left, remove_stop_words=True))
    right_tokens = set(_tokens(right, remove_stop_words=True))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def bm25_scores(query: str, chunks: list[DocumentChunkDocument]) -> list[float]:
    """Score query terms against chunks with corpus-aware BM25 weighting."""
    query_terms = set(_tokens(query, remove_stop_words=True))
    if not query_terms or not chunks:
        return [0.0] * len(chunks)

    token_counts = [
        Counter(_tokens(chunk.content, remove_stop_words=True))
        for chunk in chunks
    ]
    document_frequencies = Counter(
        term
        for counts in token_counts
        for term in counts.keys() & query_terms
    )
    average_length = sum(sum(counts.values()) for counts in token_counts) / len(token_counts)
    if average_length == 0:
        return [0.0] * len(chunks)

    document_count = len(chunks)
    k1 = 1.5
    b = 0.75
    scores = []
    for counts in token_counts:
        document_length = sum(counts.values())
        score = 0.0
        for term in query_terms:
            term_frequency = counts[term]
            if not term_frequency:
                continue
            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_normalizer = k1 * (1 - b + b * document_length / average_length)
            score += inverse_document_frequency * (
                term_frequency * (k1 + 1) / (term_frequency + length_normalizer)
            )
        scores.append(score)
    return scores


def rerank_hybrid(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Fuse available vector, BM25, and keyword rankings, then diversify results.

    Lexical-only mode remains independent of external model or search services.
    When semantic candidates are supplied, reciprocal-rank fusion combines
    dense and sparse rankings before the MMR-style diversity pass.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not candidates:
        return []

    chunks = [candidate["chunk"] for candidate in candidates]
    phrases = _quoted_phrases(query)
    lexical_scores = bm25_scores(query, chunks)
    overlap_scores = [
        lexical_similarity(query, chunk.content)
        for chunk in chunks
    ]
    phrase_scores = [
        sum(
            _contains_phrase(_tokens(chunk.content), phrase)
            for phrase in phrases
        ) / len(phrases)
        if phrases else 0.0
        for chunk in chunks
    ]
    phrase_ranking = sorted(
        ((index, score) for index, score in enumerate(phrase_scores) if score > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    overlap_ranking = sorted(
        ((index, score) for index, score in enumerate(overlap_scores) if score > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    lexical_ranking = sorted(
        ((index, score) for index, score in enumerate(lexical_scores) if score > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    candidate_pool_size = min(150, max(top_k * 4, top_k))
    overlap_ranks = {
        index: rank
        for rank, (index, _) in enumerate(overlap_ranking[:candidate_pool_size], start=1)
    }
    semantic_ranking = sorted(
        (
            (index, candidate["semantic_score"])
            for index, candidate in enumerate(candidates)
            if isinstance(candidate.get("semantic_score"), (int, float))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    semantic_ranks = {
        index: rank
        for rank, (index, _) in enumerate(semantic_ranking[:candidate_pool_size], start=1)
    }
    lexical_ranks = {
        index: rank
        for rank, (index, _) in enumerate(lexical_ranking[:candidate_pool_size], start=1)
    }
    selected_indexes = dict.fromkeys([
        *(index for index, _ in phrase_ranking[:candidate_pool_size]),
        *overlap_ranks,
        *lexical_ranks,
        *semantic_ranks,
    ])
    scored = []
    for index in selected_indexes:
        overlap_rank = overlap_ranks.get(index)
        lexical_rank = lexical_ranks.get(index)
        lexical_score = lexical_scores[index]
        overlap_score = overlap_scores[index]
        sparse_score = (
            (0.5 * 61 / (60 + overlap_rank) if overlap_rank is not None else 0)
            + (0.5 * 61 / (60 + lexical_rank) if lexical_rank is not None else 0)
        )
        semantic_rank = semantic_ranks.get(index)
        semantic_score = candidates[index].get("semantic_score")
        if semantic_rank is not None:
            semantic_rrf = 61 / (60 + semantic_rank)
            hybrid_score = 0.65 * semantic_rrf + 0.35 * sparse_score
            if phrases:
                hybrid_score = 0.55 * semantic_rrf + 0.30 * sparse_score + 0.15 * phrase_scores[index]
        else:
            hybrid_score = 0.7 * phrase_scores[index] + 0.3 * sparse_score if phrases else sparse_score
        scored.append({
            **candidates[index],
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "overlap_score": overlap_score,
            "phrase_score": phrase_scores[index],
            "hybrid_score": hybrid_score,
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
    def __init__(
        self,
        database: Database,
        embedding_provider: EmbeddingProvider | None = None,
        *,
        vector_index_name: str = "document_chunks_vector_v1",
    ) -> None:
        self.database = database
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.semantic_enabled = not isinstance(self.embedding_provider, HashEmbeddingProvider)
        self.vector_store = MongoVectorStore(database, vector_index_name=vector_index_name)

    def _validate_embedding(self, vector: list[float]) -> list[float]:
        dimensions = getattr(self.embedding_provider, "dimensions", None)
        if not isinstance(dimensions, int) or len(vector) != dimensions:
            raise EmbeddingProviderError("The embedding provider returned a vector with an unexpected size.")
        if not all(math.isfinite(value) for value in vector) or not any(vector):
            raise EmbeddingProviderError("The embedding provider returned a non-finite or zero vector.")
        return vector

    def setup_vector_index(self) -> dict[str, str | bool]:
        if not self.semantic_enabled:
            raise EmbeddingConfigurationError(
                "Configure an embedding API URL, API key, model, and vector dimensions before setting up semantic retrieval."
            )
        dimensions = getattr(self.embedding_provider, "dimensions", None)
        if not isinstance(dimensions, int):
            raise EmbeddingConfigurationError("The configured embedding provider must expose its vector dimensions.")
        return self.vector_store.setup_vector_index(dimensions=dimensions)

    def vector_index_status(self) -> dict[str, str | bool]:
        if not self.semantic_enabled:
            raise EmbeddingConfigurationError(
                "Configure an embedding API URL, API key, model, and vector dimensions before checking semantic retrieval."
            )
        dimensions = getattr(self.embedding_provider, "dimensions", None)
        if not isinstance(dimensions, int):
            raise EmbeddingConfigurationError("The configured embedding provider must expose its vector dimensions.")
        return self.vector_store.vector_index_status(dimensions=dimensions)

    def index_material(self, material_id: str) -> int:
        if self.semantic_enabled:
            dimensions = getattr(self.embedding_provider, "dimensions", None)
            if not isinstance(dimensions, int):
                raise EmbeddingConfigurationError("The configured embedding provider must expose its vector dimensions.")
            self.vector_store.require_vector_index_ready(dimensions=dimensions)
        material = self.database.study_materials.find_one({"_id": material_id})
        if material is None:
            material = self.database.study_materials.find_one({"id": material_id})
        if material is None:
            raise ValueError("Material not found.")
        resolved_material_id = str(material.get("id", material.get("_id", material_id)))
        chunks = list(self.database.document_chunks.find({"study_material_id": resolved_material_id}))
        indexed_chunks: list[DocumentChunkDocument] = []
        for start in range(0, len(chunks), 64):
            batch = [DocumentChunkDocument.model_validate(item) for item in chunks[start:start + 64]]
            embeddings = self.embedding_provider.embed_many([chunk.content for chunk in batch])
            if len(embeddings) != len(batch):
                raise EmbeddingProviderError("The embedding provider returned the wrong number of vectors.")
            for chunk, embedding in zip(batch, embeddings, strict=True):
                chunk.embedding = self._validate_embedding(embedding)
                chunk.embedding_model = self.embedding_provider.model_name
                indexed_chunks.append(chunk)
        for chunk in indexed_chunks:
            self.vector_store.upsert(chunk)
        count = len(indexed_chunks)
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
        if self.semantic_enabled:
            query_embedding = self._validate_embedding(self.embedding_provider.embed(query))
            candidates = self.vector_store.search(
                query_embedding,
                top_k=top_k,
                filters=filters,
                embedding_model=self.embedding_provider.model_name,
            )
            return rerank_hybrid(query, candidates, top_k)
        candidates = []
        for chunk in self.vector_store.candidates(filters=filters):
            candidates.append({"chunk": chunk})
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
        if self.semantic_enabled:
            query["embedding_model"] = self.embedding_provider.model_name
        query.update({f"metadata.{key}": value for key, value in filters.items()})
        return [
            {"chunk": DocumentChunkDocument.model_validate(item), "score": 1.0}
            for item in self.database.document_chunks.find(query).sort("chunk_index", 1)
        ]


def embedding_provider_from_settings(settings: object) -> EmbeddingProvider:
    api_base_url = getattr(settings, "retrieval_embedding_api_base_url", None)
    api_key = getattr(settings, "retrieval_embedding_api_key", None)
    if not api_base_url and not api_key:
        return HashEmbeddingProvider()
    if not api_base_url or not api_key:
        raise EmbeddingConfigurationError(
            "RETRIEVAL_EMBEDDING_API_BASE_URL and RETRIEVAL_EMBEDDING_API_KEY must be configured together."
        )
    model_name = getattr(settings, "retrieval_embedding_model", None)
    dimensions = getattr(settings, "retrieval_embedding_dimensions", None)
    if not model_name or dimensions is None:
        raise EmbeddingConfigurationError(
            "RETRIEVAL_EMBEDDING_MODEL and RETRIEVAL_EMBEDDING_DIMENSIONS must be configured with the embedding API."
        )
    return OpenAICompatibleEmbeddingProvider(
        api_base_url=api_base_url,
        api_key=api_key,
        model_name=model_name,
        dimensions=dimensions,
        timeout_seconds=getattr(settings, "retrieval_embedding_timeout_seconds", 30),
    )
