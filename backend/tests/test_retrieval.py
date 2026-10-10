import mongomock
import pytest
from types import SimpleNamespace

from app.models import DocumentChunkDocument, StudyMaterialDocument
from app.services.retrieval import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    HashEmbeddingProvider,
    MongoVectorStore,
    OpenAICompatibleEmbeddingProvider,
    RetrievalService,
    VectorIndexNotFoundError,
    VectorSearchError,
    bm25_scores,
    cosine_similarity,
    embedding_provider_from_settings,
    lexical_similarity,
    rerank_hybrid,
)


@pytest.fixture
def retrieval_service(tmp_path) -> RetrievalService:
    database = mongomock.MongoClient().test
    material = StudyMaterialDocument(
        subject_id="subject-1",
        filename="algorithms.txt",
        storage_key="algorithms.txt",
        content_type="text/plain",
        size_bytes=10,
        status="processed",
    )
    database.study_materials.insert_one(material.model_dump(by_alias=True))
    chunks = [
        DocumentChunkDocument(
            study_material_id=material.id,
            chunk_index=0,
            page_number=1,
            content="Binary trees have nodes and child pointers.",
            metadata={"source_file": "algorithms.txt", "subject_id": "subject-1"},
        ),
        DocumentChunkDocument(
            study_material_id=material.id,
            chunk_index=1,
            page_number=2,
            content="Relational databases organize data into tables.",
            metadata={"source_file": "algorithms.txt", "subject_id": "subject-1"},
        ),
        DocumentChunkDocument(
            study_material_id=material.id,
            chunk_index=2,
            page_number=3,
            content="Binary trees are also used in another subject.",
            metadata={"source_file": "other.txt", "subject_id": "subject-2"},
        ),
    ]
    database.document_chunks.insert_many([chunk.model_dump(by_alias=True) for chunk in chunks])
    return RetrievalService(database)


def test_hash_embeddings_are_normalized_and_deterministic() -> None:
    provider = HashEmbeddingProvider(dimensions=32)

    first = provider.embed("Binary trees")
    second = provider.embed("Binary trees")

    assert first == second
    assert cosine_similarity(first, first) == pytest.approx(1.0)


def test_embedding_provider_batches_and_restores_response_order(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
            }

    captured: dict = {}

    def fake_post(url: str, **kwargs) -> Response:
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("app.services.retrieval.httpx.post", fake_post)
    provider = OpenAICompatibleEmbeddingProvider(
        api_base_url="https://embedding.example/v1",
        api_key="test-only-secret",
        model_name="test-embedder",
        dimensions=2,
    )

    vectors = provider.embed_many(["first", "second"])

    assert captured["url"] == "https://embedding.example/v1/embeddings"
    assert captured["json"] == {"model": "test-embedder", "input": ["first", "second"]}
    assert captured["headers"]["Authorization"] == "Bearer test-only-secret"
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_embedding_provider_requires_secure_remote_base_url() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="HTTPS"):
        OpenAICompatibleEmbeddingProvider(
            api_base_url="http://embedding.example/v1",
            api_key="test-only-secret",
            model_name="test-embedder",
            dimensions=2,
        )

    local_provider = OpenAICompatibleEmbeddingProvider(
        api_base_url="http://localhost:8001/v1/",
        api_key="test-only-secret",
        model_name="test-embedder",
        dimensions=2,
    )
    assert local_provider.endpoint == "http://localhost:8001/v1/embeddings"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:password@embedding.example/v1",
        "https://embedding.example/v1?api-version=1",
        "https://embedding.example/v1/embeddings",
    ],
)
def test_embedding_provider_rejects_ambiguous_base_urls(base_url: str) -> None:
    with pytest.raises(EmbeddingConfigurationError, match="base URL"):
        OpenAICompatibleEmbeddingProvider(
            api_base_url=base_url,
            api_key="test-only-secret",
            model_name="test-embedder",
            dimensions=2,
        )


def test_embedding_provider_rejects_invalid_dimensions_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(EmbeddingConfigurationError, match="dimensions"):
        OpenAICompatibleEmbeddingProvider(
            api_base_url="https://embedding.example/v1",
            api_key="test-only-secret",
            model_name="test-embedder",
            dimensions=0,
        )

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [float("nan"), 1.0]}]}

    monkeypatch.setattr("app.services.retrieval.httpx.post", lambda *args, **kwargs: Response())
    provider = OpenAICompatibleEmbeddingProvider(
        api_base_url="https://embedding.example/v1",
        api_key="test-only-secret",
        model_name="test-embedder",
        dimensions=2,
    )
    with pytest.raises(EmbeddingProviderError, match="invalid vectors"):
        provider.embed("query")


def test_embedding_provider_configuration_is_opt_in() -> None:
    lexical = embedding_provider_from_settings(SimpleNamespace(
        retrieval_embedding_api_base_url=None,
        retrieval_embedding_api_key=None,
    ))
    assert isinstance(lexical, HashEmbeddingProvider)

    with pytest.raises(EmbeddingConfigurationError, match="configured together"):
        embedding_provider_from_settings(SimpleNamespace(
            retrieval_embedding_api_base_url="https://embedding.example/v1",
            retrieval_embedding_api_key=None,
        ))
    with pytest.raises(EmbeddingConfigurationError, match="MODEL and RETRIEVAL_EMBEDDING_DIMENSIONS"):
        embedding_provider_from_settings(SimpleNamespace(
            retrieval_embedding_api_base_url="https://embedding.example/v1",
            retrieval_embedding_api_key="test-only-secret",
            retrieval_embedding_model=None,
            retrieval_embedding_dimensions=None,
        ))
    configured = embedding_provider_from_settings(SimpleNamespace(
        retrieval_embedding_api_base_url="https://embedding.example/v1",
        retrieval_embedding_api_key="test-only-secret",
        retrieval_embedding_model="test-embedder",
        retrieval_embedding_dimensions=768,
        retrieval_embedding_timeout_seconds=45,
    ))
    assert isinstance(configured, OpenAICompatibleEmbeddingProvider)
    assert configured.dimensions == 768
    assert configured.timeout_seconds == 45


def test_hybrid_reranking_exposes_component_scores_and_diversifies_results() -> None:
    candidates = [
        {"chunk": DocumentChunkDocument(study_material_id="m", chunk_index=0, page_number=1, content="Binary tree traversal visits nodes."), "score": 0.9},
        {"chunk": DocumentChunkDocument(study_material_id="m", chunk_index=1, page_number=2, content="Binary tree traversal visits nodes in order."), "score": 0.89},
        {"chunk": DocumentChunkDocument(study_material_id="m", chunk_index=2, page_number=3, content="A database table stores related records."), "score": 0.6},
    ]

    results = rerank_hybrid("binary tree traversal", candidates, top_k=2)

    assert len(results) == 2
    assert all("semantic_score" in result and "lexical_score" in result for result in results)
    assert results[0]["chunk"].content.startswith("Binary tree")


def test_hybrid_reranking_uses_semantic_rank_for_paraphrased_evidence() -> None:
    candidates = [
        {
            "chunk": DocumentChunkDocument(
                study_material_id="m",
                chunk_index=0,
                page_number=1,
                content="Felines are obligate carnivorous mammals.",
            ),
            "semantic_score": 0.94,
        },
        {
            "chunk": DocumentChunkDocument(
                study_material_id="m",
                chunk_index=1,
                page_number=2,
                content="Cats are commonly kept as household pets.",
            ),
            "semantic_score": 0.38,
        },
    ]

    results = rerank_hybrid("What kind of animal is a cat?", candidates, top_k=2)

    assert results[0]["chunk"].chunk_index == 0
    assert results[0]["semantic_score"] == 0.94
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


def test_mongo_vector_store_uses_atlas_ann_filter_and_score() -> None:
    chunk = DocumentChunkDocument(
        study_material_id="material-1",
        chunk_index=2,
        page_number=3,
        content="Cats are mammals.",
        embedding=[0.1, 0.9],
        embedding_model="test-embedder",
    )

    class Collection:
        pipeline: list[dict] | None = None

        def aggregate(self, pipeline: list[dict]) -> list[dict]:
            self.pipeline = pipeline
            return [{**chunk.model_dump(by_alias=True), "vector_score": 0.88}]

    collection = Collection()
    store = MongoVectorStore(SimpleNamespace(document_chunks=collection), vector_index_name="test-vector-index")

    results = store.search(
        [0.2, 0.8],
        top_k=5,
        filters={"subject_id": "subject-1"},
        embedding_model="test-embedder",
    )

    vector_stage = collection.pipeline[0]["$vectorSearch"]
    assert vector_stage["index"] == "test-vector-index"
    assert vector_stage["path"] == "embedding"
    assert vector_stage["numCandidates"] >= vector_stage["limit"] * 20
    assert vector_stage["filter"] == {
        "embedding_model": {"$eq": "test-embedder"},
        "metadata.subject_id": {"$eq": "subject-1"},
    }
    assert results[0]["chunk"].id == chunk.id
    assert results[0]["semantic_score"] == pytest.approx(0.88)


def test_vector_index_setup_creates_filterable_atlas_vector_index() -> None:
    class Collection:
        created_model = None

        def list_search_indexes(self) -> list[dict]:
            return []

        def create_search_index(self, model) -> str:
            self.created_model = model
            return "document_chunks_vector_v1"

    collection = Collection()
    store = MongoVectorStore(SimpleNamespace(document_chunks=collection))

    state = store.setup_vector_index(dimensions=768)

    assert state == {
        "name": "document_chunks_vector_v1",
        "status": "BUILDING",
        "queryable": False,
    }
    assert collection.created_model is not None


def test_vector_index_status_requires_matching_ready_definition() -> None:
    class Collection:
        def list_search_indexes(self) -> list[dict]:
            return [{
                "name": "document_chunks_vector_v1",
                "type": "vectorSearch",
                "status": "READY",
                "queryable": True,
                "latestDefinition": {"fields": [
                    {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine"},
                    *({"type": "filter", "path": path} for path in MongoVectorStore._FILTER_PATHS),
                ]},
            }]

    store = MongoVectorStore(SimpleNamespace(document_chunks=Collection()))
    assert store.vector_index_status(dimensions=768)["queryable"] is True
    with pytest.raises(VectorSearchError, match="incompatible"):
        store.vector_index_status(dimensions=1024)

    class MissingCollection:
        def list_search_indexes(self) -> list[dict]:
            return []

    with pytest.raises(VectorIndexNotFoundError, match="does not exist"):
        MongoVectorStore(SimpleNamespace(document_chunks=MissingCollection())).vector_index_status(dimensions=768)


def test_stop_words_do_not_change_bm25_or_keyword_overlap() -> None:
    chunks = [
        DocumentChunkDocument(
            study_material_id="m",
            chunk_index=0,
            page_number=1,
            content="The binary tree stores nodes and child pointers.",
        ),
        DocumentChunkDocument(
            study_material_id="m",
            chunk_index=1,
            page_number=2,
            content="A relational database stores rows in tables.",
        ),
    ]

    assert bm25_scores("the binary tree", chunks) == bm25_scores("binary tree", chunks)
    assert lexical_similarity("the binary tree", chunks[0].content) == lexical_similarity(
        "binary tree", chunks[0].content
    )


def test_quoted_phrase_ranks_exact_match_above_term_only_match() -> None:
    candidates = [
        {
            "chunk": DocumentChunkDocument(
                study_material_id="m",
                chunk_index=0,
                page_number=1,
                content="Traversal can inspect a binary tree in several distinct orders.",
            ),
        },
        {
            "chunk": DocumentChunkDocument(
                study_material_id="m",
                chunk_index=1,
                page_number=2,
                content="Binary tree traversal visits each node exactly once.",
            ),
        },
    ]

    results = rerank_hybrid('"binary tree traversal"', candidates, top_k=2)

    assert results[0]["chunk"].content.startswith("Binary tree traversal")
    assert results[0]["phrase_score"] == 1.0
    assert results[1]["phrase_score"] == 0.0


def test_quoted_phrase_can_match_stop_word_only_phrase() -> None:
    candidates = [
        {
            "chunk": DocumentChunkDocument(
                study_material_id="m",
                chunk_index=0,
                page_number=1,
                content="To be or not to be is the central question.",
            ),
        },
        {
            "chunk": DocumentChunkDocument(
                study_material_id="m",
                chunk_index=1,
                page_number=2,
                content="Binary tree traversal visits nodes.",
            ),
        },
    ]

    results = rerank_hybrid('"to be"', candidates, top_k=2)

    assert len(results) == 1
    assert results[0]["phrase_score"] == 1.0


def test_retrieval_finds_relevant_unindexed_chunks(retrieval_service: RetrievalService) -> None:
    chunk = DocumentChunkDocument(
        study_material_id="unindexed-material",
        chunk_index=0,
        page_number=1,
        content="Topological sorting orders vertices in a directed acyclic graph.",
        metadata={"subject_id": "subject-1"},
    )
    retrieval_service.database.document_chunks.insert_one(chunk.model_dump(by_alias=True))

    results = retrieval_service.retrieve(
        "directed acyclic graph topological sorting",
        top_k=2,
        subject_id="subject-1",
    )

    assert results
    assert results[0]["chunk"].id == chunk.id
    assert results[0]["lexical_score"] > 0


def test_retrieval_returns_no_unrelated_chunks(retrieval_service: RetrievalService) -> None:
    results = retrieval_service.retrieve("quantum algorithms", top_k=5)

    assert results == []


def test_indexing_persists_embeddings_and_retrieval_ranks_matching_content(retrieval_service: RetrievalService) -> None:
    count = retrieval_service.index_material(next(retrieval_service.database.study_materials.find()) ["_id"])

    results = retrieval_service.retrieve("binary trees", top_k=2, subject_id="subject-1")

    assert count == 3
    assert results[0]["chunk"].content.startswith("Binary trees")
    assert len(results) == 1
    assert all(result["chunk"].embedding_model == "hash-embedding-v1" for result in results)


def test_semantic_retrieval_uses_query_embedding_and_atlas_candidates(
    retrieval_service: RetrievalService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TestEmbeddingProvider:
        model_name = "test-embedder"
        dimensions = 2

        def embed(self, text: str) -> list[float]:
            assert text == "household feline"
            return [0.4, 0.6]

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            return [[0.4, 0.6] for _ in texts]

    service = RetrievalService(retrieval_service.database, TestEmbeddingProvider())
    vector_chunk = DocumentChunkDocument(
        study_material_id="material-1",
        chunk_index=10,
        page_number=10,
        content="Felines are obligate carnivorous mammals.",
    )
    observed: dict = {}

    def search(query_embedding, *, top_k, filters, embedding_model):
        observed.update({
            "query_embedding": query_embedding,
            "top_k": top_k,
            "filters": filters,
            "embedding_model": embedding_model,
        })
        return [{"chunk": vector_chunk, "score": 0.9, "semantic_score": 0.9}]

    monkeypatch.setattr(service.vector_store, "search", search)
    results = service.retrieve("household feline", top_k=3, subject_id="subject-1")

    assert service.semantic_enabled
    assert observed == {
        "query_embedding": [0.4, 0.6],
        "top_k": 3,
        "filters": {"subject_id": "subject-1"},
        "embedding_model": "test-embedder",
    }
    assert results[0]["chunk"].id == vector_chunk.id


def test_semantic_material_indexing_batches_embeddings(
    retrieval_service: RetrievalService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BatchEmbeddingProvider:
        model_name = "batch-test-embedder"
        dimensions = 2

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            self.batch_sizes.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

    provider = BatchEmbeddingProvider()
    service = RetrievalService(retrieval_service.database, provider)
    monkeypatch.setattr(service.vector_store, "require_vector_index_ready", lambda **kwargs: None)

    material_id = next(retrieval_service.database.study_materials.find())["_id"]
    assert service.index_material(material_id) == 3
    assert provider.batch_sizes == [3]
    indexed = list(retrieval_service.database.document_chunks.find({"study_material_id": material_id}))
    assert all(item["embedding_model"] == "batch-test-embedder" for item in indexed)
    assert all(item["embedding"] == [1.0, 0.0] for item in indexed)


def test_semantic_indexing_does_not_replace_vectors_if_a_later_batch_fails(
    retrieval_service: RetrievalService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingBatchProvider:
        model_name = "replacement-model"
        dimensions = 2

        def __init__(self) -> None:
            self.batch_count = 0

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            self.batch_count += 1
            if self.batch_count == 2:
                raise EmbeddingProviderError("The provider rejected the second batch.")
            return [[1.0, 0.0] for _ in texts]

    material_id = next(retrieval_service.database.study_materials.find())["_id"]
    retrieval_service.database.document_chunks.update_many(
        {"study_material_id": material_id},
        {"$set": {"embedding": [0.0, 1.0], "embedding_model": "existing-model"}},
    )
    extra_chunks = [
        DocumentChunkDocument(
            study_material_id=material_id,
            chunk_index=index + 3,
            page_number=index + 4,
            content=f"Additional indexed passage {index}.",
            embedding=[0.0, 1.0],
            embedding_model="existing-model",
        )
        for index in range(62)
    ]
    retrieval_service.database.document_chunks.insert_many([
        chunk.model_dump(by_alias=True) for chunk in extra_chunks
    ])

    service = RetrievalService(retrieval_service.database, FailingBatchProvider())
    monkeypatch.setattr(service.vector_store, "require_vector_index_ready", lambda **kwargs: None)

    with pytest.raises(EmbeddingProviderError, match="second batch"):
        service.index_material(material_id)

    stored_chunks = list(retrieval_service.database.document_chunks.find({"study_material_id": material_id}))
    assert len(stored_chunks) == 65
    assert all(chunk["embedding_model"] == "existing-model" for chunk in stored_chunks)
    assert all(chunk["embedding"] == [0.0, 1.0] for chunk in stored_chunks)


def test_retrieval_filters_out_chunks_from_other_subjects(retrieval_service: RetrievalService) -> None:
    material_id = next(retrieval_service.database.study_materials.find())["_id"]
    retrieval_service.index_material(material_id)

    results = retrieval_service.retrieve("binary trees", top_k=10, subject_id="subject-1")

    assert len(results) == 1
    assert all(result["chunk"].metadata["subject_id"] == "subject-1" for result in results)


def test_indexing_accepts_material_documents_with_id_field(retrieval_service: RetrievalService) -> None:
    material = next(retrieval_service.database.study_materials.find())
    retrieval_service.database.study_materials.delete_many({})
    material["id"] = material.pop("_id")
    retrieval_service.database.study_materials.insert_one(material)

    assert retrieval_service.index_material(material["id"]) == 3
    assert retrieval_service.database.study_materials.find_one({"id": material["id"]})["status"] == "indexed"
