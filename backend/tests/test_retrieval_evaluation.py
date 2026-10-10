import mongomock
import pytest
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.core.database import get_database
from app.main import app
from app.models import DocumentChunkDocument, UserDocument
from app.services.retrieval import RetrievalService
from app.services.retrieval_evaluation import evaluate_ranking, evaluate_retrieval_benchmark


def test_evaluate_ranking_calculates_precision_recall_mrr_and_ndcg() -> None:
    metrics = evaluate_ranking(
        ["irrelevant", "relevant-a", "relevant-b"],
        {"relevant-a", "relevant-b"},
        k=3,
    )

    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert metrics["mrr@3"] == pytest.approx(0.5)
    assert metrics["ndcg@3"] == pytest.approx(
        (1 / 2 + 1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)
    )


@pytest.mark.parametrize(
    ("retrieved_ids", "relevant_ids", "k"),
    [
        (["relevant"], {"relevant"}, 0),
        (["irrelevant"], set(), 5),
    ],
)
def test_evaluate_ranking_rejects_invalid_inputs(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> None:
    with pytest.raises(ValueError):
        evaluate_ranking(retrieved_ids, relevant_ids, k=k)


def test_evaluate_retrieval_benchmark_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="At least one benchmark case"):
        evaluate_retrieval_benchmark(
            [],
            lambda query, k: [{"query": query, "k": k}],
            k=5,
        )


def test_retrieval_benchmark_measures_labeled_search_results() -> None:
    database = mongomock.MongoClient().test
    chunks = [
        DocumentChunkDocument(
            id="tree-intro",
            study_material_id="trees",
            chunk_index=0,
            page_number=1,
            content="Binary trees consist of nodes connected by parent and child links.",
            metadata={"subject_id": "algorithms"},
        ),
        DocumentChunkDocument(
            id="tree-traversal",
            study_material_id="trees",
            chunk_index=1,
            page_number=2,
            content="Tree traversal visits each node using preorder, inorder, or postorder.",
            metadata={"subject_id": "algorithms"},
        ),
        DocumentChunkDocument(
            id="database-index",
            study_material_id="databases",
            chunk_index=0,
            page_number=1,
            content="Database indexes speed up record lookup with an ordered structure.",
            metadata={"subject_id": "databases"},
        ),
        DocumentChunkDocument(
            id="graph-traversal",
            study_material_id="graphs",
            chunk_index=0,
            page_number=1,
            content="Graph traversal explores vertices using breadth-first or depth-first search.",
            metadata={"subject_id": "algorithms"},
        ),
    ]
    database.document_chunks.insert_many(
        [chunk.model_dump(by_alias=True) for chunk in chunks]
    )
    service = RetrievalService(database)
    cases = [
        {"query": "binary tree nodes", "relevant_chunk_ids": ["tree-intro"]},
        {"query": "tree traversal preorder", "relevant_chunk_ids": ["tree-traversal"]},
        {"query": "database indexes lookup", "relevant_chunk_ids": ["database-index"]},
        {"query": "graph traversal breadth-first", "relevant_chunk_ids": ["graph-traversal"]},
    ]

    result = evaluate_retrieval_benchmark(
        cases,
        lambda query, k: service.retrieve(query, top_k=k),
        k=2,
    )

    assert result["query_count"] == 4
    assert result["k"] == 2
    assert result["metrics"]["recall@2"] == pytest.approx(1.0)
    assert result["metrics"]["mrr@2"] >= 0.875
    assert result["metrics"]["ndcg@2"] >= 0.9
    assert len(result["per_query"]) == 4
    assert result["per_query"][0]["retrieved_chunk_ids"]
    assert result["per_query"][0]["relevant_chunk_ids"] == ["tree-intro"]


def test_retrieval_benchmark_api_is_scoped_and_restricted_to_faculty_or_admin() -> None:
    database = mongomock.MongoClient().test
    database.document_chunks.insert_one(
        DocumentChunkDocument(
            id="tree-intro",
            study_material_id="trees",
            chunk_index=0,
            page_number=1,
            content="Binary trees consist of nodes connected by parent and child links.",
            metadata={"subject_id": "algorithms"},
        ).model_dump(by_alias=True)
    )
    faculty = UserDocument(
        email="faculty@example.edu",
        display_name="Faculty",
        password_hash="unused",
        roles=["faculty"],
    )
    student = UserDocument(
        email="student@example.edu",
        display_name="Student",
        password_hash="unused",
        roles=["student"],
    )
    app.dependency_overrides[get_database] = lambda: database
    app.dependency_overrides[get_current_user] = lambda: faculty
    try:
        with TestClient(app) as client:
            payload = {
                "cases": [{
                    "query": "binary trees nodes",
                    "relevant_chunk_ids": ["tree-intro"],
                }],
                "k": 1,
                "subject_id": "algorithms",
            }
            response = client.post("/api/retrieval/evaluate", json=payload)

            assert response.status_code == 200
            assert response.json()["metrics"]["recall@1"] == 1.0
            assert response.json()["per_query"][0]["retrieved_chunk_ids"] == ["tree-intro"]

            unscoped = client.post(
                "/api/retrieval/evaluate",
                json={**payload, "subject_id": None},
            )
            assert unscoped.status_code == 422
            validation_error = unscoped.json()
            assert validation_error["error"] == "validation_error"
            assert validation_error["details"][0]["loc"] == ["body"]
            assert "ctx" not in validation_error["details"][0]
            assert "input" not in validation_error["details"][0]

            app.dependency_overrides[get_current_user] = lambda: student
            forbidden = client.post("/api/retrieval/evaluate", json=payload)
            assert forbidden.status_code == 403
    finally:
        app.dependency_overrides.clear()
