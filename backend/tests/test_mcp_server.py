import asyncio
import json
import os
from pathlib import Path
import sys

import httpx
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.mcp_server import (
    TonyExamApiClient,
    TonyExamApiError,
    TonyExamMcpTools,
    create_mcp_server,
)


def _run(coroutine):
    return asyncio.run(coroutine)


def _api_client(handler) -> TonyExamApiClient:
    return TonyExamApiClient(
        api_url="http://localhost:8000",
        access_token="Bearer test-access-token",
        transport=httpx.MockTransport(handler),
    )


def test_search_materials_calls_authenticated_retrieval_api() -> None:
    observed: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["method"] = request.method
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers["Authorization"]
        observed["body"] = json.loads(request.content)
        return httpx.Response(200, json={"strategy": "hybrid", "results": []})

    tools = TonyExamMcpTools(_api_client(handler))
    result = _run(tools.search_materials(
        "binary tree traversal",
        top_k=3,
        subject_id="subject-1",
    ))

    assert observed == {
        "method": "POST",
        "url": "http://localhost:8000/api/retrieval/search",
        "authorization": "Bearer test-access-token",
        "body": {
            "query": "binary tree traversal",
            "top_k": 3,
            "subject_id": "subject-1",
            "course_id": None,
            "syllabus_id": None,
            "topic_id": None,
        },
    }
    assert result == {"strategy": "hybrid", "results": []}


def test_generate_question_starts_a_single_async_run_with_validation_mode() -> None:
    observed: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "run-1", "status": "queued"})

    tools = TonyExamMcpTools(_api_client(handler))
    result = _run(tools.generate_question(
        template_id="template-1",
        query="binary tree",
        difficulty="Medium",
        bloom_level="Apply",
        validation_mode="fast",
        topic_id="topic-1",
    ))

    assert observed["path"] == "/api/questions/generate/start"
    assert observed["body"] == {
        "template_id": "template-1",
        "query": "binary tree",
        "difficulty": "Medium",
        "bloom_level": "Apply",
        "candidate_count": 1,
        "top_k": 5,
        "validation_mode": "fast",
        "subject_id": None,
        "course_id": None,
        "syllabus_id": None,
        "topic_id": "topic-1",
    }
    assert result == {"id": "run-1", "status": "queued"}


def test_get_generation_run_requests_authenticated_status() -> None:
    observed: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        return httpx.Response(200, json={"id": "run-1", "status": "completed"})

    tools = TonyExamMcpTools(_api_client(handler))
    result = _run(tools.get_generation_run("run-1"))

    assert observed["path"] == "/api/questions/generate/runs/run-1"
    assert result["status"] == "completed"


def test_get_generation_run_rejects_path_injection() -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request for unsafe run ID: {_request.url}")

    tools = TonyExamMcpTools(_api_client(unexpected_request))

    with pytest.raises(ValueError, match="run_id"):
        _run(tools.get_generation_run("../another-user"))


@pytest.mark.parametrize(
    ("api_url", "token"),
    [
        ("http://example.com", "token"),
        ("https://user:password@example.com", "token"),
        ("https://example.com?token=secret", "token"),
        ("https://example.com/api", "token"),
        ("https://example.com", ""),
    ],
)
def test_api_client_rejects_unsafe_or_incomplete_configuration(api_url: str, token: str) -> None:
    with pytest.raises(ValueError):
        TonyExamApiClient(api_url=api_url, access_token=token)


def test_api_client_reports_forbidden_without_echoing_credentials() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-access-token"
        return httpx.Response(
            403,
            json={"detail": "Token test-access-token does not have the required role."},
        )

    client = _api_client(handler)

    with pytest.raises(TonyExamApiError, match="HTTP 403: Token \\[redacted\\]") as error:
        _run(client.request("POST", "/api/retrieval/search", {}))

    assert "test-access-token" not in str(error.value)


def test_search_materials_validates_query_and_top_k_before_api_call() -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request for invalid arguments: {_request.url}")

    tools = TonyExamMcpTools(_api_client(unexpected_request))

    with pytest.raises(ValueError, match="query"):
        _run(tools.search_materials("  "))
    with pytest.raises(ValueError, match="top_k"):
        _run(tools.search_materials("binary trees", top_k=51))


def test_mcp_server_registers_only_the_supported_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-access-token"
        return httpx.Response(200, json={})

    server = create_mcp_server(_api_client(handler))

    tools = _run(server.list_tools())

    assert {tool.name for tool in tools} == {
        "search_materials",
        "generate_question",
        "get_generation_run",
    }


def test_stdio_server_completes_mcp_handshake_and_publishes_tools() -> None:
    backend_path = str(Path(__file__).parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": backend_path,
        "TONY_EXAM_API_URL": "http://localhost:8000",
        "TONY_EXAM_API_TOKEN": "test-access-token",
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env=env,
        cwd=backend_path,
    )

    async def inspect_tools() -> dict:
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.list_tools()
                names = {tool.name for tool in response.tools}
                generate_schema = next(
                    tool.inputSchema
                    for tool in response.tools
                    if tool.name == "generate_question"
                )
                return {"names": names, "generate_schema": generate_schema}

    result = _run(inspect_tools())
    assert result["names"] == {
        "search_materials",
        "generate_question",
        "get_generation_run",
    }
    assert result["generate_schema"]["properties"]["validation_mode"]["enum"] == [
        "fast",
        "strict",
    ]


def test_generate_question_rejects_blank_template_before_api_call() -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request for invalid arguments: {_request.url}")

    tools = TonyExamMcpTools(_api_client(unexpected_request))

    with pytest.raises(ValueError, match="template_id"):
        _run(tools.generate_question("", "topic", "Medium", "Apply"))
