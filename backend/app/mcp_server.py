from __future__ import annotations

import ipaddress
import os
import re
from typing import Literal
from urllib.parse import urlsplit

import httpx
from mcp.server.fastmcp import FastMCP


class TonyExamApiError(RuntimeError):
    pass


class TonyExamApiClient:
    def __init__(
        self,
        *,
        api_url: str,
        access_token: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 65.0,
    ) -> None:
        normalized_url = api_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("TONY_EXAM_API_URL must be an absolute http or https URL.")
        if parsed_url.path not in {"", "/"}:
            raise ValueError("TONY_EXAM_API_URL must be the API origin without a path.")
        if parsed_url.query or parsed_url.fragment:
            raise ValueError("TONY_EXAM_API_URL must not include a query string or fragment.")
        if parsed_url.username or parsed_url.password:
            raise ValueError("TONY_EXAM_API_URL must not embed credentials.")
        if parsed_url.scheme != "https":
            hostname = (parsed_url.hostname or "").casefold()
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = hostname == "localhost"
            if not is_loopback:
                raise ValueError("TONY_EXAM_API_URL must use HTTPS except for localhost.")
        token = access_token.strip()
        if not token:
            raise ValueError("TONY_EXAM_API_TOKEN must contain an API bearer token.")
        self.api_url = normalized_url
        self.access_token = token[7:].strip() if token[:7].casefold() == "bearer " else token
        if not self.access_token:
            raise ValueError("TONY_EXAM_API_TOKEN must contain an API bearer token.")
        self.transport = transport
        self.timeout_seconds = timeout_seconds

    async def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            async with httpx.AsyncClient(
                base_url=self.api_url,
                headers={"Authorization": f"Bearer {self.access_token}"},
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.request(method, path, json=payload)
        except httpx.HTTPError as error:
            raise TonyExamApiError(
                "Could not reach the TonyExam API. Check TONY_EXAM_API_URL and API availability."
            ) from error

        if not response.is_success:
            detail: object = "request failed"
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = body.get("detail", body.get("message", detail))
            except ValueError:
                pass
            if not isinstance(detail, str):
                detail = "request failed"
            detail = detail.replace(self.access_token, "[redacted]")
            raise TonyExamApiError(
                f"TonyExam API returned HTTP {response.status_code}: {detail}"
            )
        try:
            result = response.json()
        except ValueError as error:
            raise TonyExamApiError("TonyExam API returned an invalid JSON response.") from error
        if not isinstance(result, dict):
            raise TonyExamApiError("TonyExam API returned an unexpected response shape.")
        return result


class TonyExamMcpTools:
    def __init__(self, api: TonyExamApiClient) -> None:
        self.api = api

    async def search_materials(
        self,
        query: str,
        top_k: int = 5,
        subject_id: str | None = None,
        course_id: str | None = None,
        syllabus_id: str | None = None,
        topic_id: str | None = None,
    ) -> dict:
        if not query.strip() or len(query) > 2_000:
            raise ValueError("query must contain between 1 and 2,000 characters.")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50.")
        return await self.api.request(
            "POST",
            "/api/retrieval/search",
            {
                "query": query,
                "top_k": top_k,
                "subject_id": subject_id,
                "course_id": course_id,
                "syllabus_id": syllabus_id,
                "topic_id": topic_id,
            },
        )

    async def generate_question(
        self,
        template_id: str,
        query: str,
        difficulty: str,
        bloom_level: str,
        validation_mode: Literal["fast", "strict"] = "strict",
        subject_id: str | None = None,
        course_id: str | None = None,
        syllabus_id: str | None = None,
        topic_id: str | None = None,
    ) -> dict:
        if not template_id.strip():
            raise ValueError("template_id is required.")
        if not query.strip() or len(query) > 2_000:
            raise ValueError("query must contain between 1 and 2,000 characters.")
        if not difficulty.strip() or not bloom_level.strip():
            raise ValueError("difficulty and bloom_level are required.")
        return await self.api.request(
            "POST",
            "/api/questions/generate/start",
            {
                "template_id": template_id,
                "query": query,
                "difficulty": difficulty,
                "bloom_level": bloom_level,
                "candidate_count": 1,
                "top_k": 5,
                "validation_mode": validation_mode,
                "subject_id": subject_id,
                "course_id": course_id,
                "syllabus_id": syllabus_id,
                "topic_id": topic_id,
            },
        )

    async def get_generation_run(self, run_id: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise ValueError("run_id must contain only letters, numbers, underscores, or hyphens.")
        return await self.api.request(
            "GET",
            f"/api/questions/generate/runs/{run_id}",
        )


def create_mcp_server(api: TonyExamApiClient) -> FastMCP:
    server = FastMCP(
        "TonyExamHelper",
        instructions=(
            "Use search_materials to retrieve scoped source passages before generation. "
            "Use generate_question to create a single question asynchronously, then poll "
            "get_generation_run until it completes. The TonyExam API enforces the caller's "
            "account roles and ownership permissions."
        ),
    )
    tools = TonyExamMcpTools(api)
    server.tool()(tools.search_materials)
    server.tool()(tools.generate_question)
    server.tool()(tools.get_generation_run)
    return server


def main() -> None:
    api_url = os.environ.get("TONY_EXAM_API_URL", "http://localhost:8000")
    access_token = os.environ.get("TONY_EXAM_API_TOKEN", "")
    api = TonyExamApiClient(api_url=api_url, access_token=access_token)
    create_mcp_server(api).run(transport="stdio")


if __name__ == "__main__":
    main()
