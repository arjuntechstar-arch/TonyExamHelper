from __future__ import annotations

import hashlib
from urllib.parse import urlparse

import httpx

from app.models import DocumentChunkDocument


class WebSearchError(RuntimeError):
    pass


class TavilySearchProvider:
    endpoint = "https://api.tavily.com/search"
    timeout_seconds = 15

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        try:
            response = httpx.post(
                self.endpoint,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as error:
            raise WebSearchError("Web search timed out. Please retry or use indexed course material.") from error
        except httpx.HTTPStatusError as error:
            raise WebSearchError(f"Web search provider returned HTTP {error.response.status_code}.") from error
        except httpx.HTTPError as error:
            raise WebSearchError("Could not reach the web search provider.") from error
        except ValueError as error:
            raise WebSearchError("Web search provider returned invalid JSON.") from error

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise WebSearchError("Web search provider returned an invalid result set.")

        chunks: list[dict] = []
        seen_urls: set[str] = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            title = result.get("title")
            url = result.get("url")
            content = result.get("content")
            if not all(isinstance(value, str) and value.strip() for value in (title, url, content)):
                continue
            if len(url) > 2_000:
                continue
            parsed_url = urlparse(url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc or url in seen_urls:
                continue
            seen_urls.add(url)

            chunk_id = "web-" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
            chunk = DocumentChunkDocument(
                id=chunk_id,
                study_material_id="web-search",
                chunk_index=len(chunks),
                page_number=1,
                content=content.strip()[:5_000],
                metadata={"source": "web_search", "title": title.strip()[:300], "url": url},
            )
            try:
                score = float(result.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            chunks.append({"chunk": chunk, "score": score})
            if len(chunks) >= max_results:
                break
        if not chunks:
            raise WebSearchError("Web search returned no usable sources for this topic.")
        return chunks
