from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import Settings, get_settings


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
    model_status: str
    model_provider: str | None = None
    model_name: str | None = None


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = get_settings()) -> HealthResponse:
    provider = settings.llm_provider.casefold()
    openrouter_key_count = sum(bool(key) for key in (
        settings.openrouter_api_key,
        settings.openrouter2_api_key,
        settings.openrouter3_api_key,
    ))
    if openrouter_key_count or settings.nvidia_api_key:
        route_labels = ([f"OpenRouter ({openrouter_key_count} keys)"] if openrouter_key_count else [])
        if settings.nvidia_api_key:
            route_labels.append("NVIDIA fallback")
        return HealthResponse(
            status="ok",
            service="api",
            timestamp=datetime.now(UTC),
            model_status="configured",
            model_provider=" → ".join(route_labels),
            model_name=(settings.openrouter_model if openrouter_key_count else settings.nvidia_model),
        )
    configured = {
        "openai": bool(settings.openai_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "nvidia": bool(settings.nvidia_api_key),
        "nvidia-nim": bool(settings.nvidia_api_key),
        "kimi-k3": bool(settings.nvidia_api_key),
    }.get(provider, False)
    model_name = {
        "openai": settings.openai_model,
        "openrouter": settings.openrouter_model,
        "nvidia": settings.nvidia_model,
        "nvidia-nim": settings.nvidia_model,
        "kimi-k3": settings.nvidia_model,
    }.get(provider)
    return HealthResponse(
        status="ok",
        service="api",
        timestamp=datetime.now(UTC),
        model_status="configured" if configured else "unavailable",
        model_provider=provider if configured else None,
        model_name=model_name if configured else None,
    )
