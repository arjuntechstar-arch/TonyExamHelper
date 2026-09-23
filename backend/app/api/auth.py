from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.config import get_settings
from app.core.database import get_database
from app.models import UserDocument
from app.schemas import CurrentUserResponse, LoginRequest, TokenResponse
from app.services.auth import AuthService, AuthenticationError, AuthorizationError

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth_service() -> AuthService:
    settings = get_settings()
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be configured before authentication can be used.")
    return AuthService(get_database(), settings.jwt_secret_key, settings.jwt_algorithm, settings.access_token_expire_minutes)


def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    service: AuthService = Depends(get_auth_service),
) -> UserDocument:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication is required.")
    try:
        return service.current_user(authorization.removeprefix("Bearer "))
    except AuthenticationError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error


def require_roles(*roles: str):
    def dependency(user: UserDocument = Depends(get_current_user)) -> UserDocument:
        try:
            return AuthService.require_roles(user, *roles)
        except AuthorizationError as error:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(error)) from error
    return dependency


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, service: AuthService = Depends(get_auth_service)) -> TokenResponse:
    try:
        user = service.authenticate(payload.email, payload.password)
    except AuthenticationError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error
    return TokenResponse(access_token=service.issue_token(user))


@router.get("/me", response_model=CurrentUserResponse)
def me(user: UserDocument = Depends(get_current_user)) -> CurrentUserResponse:
    return CurrentUserResponse(id=user.id, email=user.email, display_name=user.display_name, roles=user.roles)
