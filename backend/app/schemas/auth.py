from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    roles: list[str]
    bio: str | None = None
    institution: str | None = None


class ProfileUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=150)
    bio: str | None = Field(default=None, max_length=500)
    institution: str | None = Field(default=None, max_length=150)


class PasswordUpdateRequest(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class UserCreateRequest(BaseModel):
    email: str = Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    display_name: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=8, max_length=128)
    roles: list[str] = Field(default_factory=lambda: ["student"], min_length=1)


class UserSummaryResponse(BaseModel):
    id: str
    email: str
    display_name: str
    roles: list[str]
    is_active: bool


class UserStatusRequest(BaseModel):
    is_active: bool
