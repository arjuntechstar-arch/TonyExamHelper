import mongomock
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.auth import register
from app.schemas import RegisterRequest
from app.services.auth import AuthService, AuthenticationError, AuthorizationError


@pytest.fixture
def service() -> AuthService:
    database = mongomock.MongoClient().test
    database.users.create_index("email", unique=True)
    return AuthService(database, "test-secret-that-is-at-least-32-characters", "HS256", 30)


def test_registration_creates_selected_role_and_returns_a_token(service: AuthService) -> None:
    response = register(
        RegisterRequest(email="New.Faculty@example.edu", password="secure-passphrase", role="faculty"),
        service=service,
    )
    user = service.current_user(response.access_token)

    assert user.email == "new.faculty@example.edu"
    assert user.display_name == "New.Faculty"
    assert user.roles == ["faculty"]
    assert service.verify_password("secure-passphrase", user.password_hash)


def test_registration_defaults_to_student_and_rejects_duplicate_email(service: AuthService) -> None:
    payload = RegisterRequest(email="student@example.edu", password="secure-passphrase")
    register(payload, service=service)

    with pytest.raises(HTTPException) as error:
        register(payload, service=service)
    assert error.value.status_code == 409
    assert "already exists" in error.value.detail


def test_registration_rejects_admin_role() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="admin@example.edu", password="secure-passphrase", role="admin")


def test_authentication_issues_a_token_and_resolves_the_user(service: AuthService) -> None:
    created = service.create_user(email="faculty@example.edu", display_name="Faculty", password="secure-passphrase", roles=["faculty"])
    authenticated = service.authenticate("faculty@example.edu", "secure-passphrase")
    token = service.issue_token(authenticated)

    assert service.current_user(token).id == created.id


def test_authentication_rejects_an_invalid_password(service: AuthService) -> None:
    service.create_user(email="student@example.edu", display_name="Student", password="secure-passphrase", roles=["student"])

    with pytest.raises(AuthenticationError):
        service.authenticate("student@example.edu", "wrong-password")


def test_role_check_rejects_unauthorized_users(service: AuthService) -> None:
    student = service.create_user(email="student@example.edu", display_name="Student", password="secure-passphrase", roles=["student"])

    with pytest.raises(AuthorizationError):
        service.require_roles(student, "admin", "faculty")
