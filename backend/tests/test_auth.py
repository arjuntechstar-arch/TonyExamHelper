import mongomock
import pytest

from app.services.auth import AuthService, AuthenticationError, AuthorizationError


@pytest.fixture
def service() -> AuthService:
    return AuthService(mongomock.MongoClient().test, "test-secret-that-is-at-least-32-characters", "HS256", 30)


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
