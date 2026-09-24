import mongomock
import pytest

from app.api.users import create_user, list_users, update_user_status
from app.models import UserDocument
from app.schemas import UserCreateRequest, UserStatusRequest


def test_admin_user_management_returns_safe_summaries() -> None:
    database = mongomock.MongoClient().test
    database.users.create_index("email", unique=True)
    admin = UserDocument(email="admin@example.edu", display_name="Admin", password_hash="hash", roles=["admin"])

    created = create_user(
        UserCreateRequest(email="student@example.edu", display_name="Student", password="secure-passphrase", roles=["student"]),
        database=database,
        _=admin,
    )
    assert created.email == "student@example.edu"
    assert not hasattr(created, "password_hash")
    assert [user.email for user in list_users(database=database, _=admin)] == ["student@example.edu"]

    updated = update_user_status(created.id, UserStatusRequest(is_active=False), database=database, _=admin)
    assert updated.is_active is False


def test_admin_user_management_rejects_duplicate_email() -> None:
    database = mongomock.MongoClient().test
    database.users.create_index("email", unique=True)
    admin = UserDocument(email="admin@example.edu", display_name="Admin", password_hash="hash", roles=["admin"])
    payload = UserCreateRequest(email="student@example.edu", display_name="Student", password="secure-passphrase")
    create_user(payload, database=database, _=admin)

    with pytest.raises(Exception, match="already exists"):
        create_user(payload, database=database, _=admin)