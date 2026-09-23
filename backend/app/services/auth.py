from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from pymongo.database import Database

from app.models import UserDocument


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class AuthService:
    def __init__(self, database: Database, secret_key: str, algorithm: str, expiry_minutes: int) -> None:
        self.users = database.users
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiry_minutes = expiry_minutes

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        return bcrypt.checkpw(password.encode(), password_hash.encode())

    def create_user(self, *, email: str, display_name: str, password: str, roles: list[str]) -> UserDocument:
        user = UserDocument(email=email.lower(), display_name=display_name, password_hash=self.hash_password(password), roles=roles)
        self.users.insert_one(user.model_dump(by_alias=True))
        return user

    def authenticate(self, email: str, password: str) -> UserDocument:
        document = self.users.find_one({"email": email.lower()})
        if not document or not document.get("is_active") or not self.verify_password(password, document["password_hash"]):
            raise AuthenticationError("Invalid email or password.")
        return UserDocument.model_validate(document)

    def issue_token(self, user: UserDocument) -> str:
        now = datetime.now(UTC)
        payload: dict[str, Any] = {"sub": user.id, "roles": user.roles, "iat": now, "exp": now + timedelta(minutes=self.expiry_minutes)}
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def current_user(self, token: str) -> UserDocument:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload["sub"]
        except (jwt.InvalidTokenError, KeyError) as error:
            raise AuthenticationError("Invalid or expired access token.") from error
        document = self.users.find_one({"_id": user_id})
        if not document or not document.get("is_active"):
            raise AuthenticationError("Invalid or expired access token.")
        return UserDocument.model_validate(document)

    @staticmethod
    def require_roles(user: UserDocument, *roles: str) -> UserDocument:
        if not set(user.roles).intersection(roles):
            raise AuthorizationError("Insufficient permissions.")
        return user
