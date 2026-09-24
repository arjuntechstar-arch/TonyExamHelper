from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.api.auth import require_roles
from app.core.database import get_database
from app.models import UserDocument
from app.schemas import UserCreateRequest, UserStatusRequest, UserSummaryResponse
from app.services.auth import AuthService

router = APIRouter(prefix="/users", tags=["users"])
AdminUser = Depends(require_roles("admin"))


def _summary(user: UserDocument) -> UserSummaryResponse:
    return UserSummaryResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        roles=user.roles,
        is_active=user.is_active,
    )


@router.get("", response_model=list[UserSummaryResponse])
def list_users(database: Database = Depends(get_database), _: UserDocument = AdminUser) -> list[UserSummaryResponse]:
    return [_summary(UserDocument.model_validate(document)) for document in database.users.find({}).sort("email", 1)]


@router.post("", response_model=UserSummaryResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreateRequest,
    database: Database = Depends(get_database),
    _: UserDocument = AdminUser,
) -> UserSummaryResponse:
    service = AuthService(database, "unused-secret", "HS256", 30)
    try:
        user = service.create_user(
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
            roles=payload.roles,
        )
    except DuplicateKeyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A user with this email already exists.") from error
    return _summary(user)


@router.patch("/{user_id}/status", response_model=UserSummaryResponse)
def update_user_status(
    user_id: str,
    payload: UserStatusRequest,
    database: Database = Depends(get_database),
    _: UserDocument = AdminUser,
) -> UserSummaryResponse:
    result = database.users.update_one({"_id": user_id}, {"$set": {"is_active": payload.is_active}})
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return _summary(UserDocument.model_validate(database.users.find_one({"_id": user_id})))
