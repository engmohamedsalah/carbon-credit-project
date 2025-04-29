from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import verify_password, get_password_hash
from app.core.database import get_db
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserUpdate, TokenPayload
from app.services.base import CRUDBase

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

# Constants
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        db_obj = User(
            email=obj_in.email,
            hashed_password=get_password_hash(obj_in.password),
            full_name=obj_in.full_name,
            role=obj_in.role or UserRole.VIEWER,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self, db: Session, *, db_obj: User, obj_in: Union[UserUpdate, Dict[str, Any]]
    ) -> User:
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if update_data.get("password"):
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def authenticate(self, db: Session, *, email: str, password: str) -> Optional[User]:
        user = self.get_by_email(db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def is_active(self, user: User) -> bool:
        return user.is_active

    def is_admin(self, user: User) -> bool:
        return user.role == UserRole.ADMIN

    def is_verifier(self, user: User) -> bool:
        return user.role == UserRole.VERIFIER or user.role == UserRole.ADMIN

    def is_project_developer(self, user: User) -> bool:
        return user.role == UserRole.PROJECT_DEVELOPER or user.role == UserRole.ADMIN


user_service = CRUDUser(User)

# Auth service functions
def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=["HS256"]
        )
        token_data = TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = user_service.get(db, id=token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user_service.is_active(user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not user_service.is_active(current_user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_current_active_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    if not user_service.is_admin(current_user):
        raise HTTPException(
            status_code=403, detail="The user doesn't have enough privileges"
        )
    return current_user

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return user_service.get_by_email(db, email=email)

def create_user(db: Session, obj_in: UserCreate) -> User:
    return user_service.create(db, obj_in=obj_in)

def authenticate(db: Session, email: str, password: str) -> Optional[User]:
    return user_service.authenticate(db, email=email, password=password)

def is_active(user: User) -> bool:
    return user_service.is_active(user)

def is_admin(user: User) -> bool:
    return user_service.is_admin(user)

def is_verifier(user: User) -> bool:
    return user_service.is_verifier(user)

def is_project_developer(user: User) -> bool:
    return user_service.is_project_developer(user)
