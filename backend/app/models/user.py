"""
User models and schemas
"""
from typing import Optional
from pydantic import BaseModel, validator
from datetime import datetime


class UserBase(BaseModel):
    """Base user model"""
    email: str
    full_name: str
    role: str = "Project Developer"
    is_active: bool = True


class UserCreate(UserBase):
    """User creation model"""
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ["Project Developer", "Verifier", "Admin"]
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v


class UserUpdate(BaseModel):
    """User update model"""
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    
    @validator('role')
    def validate_role(cls, v):
        if v is not None:
            allowed_roles = ["Project Developer", "Verifier", "Admin"]
            if v not in allowed_roles:
                raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v


class UserResponse(UserBase):
    """User response model"""
    id: int
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """User login model"""
    email: str
    password: str


class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model"""
    email: Optional[str] = None
