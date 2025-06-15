"""
Authentication service layer
"""
import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import HTTPException, status
from app.core.security import (
    verify_password, 
    get_password_hash, 
    create_access_token,
    verify_token,
    validate_password_strength
)
from app.db.database import db_manager
from app.models.user import UserCreate, UserResponse, Token
from app.core.config import settings

logger = logging.getLogger(__name__)


class AuthService:
    """Professional authentication service"""
    
    @staticmethod
    def register_user(user_data: UserCreate) -> Token:
        """Register a new user"""
        try:
            # Validate password strength
            validate_password_strength(user_data.password)
            
            # Check if user already exists
            existing_user = db_manager.execute_query(
                "SELECT email FROM users WHERE email = ?",
                (user_data.email,)
            )
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Hash password
            hashed_password = get_password_hash(user_data.password)
            
            # Insert user
            user_id = db_manager.execute_insert(
                """
                INSERT INTO users (email, hashed_password, full_name, role, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_data.email, hashed_password, user_data.full_name, user_data.role, True)
            )
            
            # Create access token
            access_token = create_access_token(subject=user_data.email)
            
            # Store token in database
            expires_at = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            db_manager.execute_insert(
                """
                INSERT INTO tokens (token, email, expires_at, is_active)
                VALUES (?, ?, ?, ?)
                """,
                (access_token, user_data.email, expires_at, True)
            )
            
            logger.info(f"User registered successfully: {user_data.email}")
            
            return Token(
                access_token=access_token,
                token_type="bearer",
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    @staticmethod
    def authenticate_user(email: str, password: str) -> Token:
        """Authenticate user and return token"""
        try:
            # Get user from database
            user = db_manager.execute_query(
                """
                SELECT id, email, hashed_password, is_active 
                FROM users 
                WHERE email = ?
                """,
                (email,)
            )
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            if not user["is_active"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is deactivated"
                )
            
            # Verify password
            if not verify_password(password, user["hashed_password"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Create access token
            access_token = create_access_token(subject=email)
            
            # Store token in database
            expires_at = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            db_manager.execute_insert(
                """
                INSERT INTO tokens (token, email, expires_at, is_active)
                VALUES (?, ?, ?, ?)
                """,
                (access_token, email, expires_at, True)
            )
            
            logger.info(f"User authenticated successfully: {email}")
            
            return Token(
                access_token=access_token,
                token_type="bearer",
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )
    
    @staticmethod
    def get_current_user(token: str) -> UserResponse:
        """Get current user from token"""
        try:
            # Remove Bearer prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Verify token
            email = verify_token(token)
            if not email:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            # Check if token exists and is active in database
            token_data = db_manager.execute_query(
                """
                SELECT email, expires_at, is_active 
                FROM tokens 
                WHERE token = ?
                """,
                (token,)
            )
            
            if not token_data or not token_data["is_active"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token is invalid or expired"
                )
            
            # Check if token is expired
            if token_data["expires_at"] and datetime.fromisoformat(token_data["expires_at"]) < datetime.utcnow():
                # Deactivate expired token
                db_manager.execute_update(
                    "UPDATE tokens SET is_active = FALSE WHERE token = ?",
                    (token,)
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            # Get user details
            user = db_manager.execute_query(
                """
                SELECT id, email, full_name, role, is_active 
                FROM users 
                WHERE email = ?
                """,
                (email,)
            )
            
            if not user or not user["is_active"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or deactivated"
                )
            
            return UserResponse(
                id=user["id"],
                email=user["email"],
                full_name=user["full_name"],
                role=user["role"],
                is_active=user["is_active"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get current user failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get user information"
            )
    
    @staticmethod
    def logout_user(token: str) -> bool:
        """Logout user by deactivating token"""
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Deactivate token
            affected_rows = db_manager.execute_update(
                "UPDATE tokens SET is_active = FALSE WHERE token = ?",
                (token,)
            )
            
            return affected_rows > 0
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False


# Global auth service instance
auth_service = AuthService()
