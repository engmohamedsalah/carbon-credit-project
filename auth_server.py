"""
Authentication API server for testing.
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import os
import hashlib
import secrets
import time
from datetime import datetime, timedelta

app = FastAPI(
    title="Carbon Credit Verification Auth API",
    description="Authentication API for the Carbon Credit Verification SaaS",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
def get_db_connection():
    os.makedirs('data', exist_ok=True)
    db_path = 'data/carbon_credits.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Create users table if it doesn't exist
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'Project Developer',
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        is_admin BOOLEAN NOT NULL DEFAULT FALSE
    )
    ''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Models
class User(BaseModel):
    email: str
    full_name: str
    password: str
    role: str = "Project Developer"

class UserInDB(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    is_admin: bool

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    full_name: str
    email: str
    role: str

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_token() -> str:
    return secrets.token_hex(32)

# User functions
def get_user_by_email(email: str) -> Optional[UserInDB]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return UserInDB(
            id=user['id'],
            email=user['email'],
            full_name=user['full_name'],
            role=user['role'],
            is_active=user['is_active'],
            is_admin=user['is_admin']
        )
    return None

def create_user(user: User) -> Optional[UserInDB]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        hashed_password = hash_password(user.password)
        cursor.execute(
            "INSERT INTO users (email, full_name, hashed_password, role) VALUES (?, ?, ?, ?)",
            (user.email, user.full_name, hashed_password, user.role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        return UserInDB(
            id=user_id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=True,
            is_admin=False
        )
    except sqlite3.IntegrityError:
        conn.close()
        return None

def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
    
    hashed_password = hash_password(password)
    if hashed_password != user['hashed_password']:
        return None
    
    return UserInDB(
        id=user['id'],
        email=user['email'],
        full_name=user['full_name'],
        role=user['role'],
        is_active=user['is_active'],
        is_admin=user['is_admin']
    )

# Routes
@app.post("/auth/register", response_model=Token)
async def register(user: User):
    db_user = get_user_by_email(user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    created_user = create_user(user)
    if not created_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    
    access_token = create_token()
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=created_user.id,
        full_name=created_user.full_name,
        email=created_user.email,
        role=created_user.role
    )

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_token()
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        full_name=user.full_name,
        email=user.email,
        role=user.role
    )

@app.get("/auth/user")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # This is just a mock that returns a successful response
    return {
        "id": 1,
        "email": "user@example.com",
        "full_name": "Test User",
        "role": "Project Developer",
        "is_active": True
    }

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Auth API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 