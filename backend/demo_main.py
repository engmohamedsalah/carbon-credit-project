"""
Carbon Credit Verification API - Demo Version
Professional FastAPI application demonstrating clean architecture
"""
import logging
import hashlib
import secrets
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Carbon Credit Verification API",
    description="Professional API for carbon credit verification and management",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

# Simple in-memory storage (for demo)
users_db = {}
tokens_db = {}
projects_db = []

# Models
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "Project Developer"
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 28800

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    location_name: str
    area_size: float
    project_type: str = "Reforestation"
    
    @validator('name')
    def validate_name(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Project name must be at least 3 characters long')
        return v.strip()

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    location_name: str
    area_size: float
    project_type: str
    status: str = "Pending"
    user_id: int
    created_at: str

# Utility functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_token() -> str:
    return secrets.token_urlsafe(32)

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    if token.startswith('Bearer '):
        token = token[7:]
    
    email = tokens_db.get(token)
    if not email or email not in users_db:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = users_db[email]
    return UserResponse(**user)

# Health endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Carbon Credit Verification API",
        "version": "1.0.0",
        "architecture": "Professional Clean Architecture"
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Carbon Credit Verification API",
        "version": "1.0.0",
        "docs": "/api/v1/docs",
        "health": "/health",
        "architecture": "Professional Clean Architecture with:",
        "features": [
            "Modular structure",
            "Proper error handling", 
            "Input validation",
            "Security best practices",
            "API documentation",
            "Logging",
            "CORS configuration"
        ]
    }

# Authentication endpoints
@app.post("/api/v1/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user with proper validation"""
    if user_data.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Store user
    user_id = len(users_db) + 1
    users_db[user_data.email] = {
        "id": user_id,
        "email": user_data.email,
        "hashed_password": hash_password(user_data.password),
        "full_name": user_data.full_name,
        "role": user_data.role,
        "is_active": True
    }
    
    # Create token
    token = create_token()
    tokens_db[token] = user_data.email
    
    logger.info(f"User registered: {user_data.email}")
    return Token(access_token=token)

@app.post("/api/v1/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user with proper authentication"""
    user = users_db.get(form_data.username)
    if not user or user["hashed_password"] != hash_password(form_data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token()
    tokens_db[token] = form_data.username
    
    logger.info(f"User logged in: {form_data.username}")
    return Token(access_token=token)

@app.get("/api/v1/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_user)):
    """Get current user information"""
    return current_user

# Project endpoints
@app.get("/api/v1/projects")
async def get_projects(current_user: UserResponse = Depends(get_current_user)):
    """Get user's projects with proper authorization"""
    user_projects = [p for p in projects_db if p["user_id"] == current_user.id]
    return {
        "projects": user_projects,
        "total": len(user_projects),
        "page": 1,
        "page_size": 20
    }

@app.post("/api/v1/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(project_data: ProjectCreate, current_user: UserResponse = Depends(get_current_user)):
    """Create a new project with proper validation"""
    project_id = len(projects_db) + 1
    
    new_project = {
        "id": project_id,
        "name": project_data.name,
        "description": project_data.description,
        "location_name": project_data.location_name,
        "area_size": project_data.area_size,
        "project_type": project_data.project_type,
        "status": "Pending",
        "user_id": current_user.id,
        "created_at": datetime.now().isoformat()
    }
    
    projects_db.append(new_project)
    logger.info(f"Project created: {project_data.name} by {current_user.email}")
    
    return ProjectResponse(**new_project)

@app.get("/api/v1/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, current_user: UserResponse = Depends(get_current_user)):
    """Get project by ID with proper authorization"""
    project = next((p for p in projects_db if p["id"] == project_id), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project["user_id"] != current_user.id and current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return ProjectResponse(**project)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return {"error": True, "message": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Carbon Credit Verification API (Demo)")
    uvicorn.run("demo_main:app", host="0.0.0.0", port=8000, reload=True) 