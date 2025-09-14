"""
Main API handler for Vercel serverless functions
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from passlib.context import CryptContext
import secrets
import logging
from datetime import datetime, timedelta
from database import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Carbon Credit Verification API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "Project Developer"

class UserLogin(BaseModel):
    email: str
    password: str

class ProjectCreate(BaseModel):
    name: str
    description: str
    location_name: str
    area_hectares: float
    project_type: str = "Reforestation"

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Carbon Credit Verification API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Authentication endpoints
@app.post("/api/v1/auth/register")
async def register(user: UserCreate):
    try:
        with db_manager.get_connection() as conn:
            # Check if user already exists
            cursor = conn.execute("SELECT id FROM users WHERE email = ?", (user.email,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Hash password and create user
            hashed_password = pwd_context.hash(user.password)
            cursor = conn.execute("""
                INSERT INTO users (email, hashed_password, full_name, role)
                VALUES (?, ?, ?, ?)
            """, (user.email, hashed_password, user.full_name, user.role))
            
            user_id = cursor.lastrowid
            
            return {
                "message": "User registered successfully",
                "user_id": user_id,
                "email": user.email
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/v1/auth/login")
async def login(user: UserLogin):
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, email, hashed_password, full_name, role 
                FROM users WHERE email = ? AND is_active = TRUE
            """, (user.email,))
            
            db_user = cursor.fetchone()
            if not db_user or not pwd_context.verify(user.password, db_user['hashed_password']):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Create access token
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)
            
            conn.execute("""
                INSERT INTO auth_tokens (token, user_id, expires_at)
                VALUES (?, ?, ?)
            """, (token, db_user['id'], expires_at))
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "user": {
                    "id": db_user['id'],
                    "email": db_user['email'],
                    "full_name": db_user['full_name'],
                    "role": db_user['role']
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

# Get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT u.id, u.email, u.full_name, u.role
                FROM users u
                JOIN auth_tokens t ON u.id = t.user_id
                WHERE t.token = ? AND t.expires_at > ? AND u.is_active = TRUE
            """, (token, datetime.now()))
            
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            
            return {
                "id": user['id'],
                "email": user['email'],
                "full_name": user['full_name'],
                "role": user['role']
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")

# Project endpoints
@app.get("/api/v1/projects")
async def get_projects(current_user: dict = Depends(get_current_user)):
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, name, description, location_name, area_hectares, 
                       project_type, status, created_at, start_date, end_date,
                       estimated_carbon_credits
                FROM projects
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (current_user['id'],))
            
            projects = []
            for row in cursor.fetchall():
                projects.append({
                    "id": row['id'],
                    "name": row['name'],
                    "description": row['description'],
                    "location_name": row['location_name'],
                    "area_hectares": row['area_hectares'],
                    "project_type": row['project_type'],
                    "status": row['status'],
                    "created_at": row['created_at'],
                    "start_date": row['start_date'],
                    "end_date": row['end_date'],
                    "estimated_carbon_credits": row['estimated_carbon_credits']
                })
            
            return {"projects": projects}
    except Exception as e:
        logger.error(f"Get projects error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch projects")

@app.post("/api/v1/projects")
async def create_project(project: ProjectCreate, current_user: dict = Depends(get_current_user)):
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO projects (name, description, location_name, area_hectares, 
                                    project_type, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (project.name, project.description, project.location_name,
                  project.area_hectares, project.project_type, current_user['id']))
            
            project_id = cursor.lastrowid
            
            return {
                "message": "Project created successfully",
                "project_id": project_id,
                "name": project.name
            }
    except Exception as e:
        logger.error(f"Create project error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create project")

# For Vercel deployment
def handler(request):
    return app(request)