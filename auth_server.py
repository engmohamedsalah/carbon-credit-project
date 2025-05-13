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

# Import the mock satellite module
try:
    from mock_satellite import SatelliteImageProcessor
    satellite_processor = SatelliteImageProcessor()
except ImportError:
    print("Warning: mock_satellite module not found. Verification functionality will be limited.")
    satellite_processor = None

# Project Models
class ProjectBase(BaseModel):
    name: str
    location: str
    area_size: float
    description: str
    project_type: str = "Reforestation"
    
class Project(ProjectBase):
    id: int
    user_id: int
    created_at: str
    status: str = "Pending"
    
class ProjectCreate(ProjectBase):
    pass

# Initialize projects table
def init_projects_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        area_size REAL NOT NULL,
        description TEXT NOT NULL,
        project_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Pending',
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    conn.commit()
    conn.close()

# Initialize projects table
init_projects_db()

# Project routes
@app.post("/projects/new", response_model=Project)
async def create_project(project: ProjectCreate, token: str = Depends(oauth2_scheme)):
    # In a real implementation, we would validate the token
    # For simplicity, we'll just extract the user ID
    user_id = 1  # Mock user ID
    
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO projects (user_id, name, location, area_size, description, project_type, created_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, project.name, project.location, project.area_size, project.description, project.project_type, created_at, "Pending")
    )
    
    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "id": project_id,
        "user_id": user_id,
        "name": project.name,
        "location": project.location,
        "area_size": project.area_size,
        "description": project.description,
        "project_type": project.project_type,
        "created_at": created_at,
        "status": "Pending"
    }

@app.get("/projects")
async def get_projects(token: str = Depends(oauth2_scheme)):
    """Get a list of projects for the current user"""
    # In a real implementation, we would extract the user ID from the token
    user_id = 1  # Mock user ID
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM projects WHERE user_id = ?", (user_id,))
    projects = cursor.fetchall()
    conn.close()
    
    # Convert to a list of dictionaries
    result = []
    for project in projects:
        result.append({
            "id": project["id"],
            "user_id": project["user_id"],
            "name": project["name"],
            "location": project["location"],
            "area_size": project["area_size"],
            "description": project["description"],
            "project_type": project["project_type"],
            "created_at": project["created_at"],
            "status": project["status"]
        })
    
    return result

@app.get("/projects/{project_id}")
async def get_project(project_id: int, token: str = Depends(oauth2_scheme)):
    """Get a specific project by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    project = cursor.fetchone()
    conn.close()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found"
        )
    
    # Return the project
    return {
        "id": project["id"],
        "user_id": project["user_id"],
        "name": project["name"],
        "location": project["location"],
        "area_size": project["area_size"],
        "description": project["description"],
        "project_type": project["project_type"],
        "created_at": project["created_at"],
        "status": project["status"]
    }

@app.get("/projects/new")
async def get_new_project_form():
    # This endpoint serves the form for creating a new project
    return {
        "message": "New project form endpoint",
        "project_types": ["Reforestation", "Afforestation", "Forest Conservation", "Improved Forest Management"]
    }

@app.get("/verification")
async def get_verification(project_id: str, token: str = Depends(oauth2_scheme)):
    # Try to find the project in the database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if project_id == "new":
        # Just return a mock response for the form
        verification_data = {
            "message": "Verification process form",
            "project_id": "new",
            "status": "Form",
            "providers": ["Sentinel-2", "Landsat-8", "PlanetScope"],
            "models": {
                "land_cover": ["U-Net", "Random Forest"],
                "carbon_estimation": ["Random Forest", "Gradient Boosting"]
            }
        }
    else:
        try:
            project_id_int = int(project_id)
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id_int,))
            project = cursor.fetchone()
            
            if not project:
                conn.close()
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project with ID {project_id} not found"
                )
            
            # Use the mock satellite processor if available
            if satellite_processor:
                # Process the project location to get imagery and analysis
                location = project["location"]
                imagery_info = satellite_processor.acquire_imagery(location)
                analysis_results = satellite_processor.process_imagery(location)
                
                # Update the project status to "Verified" in the database
                cursor.execute(
                    "UPDATE projects SET status = ? WHERE id = ?",
                    ("Verified", project_id_int)
                )
                conn.commit()
                
                verification_data = {
                    "message": "Verification process completed",
                    "project_id": project_id,
                    "project_name": project["name"],
                    "status": "Verified",
                    "satellite_imagery": imagery_info,
                    "analysis_results": analysis_results
                }
            else:
                # Fallback to mock data if satellite processor is not available
                verification_data = {
                    "message": "Verification process started (mock)",
                    "project_id": project_id,
                    "project_name": project["name"],
                    "status": "In Progress",
                    "satellite_imagery": {
                        "status": "Acquiring",
                        "provider": "Sentinel-2",
                        "resolution": "10m"
                    },
                    "ml_models": {
                        "land_cover": "U-Net",
                        "carbon_estimation": "Random Forest"
                    }
                }
        except ValueError:
            verification_data = {
                "message": "Invalid project ID",
                "project_id": project_id,
                "status": "Error"
            }
    
    conn.close()
    return verification_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 