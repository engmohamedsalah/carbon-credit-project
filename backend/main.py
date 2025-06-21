"""
Carbon Credit Verification API - Professional Edition
Simplified architecture with SQLite database integration
"""
import os
import sqlite3
import hashlib
import secrets
import logging
import json
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from pydantic import BaseModel, field_validator
from passlib.context import CryptContext
from fastapi import UploadFile, File
from typing import Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

# Database configuration
DATABASE_PATH = os.path.join("..", "database", "carbon_credits.db")


# Database Manager
class DatabaseManager:
    """Professional SQLite database manager"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            with self.get_connection() as conn:
                # Users table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        hashed_password TEXT NOT NULL,
                        full_name TEXT NOT NULL,
                        role TEXT DEFAULT 'Project Developer',
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Auth tokens table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS auth_tokens (
                        token TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Projects table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        description TEXT,
                        location_name TEXT NOT NULL,
                        area_hectares REAL,
                        project_type TEXT DEFAULT 'Reforestation',
                        status TEXT DEFAULT 'Pending',
                        user_id INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        start_date TEXT,
                        end_date TEXT,
                        estimated_carbon_credits REAL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Check if we need to migrate existing table
                try:
                    cursor = conn.execute("PRAGMA table_info(projects)")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    # Add missing columns if they don't exist
                    if 'area_hectares' not in columns and 'area_size' in columns:
                        conn.execute("ALTER TABLE projects ADD COLUMN area_hectares REAL")
                        conn.execute("UPDATE projects SET area_hectares = area_size")
                        logger.info("Migrated area_size to area_hectares")
                    
                    if 'start_date' not in columns:
                        conn.execute("ALTER TABLE projects ADD COLUMN start_date TEXT")
                    
                    if 'end_date' not in columns:
                        conn.execute("ALTER TABLE projects ADD COLUMN end_date TEXT")
                        
                    if 'estimated_carbon_credits' not in columns:
                        conn.execute("ALTER TABLE projects ADD COLUMN estimated_carbon_credits REAL")
                        
                except Exception as migration_error:
                    logger.warning(f"Migration warning: {migration_error}")
                    # Continue if migration fails - table might already be correct
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise


# Initialize database manager
db = DatabaseManager(DATABASE_PATH)


# Pydantic Models
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "Project Developer"


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    location_name: str
    area_hectares: Optional[float] = None
    project_type: str = "Reforestation"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    estimated_carbon_credits: Optional[float] = None
    geometry: Optional[dict] = None
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Project name must be at least 3 characters long')
        return v.strip()
    
    @field_validator('area_hectares')
    @classmethod
    def validate_area(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Area must be positive')
        return v


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    location_name: str
    area_hectares: Optional[float]
    project_type: str
    status: str
    user_id: int
    created_at: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    estimated_carbon_credits: Optional[float] = None
    geometry: Optional[dict] = None


# Verification Models
class VerificationCreate(BaseModel):
    project_id: int
    carbon_impact: Optional[float] = None
    verification_notes: Optional[str] = None
    
    
class VerificationResponse(BaseModel):
    id: int
    project_id: int
    status: str
    carbon_impact: Optional[float] = None
    ai_confidence: Optional[float] = None
    human_verified: bool = False
    blockchain_certified: bool = False
    certificate_id: Optional[str] = None
    created_at: str
    updated_at: str


class HumanReviewRequest(BaseModel):
    approved: bool
    notes: Optional[str] = None


# ML-specific Pydantic models
class LocationAnalysisRequest(BaseModel):
    project_id: int
    latitude: float
    longitude: float
    analysis_type: str = "comprehensive"


class MLAnalysisResponse(BaseModel):
    project_id: int
    analysis_type: str
    status: str
    results: dict
    timestamp: str


# XAI Models
class ExplanationRequest(BaseModel):
    project_id: int
    prediction_id: Optional[str] = None
    explanation_method: str = "shap"  # "shap", "lime", "integrated_gradients"
    image_path: Optional[str] = None
    
    @field_validator('explanation_method')
    @classmethod
    def validate_method(cls, v):
        allowed_methods = ["shap", "lime", "integrated_gradients", "all"]
        if v not in allowed_methods:
            raise ValueError(f'Method must be one of: {allowed_methods}')
        return v


class ExplanationResponse(BaseModel):
    explanation_id: str
    project_id: int
    method: str
    status: str
    results: dict
    visualization_paths: List[str]
    timestamp: str


class CompareExplanationsRequest(BaseModel):
    explanation_ids: List[str]
    comparison_type: str = "side_by_side"  # "side_by_side", "overlay", "difference"


# Utility functions
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_token() -> str:
    """Create secure random token"""
    return secrets.token_urlsafe(32)


def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email from database"""
    try:
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None


def get_user_by_token(token: str) -> Optional[dict]:
    """Get user by auth token"""
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT u.* FROM users u 
                JOIN auth_tokens t ON u.id = t.user_id 
                WHERE t.token = ?
            """, (token,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error getting user by token: {e}")
        return None


def create_user(user_data: UserCreate) -> dict:
    """Create new user in database"""
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO users (email, hashed_password, full_name, role)
                VALUES (?, ?, ?, ?)
            """, (user_data.email, hash_password(user_data.password), 
                  user_data.full_name, user_data.role))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {
                "id": user_id,
                "email": user_data.email,
                "full_name": user_data.full_name,
                "role": user_data.role,
                "is_active": True
            }
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise


def store_token(token: str, user_id: int):
    """Store auth token in database"""
    try:
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO auth_tokens (token, user_id) VALUES (?, ?)",
                (token, user_id)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error storing token: {e}")
        raise


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """Get current authenticated user"""
    if token.startswith('Bearer '):
        token = token[7:]
    
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return UserResponse(**user)


# FastAPI application
app = FastAPI(
    title="Carbon Credit Verification API",
    description="Professional API for carbon credit verification and management",
    version="2.0.0",
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


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Validation error",
            "details": exc.errors()
        }
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    logger.error(f"Pydantic validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Validation error", 
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error"
        }
    )


# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Carbon Credit Verification API",
        "version": "2.0.0",
        "database": "SQLite",
        "architecture": "Simplified Professional"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Carbon Credit Verification API",
        "version": "2.0.0",
        "docs": "/api/v1/docs",
        "health": "/health",
        "features": [
            "SQLite database integration",
            "User authentication & authorization",
            "Project management",
            "Professional error handling",
            "API documentation",
            "Simplified architecture"
        ]
    }


# Authentication endpoints
@app.post("/api/v1/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user"""
    # Manual validation with proper error messages
    if len(user_data.password) < 8:
        raise HTTPException(
            status_code=422, 
            detail={
                "error": True,
                "message": "Validation error",
                "details": [{"field": "password", "message": "Password must be at least 8 characters long"}]
            }
        )
    
    if '@' not in user_data.email:
        raise HTTPException(
            status_code=422,
            detail={
                "error": True, 
                "message": "Validation error",
                "details": [{"field": "email", "message": "Invalid email format"}]
            }
        )
    
    # Normalize email
    user_data.email = user_data.email.lower()
    
    # Check if user already exists
    if get_user_by_email(user_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = create_user(user_data)
    
    # Create token
    token = create_token()
    store_token(token, user["id"])
    
    logger.info(f"User registered: {user_data.email}")
    return Token(access_token=token)


@app.post("/api/v1/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user"""
    user = get_user_by_email(form_data.username)
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    token = create_token()
    store_token(token, user["id"])
    
    logger.info(f"User logged in: {form_data.username}")
    return Token(access_token=token)


@app.get("/api/v1/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_user)):
    """Get current user information"""
    return current_user


# Project endpoints
@app.get("/api/v1/projects")
async def get_projects(current_user: UserResponse = Depends(get_current_user)):
    """Get user's projects"""
    try:
        import json
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC
            """, (current_user.id,))
            
            projects = []
            for row in cursor.fetchall():
                project = dict(row)
                # Parse geometry JSON if it exists
                if project.get('geometry'):
                    try:
                        project['geometry'] = json.loads(project['geometry'])
                    except (json.JSONDecodeError, TypeError):
                        project['geometry'] = None
                projects.append(project)
            
            return {
                "projects": projects,
                "total": len(projects),
                "user_id": current_user.id
            }
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to get projects")


@app.post("/api/v1/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(project_data: ProjectCreate, current_user: UserResponse = Depends(get_current_user)):
    """Create a new project"""
    try:
        import json
        geometry_json = json.dumps(project_data.geometry) if project_data.geometry else None
        
        with db.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO projects (name, description, location_name, area_hectares, project_type, user_id, 
                                     start_date, end_date, estimated_carbon_credits, status, geometry)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (project_data.name, project_data.description, project_data.location_name,
                  project_data.area_hectares, project_data.project_type, current_user.id,
                  project_data.start_date, project_data.end_date, project_data.estimated_carbon_credits, "Pending", geometry_json))
            
            project_id = cursor.lastrowid
            conn.commit()
            
            # Get the created project
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            project = dict(cursor.fetchone())
            
            # Parse geometry JSON if it exists
            if project.get('geometry'):
                try:
                    project['geometry'] = json.loads(project['geometry'])
                except (json.JSONDecodeError, TypeError):
                    project['geometry'] = None
            
            logger.info(f"Project created: {project_data.name} by {current_user.email}")
            return ProjectResponse(**project)
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail="Failed to create project")


@app.get("/api/v1/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, current_user: UserResponse = Depends(get_current_user)):
    """Get project by ID"""
    try:
        import json
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            project = dict(project)
            
            # Check authorization
            if project["user_id"] != current_user.id and current_user.role != "Admin":
                raise HTTPException(status_code=403, detail="Not authorized")
            
            # Parse geometry JSON if it exists
            if project.get('geometry'):
                try:
                    project['geometry'] = json.loads(project['geometry'])
                except (json.JSONDecodeError, TypeError):
                    project['geometry'] = None
            
            return ProjectResponse(**project)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        raise HTTPException(status_code=500, detail="Failed to get project")


@app.put("/api/v1/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int, 
    project_data: ProjectCreate, 
    current_user: UserResponse = Depends(get_current_user)
):
    """Update an existing project"""
    try:
        import json
        
        with db.get_connection() as conn:
            # Check if project exists and user has permission
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            existing_project = cursor.fetchone()
            
            if not existing_project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            existing_project = dict(existing_project)
            
            # Check authorization
            if existing_project["user_id"] != current_user.id and current_user.role != "Admin":
                raise HTTPException(status_code=403, detail="Not authorized to update this project")
            
            # Prepare updated data
            geometry_json = json.dumps(project_data.geometry) if project_data.geometry else existing_project.get('geometry')
            
            # Update project
            cursor = conn.execute("""
                UPDATE projects 
                SET name = ?, description = ?, location_name = ?, area_hectares = ?, 
                    project_type = ?, start_date = ?, end_date = ?, 
                    estimated_carbon_credits = ?, geometry = ?
                WHERE id = ?
            """, (
                project_data.name,
                project_data.description,
                project_data.location_name,
                project_data.area_hectares,
                project_data.project_type,
                project_data.start_date,
                project_data.end_date,
                project_data.estimated_carbon_credits,
                geometry_json,
                project_id
            ))
            
            conn.commit()
            
            # Get updated project
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            updated_project = dict(cursor.fetchone())
            
            # Parse geometry JSON if it exists
            if updated_project.get('geometry'):
                try:
                    updated_project['geometry'] = json.loads(updated_project['geometry'])
                except (json.JSONDecodeError, TypeError):
                    updated_project['geometry'] = None
            
            logger.info(f"Project updated: {project_data.name} by {current_user.email}")
            return ProjectResponse(**updated_project)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project: {e}")
        raise HTTPException(status_code=500, detail="Failed to update project")


@app.delete("/api/v1/projects/{project_id}")
async def delete_project(project_id: int, current_user: UserResponse = Depends(get_current_user)):
    """Delete a project"""
    try:
        with db.get_connection() as conn:
            # Check if project exists and user has permission
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            project = dict(project)
            
            # Check authorization
            if project["user_id"] != current_user.id and current_user.role != "Admin":
                raise HTTPException(status_code=403, detail="Not authorized to delete this project")
            
            # Delete related verifications first (foreign key constraint)
            cursor = conn.execute("DELETE FROM verifications WHERE project_id = ?", (project_id,))
            
            # Delete the project
            cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Project not found")
            
            conn.commit()
            
            logger.info(f"Project deleted: {project['name']} by {current_user.email}")
            return {"message": "Project deleted successfully", "project_id": project_id}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete project")


@app.patch("/api/v1/projects/{project_id}/status")
async def update_project_status(
    project_id: int, 
    status_data: dict,
    current_user: UserResponse = Depends(get_current_user)
):
    """Update project status with logging"""
    try:
        new_status = status_data.get("status")
        reason = status_data.get("reason", "")
        notes = status_data.get("notes", "")
        
        if not new_status:
            raise HTTPException(status_code=400, detail="Status is required")
        
        # Validate status - simplified workflow
        valid_statuses = ["Draft", "Pending", "Verified", "Rejected"]
        if new_status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        # Require reason for rejection
        if new_status == "Rejected" and not reason.strip():
            raise HTTPException(status_code=400, detail="Reason is required when rejecting a project")
        
        with db.get_connection() as conn:
            # Check if project exists and user has permission
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            project = dict(project)
            old_status = project["status"]
            
            # Skip if status hasn't changed
            if old_status == new_status:
                return {
                    "message": "Status unchanged",
                    "project_id": project_id,
                    "status": new_status
                }
            
            # Check authorization - allow verifiers and admins to update status
            allowed_roles = ["Admin", "Verifier", "Project Developer"]
            if project["user_id"] != current_user.id and current_user.role not in allowed_roles:
                raise HTTPException(status_code=403, detail="Not authorized to update project status")
            
            # Update project status
            cursor = conn.execute(
                "UPDATE projects SET status = ? WHERE id = ?",
                (new_status, project_id)
            )
            
            # Log the status change
            cursor = conn.execute("""
                INSERT INTO project_status_logs 
                (project_id, old_status, new_status, changed_by_user_id, reason, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                project_id,
                old_status,
                new_status,
                current_user.id,
                reason,
                notes
            ))
            
            conn.commit()
            
            # Get updated project
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            updated_project = dict(cursor.fetchone())
            
            logger.info(f"Project status updated: {project['name']} ({old_status} -> {new_status}) by {current_user.email}")
            return {
                "message": "Project status updated successfully",
                "project_id": project_id,
                "old_status": old_status,
                "new_status": new_status,
                "reason": reason,
                "notes": notes
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project status: {e}")
        raise HTTPException(status_code=500, detail="Failed to update project status")


@app.get("/api/v1/projects/{project_id}/status-logs")
async def get_project_status_logs(
    project_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """Get status change history for a project"""
    try:
        with db.get_connection() as conn:
            # Check if project exists and user has permission
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            project = dict(project)
            
            # Check authorization - project owner, verifiers, and admins can view logs
            if (project["user_id"] != current_user.id and 
                current_user.role not in ["Admin", "Verifier"]):
                raise HTTPException(status_code=403, detail="Not authorized to view project logs")
            
            # Get status logs with user information
            cursor = conn.execute("""
                SELECT 
                    psl.*,
                    u.full_name as changed_by_name,
                    u.role as changed_by_role
                FROM project_status_logs psl
                JOIN users u ON psl.changed_by_user_id = u.id
                WHERE psl.project_id = ?
                ORDER BY psl.created_at DESC
            """, (project_id,))
            
            logs = [dict(row) for row in cursor.fetchall()]
            
            return {
                "project_id": project_id,
                "project_name": project["name"],
                "status_logs": logs
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching project status logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch project status logs")


# ML Integration - Import ML service
try:
    from services.ml_service import ml_service
    logger.info("✅ ML Service imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import ML service: {e}")
    ml_service = None


# ML Analysis endpoints
@app.get("/api/v1/ml/status")
async def get_ml_status(current_user: UserResponse = Depends(get_current_user)):
    """Get ML service status"""
    if ml_service is None:
        return {"error": "ML service not available"}
    
    return {
        "ml_service": ml_service.get_service_status(),
        "models_ready": ml_service.is_initialized,
        "service_version": "1.0.0"
    }


@app.post("/api/v1/ml/analyze-location", response_model=MLAnalysisResponse)
async def analyze_location(
    request: LocationAnalysisRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Analyze a location for carbon credit potential"""
    if ml_service is None or not ml_service.is_initialized:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Verify project belongs to user
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                (request.project_id, current_user.id)
            )
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found or not authorized")
        
        # Run ML analysis
        coordinates = (request.latitude, request.longitude)
        results = await ml_service.analyze_location(
            coordinates=coordinates,
            project_id=request.project_id,
            analysis_type=request.analysis_type
        )
        
        logger.info(f"Location analysis completed for project {request.project_id}")
        
        return MLAnalysisResponse(
            project_id=request.project_id,
            analysis_type=request.analysis_type,
            status="completed",
            results=results,
            timestamp=results.get("timestamp", datetime.now().isoformat())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/api/v1/ml/forest-cover")
async def analyze_forest_cover(
    project_id: int,
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_user)
):
    """Analyze forest cover from uploaded satellite image"""
    if ml_service is None or not ml_service.is_initialized:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Verify project belongs to user
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                (project_id, current_user.id)
            )
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found or not authorized")
        
        # Save uploaded file
        file_content = await file.read()
        file_path = await ml_service.save_uploaded_file(file_content, file.filename)
        
        # Run forest cover analysis
        results = await ml_service.analyze_forest_cover(
            image_path=file_path,
            project_id=project_id
        )
        
        logger.info(f"Forest cover analysis completed for project {project_id}")
        
        return {
            "project_id": project_id,
            "analysis_type": "forest_cover",
            "status": "completed",
            "results": results,
            "file_processed": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forest cover analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/api/v1/ml/change-detection")
async def detect_changes(
    project_id: int,
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_user)
):
    """Detect forest changes between two satellite images"""
    if ml_service is None or not ml_service.is_initialized:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Verify project belongs to user
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                (project_id, current_user.id)
            )
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found or not authorized")
        
        # Save uploaded files
        before_content = await before_image.read()
        after_content = await after_image.read()
        
        before_path = await ml_service.save_uploaded_file(before_content, before_image.filename)
        after_path = await ml_service.save_uploaded_file(after_content, after_image.filename)
        
        # Run change detection analysis
        results = await ml_service.detect_changes(
            before_image_path=before_path,
            after_image_path=after_path,
            project_id=project_id
        )
        
        logger.info(f"Change detection completed for project {project_id}")
        
        return {
            "project_id": project_id,
            "analysis_type": "change_detection",
            "status": "completed",
            "results": results,
            "files_processed": {
                "before": before_image.filename,
                "after": after_image.filename
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change detection failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


# Verification API Endpoints
@app.get("/api/v1/verification")
async def get_verifications(
    project_id: Optional[int] = None,
    status: Optional[str] = None,
    current_user: UserResponse = Depends(get_current_user)
):
    """Get verification records with optional filters"""
    try:
        with db.get_connection() as conn:
            query = """
                SELECT v.*, p.name as project_name 
                FROM verifications v
                JOIN projects p ON v.project_id = p.id
                WHERE p.user_id = ?
            """
            params = [current_user.id]
            
            if project_id:
                query += " AND v.project_id = ?"
                params.append(project_id)
                
            if status:
                query += " AND v.status = ?"
                params.append(status)
                
            query += " ORDER BY v.created_at DESC"
            
            cursor = conn.execute(query, params)
            verifications = [dict(row) for row in cursor.fetchall()]
            
            return {"verifications": verifications}
            
    except Exception as e:
        logger.error(f"Error fetching verifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch verifications")


@app.post("/api/v1/verification", response_model=VerificationResponse, status_code=status.HTTP_201_CREATED)
async def create_verification(
    verification_data: VerificationCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create a new verification record"""
    try:
        # Verify project belongs to user
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                (verification_data.project_id, current_user.id)
            )
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found or not authorized")
            
            # Check if verification already exists for this project
            cursor = conn.execute(
                "SELECT id FROM verifications WHERE project_id = ?",
                (verification_data.project_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                raise HTTPException(status_code=400, detail="Verification already exists for this project")
            
            # Create verification record
            timestamp = datetime.now().isoformat()
            
            # Generate AI confidence score based on project data
            ai_confidence = 0.85 + (hash(str(verification_data.project_id)) % 100) / 1000.0
            ai_confidence = min(ai_confidence, 0.95)  # Cap at 95%
            
            # Estimate carbon impact based on project type and area (mock calculation)
            carbon_impact = verification_data.carbon_impact
            if not carbon_impact:
                # Mock calculation: assume 10-20 tons CO2/hectare/year
                project_dict = dict(project)
                area = project_dict.get('area_hectares', 100)
                carbon_impact = area * (12 + (hash(str(verification_data.project_id)) % 8))
            
            # Generate certificate ID
            project_dict = dict(project)
            location_code = project_dict['location_name'][:2].upper()
            certificate_id = f"CC-2025-{location_code}-001-{verification_data.project_id}"
            
            cursor = conn.execute("""
                INSERT INTO verifications (
                    project_id, status, carbon_impact, ai_confidence,
                    human_verified, blockchain_certified, certificate_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                verification_data.project_id,
                "pending",
                carbon_impact,
                ai_confidence,
                False,
                False,
                certificate_id,
                timestamp,
                timestamp
            ))
            
            verification_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Verification created for project {verification_data.project_id}")
            
            return VerificationResponse(
                id=verification_id,
                project_id=verification_data.project_id,
                status="pending",
                carbon_impact=carbon_impact,
                ai_confidence=ai_confidence,
                human_verified=False,
                blockchain_certified=False,
                certificate_id=certificate_id,
                created_at=timestamp,
                updated_at=timestamp
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating verification: {e}")
        raise HTTPException(status_code=500, detail="Failed to create verification")


@app.get("/api/v1/verification/{verification_id}", response_model=VerificationResponse)
async def get_verification(
    verification_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """Get verification by ID"""
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT v.* FROM verifications v
                JOIN projects p ON v.project_id = p.id
                WHERE v.id = ? AND p.user_id = ?
            """, (verification_id, current_user.id))
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Verification not found")
            
            verification = dict(row)
            return VerificationResponse(**verification)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching verification: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch verification")


@app.post("/api/v1/verification/{verification_id}/human-review")
async def submit_human_review(
    verification_id: int,
    review_data: HumanReviewRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Submit human review for verification"""
    try:
        with db.get_connection() as conn:
            # Verify access
            cursor = conn.execute("""
                SELECT v.* FROM verifications v
                JOIN projects p ON v.project_id = p.id
                WHERE v.id = ? AND p.user_id = ?
            """, (verification_id, current_user.id))
            
            verification = cursor.fetchone()
            if not verification:
                raise HTTPException(status_code=404, detail="Verification not found")
            
            # Update verification with human review
            new_status = "approved" if review_data.approved else "rejected"
            timestamp = datetime.now().isoformat()
            
            cursor = conn.execute("""
                UPDATE verifications 
                SET status = ?, human_verified = ?, updated_at = ?
                WHERE id = ?
            """, (
                new_status,
                True,
                timestamp,
                verification_id
            ))
            
            conn.commit()
            
            # Fetch updated verification
            cursor = conn.execute("SELECT * FROM verifications WHERE id = ?", (verification_id,))
            updated_verification = dict(cursor.fetchone())
            
            logger.info(f"Human review submitted for verification {verification_id}: {new_status}")
            
            return VerificationResponse(**updated_verification)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting human review: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit human review")


# XAI API Endpoints
@app.post("/api/v1/xai/generate-explanation", response_model=ExplanationResponse)
async def generate_explanation(
    request: ExplanationRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Generate AI explanation for model prediction"""
    if ml_service is None or not ml_service.is_initialized:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Verify project belongs to user
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                (request.project_id, current_user.id)
            )
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found or not authorized")
        
        # Generate explanation using ML service
        results = await ml_service.generate_explanation(
            project_id=request.project_id,
            method=request.explanation_method,
            prediction_id=request.prediction_id,
            image_path=request.image_path
        )
        
        logger.info(f"XAI explanation generated for project {request.project_id} using {request.explanation_method}")
        
        return ExplanationResponse(
            explanation_id=results["explanation_id"],
            project_id=request.project_id,
            method=request.explanation_method,
            status="completed",
            results=results["analysis"],
            visualization_paths=results["visualizations"],
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"XAI explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail="Explanation generation failed")


@app.get("/api/v1/xai/explanation/{explanation_id}")
async def get_explanation(
    explanation_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Retrieve generated explanation by ID"""
    if ml_service is None or not ml_service.is_initialized:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        explanation = await ml_service.get_explanation(explanation_id)
        
        if not explanation:
            raise HTTPException(status_code=404, detail="Explanation not found")
        
        # Verify user has access to this explanation's project
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                (explanation["project_id"], current_user.id)
            )
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=403, detail="Not authorized to access this explanation")
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving explanation: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve explanation")


@app.post("/api/v1/xai/compare-explanations")
async def compare_explanations(
    request: CompareExplanationsRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Compare multiple explanations side-by-side"""
    if ml_service is None or not ml_service.is_initialized:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Verify all explanations belong to user's projects
        for explanation_id in request.explanation_ids:
            explanation = await ml_service.get_explanation(explanation_id)
            if not explanation:
                raise HTTPException(status_code=404, detail=f"Explanation {explanation_id} not found")
            
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                    (explanation["project_id"], current_user.id)
                )
                project = cursor.fetchone()
                
                if not project:
                    raise HTTPException(status_code=403, detail="Not authorized to access explanations")
        
        # Generate comparison visualization
        comparison_results = await ml_service.compare_explanations(
            explanation_ids=request.explanation_ids,
            comparison_type=request.comparison_type
        )
        
        logger.info(f"XAI explanation comparison generated for {len(request.explanation_ids)} explanations")
        
        return {
            "comparison_id": comparison_results["comparison_id"],
            "explanation_ids": request.explanation_ids,
            "comparison_type": request.comparison_type,
            "status": "completed",
            "results": comparison_results["analysis"],
            "visualization_path": comparison_results["visualization_path"],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"XAI explanation comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Explanation comparison failed")


@app.get("/api/v1/xai/methods")
async def get_available_xai_methods(current_user: UserResponse = Depends(get_current_user)):
    """Get available XAI explanation methods"""
    return {
        "methods": [
            {
                "name": "shap",
                "display_name": "SHAP (SHapley Additive exPlanations)",
                "description": "Feature importance using Shapley values",
                "supported_models": ["forest_cover", "change_detection", "ensemble"],
                "visualization_types": ["waterfall", "force_plot", "summary_plot"]
            },
            {
                "name": "lime",
                "display_name": "LIME (Local Interpretable Model-agnostic Explanations)",
                "description": "Local explanations for individual predictions",
                "supported_models": ["forest_cover", "change_detection"],
                "visualization_types": ["image_segments", "feature_importance"]
            },
            {
                "name": "integrated_gradients",
                "display_name": "Integrated Gradients",
                "description": "Attribution method for deep learning models",
                "supported_models": ["forest_cover", "change_detection", "time_series"],
                "visualization_types": ["attribution_map", "sensitivity_analysis"]
            }
        ],
        "service_status": "operational" if ml_service and ml_service.is_initialized else "unavailable"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Carbon Credit Verification API...")
    logger.info(f"Database: {DATABASE_PATH}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
