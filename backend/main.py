"""
Carbon Credit Verification API - Professional Edition
Simplified architecture with SQLite database integration
"""
import os
import hashlib
import secrets
import logging
import json
import time
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, field_validator
from passlib.context import CryptContext
from fastapi import UploadFile, File
from typing import Tuple

# Import role utilities
from utils.role_utils import (
    is_admin,
    can_manage_projects,
    can_verify_projects,
    can_access_analytics,
    can_manage_users,
    can_access_blockchain,
    normalize_role,
    RolePermissions,
    Roles,
)

# Runtime configuration (env-driven; no secrets in the repo)
from config import settings
# DB connection factory (Turso/libSQL in prod, sqlite3 locally)
import dbdriver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

# Database configuration (absolute path from env-driven config)
DATABASE_PATH = settings.DATABASE_PATH


# Database Manager
class DatabaseManager:
    """Professional SQLite database manager"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup (Turso in prod, sqlite3 locally)."""
        conn = None
        try:
            # Only the local sqlite3 file needs a directory; Turso is remote.
            if not dbdriver.TURSO_URL:
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = dbdriver.connect(self.db_path)
            yield conn
            # Explicitly commit the transaction
            conn.commit()
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

                # Password reset tokens table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS password_reset_tokens (
                        token TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        used INTEGER NOT NULL DEFAULT 0,
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
                
                # XAI Explanations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS xai_explanations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        explanation_id TEXT UNIQUE NOT NULL,
                        project_id INTEGER NOT NULL,
                        user_id INTEGER NOT NULL,
                        method TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        confidence_score REAL,
                        business_summary TEXT,
                        explanation_data TEXT, -- JSON data for the full explanation
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id),
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # IoT Sensors table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS iot_sensors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sensor_id TEXT UNIQUE NOT NULL,
                        sensor_type TEXT NOT NULL,
                        location_lat REAL NOT NULL,
                        location_lng REAL NOT NULL,
                        project_id INTEGER NOT NULL,
                        status TEXT DEFAULT 'active',
                        last_reading TEXT,
                        installation_date TEXT,
                        calibration_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                """)
                
                # Sensor Readings table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_readings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sensor_id TEXT NOT NULL,
                        reading_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (sensor_id) REFERENCES iot_sensors (sensor_id)
                    )
                """)
                
                # User Settings table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        setting_key TEXT NOT NULL,
                        setting_value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id),
                        UNIQUE(user_id, setting_key)
                    )
                """)

                # Verifications table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS verifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        carbon_impact REAL,
                        ai_confidence REAL,
                        human_verified BOOLEAN DEFAULT FALSE,
                        blockchain_certified BOOLEAN DEFAULT FALSE,
                        certificate_id TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                """)

                # Project Status Logs table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS project_status_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER NOT NULL,
                        old_status TEXT NOT NULL,
                        new_status TEXT NOT NULL,
                        changed_by_user_id INTEGER NOT NULL,
                        reason TEXT,
                        notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id),
                        FOREIGN KEY (changed_by_user_id) REFERENCES users (id)
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

                    # Add geometry column if missing
                    if 'geometry' not in columns:
                        conn.execute("ALTER TABLE projects ADD COLUMN geometry TEXT")

                    # Store offline-computed carbon analysis (JSON) per project
                    if 'carbon_analysis' not in columns:
                        conn.execute("ALTER TABLE projects ADD COLUMN carbon_analysis TEXT")

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


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


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


class CarbonAnalysisData(BaseModel):
    """An offline-computed carbon analysis (ml/analyze) stored on a project.

    torch/rasterio cannot run on the serverless backend, so the ML runs off-platform
    and posts its result here; the app then stores, displays, and reports it.
    """
    model_config = {"protected_namespaces": ()}  # allow the model_version field name

    scene_id: Optional[str] = None
    scene_date: Optional[str] = None
    cloud_cover: Optional[float] = None
    total_area_hectares: Optional[float] = None
    forest_coverage_percent: Optional[float] = None
    forest_area_hectares: Optional[float] = None
    total_co2e_tonnes: Optional[float] = None
    carbon_density_tco2e_per_ha: Optional[float] = None
    biome: Optional[str] = None
    mean_probability: Optional[float] = None
    model_version: Optional[str] = None


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


# IoT Sensor Models
class IoTSensorCreate(BaseModel):
    sensor_id: str
    sensor_type: str  # "soil_moisture", "co2_flux", "temperature", "tree_growth"
    location_lat: float
    location_lng: float
    project_id: int
    installation_date: Optional[str] = None
    calibration_data: Optional[dict] = None

class IoTSensorResponse(BaseModel):
    id: int
    sensor_id: str
    sensor_type: str
    location_lat: float
    location_lng: float
    project_id: int
    status: str
    last_reading: Optional[dict] = None
    installation_date: Optional[str] = None
    created_at: str

class SensorReadingCreate(BaseModel):
    sensor_id: str
    reading_type: str
    value: float
    unit: str
    timestamp: Optional[str] = None
    metadata: Optional[dict] = None

class SensorReadingResponse(BaseModel):
    id: int
    sensor_id: str
    reading_type: str
    value: float
    unit: str
    timestamp: str
    metadata: Optional[dict] = None


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


def send_password_reset_email(to_email: str, reset_link: str) -> bool:
    """Send a password-reset email via SMTP. Returns True if sent.

    If SMTP is not configured (no SMTP_HOST), returns False without raising — the
    caller then falls back to logging the link (and optionally returning it in the
    response when PASSWORD_RESET_EXPOSE_LINK is enabled for demo use).
    """
    if not settings.SMTP_HOST:
        return False
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg["Subject"] = "Reset your Carbon Credit Verification password"
        msg["From"] = settings.SMTP_FROM or settings.SMTP_USER
        msg["To"] = to_email
        msg.set_content(
            "We received a request to reset your password.\n\n"
            f"Reset it here (valid for {settings.PASSWORD_RESET_TTL_MINUTES} minutes):\n{reset_link}\n\n"
            "If you didn't request this, you can ignore this email."
        )
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as smtp:
            smtp.starttls()
            if settings.SMTP_USER:
                smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"Failed to send reset email: {e}")
        return False


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
    """Get user by auth token, rejecting expired tokens.

    Strict expiry: a token with a NULL or past ``expires_at`` is not honored, so
    leaked/legacy tokens (e.g. from an old database) can never authenticate.

    On serverless + a remote DB, a just-issued token can momentarily be invisible
    to a read served by a different instance (read-after-write lag), so we retry a
    few times before rejecting — otherwise a user gets a spurious 401 right after
    logging in.
    """
    query = """
        SELECT u.* FROM users u
        JOIN auth_tokens t ON u.id = t.user_id
        WHERE t.token = ?
          AND t.expires_at IS NOT NULL
          AND t.expires_at > datetime('now')
    """
    attempts = 4 if dbdriver.TURSO_URL else 1  # only the remote DB has the lag window
    for attempt in range(attempts):
        try:
            with db.get_connection() as conn:
                row = conn.execute(query, (token,)).fetchone()
                if row:
                    return dict(row)
        except Exception as e:
            logger.error(f"Error getting user by token: {e}")
            return None
        if attempt < attempts - 1:
            time.sleep(0.25)
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


def seed_admin() -> None:
    """Create a single admin from env vars if (and only if) no users exist.

    Keeps credentials out of the repository: the database is created fresh at
    runtime and the first admin comes from ADMIN_EMAIL / ADMIN_PASSWORD.
    """
    if not settings.ADMIN_EMAIL or not settings.ADMIN_PASSWORD:
        return
    try:
        with db.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count:
                return
            conn.execute(
                "INSERT INTO users (email, hashed_password, full_name, role) "
                "VALUES (?, ?, ?, ?)",
                (
                    settings.ADMIN_EMAIL.lower(),
                    hash_password(settings.ADMIN_PASSWORD),
                    settings.ADMIN_NAME,
                    Roles.ADMINISTRATOR.value,
                ),
            )
            conn.commit()
            logger.info(f"Seeded initial admin user: {settings.ADMIN_EMAIL.lower()}")
    except Exception as e:
        logger.error(f"Admin seed failed: {e}")


def store_token(token: str, user_id: int, ttl_minutes: Optional[int] = None):
    """Store auth token in database with an expiry timestamp."""
    if ttl_minutes is None:
        ttl_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    try:
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO auth_tokens (token, user_id, expires_at) "
                "VALUES (?, ?, datetime('now', ?))",
                (token, user_id, f"+{int(ttl_minutes)} minutes")
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error storing token: {e}")
        raise


def delete_token(token: str):
    """Delete auth token from database"""
    try:
        with db.get_connection() as conn:
            conn.execute(
                "DELETE FROM auth_tokens WHERE token = ?",
                (token,)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error deleting token: {e}")
        # Don't raise further to avoid logout errors bubbling


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """Get current authenticated user"""
    if token.startswith('Bearer '):
        token = token[7:]
    
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return UserResponse(**user)


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI application
app = FastAPI(
    title="Carbon Credit Verification API",
    description="Professional API for carbon credit verification and management",
    version="2.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _seed_admin_on_startup() -> None:
    seed_admin()


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
    blockchain_status = "connected" if blockchain_service and blockchain_service.is_connected() else "disconnected"
    return {
        "status": "healthy",
        "service": "Carbon Credit Verification API",
        "version": "2.0.0",
        "database": "Turso (libSQL)" if dbdriver.TURSO_URL else "SQLite",
        "architecture": "Simplified Professional",
        "blockchain": blockchain_status
    }

# Blockchain endpoints
@app.get("/api/v1/blockchain/status")
async def get_blockchain_status(current_user: UserResponse = Depends(get_current_user)):
    """Get blockchain network status"""
    if not blockchain_service:
        return {"error": "Blockchain service not available"}
    
    return blockchain_service.get_network_info()

@app.get("/api/v1/blockchain/enabled")
async def blockchain_enabled(current_user: UserResponse = Depends(get_current_user)):
    """Report whether on-chain certification is configured (used by the UI to gate features)."""
    return {"enabled": bool(blockchain_service and getattr(blockchain_service, "enabled", False))}


@app.get("/api/v1/blockchain/verify")
async def verify_certificate(
    token_id_or_hash: str = Query(..., description="Token ID or transaction hash to verify"),
    current_user: UserResponse = Depends(get_current_user)
):
    """Verify a carbon credit certificate"""
    if not blockchain_service or not getattr(blockchain_service, "enabled", False):
        raise HTTPException(status_code=503, detail="Blockchain certification is not configured")

    result = blockchain_service.verify_certificate(token_id_or_hash)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result

@app.post("/api/v1/blockchain/mint")
async def mint_carbon_credit(
    project_id: int,
    carbon_amount: int,
    recipient_address: str = None,
    current_user: UserResponse = Depends(get_current_user)
):
    """Mint a carbon credit NFT (Admin only)"""
    if not is_admin(current_user.role):
        raise HTTPException(status_code=403, detail="Admin access required")

    if not blockchain_service or not getattr(blockchain_service, "enabled", False):
        raise HTTPException(status_code=503, detail="Blockchain certification is not configured")

    # Validate project and verification state
    with db.get_connection() as conn:
        # Ensure project exists
        cursor = conn.execute("SELECT name, location_name FROM projects WHERE id = ?", (project_id,))
        project = cursor.fetchone()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Ensure there's an approved verification and not already certified
        cursor = conn.execute(
            "SELECT id, status, blockchain_certified FROM verifications WHERE project_id = ?",
            (project_id,)
        )
        verification = cursor.fetchone()
        if not verification:
            raise HTTPException(status_code=404, detail="No verification found for this project")
        verification = dict(verification)
        if verification.get("blockchain_certified"):
            raise HTTPException(status_code=400, detail="Verification already blockchain certified")
        if str(verification.get("status", "")).lower() not in ["approved", "verified"]:
            raise HTTPException(status_code=400, detail="Verification must be approved before minting")
    
    # A real recipient wallet is required — never mint to a placeholder address.
    if not recipient_address:
        raise HTTPException(status_code=400, detail="recipient_address is required")

    # Generate verification hash
    import hashlib
    verification_data = f"{project_id}-{carbon_amount}-{current_user.id}"
    verification_hash = hashlib.sha256(verification_data.encode()).hexdigest()
    
    # Perform mint
    result = blockchain_service.mint_carbon_credit_nft(
        recipient_address=recipient_address,
        project_id=project_id,
        carbon_amount=int(carbon_amount),
        project_name=project['name'],
        location=project['location_name'],
        verification_hash=verification_hash
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Persist certification result in database
    try:
        with db.get_connection() as conn:
            from datetime import datetime
            tx_hash = result.get("transaction_hash")
            # Mark as certified and store blockchain reference (tx hash)
            conn.execute(
                """
                UPDATE verifications
                SET blockchain_certified = ?,
                    certificate_id = ?,
                    status = ?,
                    updated_at = ?
                WHERE project_id = ?
                """,
                (
                    True,
                    tx_hash or str(result.get("token_id")),
                    "verified",
                    datetime.now().isoformat(),
                    project_id,
                )
            )
            conn.commit()
            cursor = conn.execute("SELECT id FROM verifications WHERE project_id = ?", (project_id,))
            ver_row = cursor.fetchone()
            if ver_row:
                result["verification_id"] = ver_row[0]
    except Exception as e:
        logger.warning(f"Failed to persist blockchain certification: {e}")
    
    return result


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

    # Security: never trust a client-supplied role at open registration.
    # All self-registered users get the lowest-privilege role; elevated roles
    # are assigned only out-of-band (e.g. the seeded admin).
    user_data.role = Roles.PROJECT_DEVELOPER.value

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
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
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


@app.post("/api/v1/auth/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """Logout current token"""
    # oauth2_scheme returns the raw token string
    delete_token(token)
    return {"message": "Logged out"}


@app.post("/api/v1/auth/forgot-password")
@limiter.limit("5/minute")
async def forgot_password(request: Request, data: ForgotPasswordRequest):
    """Start a password reset: issue a single-use, time-limited token and email it.

    Always returns the same generic message (no account enumeration). When SMTP is
    not configured, the link is logged server-side (and returned in the response only
    if PASSWORD_RESET_EXPOSE_LINK is enabled — for local/demo use, never production).
    """
    generic = {"message": "If an account with that email exists, a reset link has been sent."}
    email = data.email.strip().lower()
    user = get_user_by_email(email)
    if not user:
        return generic  # do not reveal whether the email exists

    reset_token = create_token()
    try:
        with db.get_connection() as conn:
            # Invalidate any prior unused tokens for this user, then store the new one.
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user["id"],))
            conn.execute(
                "INSERT INTO password_reset_tokens (token, user_id, expires_at) "
                "VALUES (?, ?, datetime('now', ?))",
                (reset_token, user["id"], f"+{settings.PASSWORD_RESET_TTL_MINUTES} minutes"),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error creating reset token: {e}")
        raise HTTPException(status_code=500, detail="Failed to start password reset")

    reset_link = f"{settings.FRONTEND_URL.rstrip('/')}/reset-password?token={reset_token}"
    sent = send_password_reset_email(email, reset_link)
    if not sent:
        logger.info(f"Password reset link for {email} (email not sent): {reset_link}")

    response = dict(generic)
    if not sent and settings.PASSWORD_RESET_EXPOSE_LINK:
        # Demo convenience only — insecure, off by default.
        response["reset_link"] = reset_link
    return response


@app.post("/api/v1/auth/reset-password")
@limiter.limit("5/minute")
async def reset_password(request: Request, data: ResetPasswordRequest):
    """Complete a password reset with a valid, unused, unexpired token."""
    try:
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT user_id FROM password_reset_tokens "
                "WHERE token = ? AND used = 0 AND expires_at > datetime('now')",
                (data.token,),
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Invalid or expired reset token")
            user_id = row["user_id"]

            conn.execute(
                "UPDATE users SET hashed_password = ? WHERE id = ?",
                (hash_password(data.new_password), user_id),
            )
            conn.execute("UPDATE password_reset_tokens SET used = 1 WHERE token = ?", (data.token,))
            # Invalidate existing sessions so the account is fully re-secured.
            conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
            conn.commit()

        return {"message": "Password has been reset. You can now log in with your new password."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset password")


# Project endpoints
@app.get("/api/v1/projects")
async def get_projects(current_user: UserResponse = Depends(get_current_user)):
    """Get projects based on user role"""
    try:
        import json
        with db.get_connection() as conn:
            # Role-based access control
            if can_verify_projects(current_user.role):
                # Administrator and Verifier users can see all projects
                cursor = conn.execute("""
                    SELECT * FROM projects ORDER BY created_at DESC
                """)
            else:
                # Project Developer and other users see only their own projects
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
            if project["user_id"] != current_user.id and not is_admin(current_user.role):
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
            if existing_project["user_id"] != current_user.id and not is_admin(current_user.role):
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
            if project["user_id"] != current_user.id and not is_admin(current_user.role):
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
            if project["user_id"] != current_user.id and not can_manage_projects(current_user.role) and not can_verify_projects(current_user.role):
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
                not can_verify_projects(current_user.role)):
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

# Report Service - Import report service
try:
    from services.report_service import ReportService
    report_service = ReportService(DATABASE_PATH)
    logger.info("✅ Report Service imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import Report service: {e}")
    report_service = None

# Blockchain Service - Import blockchain service
try:
    from services.blockchain_service import blockchain_service
    logger.info("✅ Blockchain Service imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import Blockchain service: {e}")
    blockchain_service = None


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
            if is_admin(current_user.role):
                # Administrator users see all verifications
                query = """
                    SELECT v.*, p.name as project_name 
                    FROM verifications v
                    JOIN projects p ON v.project_id = p.id
                    WHERE 1=1
                """
                params = []
            else:
                # Other users see only verifications for their projects
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
        # Verify project exists and user is authorized
        with db.get_connection() as conn:
            if can_verify_projects(current_user.role):
                # Administrator and Verifier users can verify any project
                cursor = conn.execute(
                    "SELECT * FROM projects WHERE id = ?",
                    (verification_data.project_id,)
                )
            else:
                # Project Developer and other users can only verify their own projects
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

            # Honest human-in-the-loop record. No AI score is fabricated: ai_confidence
            # is populated only by a real ML analysis of the project's imagery (not wired
            # into this endpoint), so it stays null here. carbon_impact is the applicant's
            # own estimate if supplied, never a synthetic value.
            ai_confidence = None
            carbon_impact = verification_data.carbon_impact

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
            if is_admin(current_user.role):
                # Administrator can review any verification
                cursor = conn.execute("""
                    SELECT v.* FROM verifications v
                    JOIN projects p ON v.project_id = p.id
                    WHERE v.id = ?
                """, (verification_id,))
            else:
                # Other users can only review verifications for their projects
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


# Import Enhanced XAI Service
try:
    from services.xai_service import xai_service
    logger.info("✅ XAI Service imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import XAI service: {e}")
    xai_service = None

# Enhanced XAI Models
class EnhancedExplanationRequest(BaseModel):
    model_id: str = "forest_cover_ensemble"
    instance_data: dict
    explanation_method: str = "shap"
    business_friendly: bool = True
    include_uncertainty: bool = True

class ReportRequest(BaseModel):
    explanation_id: str
    format: str = "pdf"
    include_business_summary: bool = True


class UserSettingsUpdate(BaseModel):
    email_notifications: Optional[bool] = None
    project_notifications: Optional[bool] = None
    security_alerts: Optional[bool] = None
    api_key_enabled: Optional[bool] = None
    theme: Optional[str] = None  # light, dark, auto
    language: Optional[str] = None
    timezone: Optional[str] = None


class UserSettingsResponse(BaseModel):
    email_notifications: bool
    project_notifications: bool 
    security_alerts: bool
    api_key_enabled: bool
    theme: str
    language: str
    timezone: str
    api_key: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

# Enhanced XAI API Endpoints
@app.post("/api/v1/xai/generate-explanation")
async def generate_enhanced_explanation(
    request: EnhancedExplanationRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Generate enhanced AI explanation with business context"""
    if xai_service is None:
        raise HTTPException(status_code=503, detail="XAI service not available")
    try:
        # Generate explanation using enhanced XAI service
        explanation = await xai_service.generate_explanation(
            model_id=request.model_id,
            instance_data=request.instance_data,
            explanation_method=request.explanation_method,
            business_friendly=request.business_friendly,
            include_uncertainty=request.include_uncertainty
        )
        
        if explanation.get("disabled"):
            raise HTTPException(status_code=503, detail=explanation["error"])
        if "error" in explanation:
            raise HTTPException(status_code=500, detail=explanation["error"])

        # Save explanation to database for persistence
        try:
            with db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO xai_explanations (
                        explanation_id, project_id, user_id, method, timestamp, 
                        confidence_score, business_summary, explanation_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    explanation['explanation_id'],
                    request.instance_data.get('project_id'),
                    current_user.id,
                    explanation['method'],
                    explanation['timestamp'],
                    explanation.get('confidence_score'),
                    explanation.get('business_summary', ''),
                    json.dumps(explanation)  # Store full explanation as JSON
                ))
                conn.commit()
                logger.info(f"Explanation saved to database: {explanation['explanation_id']}")
        except Exception as e:
            logger.warning(f"Failed to save explanation to database: {e}")
            # Don't fail the request if database save fails
        
        logger.info(f"Enhanced XAI explanation generated: {explanation['explanation_id']}")
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced XAI explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail="Explanation generation failed")

@app.post("/api/v1/xai/generate-explanation-legacy", response_model=ExplanationResponse)
async def generate_explanation_legacy(
    request: ExplanationRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Legacy explanation endpoint — disabled (no fabricated explanations are served)."""
    raise HTTPException(
        status_code=503, detail="Explainability is not available in this build"
    )


@app.get("/api/v1/xai/explanation/{explanation_id}")
async def get_enhanced_explanation(
    explanation_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Retrieve enhanced explanation by ID"""
    try:
        # Get explanation from database
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT explanation_data FROM xai_explanations 
                WHERE explanation_id = ? AND user_id = ?
            """, (explanation_id, current_user.id))
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Explanation not found")
            
            # Parse the JSON explanation data
            explanation = json.loads(row[0])
            return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving explanation: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve explanation")

@app.get("/api/v1/xai/explanation-legacy/{explanation_id}")
async def get_explanation_legacy(
    explanation_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Legacy endpoint — disabled (its generator no longer serves fabricated explanations)."""
    raise HTTPException(
        status_code=503, detail="Explainability is not available in this build"
    )


@app.post("/api/v1/xai/compare-explanations")
async def compare_enhanced_explanations(
    request: CompareExplanationsRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Compare multiple enhanced explanations with business analysis"""
    if xai_service is None:
        raise HTTPException(status_code=503, detail="XAI service not available")
    try:
        # Generate comparison using enhanced XAI service
        comparison = await xai_service.compare_explanations(request.explanation_ids)
        
        if "error" in comparison:
            raise HTTPException(status_code=404, detail=comparison["error"])
        
        logger.info(f"Enhanced XAI explanation comparison generated for {len(request.explanation_ids)} explanations")
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced XAI explanation comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Explanation comparison failed")

@app.post("/api/v1/xai/generate-report")
async def generate_explanation_report(
    request: ReportRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Generate professional report from explanation"""
    if xai_service is None:
        raise HTTPException(status_code=503, detail="XAI service not available")
    try:
        report = await xai_service.generate_report(
            explanation_id=request.explanation_id,
            format=request.format,
            include_business_summary=request.include_business_summary
        )
        
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])
        
        logger.info(f"XAI report generated: {report['report_id']}")
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail="Report generation failed")

@app.get("/api/v1/xai/explanation-history/{project_id}")
async def get_explanation_history(
    project_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """Get explanation history for a project"""
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
        
        # Get all explanations for this project from database
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT explanation_id, method, timestamp, confidence_score, business_summary, explanation_data
                FROM xai_explanations 
                WHERE project_id = ? AND user_id = ?
                ORDER BY created_at DESC
            """, (project_id, current_user.id))
            
            project_explanations = []
            for row in cursor.fetchall():
                # Truncate business summary if too long
                summary = row[4] or ""
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                    
                project_explanations.append({
                    "explanation_id": row[0],
                    "method": row[1],
                    "timestamp": row[2],
                    "confidence_score": row[3],
                    "business_summary": summary
                })
        
        return {
            "project_id": project_id,
            "explanation_count": len(project_explanations),
            "explanations": project_explanations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving explanation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve explanation history")

@app.post("/api/v1/xai/compare-explanations-legacy")
async def compare_explanations_legacy(
    request: CompareExplanationsRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Legacy explanation-comparison endpoint — disabled (no fabricated explanations)."""
    raise HTTPException(
        status_code=503, detail="Explainability is not available in this build"
    )


@app.get("/api/v1/xai/methods")
async def get_available_xai_methods(current_user: UserResponse = Depends(get_current_user)):
    """Get available enhanced XAI explanation methods"""
    return {
        "methods": [
            {
                "name": "shap",
                "display_name": "SHAP (SHapley Additive exPlanations)",
                "description": "Feature importance using Shapley values with business context",
                "supported_models": ["forest_cover", "change_detection", "ensemble"],
                "visualization_types": ["waterfall", "force_plot", "summary_plot", "business_dashboard"],
                "business_features": ["plain_language_explanations", "financial_impact", "risk_assessment"]
            },
            {
                "name": "lime",
                "display_name": "LIME (Local Interpretable Model-agnostic Explanations)",
                "description": "Local explanations with regulatory compliance features",
                "supported_models": ["forest_cover", "change_detection"],
                "visualization_types": ["image_segments", "feature_importance", "business_metrics"],
                "business_features": ["stakeholder_reports", "uncertainty_quantification", "compliance_notes"]
            },
            {
                "name": "integrated_gradients",
                "display_name": "Integrated Gradients",
                "description": "Attribution method with professional reporting capabilities",
                "supported_models": ["forest_cover", "change_detection", "time_series"],
                "visualization_types": ["attribution_map", "sensitivity_analysis", "confidence_intervals"],
                "business_features": ["executive_summaries", "audit_trails", "regulatory_compliance"]
            }
        ],
        "enhanced_features": [
            "Business-friendly explanations",
            "Professional PDF reports",
            "Regulatory compliance documentation",
            "Uncertainty quantification",
            "Risk assessment",
            "Explanation comparison and history",
            "Executive summaries",
            "Audit trails for regulatory review"
        ],
        "service_status": "enhanced_operational",
        "compliance": {
            "eu_ai_act": "compliant",
            "carbon_standards": "vcs_gold_standard_ready",
            "audit_ready": True
        }
    }


# Report Generation API Endpoints
@app.get("/api/v1/reports/certificate/{verification_id}")
async def download_verification_certificate(
    verification_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """Download verification certificate PDF"""
    if report_service is None:
        raise HTTPException(status_code=503, detail="Report service not available")
    
    try:
        logger.info(f"Certificate request - verification_id: {verification_id}, user: {current_user.email}, role: {current_user.role}")
        logger.info(f"is_admin check result: {is_admin(current_user.role)}")
        
        with db.get_connection() as conn:
            # Verify the verification exists and user has access
            if is_admin(current_user.role):
                logger.info("Using admin query")
                cursor = conn.execute(
                    "SELECT project_id FROM verifications WHERE id = ?",
                    (verification_id,)
                )
            else:
                logger.info(f"Using user query for user_id: {current_user.id}")
                cursor = conn.execute("""
                    SELECT v.project_id 
                    FROM verifications v
                    JOIN projects p ON v.project_id = p.id
                    WHERE v.id = ? AND p.user_id = ?
                """, (verification_id, current_user.id))
            
            result = cursor.fetchone()
            logger.info(f"Database query result: {result}")
            if not result:
                raise HTTPException(status_code=404, detail="Verification not found or not authorized")
            
            project_id = result[0]
            
            # Generate certificate PDF
            pdf_buffer = report_service.generate_verification_certificate(verification_id, project_id)
            
            # Return PDF as response
            from fastapi.responses import StreamingResponse
            
            return StreamingResponse(
                pdf_buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=verification_certificate_{verification_id}.pdf"}
            )
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating certificate: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate certificate")


@app.get("/api/v1/reports/analytics")
async def download_analytics_report(current_user: UserResponse = Depends(get_current_user)):
    """Download comprehensive analytics report PDF"""
    if report_service is None:
        raise HTTPException(status_code=503, detail="Report service not available")
    
    # Only admins can generate analytics reports
    if not is_admin(current_user.role):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Gather all analytics data
        with db.get_connection() as conn:
            analytics_data = {}
            
            # Dashboard data
            cursor = conn.execute("SELECT COUNT(*) FROM projects")
            total_projects = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT status, COUNT(*) FROM projects GROUP BY status")
            project_status = dict(cursor.fetchall())
            
            cursor = conn.execute("SELECT COUNT(*) FROM verifications")
            total_verifications = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT SUM(carbon_impact) FROM verifications WHERE carbon_impact IS NOT NULL")
            result = cursor.fetchone()
            total_carbon_impact = result[0] if result[0] else 0
            
            analytics_data['dashboard'] = {
                'total_projects': total_projects,
                'project_status': project_status,
                'total_verifications': total_verifications,
                'total_carbon_impact': total_carbon_impact,
                'ml_performance': {
                    'source': 'offline_evaluation',
                    'forest_cover_f1': 0.4911,
                    'change_detection_f1': 0.6006
                }
            }
            
            # Generate report PDF
            pdf_buffer = report_service.generate_analytics_report(analytics_data, current_user.name)
            
            # Return PDF as response
            from fastapi.responses import StreamingResponse
            
            return StreamingResponse(
                pdf_buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=analytics_report_{datetime.now().strftime('%Y%m%d')}.pdf"}
            )
            
    except Exception as e:
        logger.error(f"Error generating analytics report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics report")


@app.put("/api/v1/projects/{project_id}/carbon-analysis")
async def set_carbon_analysis(
    project_id: int,
    analysis: CarbonAnalysisData,
    current_user: UserResponse = Depends(get_current_user)
):
    """Store an offline-computed carbon analysis for a project (owner or admin).

    The heavy ML (torch/rasterio) runs off-platform via ml/analyze and posts its
    result here so the serverless app can display and report it.
    """
    with db.get_connection() as conn:
        if is_admin(current_user.role):
            cursor = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
        else:
            cursor = conn.execute(
                "SELECT id FROM projects WHERE id = ? AND user_id = ?",
                (project_id, current_user.id)
            )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Project not found or not authorized")
        conn.execute(
            "UPDATE projects SET carbon_analysis = ? WHERE id = ?",
            (json.dumps(analysis.model_dump()), project_id)
        )
        conn.commit()
    logger.info(f"Carbon analysis stored for project {project_id} by {current_user.email}")
    return {"message": "Carbon analysis stored", "project_id": project_id}


@app.get("/api/v1/projects/{project_id}/carbon-analysis")
async def get_carbon_analysis(
    project_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """Return the stored carbon analysis for a project (owner or admin), or null."""
    with db.get_connection() as conn:
        if is_admin(current_user.role):
            cursor = conn.execute("SELECT carbon_analysis FROM projects WHERE id = ?", (project_id,))
        else:
            cursor = conn.execute(
                "SELECT carbon_analysis FROM projects WHERE id = ? AND user_id = ?",
                (project_id, current_user.id)
            )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found or not authorized")
        raw = row["carbon_analysis"]
    return {"project_id": project_id, "carbon_analysis": json.loads(raw) if raw else None}


@app.get("/api/v1/reports/project/{project_id}")
async def download_project_report(
    project_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """Download project summary report PDF"""
    if report_service is None:
        raise HTTPException(status_code=503, detail="Report service not available")
    
    try:
        with db.get_connection() as conn:
            # Verify project exists and user has access
            if is_admin(current_user.role):
                cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            else:
                cursor = conn.execute(
                    "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                    (project_id, current_user.id)
                )
            
            project = cursor.fetchone()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found or not authorized")
            
            # Get project verifications
            cursor = conn.execute(
                "SELECT * FROM verifications WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,)
            )
            verifications = cursor.fetchall()
            
            # Create simple project report (could be enhanced)
            from io import BytesIO
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.pagesizes import A4
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            story.append(Paragraph(f"Project Report: {project[1]}", styles['Title']))
            story.append(Spacer(1, 20))
            
            # Project details
            story.append(Paragraph(f"Location: {project[2]}", styles['Normal']))
            story.append(Paragraph(f"Type: {project[3]}", styles['Normal']))
            story.append(Paragraph(f"Status: {project[4]}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Verifications
            story.append(Paragraph(f"Verifications: {len(verifications)}", styles['Heading2']))
            for v in verifications:
                story.append(Paragraph(f"• Status: {v[2]}, Carbon Impact: {v[3] or 'N/A'} tonnes", styles['Normal']))

            # Carbon analysis section (offline ML result stored on the project)
            ca_raw = dict(project).get('carbon_analysis')
            if ca_raw:
                ca = json.loads(ca_raw)
                story.append(Spacer(1, 20))
                story.append(Paragraph("Carbon Analysis (experimental)", styles['Heading2']))
                for label, value in [
                    ("Imagery", ca.get('scene_id')),
                    ("Acquired", ca.get('scene_date')),
                    ("Analyzed area (ha)", ca.get('total_area_hectares')),
                    ("Forest coverage (%)", ca.get('forest_coverage_percent')),
                    ("Forest area (ha)", ca.get('forest_area_hectares')),
                    ("Carbon stock (tCO2e)", ca.get('total_co2e_tonnes')),
                    ("Biome", ca.get('biome')),
                ]:
                    if value is not None:
                        story.append(Paragraph(f"• {label}: {value}", styles['Normal']))
                story.append(Spacer(1, 8))
                story.append(Paragraph(
                    "Experimental estimate — the forest model is a research prototype "
                    "(F1~0.49, conservative/under-estimating out-of-region). Requires human "
                    "verification; not for compliance-grade crediting.",
                    styles['Italic']))
            
            doc.build(story)
            buffer.seek(0)
            
            # Return PDF as response
            from fastapi.responses import StreamingResponse
            
            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=project_report_{project_id}.pdf"}
            )
            
    except Exception as e:
        logger.error(f"Error generating project report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate project report")


# Analytics API Endpoints
@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_analytics(current_user: UserResponse = Depends(get_current_user)):
    """Get comprehensive dashboard analytics"""
    try:
        with db.get_connection() as conn:
            analytics = {}
            
            # Project Statistics
            cursor = conn.execute("SELECT COUNT(*) FROM projects")
            analytics['total_projects'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT status, COUNT(*) FROM projects GROUP BY status")
            status_counts = dict(cursor.fetchall())
            analytics['project_status'] = status_counts
            
            # Verification Statistics
            cursor = conn.execute("SELECT COUNT(*) FROM verifications")
            analytics['total_verifications'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT status, COUNT(*) FROM verifications GROUP BY status")
            verification_status = dict(cursor.fetchall())
            analytics['verification_status'] = verification_status
            
            # ML Model Performance — real recorded offline-evaluation metrics.
            # Source: ml/inference/production_inference.py / ml/evaluation_results/postprocessing_evaluation_results.csv
            analytics['ml_performance'] = {
                'source': 'offline_evaluation',
                'forest_cover_f1': 0.4911,
                'change_detection_f1': 0.6006
            }
            
            # Carbon Impact Analytics
            cursor = conn.execute("SELECT SUM(carbon_impact) FROM verifications WHERE carbon_impact IS NOT NULL")
            result = cursor.fetchone()
            analytics['total_carbon_impact'] = result[0] if result[0] else 0
            
            # Recent Activity (last 30 days)
            cursor = conn.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count 
                FROM projects 
                WHERE created_at >= date('now', '-30 days')
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            analytics['daily_activity'] = dict(cursor.fetchall())
            
            # User Activity by Role
            cursor = conn.execute("SELECT role, COUNT(*) FROM users GROUP BY role")
            analytics['user_roles'] = dict(cursor.fetchall())
            
            # XAI Usage Statistics
            cursor = conn.execute("SELECT COUNT(*) FROM xai_explanations")
            analytics['total_explanations'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT method, COUNT(*) FROM xai_explanations GROUP BY method")
            analytics['xai_methods_usage'] = dict(cursor.fetchall())
            
            logger.info("Dashboard analytics generated successfully")
            return analytics
            
    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics")


@app.get("/api/v1/analytics/performance")
async def get_performance_analytics(current_user: UserResponse = Depends(get_current_user)):
    """Get ML model performance analytics"""
    try:
        with db.get_connection() as conn:
            performance = {}

            # Model performance — real recorded offline-evaluation metrics (single measured
            # run, no per-day trend exists). Source: ml/inference/production_inference.py /
            # ml/evaluation_results/postprocessing_evaluation_results.csv
            performance['model_metrics'] = {
                'source': 'offline_evaluation',
                'forest_cover_f1': 0.4911,
                'change_detection_f1': 0.6006
            }

            # Real DB-derived activity: projects and verifications created over the last 90 days.
            cursor = conn.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as projects
                FROM projects
                WHERE created_at >= date('now', '-90 days')
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            performance['daily_projects'] = dict(cursor.fetchall())

            cursor = conn.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as verifications
                FROM verifications
                WHERE created_at >= date('now', '-90 days')
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            performance['daily_verifications'] = dict(cursor.fetchall())

            # Model confidence distribution
            cursor = conn.execute("SELECT ai_confidence FROM verifications WHERE ai_confidence IS NOT NULL")
            confidences = [row[0] for row in cursor.fetchall()]
            
            performance['confidence_distribution'] = {
                'high_confidence': len([c for c in confidences if c > 0.9]),
                'medium_confidence': len([c for c in confidences if 0.7 <= c <= 0.9]),
                'low_confidence': len([c for c in confidences if c < 0.7]),
                'average_confidence': sum(confidences) / len(confidences) if confidences else 0
            }
            
            return performance
            
    except Exception as e:
        logger.error(f"Error generating performance analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate performance analytics")


@app.get("/api/v1/analytics/carbon-impact")
async def get_carbon_impact_analytics(current_user: UserResponse = Depends(get_current_user)):
    """Get carbon impact analytics"""
    try:
        with db.get_connection() as conn:
            carbon_analytics = {}
            
            # Total carbon impact
            cursor = conn.execute("SELECT SUM(carbon_impact) FROM verifications WHERE status = 'verified'")
            result = cursor.fetchone()
            carbon_analytics['total_verified_carbon'] = result[0] if result[0] else 0
            
            # Carbon impact by project type
            cursor = conn.execute("""
                SELECT p.project_type, SUM(v.carbon_impact) as total_impact, COUNT(*) as project_count
                FROM verifications v
                JOIN projects p ON v.project_id = p.id
                WHERE v.status = 'verified' AND v.carbon_impact IS NOT NULL
                GROUP BY p.project_type
            """)
            carbon_analytics['impact_by_type'] = [
                {'type': row[0], 'impact': row[1], 'projects': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Monthly carbon trends
            cursor = conn.execute("""
                SELECT strftime('%Y-%m', v.created_at) as month, 
                       SUM(v.carbon_impact) as monthly_impact,
                       COUNT(*) as verifications
                FROM verifications v
                WHERE v.status = 'verified' AND v.carbon_impact IS NOT NULL
                AND v.created_at >= date('now', '-12 months')
                GROUP BY strftime('%Y-%m', v.created_at)
                ORDER BY month
            """)
            carbon_analytics['monthly_trends'] = [
                {'month': row[0], 'impact': row[1], 'verifications': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Top performing projects
            cursor = conn.execute("""
                SELECT p.name, p.location_name, v.carbon_impact, v.ai_confidence
                FROM verifications v
                JOIN projects p ON v.project_id = p.id
                WHERE v.status = 'verified' AND v.carbon_impact IS NOT NULL
                ORDER BY v.carbon_impact DESC
                LIMIT 10
            """)
            carbon_analytics['top_projects'] = [
                {
                    'name': row[0],
                    'location': row[1], 
                    'impact': row[2],
                    'confidence': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            return carbon_analytics
            
    except Exception as e:
        logger.error(f"Error generating carbon impact analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate carbon impact analytics")


# IoT Sensor Endpoints
@app.get("/api/v1/iot/sensors")
async def get_iot_sensors(
    project_id: Optional[int] = None,
    sensor_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: UserResponse = Depends(get_current_user)
):
    """Get IoT sensors with optional filtering"""
    try:
        with db.get_connection() as conn:
            query = """
                SELECT s.*, p.name as project_name, p.location_name
                FROM iot_sensors s
                JOIN projects p ON s.project_id = p.id
                WHERE 1=1
            """
            params = []
            
            if project_id:
                query += " AND s.project_id = ?"
                params.append(project_id)
            
            if sensor_type:
                query += " AND s.sensor_type = ?"
                params.append(sensor_type)
            
            if status:
                query += " AND s.status = ?"
                params.append(status)
            
            # Check user permissions
            if not is_admin(current_user.role):
                query += " AND p.user_id = ?"
                params.append(current_user.id)
            
            query += " ORDER BY s.created_at DESC"
            
            cursor = conn.execute(query, params)
            sensors = []
            
            for row in cursor.fetchall():
                sensor = {
                    'id': row[0],
                    'sensor_id': row[1],
                    'sensor_type': row[2],
                    'location_lat': row[3],
                    'location_lng': row[4],
                    'project_id': row[5],
                    'status': row[6],
                    'last_reading': json.loads(row[7]) if row[7] else None,
                    'installation_date': row[8],
                    'calibration_data': json.loads(row[9]) if row[9] else None,
                    'created_at': row[10],
                    'project_name': row[11],
                    'location_name': row[12]
                }
                sensors.append(sensor)
            
            return sensors
            
    except Exception as e:
        logger.error(f"Error fetching IoT sensors: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch IoT sensors")


@app.post("/api/v1/iot/sensors", response_model=IoTSensorResponse, status_code=status.HTTP_201_CREATED)
async def create_iot_sensor(
    sensor_data: IoTSensorCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create a new IoT sensor"""
    try:
        logger.info(f"Creating IoT sensor: {sensor_data.sensor_id} for project {sensor_data.project_id}")
        with db.get_connection() as conn:
            # Verify project exists and user has access
            cursor = conn.execute("SELECT id, user_id FROM projects WHERE id = ?", (sensor_data.project_id,))
            project = cursor.fetchone()
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            if not is_admin(current_user.role) and project[1] != current_user.id:
                raise HTTPException(status_code=403, detail="Not authorized to add sensors to this project")
            
            logger.info(f"Project verified, inserting sensor...")
            
            # Insert sensor
            conn.execute("""
                INSERT INTO iot_sensors (
                    sensor_id, sensor_type, location_lat, location_lng, 
                    project_id, installation_date, calibration_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sensor_data.sensor_id,
                sensor_data.sensor_type,
                sensor_data.location_lat,
                sensor_data.location_lng,
                sensor_data.project_id,
                sensor_data.installation_date,
                json.dumps(sensor_data.calibration_data) if sensor_data.calibration_data else None
            ))
            
            logger.info(f"Sensor inserted, fetching created sensor...")
            
            # Get created sensor
            cursor = conn.execute("SELECT * FROM iot_sensors WHERE sensor_id = ?", (sensor_data.sensor_id,))
            sensor_row = cursor.fetchone()
            
            logger.info(f"Sensor fetched: {sensor_row}")
            
            return IoTSensorResponse(
                id=sensor_row[0],
                sensor_id=sensor_row[1],
                sensor_type=sensor_row[2],
                location_lat=sensor_row[3],
                location_lng=sensor_row[4],
                project_id=sensor_row[5],
                status=sensor_row[6],
                last_reading=json.loads(sensor_row[7]) if sensor_row[7] else None,
                installation_date=sensor_row[8],
                created_at=sensor_row[10]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating IoT sensor: {e}")
        raise HTTPException(status_code=500, detail="Failed to create IoT sensor")


@app.put("/api/v1/iot/sensors/{sensor_id}", response_model=IoTSensorResponse)
async def update_iot_sensor(
    sensor_id: int,
    sensor_data: IoTSensorCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Update an existing IoT sensor"""
    try:
        logger.info(f"Updating IoT sensor: {sensor_id} by user {current_user.id}")
        with db.get_connection() as conn:
            # Verify sensor exists and user has access
            cursor = conn.execute("""
                SELECT s.*, p.user_id FROM iot_sensors s 
                JOIN projects p ON s.project_id = p.id 
                WHERE s.id = ?
            """, (sensor_id,))
            existing_sensor = cursor.fetchone()
            
            if not existing_sensor:
                raise HTTPException(status_code=404, detail="Sensor not found")
            
            if not is_admin(current_user.role) and existing_sensor[-1] != current_user.id:  # user_id is last column
                raise HTTPException(status_code=403, detail="Not authorized to update this sensor")
            
            logger.info(f"Sensor verified, updating...")
            
            # Update sensor
            conn.execute("""
                UPDATE iot_sensors 
                SET sensor_type = ?, location_lat = ?, location_lng = ?, 
                    installation_date = ?, calibration_data = ?
                WHERE id = ?
            """, (
                sensor_data.sensor_type,
                sensor_data.location_lat,
                sensor_data.location_lng,
                sensor_data.installation_date,
                json.dumps(sensor_data.calibration_data) if sensor_data.calibration_data else None,
                sensor_id
            ))
            
            logger.info(f"Sensor updated, fetching updated sensor...")
            
            # Get updated sensor
            cursor = conn.execute("SELECT * FROM iot_sensors WHERE id = ?", (sensor_id,))
            sensor_row = cursor.fetchone()
            
            logger.info(f"Updated sensor fetched: {sensor_row}")
            
            return IoTSensorResponse(
                id=sensor_row[0],
                sensor_id=sensor_row[1],
                sensor_type=sensor_row[2],
                location_lat=sensor_row[3],
                location_lng=sensor_row[4],
                project_id=sensor_row[5],
                status=sensor_row[6],
                last_reading=json.loads(sensor_row[7]) if sensor_row[7] else None,
                installation_date=sensor_row[8],
                created_at=sensor_row[10]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating IoT sensor: {e}")
        raise HTTPException(status_code=500, detail="Failed to update IoT sensor")


@app.delete("/api/v1/iot/sensors/{sensor_id}")
async def delete_iot_sensor(
    sensor_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """Delete an IoT sensor"""
    try:
        logger.info(f"Deleting IoT sensor: {sensor_id} by user {current_user.id}")
        with db.get_connection() as conn:
            # Verify sensor exists and user has access
            cursor = conn.execute("""
                SELECT s.*, p.user_id FROM iot_sensors s 
                JOIN projects p ON s.project_id = p.id 
                WHERE s.id = ?
            """, (sensor_id,))
            sensor = cursor.fetchone()
            
            if not sensor:
                raise HTTPException(status_code=404, detail="Sensor not found")
            
            if not is_admin(current_user.role) and sensor[-1] != current_user.id:  # user_id is last column
                raise HTTPException(status_code=403, detail="Not authorized to delete this sensor")
            
            # Delete related readings first (foreign key constraint)
            cursor = conn.execute("DELETE FROM sensor_readings WHERE sensor_id = ?", (sensor[1],))  # sensor[1] is sensor_id
            readings_deleted = cursor.rowcount
            
            # Delete the sensor
            cursor = conn.execute("DELETE FROM iot_sensors WHERE id = ?", (sensor_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Sensor not found")
            
            logger.info(f"Sensor {sensor[1]} deleted successfully with {readings_deleted} readings")
            
            return {
                "message": "Sensor deleted successfully",
                "sensor_id": sensor_id,
                "readings_deleted": readings_deleted
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting IoT sensor: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete IoT sensor")


@app.post("/api/v1/iot/readings", response_model=SensorReadingResponse, status_code=status.HTTP_201_CREATED)
async def create_sensor_reading(
    reading_data: SensorReadingCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create a new sensor reading"""
    try:
        with db.get_connection() as conn:
            # Verify sensor exists
            cursor = conn.execute("SELECT s.id, p.user_id FROM iot_sensors s JOIN projects p ON s.project_id = p.id WHERE s.sensor_id = ?", (reading_data.sensor_id,))
            sensor = cursor.fetchone()
            
            if not sensor:
                raise HTTPException(status_code=404, detail="Sensor not found")
            
            if not is_admin(current_user.role) and sensor[1] != current_user.id:
                raise HTTPException(status_code=403, detail="Not authorized to add readings to this sensor")
            
            # Insert reading
            conn.execute("""
                INSERT INTO sensor_readings (
                    sensor_id, reading_type, value, unit, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                reading_data.sensor_id,
                reading_data.reading_type,
                reading_data.value,
                reading_data.unit,
                reading_data.timestamp or datetime.now().isoformat(),
                json.dumps(reading_data.metadata) if reading_data.metadata else None
            ))
            
            # Update last reading in sensor table
            last_reading = {
                'type': reading_data.reading_type,
                'value': reading_data.value,
                'unit': reading_data.unit,
                'timestamp': reading_data.timestamp or datetime.now().isoformat()
            }
            
            conn.execute("""
                UPDATE iot_sensors 
                SET last_reading = ? 
                WHERE sensor_id = ?
            """, (json.dumps(last_reading), reading_data.sensor_id))
            
            # Get created reading
            cursor = conn.execute("SELECT * FROM sensor_readings WHERE id = last_insert_rowid()")
            reading_row = cursor.fetchone()
            
            return SensorReadingResponse(
                id=reading_row[0],
                sensor_id=reading_row[1],
                reading_type=reading_row[2],
                value=reading_row[3],
                unit=reading_row[4],
                timestamp=reading_row[5],
                metadata=json.loads(reading_row[6]) if reading_row[6] else None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating sensor reading: {e}")
        raise HTTPException(status_code=500, detail="Failed to create sensor reading")


@app.get("/api/v1/iot/readings/{sensor_id}")
async def get_sensor_readings(
    sensor_id: str,
    limit: int = Query(100, ge=1, le=1000),
    current_user: UserResponse = Depends(get_current_user)
):
    """Get sensor readings for a specific sensor"""
    try:
        with db.get_connection() as conn:
            # Verify sensor exists and user has access
            cursor = conn.execute("""
                SELECT s.id, p.user_id FROM iot_sensors s 
                JOIN projects p ON s.project_id = p.id 
                WHERE s.sensor_id = ?
            """, (sensor_id,))
            sensor = cursor.fetchone()
            
            if not sensor:
                raise HTTPException(status_code=404, detail="Sensor not found")
            
            if not is_admin(current_user.role) and sensor[1] != current_user.id:
                raise HTTPException(status_code=403, detail="Not authorized to access this sensor")
            
            # Get readings
            cursor = conn.execute("""
                SELECT * FROM sensor_readings 
                WHERE sensor_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (sensor_id, limit))
            
            readings = []
            for row in cursor.fetchall():
                reading = {
                    'id': row[0],
                    'sensor_id': row[1],
                    'reading_type': row[2],
                    'value': row[3],
                    'unit': row[4],
                    'timestamp': row[5],
                    'metadata': json.loads(row[6]) if row[6] else None
                }
                readings.append(reading)
            
            return readings
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sensor readings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sensor readings")


@app.get("/api/v1/iot/analytics")
async def get_iot_analytics(current_user: UserResponse = Depends(get_current_user)):
    """Get IoT analytics and summary"""
    try:
        with db.get_connection() as conn:
            analytics = {}
            
            # Total sensors
            cursor = conn.execute("SELECT COUNT(*) FROM iot_sensors")
            analytics['total_sensors'] = cursor.fetchone()[0]
            
            # Sensors by type
            cursor = conn.execute("""
                SELECT sensor_type, COUNT(*) as count 
                FROM iot_sensors 
                GROUP BY sensor_type
            """)
            analytics['sensors_by_type'] = [
                {'type': row[0], 'count': row[1]} 
                for row in cursor.fetchall()
            ]
            
            # Active sensors (with recent readings)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM iot_sensors 
                WHERE last_reading IS NOT NULL 
                AND datetime(last_reading) > datetime('now', '-24 hours')
            """)
            analytics['active_sensors'] = cursor.fetchone()[0]
            
            # Recent readings count
            cursor = conn.execute("""
                SELECT COUNT(*) FROM sensor_readings 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            analytics['recent_readings'] = cursor.fetchone()[0]
            
            # Average readings per sensor
            cursor = conn.execute("""
                SELECT AVG(reading_count) FROM (
                    SELECT sensor_id, COUNT(*) as reading_count 
                    FROM sensor_readings 
                    GROUP BY sensor_id
                )
            """)
            analytics['avg_readings_per_sensor'] = cursor.fetchone()[0] or 0
            
            return analytics
            
    except Exception as e:
        logger.error(f"Error generating IoT analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate IoT analytics")



# Settings Management Endpoints
@app.get("/api/v1/settings", response_model=UserSettingsResponse)
async def get_user_settings(current_user: UserResponse = Depends(get_current_user)):
    """Get user settings"""
    try:
        with db.get_connection() as conn:
            # Get user settings from database
            cursor = conn.execute("""
                SELECT setting_key, setting_value 
                FROM user_settings 
                WHERE user_id = ?
            """, (current_user.id,))
            
            settings_data = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Default settings
            defaults = {
                'email_notifications': 'true',
                'project_notifications': 'true',
                'security_alerts': 'true',
                'api_key_enabled': 'false',
                'theme': 'light',
                'language': 'en',
                'timezone': 'UTC'
            }
            
            # Merge with defaults
            for key, default_value in defaults.items():
                if key not in settings_data:
                    settings_data[key] = default_value
            
            # Generate API key if enabled
            api_key = None
            if settings_data.get('api_key_enabled') == 'true':
                # Generate a simple API key (in production, use a more secure method)
                api_key = f"ccv_{secrets.token_hex(16)}"
            
            return UserSettingsResponse(
                email_notifications=settings_data['email_notifications'] == 'true',
                project_notifications=settings_data['project_notifications'] == 'true',
                security_alerts=settings_data['security_alerts'] == 'true',
                api_key_enabled=settings_data['api_key_enabled'] == 'true',
                theme=settings_data['theme'],
                language=settings_data['language'],
                timezone=settings_data['timezone'],
                api_key=api_key
            )
            
    except Exception as e:
        logger.error(f"Error getting user settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user settings")


@app.patch("/api/v1/settings", response_model=UserSettingsResponse)
async def update_user_settings(
    settings: UserSettingsUpdate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Update user settings"""
    try:
        with db.get_connection() as conn:
            # Update each setting that was provided
            for setting_key, setting_value in settings.model_dump(exclude_none=True).items():
                conn.execute("""
                    INSERT OR REPLACE INTO user_settings (user_id, setting_key, setting_value, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (current_user.id, setting_key, str(setting_value).lower(), datetime.now().isoformat()))
            
            conn.commit()
            
            # Return updated settings
            return await get_user_settings(current_user)
            
    except Exception as e:
        logger.error(f"Error updating user settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user settings")


@app.post("/api/v1/settings/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Change user password"""
    try:
        with db.get_connection() as conn:
            # Verify current password
            cursor = conn.execute("SELECT hashed_password FROM users WHERE id = ?", (current_user.id,))
            user_data = cursor.fetchone()
            
            if not user_data:
                raise HTTPException(status_code=404, detail="User not found")
            
            if not pwd_context.verify(request.current_password, user_data[0]):
                raise HTTPException(status_code=400, detail="Current password is incorrect")
            
            # Hash new password and update
            new_hashed_password = pwd_context.hash(request.new_password)
            conn.execute("""
                UPDATE users SET hashed_password = ? WHERE id = ?
            """, (new_hashed_password, current_user.id))
            
            conn.commit()
            
            return {"message": "Password changed successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")


# Seed at import time too: Vercel serverless may not fire ASGI startup events,
# and this runs once per cold start (a no-op after the first admin exists).
seed_admin()


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
