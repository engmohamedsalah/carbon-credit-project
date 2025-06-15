from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Create SQLAlchemy engine with SQLite support
if settings.DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        settings.DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(settings.DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
def init_db():
    # Import models here to ensure they are registered with SQLAlchemy
    from app.models import user, project, verification, satellite
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Check if admin user exists and create if not
    from app.models.user import User
    from app.core.security import get_password_hash
    from sqlalchemy.orm import Session
    
    db = SessionLocal()
    admin = db.query(User).filter(User.email == "admin@example.com").first()
    
    if not admin:
        admin_user = User(
            email="admin@example.com",
            hashed_password=get_password_hash("password123"),
            is_active=True,
            is_admin=True,
            full_name="Admin User"
        )
        db.add(admin_user)
        db.commit()
        print("Admin user created.")
    
    db.close()
