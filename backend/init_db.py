"""
Database initialization script.
This will create SQLite database for local development.
"""
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from app.models.user import Base as UserBase
from app.models.project import Base as ProjectBase
from app.models.verification import Base as VerificationBase
from app.models.satellite import Base as SatelliteBase

# Create SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./carbon_credits.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create all tables
def init_db():
    # Create tables
    UserBase.metadata.create_all(bind=engine)
    ProjectBase.metadata.create_all(bind=engine)
    VerificationBase.metadata.create_all(bind=engine)
    SatelliteBase.metadata.create_all(bind=engine)
    
    print("Database tables created successfully.")
    
    # Create session for adding initial data
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    # Check if admin user exists
    from app.models.user import User
    admin = db.query(User).filter(User.email == "admin@example.com").first()
    
    # Add admin user if not exists
    if not admin:
        from app.core.security import get_password_hash
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
    print("Database initialization complete.")

if __name__ == "__main__":
    init_db()
    print("SQLite database initialized. You can now start the backend server.") 