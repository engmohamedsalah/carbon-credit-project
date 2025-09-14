"""
Serverless-compatible database manager for Vercel deployment
Uses in-memory SQLite that's initialized on each cold start
"""
import os
import sqlite3
import tempfile
import shutil
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class ServerlessDBManager:
    """Database manager that works in Vercel's serverless environment"""
    
    def __init__(self):
        self.db_path = None
        self._initialized = False
    
    def _get_db_path(self):
        """Get or create temporary database path"""
        if self.db_path is None:
            # Create a temporary file for the database
            temp_dir = tempfile.gettempdir()
            self.db_path = os.path.join(temp_dir, "carbon_credits_temp.db")
            
            # Copy the original database if it exists (check multiple locations)
            possible_locations = [
                os.path.join(os.path.dirname(__file__), "carbon_credits.db"),  # Same directory as this file
                os.path.join(os.path.dirname(__file__), "..", "database", "carbon_credits.db"),  # Original location
            ]
            
            original_db = None
            for location in possible_locations:
                if os.path.exists(location):
                    original_db = location
                    break
            
            if original_db:
                shutil.copy2(original_db, self.db_path)
                logger.info(f"Copied database from {original_db} to {self.db_path}")
            else:
                logger.info(f"Original database not found in any location, will create new one at {self.db_path}")
        
        return self.db_path
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            db_path = self._get_db_path()
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # Initialize database if not done yet
            if not self._initialized:
                self._init_database(conn)
                self._initialized = True
            
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _init_database(self, conn):
        """Initialize database with required tables"""
        try:
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
            
            # Add default admin user if no users exist
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            
            if user_count == 0:
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                hashed_password = pwd_context.hash("admin123")
                
                conn.execute("""
                    INSERT INTO users (email, hashed_password, full_name, role)
                    VALUES (?, ?, ?, ?)
                """, ("admin@carbon-credit.com", hashed_password, "System Administrator", "Administrator"))
                
                logger.info("Created default admin user")
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

# Global instance
db_manager = ServerlessDBManager()