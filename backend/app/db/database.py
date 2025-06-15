"""
Database connection and session management
"""
import sqlite3
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Professional database manager with connection pooling and error handling"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url.replace("sqlite:///", "")
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tokens table if not exists
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
                ''')
                
                # Add missing columns to existing tables
                self._add_missing_columns(cursor)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _add_missing_columns(self, cursor):
        """Add missing columns to existing tables"""
        try:
            # Add role column to users table
            cursor.execute('ALTER TABLE users ADD COLUMN role TEXT DEFAULT "Project Developer"')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            # Add missing columns to projects table
            for column, default in [
                ('area_size', 'REAL DEFAULT 0'),
                ('project_type', 'TEXT DEFAULT "Reforestation"'),
                ('status', 'TEXT DEFAULT "Pending"')
            ]:
                cursor.execute(f'ALTER TABLE projects ADD COLUMN {column} {default}')
        except sqlite3.OperationalError:
            pass  # Columns already exist
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.database_url)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute a single query and return one result"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def execute_query_all(self, query: str, params: tuple = ()) -> list:
        """Execute a query and return all results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an insert query and return the last row ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an update/delete query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount


# Global database manager instance
db_manager = DatabaseManager(settings.DATABASE_URL) 