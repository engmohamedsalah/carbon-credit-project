#!/usr/bin/env python3
"""
Simple test script to verify our API setup works
"""
import sys
import os

# Add the API directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

try:
    # Test database connection
    from database import db_manager
    print("✅ Database manager imported successfully")
    
    # Test database connection
    with db_manager.get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"✅ Database connection successful. Users count: {user_count}")
    
    # Test FastAPI app import
    from index import app
    print("✅ FastAPI app imported successfully")
    print("✅ API setup is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()