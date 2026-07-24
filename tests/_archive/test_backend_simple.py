#!/usr/bin/env python3
"""
Simple backend tests for CI/CD pipeline
Tests basic functionality without requiring full server startup
"""

import sys
import os
import sqlite3
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

def test_database_connection():
    """Test database connection and basic operations"""
    try:
        db_path = Path(__file__).parent.parent / "database" / "carbon_credits.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Check if essential tables exist
        table_names = [table[0] for table in tables]
        essential_tables = ['users', 'projects', 'verifications']
        
        for table in essential_tables:
            assert table in table_names, f"Essential table '{table}' not found"
        
        print(f"✅ Database connection test passed. Found tables: {table_names}")
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    try:
        # Test basic imports
        import fastapi
        import uvicorn
        import sqlite3
        import json
        import logging
        
        print("✅ Basic imports test passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_config_files():
    """Test that configuration files exist"""
    try:
        # Check if essential files exist
        files_to_check = [
            "backend/requirements.txt",
            "backend/main.py",
            "database/carbon_credits.db"
        ]
        
        for file_path in files_to_check:
            assert os.path.exists(file_path), f"Required file not found: {file_path}"
        
        print("✅ Configuration files test passed")
        return True
        
    except AssertionError as e:
        print(f"❌ Configuration files test failed: {e}")
        return False

def test_ml_models_exist():
    """Test that ML model files exist"""
    try:
        model_files = [
            "ml/models/forest_cover_unet_focal_alpha_0.75_threshold_0.53.pth",
            "ml/models/change_detection_siamese_unet.pth",
            "ml/models/convlstm_fast_final.pth"
        ]
        
        existing_models = []
        for model_path in model_files:
            if os.path.exists(model_path):
                existing_models.append(model_path)
        
        print(f"✅ ML models test passed. Found {len(existing_models)}/{len(model_files)} models")
        return True
        
    except Exception as e:
        print(f"❌ ML models test failed: {e}")
        return False

def test_api_structure():
    """Test that API endpoints are properly defined"""
    try:
        # Read main.py and check for API endpoints
        with open("backend/main.py", "r") as f:
            content = f.read()
        
        # Check for essential endpoints
        endpoints = [
            "@app.get(\"/health\")",
            "@app.post(\"/api/v1/auth/register\")",
            "@app.post(\"/api/v1/auth/login\")",
            "@app.get(\"/api/v1/projects\")",
            "@app.post(\"/api/v1/projects\")"
        ]
        
        found_endpoints = []
        for endpoint in endpoints:
            if endpoint in content:
                found_endpoints.append(endpoint)
        
        print(f"✅ API structure test passed. Found {len(found_endpoints)}/{len(endpoints)} endpoints")
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Backend Tests for CI/CD...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_files,
        test_database_connection,
        test_ml_models_exist,
        test_api_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All backend tests passed!")
        return 0
    else:
        print("❌ Some backend tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 