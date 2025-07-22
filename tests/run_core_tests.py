#!/usr/bin/env python3
"""
Core Test Runner
Executes essential functionality tests without authentication complications
"""
import os
import sys
import subprocess
import time
import json
from datetime import datetime

def run_simple_test(test_name, test_function):
    """Run a simple test function and return results"""
    print(f"\n{'='*50}")
    print(f"Running {test_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = test_function()
        end_time = time.time()
        duration = end_time - start_time
        
        if result:
            print(f"✅ {test_name} passed in {duration:.2f}s")
            return {
                'test_name': test_name,
                'duration': duration,
                'status': 'PASSED',
                'error': None
            }
        else:
            print(f"❌ {test_name} failed in {duration:.2f}s")
            return {
                'test_name': test_name,
                'duration': duration,
                'status': 'FAILED',
                'error': 'Test returned False'
            }
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"💥 {test_name} errored in {duration:.2f}s: {e}")
        return {
            'test_name': test_name,
            'duration': duration,
            'status': 'ERROR',
            'error': str(e)
        }

def test_health_check():
    """Test health check endpoint"""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200 and response.json()["status"] == "healthy"
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    import requests
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        return response.status_code == 200 and "message" in response.json()
    except Exception as e:
        print(f"Root endpoint failed: {e}")
        return False

def test_user_registration():
    """Test user registration"""
    import requests
    try:
        user_data = {
            "email": f"testuser_{int(time.time())}@test.com",
            "password": "testpassword123",
            "full_name": "Test User",
            "role": "Project Developer"
        }
        response = requests.post("http://localhost:8000/api/v1/auth/register", json=user_data, timeout=10)
        return response.status_code == 201 and "access_token" in response.json()
    except Exception as e:
        print(f"User registration failed: {e}")
        return False

def test_ml_service_status():
    """Test ML service status"""
    import requests
    try:
        # First register a user to get a token
        user_data = {
            "email": f"mltest_{int(time.time())}@test.com",
            "password": "testpassword123",
            "full_name": "ML Test User",
            "role": "Project Developer"
        }
        register_response = requests.post("http://localhost:8000/api/v1/auth/register", json=user_data, timeout=10)
        
        if register_response.status_code != 201:
            print(f"Registration failed: {register_response.status_code}")
            return False
        
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get("http://localhost:8000/api/v1/ml/status", headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"ML status failed: {response.status_code} - {response.text}")
            return False
        
        data = response.json()
        print(f"ML Status Response: {data}")
        
        # Check if response has expected structure - the actual response has 'ml_service' and 'models_ready'
        return "ml_service" in data or "models_ready" in data or "service_version" in data
    except Exception as e:
        print(f"ML service status failed: {e}")
        return False

def test_database_connection():
    """Test database connectivity"""
    import sqlite3
    try:
        db_path = os.path.join("database", "carbon_credits.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        conn.close()
        return user_count >= 0  # Just check if we can connect and query
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def test_analytics_endpoints():
    """Test analytics endpoints"""
    import requests
    try:
        # Register a user
        user_data = {
            "email": f"analytics_{int(time.time())}@test.com",
            "password": "testpassword123",
            "full_name": "Analytics Test User",
            "role": "Project Developer"
        }
        register_response = requests.post("http://localhost:8000/api/v1/auth/register", json=user_data, timeout=10)
        
        if register_response.status_code != 201:
            return False
        
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test dashboard analytics
        response = requests.get("http://localhost:8000/api/v1/analytics/dashboard", headers=headers, timeout=10)
        if response.status_code != 200:
            return False
        
        # Test performance analytics
        response = requests.get("http://localhost:8000/api/v1/analytics/performance", headers=headers, timeout=10)
        if response.status_code != 200:
            return False
        
        return True
    except Exception as e:
        print(f"Analytics endpoints failed: {e}")
        return False

def test_xai_methods():
    """Test XAI methods endpoint"""
    import requests
    try:
        # Register a user
        user_data = {
            "email": f"xai_{int(time.time())}@test.com",
            "password": "testpassword123",
            "full_name": "XAI Test User",
            "role": "Project Developer"
        }
        register_response = requests.post("http://localhost:8000/api/v1/auth/register", json=user_data, timeout=10)
        
        if register_response.status_code != 201:
            return False
        
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get("http://localhost:8000/api/v1/xai/methods", headers=headers, timeout=10)
        return response.status_code == 200 and "methods" in response.json()
    except Exception as e:
        print(f"XAI methods failed: {e}")
        return False

def test_iot_endpoints():
    """Test IoT endpoints"""
    import requests
    try:
        # Register a user
        user_data = {
            "email": f"iot_{int(time.time())}@test.com",
            "password": "testpassword123",
            "full_name": "IoT Test User",
            "role": "Project Developer"
        }
        register_response = requests.post("http://localhost:8000/api/v1/auth/register", json=user_data, timeout=10)
        
        if register_response.status_code != 201:
            return False
        
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test IoT analytics endpoint
        response = requests.get("http://localhost:8000/api/v1/iot/analytics", headers=headers, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"IoT endpoints failed: {e}")
        return False

def main():
    """Main test runner function"""
    print("🚀 Starting Core Functionality Tests")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check if backend is running
    print("\n🔍 Checking if backend is running...")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("⚠️  Backend responded but not healthy")
            return 1
    except Exception as e:
        print(f"❌ Backend is not running: {e}")
        print("Please start the backend with: cd backend && python main.py")
        return 1
    
    # Define core tests
    core_tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("User Registration", test_user_registration),
        ("Database Connection", test_database_connection),
        ("ML Service Status", test_ml_service_status),
        ("Analytics Endpoints", test_analytics_endpoints),
        ("XAI Methods", test_xai_methods),
        ("IoT Endpoints", test_iot_endpoints),
    ]
    
    # Run all tests
    results = []
    passed = 0
    failed = 0
    errored = 0
    
    for test_name, test_function in core_tests:
        result = run_simple_test(test_name, test_function)
        results.append(result)
        
        if result['status'] == 'PASSED':
            passed += 1
        elif result['status'] == 'FAILED':
            failed += 1
        else:
            errored += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 CORE FUNCTIONALITY TEST REPORT")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"")
    print(f"📈 SUMMARY:")
    print(f"   Total Tests: {len(results)}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   💥 Errors: {errored}")
    print(f"   📊 Success Rate: {(passed/len(results)*100):.1f}%")
    print(f"")
    
    print(f"📋 TEST RESULTS:")
    for result in results:
        status_icon = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "💥"
        print(f"   {status_icon} {result['test_name']} ({result['duration']:.2f}s)")
        if result['error']:
            print(f"      Error: {result['error']}")
    
    print(f"")
    if failed == 0 and errored == 0:
        print(f"🎉 ALL CORE TESTS PASSED!")
    else:
        print(f"❌ Some tests failed or errored")
    
    print(f"{'='*80}")
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': len(results),
            'passed': passed,
            'failed': failed,
            'errored': errored,
            'success_rate': (passed/len(results)*100) if len(results) > 0 else 0
        },
        'results': results
    }
    
    with open("core_test_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Core test report saved to core_test_report.json")
    
    # Return appropriate exit code
    if failed > 0 or errored > 0:
        print(f"\n❌ Core tests completed with failures")
        return 1
    else:
        print(f"\n🎉 All core tests passed successfully!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 