import requests
import json
import time

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"

def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✅ Health check passed")

def test_auth():
    """Test authentication endpoints"""
    # Register a new user
    register_data = {
        "email": "testuser@example.com",
        "password": "password123",
        "full_name": "Test User",
        "role": "Project Developer"
    }
    response = requests.post(f"{API_URL}/auth/register", json=register_data)
    assert response.status_code in [200, 201, 400]  # 400 if user already exists
    
    # Login
    login_data = {
        "username": "testuser@example.com",
        "password": "password123"
    }
    response = requests.post(f"{API_URL}/auth/login", data=login_data)
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    
    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get current user
    response = requests.get(f"{API_URL}/auth/me", headers=headers)
    assert response.status_code == 200
    user_data = response.json()
    assert user_data["email"] == "testuser@example.com"
    
    print("✅ Authentication tests passed")
    return headers

def test_projects(headers):
    """Test project endpoints"""
    # Create a project
    project_data = {
        "name": "Test Reforestation Project",
        "description": "A test project for reforestation",
        "location_name": "Test Location",
        "project_type": "Reforestation",
        "area_hectares": 100.5,
        "estimated_carbon_credits": 500,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [0, 0],
                    [0, 1],
                    [1, 1],
                    [1, 0],
                    [0, 0]
                ]
            ]
        }
    }
    
    response = requests.post(f"{API_URL}/projects", json=project_data, headers=headers)
    assert response.status_code in [200, 201]
    project = response.json()
    project_id = project["id"]
    
    # Get project by ID
    response = requests.get(f"{API_URL}/projects/{project_id}", headers=headers)
    assert response.status_code == 200
    project_detail = response.json()
    assert project_detail["name"] == "Test Reforestation Project"
    
    # Get all projects
    response = requests.get(f"{API_URL}/projects", headers=headers)
    assert response.status_code == 200
    projects = response.json()
    assert len(projects) > 0
    
    print("✅ Project tests passed")
    return project_id

def test_project_status_logging(headers, project_id):
    """Test project status logging functionality"""
    # Test status update with logging
    status_data = {
        "status": "Verified",
        "reason": "Test verification for automated testing",
        "notes": "This is a test verification note"
    }
    
    response = requests.patch(f"{API_URL}/projects/{project_id}/status", 
                             json=status_data, headers=headers)
    assert response.status_code == 200
    status_response = response.json()
    assert status_response["new_status"] == "Verified"
    assert status_response["reason"] == "Test verification for automated testing"
    
    # Test status logs retrieval
    response = requests.get(f"{API_URL}/projects/{project_id}/status-logs", headers=headers)
    assert response.status_code == 200
    logs_data = response.json()
    assert "status_logs" in logs_data
    assert len(logs_data["status_logs"]) > 0
    
    # Verify log entry
    latest_log = logs_data["status_logs"][0]
    assert latest_log["new_status"] == "Verified"
    assert latest_log["reason"] == "Test verification for automated testing"
    assert latest_log["notes"] == "This is a test verification note"
    assert "changed_by_name" in latest_log
    
    # Test rejection with required reason
    rejection_data = {
        "status": "Rejected",
        "reason": "Test rejection reason",
        "notes": "Test rejection notes"
    }
    
    response = requests.patch(f"{API_URL}/projects/{project_id}/status", 
                             json=rejection_data, headers=headers)
    assert response.status_code == 200
    
    # Test rejection without reason (should fail)
    invalid_rejection = {"status": "Rejected"}
    response = requests.patch(f"{API_URL}/projects/{project_id}/status", 
                             json=invalid_rejection, headers=headers)
    assert response.status_code == 400
    error_data = response.json()
    assert "Reason is required when rejecting" in error_data["detail"]
    
    # Test duplicate status (should be skipped)
    duplicate_status = {"status": "Rejected", "reason": "Duplicate test"}
    response = requests.patch(f"{API_URL}/projects/{project_id}/status", 
                             json=duplicate_status, headers=headers)
    assert response.status_code == 200
    duplicate_response = response.json()
    assert duplicate_response["message"] == "Status unchanged"
    
    print("✅ Project status logging tests passed")

def test_verifications(headers, project_id):
    """Test verification endpoints"""
    # Get verifications for project
    response = requests.get(f"{API_URL}/verification", 
                           params={"project_id": project_id}, headers=headers)
    assert response.status_code == 200
    verifications = response.json()
    
    print("✅ Verification tests passed")
    return len(verifications) > 0

def run_tests():
    """Run all tests"""
    print("Starting tests...")
    
    try:
        test_health()
        headers = test_auth()
        project_id = test_projects(headers)
        test_project_status_logging(headers, project_id)
        test_verifications(headers, project_id)
        
        print("\n🎉 All tests passed successfully!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
