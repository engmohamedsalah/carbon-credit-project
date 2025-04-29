import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print("✅ Health check passed")

def test_auth():
    """Test authentication endpoints"""
    # Register a new user
    register_data = {
        "email": "test@example.com",
        "password": "password123",
        "full_name": "Test User",
        "role": "project_developer"
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=register_data)
    assert response.status_code in [200, 201, 400]  # 400 if user already exists
    
    # Login
    login_data = {
        "username": "test@example.com",
        "password": "password123"
    }
    response = requests.post(f"{BASE_URL}/auth/login", data=login_data)
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    
    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get current user
    response = requests.get(f"{BASE_URL}/users/me", headers=headers)
    assert response.status_code == 200
    user_data = response.json()
    assert user_data["email"] == "test@example.com"
    
    print("✅ Authentication tests passed")
    return headers

def test_projects(headers):
    """Test project endpoints"""
    # Create a project
    project_data = {
        "name": "Test Reforestation Project",
        "description": "A test project for reforestation",
        "location_name": "Test Location",
        "project_type": "reforestation",
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
    
    response = requests.post(f"{BASE_URL}/projects", json=project_data, headers=headers)
    assert response.status_code == 200
    project = response.json()
    project_id = project["id"]
    
    # Get project by ID
    response = requests.get(f"{BASE_URL}/projects/{project_id}", headers=headers)
    assert response.status_code == 200
    project_detail = response.json()
    assert project_detail["name"] == "Test Reforestation Project"
    
    # Get all projects
    response = requests.get(f"{BASE_URL}/projects", headers=headers)
    assert response.status_code == 200
    projects = response.json()
    assert len(projects) > 0
    
    print("✅ Project tests passed")
    return project_id

def test_verifications(headers, project_id):
    """Test verification endpoints"""
    # Create a verification
    verification_data = {
        "project_id": project_id,
        "verification_notes": "Test verification",
        "verified_carbon_credits": 450,
        "confidence_score": 0.85
    }
    
    response = requests.post(f"{BASE_URL}/verification", json=verification_data, headers=headers)
    assert response.status_code == 200
    verification = response.json()
    verification_id = verification["id"]
    
    # Get verification by ID
    response = requests.get(f"{BASE_URL}/verification/{verification_id}", headers=headers)
    assert response.status_code == 200
    verification_detail = response.json()
    assert verification_detail["project_id"] == project_id
    
    # Submit human review
    review_data = {
        "approved": True,
        "notes": "Approved after human review"
    }
    
    response = requests.post(f"{BASE_URL}/verification/{verification_id}/human-review", 
                            json=review_data, headers=headers)
    assert response.status_code == 200
    updated_verification = response.json()
    assert updated_verification["human_reviewed"] == True
    assert updated_verification["status"] == "approved"
    
    print("✅ Verification tests passed")
    return verification_id

def test_blockchain(headers, verification_id):
    """Test blockchain endpoints"""
    # Certify verification
    response = requests.post(f"{BASE_URL}/blockchain/certify/{verification_id}", headers=headers)
    assert response.status_code == 200
    certification = response.json()
    assert "transaction_hash" in certification
    assert "token_id" in certification
    
    tx_hash = certification["transaction_hash"]
    token_id = certification["token_id"]
    
    # Get transaction status
    response = requests.get(f"{BASE_URL}/blockchain/transaction/{tx_hash}", headers=headers)
    assert response.status_code == 200
    tx_status = response.json()
    assert tx_status["transaction_hash"] == tx_hash
    
    # Verify token
    response = requests.get(f"{BASE_URL}/blockchain/verify/{token_id}")
    assert response.status_code == 200
    token_data = response.json()
    assert token_data["token_id"] == token_id
    assert token_data["is_valid"] == True
    
    print("✅ Blockchain tests passed")

def run_tests():
    """Run all tests"""
    print("Starting tests...")
    
    try:
        test_health()
        headers = test_auth()
        project_id = test_projects(headers)
        verification_id = test_verifications(headers, project_id)
        test_blockchain(headers, verification_id)
        
        print("\n🎉 All tests passed successfully!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error during tests: {e}")

if __name__ == "__main__":
    run_tests()
