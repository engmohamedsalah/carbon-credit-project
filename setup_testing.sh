#!/bin/bash

# Create package.json for frontend
cat > /home/ubuntu/carbon_credit_project/frontend/package.json << 'EOL'
{
  "name": "carbon-credit-verification-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@emotion/react": "^11.10.6",
    "@emotion/styled": "^11.10.6",
    "@mui/icons-material": "^5.11.16",
    "@mui/material": "^5.12.0",
    "@reduxjs/toolkit": "^1.9.3",
    "axios": "^1.3.5",
    "leaflet": "^1.9.3",
    "leaflet-draw": "^1.0.4",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-leaflet": "^4.2.1",
    "react-redux": "^8.0.5",
    "react-router-dom": "^6.10.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOL

# Create .env file for backend
cat > /home/ubuntu/carbon_credit_project/backend/.env << 'EOL'
# Database settings
DATABASE_URL=postgresql://postgres:postgres@db:5432/carbon_credits
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=carbon_credits

# Security settings
SECRET_KEY=your-secret-key-for-jwt-tokens
ACCESS_TOKEN_EXPIRE_MINUTES=60

# API settings
API_V1_STR=/api/v1

# Blockchain settings
BLOCKCHAIN_PROVIDER_URL=https://polygon-mumbai.infura.io/v3/your-infura-key
CONTRACT_ADDRESS=0x0000000000000000000000000000000000000000
EOL

# Create test script for backend
cat > /home/ubuntu/carbon_credit_project/test_backend.py << 'EOL'
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
EOL

# Create test script for frontend
cat > /home/ubuntu/carbon_credit_project/test_frontend.py << 'EOL'
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import sys

def test_frontend():
    """Test the frontend application using Selenium"""
    print("Starting frontend tests...")
    
    # Setup Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        # Initialize the driver
        driver = webdriver.Chrome(options=options)
        
        # Test login page
        print("Testing login page...")
        driver.get("http://localhost:3000/login")
        assert "Carbon Credit Verification" in driver.title
        
        # Find login form elements
        email_input = driver.find_element(By.ID, "email")
        password_input = driver.find_element(By.ID, "password")
        login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Sign In')]")
        
        # Enter credentials and login
        email_input.send_keys("test@example.com")
        password_input.send_keys("password123")
        login_button.click()
        
        # Wait for dashboard to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'Dashboard')]"))
        )
        
        print("✅ Login test passed")
        
        # Test navigation to projects page
        print("Testing navigation to projects page...")
        projects_link = driver.find_element(By.XPATH, "//span[contains(text(), 'Projects')]")
        projects_link.click()
        
        # Wait for projects page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'Projects')]"))
        )
        
        print("✅ Navigation test passed")
        
        # Test creating a new project
        print("Testing project creation...")
        new_project_button = driver.find_element(By.XPATH, "//button[contains(text(), 'New Project')]")
        new_project_button.click()
        
        # Wait for new project form to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'Create New Project')]"))
        )
        
        # Fill out project form
        driver.find_element(By.NAME, "name").send_keys("Selenium Test Project")
        driver.find_element(By.NAME, "location_name").send_keys("Test Location")
        driver.find_element(By.NAME, "description").send_keys("A project created by Selenium tests")
        
        # Submit form
        submit_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Create Project')]")
        submit_button.click()
        
        # Wait for project detail page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'Selenium Test Project')]"))
        )
        
        print("✅ Project creation test passed")
        
        # Test logout
        print("Testing logout...")
        profile_button = driver.find_element(By.XPATH, "//button[@aria-label='account of current user']")
        profile_button.click()
        
        # Wait for menu to appear and click logout
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//li[contains(text(), 'Logout')]"))
        ).click()
        
        # Wait for login page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h5[contains(text(), 'Sign In')]"))
        )
        
        print("✅ Logout test passed")
        
        print("\n🎉 All frontend tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Frontend test failed: {e}")
    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    test_frontend()
EOL

# Create validation script
cat > /home/ubuntu/carbon_credit_project/validate_implementation.py << 'EOL'
import os
import json
import subprocess
import sys

def check_file_exists(filepath):
    """Check if a file exists"""
    exists = os.path.isfile(filepath)
    print(f"{'✅' if exists else '❌'} {filepath}")
    return exists

def check_directory_exists(dirpath):
    """Check if a directory exists"""
    exists = os.path.isdir(dirpath)
    print(f"{'✅' if exists else '❌'} {dirpath}")
    return exists

def validate_backend():
    """Validate backend implementation"""
    print("\n=== Validating Backend Implementation ===")
    
    # Check core files
    backend_files = [
        "/home/ubuntu/carbon_credit_project/backend/main.py",
        "/home/ubuntu/carbon_credit_project/backend/requirements.txt",
        "/home/ubuntu/carbon_credit_project/backend/.env"
    ]
    
    backend_dirs = [
        "/home/ubuntu/carbon_credit_project/backend/app",
        "/home/ubuntu/carbon_credit_project/backend/app/api",
        "/home/ubuntu/carbon_credit_project/backend/app/core",
        "/home/ubuntu/carbon_credit_project/backend/app/models",
        "/home/ubuntu/carbon_credit_project/backend/app/schemas",
        "/home/ubuntu/carbon_credit_project/backend/app/services"
    ]
    
    files_exist = all(check_file_exists(f) for f in backend_files)
    dirs_exist = all(check_directory_exists(d) for d in backend_dirs)
    
    # Check API endpoints
    api_files = [
        "/home/ubuntu/carbon_credit_project/backend/app/api/auth.py",
        "/home/ubuntu/carbon_credit_project/backend/app/api/projects.py",
        "/home/ubuntu/carbon_credit_project/backend/app/api/verification.py",
        "/home/ubuntu/carbon_credit_project/backend/app/api/satellite.py",
        "/home/ubuntu/carbon_credit_project/backend/app/api/blockchain.py"
    ]
    
    api_exist = all(check_file_exists(f) for f in api_files)
    
    # Check models
    model_files = [
        "/home/ubuntu/carbon_credit_project/backend/app/models/user.py",
        "/home/ubuntu/carbon_credit_project/backend/app/models/project.py",
        "/home/ubuntu/carbon_credit_project/backend/app/models/verification.py",
        "/home/ubuntu/carbon_credit_project/backend/app/models/satellite.py"
    ]
    
    models_exist = all(check_file_exists(f) for f in model_files)
    
    # Check services
    service_files = [
        "/home/ubuntu/carbon_credit_project/backend/app/services/auth_service.py",
        "/home/ubuntu/carbon_credit_project/backend/app/services/project_service.py",
        "/home/ubuntu/carbon_credit_project/backend/app/services/verification_service.py",
        "/home/ubuntu/carbon_credit_project/backend/app/services/satellite_service.py",
        "/home/ubuntu/carbon_credit_project/backend/app/services/blockchain_service.py"
    ]
    
    services_exist = all(check_file_exists(f) for f in service_files)
    
    backend_valid = files_exist and dirs_exist and api_exist and models_exist and services_exist
    
    if backend_valid:
        print("\n✅ Backend implementation is valid")
    else:
        print("\n❌ Backend implementation is incomplete")
    
    return backend_valid

def validate_frontend():
    """Validate frontend implementation"""
    print("\n=== Validating Frontend Implementation ===")
    
    # Check core files
    frontend_files = [
        "/home/ubuntu/carbon_credit_project/frontend/package.json",
        "/home/ubuntu/carbon_credit_project/frontend/src/index.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/App.js"
    ]
    
    frontend_dirs = [
        "/home/ubuntu/carbon_credit_project/frontend/src",
        "/home/ubuntu/carbon_credit_project/frontend/src/components",
        "/home/ubuntu/carbon_credit_project/frontend/src/pages",
        "/home/ubuntu/carbon_credit_project/frontend/src/store",
        "/home/ubuntu/carbon_credit_project/frontend/src/services"
    ]
    
    files_exist = all(check_file_exists(f) for f in frontend_files)
    dirs_exist = all(check_directory_exists(d) for d in frontend_dirs)
    
    # Check components
    component_files = [
        "/home/ubuntu/carbon_credit_project/frontend/src/components/Layout.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/components/ProtectedRoute.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/components/MapComponent.js"
    ]
    
    components_exist = all(check_file_exists(f) for f in component_files)
    
    # Check pages
    page_files = [
        "/home/ubuntu/carbon_credit_project/frontend/src/pages/Login.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/pages/Register.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/pages/Dashboard.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/pages/ProjectDetail.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/pages/NewProject.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/pages/Verification.js"
    ]
    
    pages_exist = all(check_file_exists(f) for f in page_files)
    
    # Check store
    store_files = [
        "/home/ubuntu/carbon_credit_project/frontend/src/store/index.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/store/authSlice.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/store/projectSlice.js",
        "/home/ubuntu/carbon_credit_project/frontend/src/store/verificationSlice.js"
    ]
    
    store_exist = all(check_file_exists(f) for f in store_files)
    
    # Check services
    service_files = [
        "/home/ubuntu/carbon_credit_project/frontend/src/services/api.js"
    ]
    
    services_exist = all(check_file_exists(f) for f in service_files)
    
    frontend_valid = files_exist and dirs_exist and components_exist and pages_exist and store_exist and services_exist
    
    if frontend_valid:
        print("\n✅ Frontend implementation is valid")
    else:
        print("\n❌ Frontend implementation is incomplete")
    
    return frontend_valid

def validate_docker():
    """Validate Docker configuration"""
    print("\n=== Validating Docker Configuration ===")
    
    docker_files = [
        "/home/ubuntu/carbon_credit_project/docker/docker-compose.yml",
        "/home/ubuntu/carbon_credit_project/docker/backend.Dockerfile",
        "/home/ubuntu/carbon_credit_project/docker/frontend.Dockerfile"
    ]
    
    docker_valid = all(check_file_exists(f) for f in docker_files)
    
    if docker_valid:
        print("\n✅ Docker configuration is valid")
    else:
        print("\n❌ Docker configuration is incomplete")
    
    return docker_valid

def validate_requirements():
    """Validate implementation against requirements"""
    print("\n=== Validating Implementation Against Requirements ===")
    
    requirements = [
        {
            "name": "Human-in-the-loop verification",
            "files": [
                "/home/ubuntu/carbon_credit_project/backend/app/api/verification.py",
                "/home/ubuntu/carbon_credit_project/frontend/src/pages/Verification.js"
            ]
        },
        {
            "name": "Blockchain integration",
            "files": [
                "/home/ubuntu/carbon_credit_project/backend/app/api/blockchain.py",
                "/home/ubuntu/carbon_credit_project/backend/app/services/blockchain_service.py"
            ]
        },
        {
            "name": "XAI for transparency",
            "files": [
                "/home/ubuntu/carbon_credit_project/backend/app/models/verification.py",
                "/home/ubuntu/carbon_credit_project/frontend/src/pages/Verification.js"
            ]
        },
        {
            "name": "Project management",
            "files": [
                "/home/ubuntu/carbon_credit_project/backend/app/api/projects.py",
                "/home/ubuntu/carbon_credit_project/frontend/src/pages/ProjectDetail.js"
            ]
        },
        {
            "name": "Satellite imagery analysis",
            "files": [
                "/home/ubuntu/carbon_credit_project/backend/app/api/satellite.py",
                "/home/ubuntu/carbon_credit_project/backend/app/services/satellite_service.py"
            ]
        }
    ]
    
    all_requirements_met = True
    
    for req in requirements:
        print(f"\nChecking requirement: {req['name']}")
        files_exist = all(check_file_exists(f) for f in req['files'])
        
        if files_exist:
            print(f"✅ {req['name']} requirement is implemented")
        else:
            print(f"❌ {req['name']} requirement is not fully implemented")
            all_requirements_met = False
    
    if all_requirements_met:
        print("\n✅ All requirements are implemented")
    else:
        print("\n❌ Some requirements are not fully implemented")
    
    return all_requirements_met

def run_validation():
    """Run all validation checks"""
    print("Starting validation of Carbon Credit Verification SaaS application...")
    
    backend_valid = validate_backend()
    frontend_valid = validate_frontend()
    docker_valid = validate_docker()
    requirements_met = validate_requirements()
    
    print("\n=== Validation Summary ===")
    print(f"Backend: {'✅ Valid' if backend_valid else '❌ Invalid'}")
    print(f"Frontend: {'✅ Valid' if frontend_valid else '❌ Invalid'}")
    print(f"Docker: {'✅ Valid' if docker_valid else '❌ Invalid'}")
    print(f"Requirements: {'✅ Met' if requirements_met else '❌ Not fully met'}")
    
    if backend_valid and frontend_valid and docker_valid and requirements_met:
        print("\n🎉 Implementation is valid and meets all requirements!")
        return True
    else:
        print("\n❌ Implementation has issues that need to be addressed")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
EOL

# Create README.md
cat > /home/ubuntu/carbon_credit_project/README.md << 'EOL'
# Carbon Credit Verification SaaS Application

A Software as a Service (SaaS) application for verifying carbon credits using AI, satellite imagery, and blockchain technology.

## Features

- **Project Management**: Create and manage carbon credit projects with geospatial data
- **Satellite Imagery Analysis**: Process and analyze satellite images to verify forest cover and carbon sequestration
- **Human-in-the-loop Verification**: Combine AI predictions with human expert review
- **Blockchain Certification**: Certify verified carbon credits on the Polygon blockchain
- **Explainable AI**: Transparent AI decision-making with explanations
- **Role-based Access Control**: Different permissions for project developers, verifiers, and administrators

## Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL with PostGIS extension for geospatial data
- **AI/ML**: PyTorch for deep learning models, scikit-learn for traditional ML algorithms
- **XAI Libraries**: SHAP and LIME for model interpretability
- **Containerization**: Docker for consistent development and deployment

### Frontend
- **Framework**: React with TypeScript
- **State Management**: Redux Toolkit
- **UI Components**: Material-UI
- **Mapping**: Leaflet.js with React-Leaflet
- **Data Visualization**: D3.js and Recharts

### Blockchain
- **Platform**: Polygon (Ethereum L2)
- **Smart Contracts**: Solidity with OpenZeppelin
- **Interaction**: ethers.js

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Node.js 16+
- Python 3.10+

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/carbon-credit-verification.git
cd carbon-credit-verification
```

2. Start the application using Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up -d
```

3. Access the application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs

## Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm start
```

## Testing

Run backend tests:
```bash
python test_backend.py
```

Run frontend tests:
```bash
python test_frontend.py
```

Validate implementation:
```bash
python validate_implementation.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project was developed as part of an academic dissertation on carbon credit verification using AI and blockchain technology.
EOL

echo "Validation and testing scripts created successfully!"
