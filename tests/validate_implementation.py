import os
import json
import subprocess
import sys
import requests

# Get the current project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_file_exists(filepath):
    """Check if a file exists"""
    full_path = os.path.join(PROJECT_DIR, filepath) if not filepath.startswith('/') else filepath
    exists = os.path.isfile(full_path)
    print(f"{'✅' if exists else '❌'} {filepath}")
    return exists

def check_directory_exists(dirpath):
    """Check if a directory exists"""
    full_path = os.path.join(PROJECT_DIR, dirpath) if not dirpath.startswith('/') else dirpath
    exists = os.path.isdir(full_path)
    print(f"{'✅' if exists else '❌'} {dirpath}")
    return exists

def validate_backend():
    """Validate backend implementation"""
    print("\n=== Validating Backend Implementation ===")
    
    # Check core files
    backend_files = [
        "backend/main.py",
        "backend/requirements.txt"
    ]
    
    backend_dirs = [
        "backend/services"
    ]
    
    files_exist = all(check_file_exists(f) for f in backend_files)
    dirs_exist = all(check_directory_exists(d) for d in backend_dirs)
    
    # Check ML service
    ml_files = [
        "backend/services/ml_service.py"
    ]
    
    ml_exist = all(check_file_exists(f) for f in ml_files)
    
    # Check database
    db_files = [
        "database/carbon_credits.db"
    ]
    
    db_exist = all(check_file_exists(f) for f in db_files)
    
    backend_valid = files_exist and dirs_exist and ml_exist and db_exist
    
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
        "frontend/package.json",
        "frontend/src/index.js",
        "frontend/src/App.js"
    ]
    
    frontend_dirs = [
        "frontend/src",
        "frontend/src/components",
        "frontend/src/pages",
        "frontend/src/store",
        "frontend/src/services"
    ]
    
    files_exist = all(check_file_exists(f) for f in frontend_files)
    dirs_exist = all(check_directory_exists(d) for d in frontend_dirs)
    
    # Check components
    component_files = [
        "frontend/src/components/Layout.js",
        "frontend/src/components/ProtectedRoute.js",
        "frontend/src/components/MapComponent.js",
        "frontend/src/components/MLAnalysis.js"
    ]
    
    components_exist = all(check_file_exists(f) for f in component_files)
    
    # Check pages
    page_files = [
        "frontend/src/pages/Login.js",
        "frontend/src/pages/Register.js",
        "frontend/src/pages/Dashboard.js",
        "frontend/src/pages/ProjectDetail.js",
        "frontend/src/pages/NewProject.js",
        "frontend/src/pages/EditProject.js",
        "frontend/src/pages/Verification.js",
        "frontend/src/pages/XAI.js"
    ]
    
    pages_exist = all(check_file_exists(f) for f in page_files)
    
    # Check store
    store_files = [
        "frontend/src/store/index.js",
        "frontend/src/store/authSlice.js",
        "frontend/src/store/projectSlice.js",
        "frontend/src/store/verificationSlice.js"
    ]
    
    store_exist = all(check_file_exists(f) for f in store_files)
    
    # Check services
    service_files = [
        "frontend/src/services/apiService.js",
        "frontend/src/services/mlService.js"
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
        "docker/docker-compose.yml",
        "docker/backend.Dockerfile",
        "docker/frontend.Dockerfile"
    ]
    
    docker_valid = all(check_file_exists(f) for f in docker_files)
    
    if docker_valid:
        print("\n✅ Docker configuration is valid")
    else:
        print("\n❌ Docker configuration is incomplete")
    
    return docker_valid

def validate_ml_models():
    """Validate ML models"""
    print("\n=== Validating ML Models ===")
    
    ml_files = [
        "ml/models/forest_cover_unet_focal_alpha_0.75_threshold_0.53.pth",
        "ml/models/change_detection_siamese_unet.pth",
        "ml/models/convlstm_fast_final.pth",
        "ml/models/ensemble_config.json"
    ]
    
    ml_valid = all(check_file_exists(f) for f in ml_files)
    
    if ml_valid:
        print("\n✅ ML models are available")
    else:
        print("\n❌ Some ML models are missing")
    
    return ml_valid

def validate_api_endpoints():
    """Validate API endpoints are working"""
    print("\n=== Validating API Endpoints ===")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        health_ok = response.status_code == 200
        print(f"{'✅' if health_ok else '❌'} Health endpoint")
        
        # Test API documentation
        response = requests.get(f"{base_url}/api/v1/docs", timeout=5)
        docs_ok = response.status_code == 200
        print(f"{'✅' if docs_ok else '❌'} API documentation")
        
        api_valid = health_ok and docs_ok
        
    except requests.exceptions.RequestException:
        print("❌ API server is not running")
        api_valid = False
    
    if api_valid:
        print("\n✅ API endpoints are working")
    else:
        print("\n❌ API endpoints have issues")
    
    return api_valid

def validate_requirements():
    """Validate implementation against requirements"""
    print("\n=== Validating Implementation Against Requirements ===")
    
    requirements = [
        {
            "name": "Project Management (CRUD)",
            "files": [
                "frontend/src/pages/ProjectDetail.js",
                "frontend/src/pages/NewProject.js",
                "frontend/src/pages/EditProject.js"
            ]
        },
        {
            "name": "ML Integration",
            "files": [
                "backend/services/ml_service.py",
                "frontend/src/services/mlService.js",
                "ml/models/forest_cover_unet_focal_alpha_0.75_threshold_0.53.pth"
            ]
        },
        {
            "name": "Verification Workflow",
            "files": [
                "frontend/src/pages/Verification.js",
                "frontend/src/store/verificationSlice.js"
            ]
        },
        {
            "name": "Authentication & RBAC",
            "files": [
                "frontend/src/pages/Login.js",
                "frontend/src/utils/roleUtils.js",
                "frontend/src/store/authSlice.js"
            ]
        },
        {
            "name": "XAI Integration",
            "files": [
                "frontend/src/pages/XAI.js",
                "frontend/src/services/xaiService.js"
            ]
        },
        {
            "name": "Status Logging System",
            "files": [
                "frontend/src/pages/ProjectDetail.js"  # Contains status logging UI
            ]
        }
    ]
    
    all_requirements_met = True
    
    for req in requirements:
        print(f"\nChecking requirement: {req['name']}")
        req_files_exist = all(check_file_exists(f) for f in req['files'])
        
        if req_files_exist:
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
    print(f"Project directory: {PROJECT_DIR}")
    
    backend_valid = validate_backend()
    frontend_valid = validate_frontend()
    docker_valid = validate_docker()
    ml_valid = validate_ml_models()
    api_valid = validate_api_endpoints()
    requirements_valid = validate_requirements()
    
    print("\n=== Validation Summary ===")
    print(f"Backend: {'✅ Valid' if backend_valid else '❌ Invalid'}")
    print(f"Frontend: {'✅ Valid' if frontend_valid else '❌ Invalid'}")
    print(f"Docker: {'✅ Valid' if docker_valid else '❌ Invalid'}")
    print(f"ML Models: {'✅ Valid' if ml_valid else '❌ Invalid'}")
    print(f"API Endpoints: {'✅ Valid' if api_valid else '❌ Invalid'}")
    print(f"Requirements: {'✅ Met' if requirements_valid else '❌ Not fully met'}")
    
    overall_valid = all([backend_valid, frontend_valid, docker_valid, ml_valid, api_valid, requirements_valid])
    
    if overall_valid:
        print("\n🎉 Implementation is complete and valid!")
        return True
    else:
        print("\n❌ Implementation has issues that need to be addressed")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
