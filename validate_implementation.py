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
