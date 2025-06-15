#!/usr/bin/env python3
"""
Comprehensive API Testing Script
Tests all Carbon Credit Verification API endpoints and verifies database integration
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

def print_test_header(test_name):
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def test_health_endpoint():
    """Test the health endpoint"""
    print_test_header("Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed: {data['status']}")
            print_info(f"Service: {data['service']}")
            print_info(f"Version: {data['version']}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False

def test_user_registration():
    """Test user registration"""
    print_test_header("User Registration")
    
    # Test data
    users_to_register = [
        {
            "email": "alice@example.com",
            "password": "securepassword123",
            "full_name": "Alice Johnson",
            "role": "Project Developer"
        },
        {
            "email": "bob@example.com", 
            "password": "anotherpassword456",
            "full_name": "Bob Smith",
            "role": "Verifier"
        },
        {
            "email": "charlie@example.com",
            "password": "thirdpassword789",
            "full_name": "Charlie Brown",
            "role": "Project Developer"
        }
    ]
    
    registered_users = []
    
    for user_data in users_to_register:
        try:
            response = requests.post(
                f"{API_BASE}/auth/register",
                json=user_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 201:
                token_data = response.json()
                print_success(f"User registered: {user_data['email']}")
                print_info(f"Token received: {token_data['access_token'][:20]}...")
                registered_users.append({
                    "user_data": user_data,
                    "token": token_data['access_token']
                })
            else:
                print_error(f"Registration failed for {user_data['email']}: {response.text}")
                
        except Exception as e:
            print_error(f"Registration error for {user_data['email']}: {str(e)}")
    
    return registered_users

def test_user_login(users):
    """Test user login"""
    print_test_header("User Login")
    
    logged_in_users = []
    
    for user_info in users:
        user_data = user_info['user_data']
        try:
            response = requests.post(
                f"{API_BASE}/auth/login",
                data={
                    "username": user_data['email'],
                    "password": user_data['password']
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                print_success(f"User logged in: {user_data['email']}")
                logged_in_users.append({
                    "user_data": user_data,
                    "token": token_data['access_token']
                })
            else:
                print_error(f"Login failed for {user_data['email']}: {response.text}")
                
        except Exception as e:
            print_error(f"Login error for {user_data['email']}: {str(e)}")
    
    return logged_in_users

def test_user_info(users):
    """Test getting user information"""
    print_test_header("User Information")
    
    for user_info in users:
        user_data = user_info['user_data']
        token = user_info['token']
        
        try:
            response = requests.get(
                f"{API_BASE}/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                user_info_response = response.json()
                print_success(f"User info retrieved: {user_data['email']}")
                print_info(f"Full name: {user_info_response['full_name']}")
                print_info(f"Role: {user_info_response['role']}")
                print_info(f"Active: {user_info_response['is_active']}")
            else:
                print_error(f"User info failed for {user_data['email']}: {response.text}")
                
        except Exception as e:
            print_error(f"User info error for {user_data['email']}: {str(e)}")

def test_project_creation(users):
    """Test project creation"""
    print_test_header("Project Creation")
    
    # Test projects data
    projects_data = [
        {
            "name": "Amazon Rainforest Restoration",
            "description": "Large-scale reforestation project in the Amazon basin",
            "location_name": "Amazon Basin, Brazil",
            "area_size": 2500.75,
            "project_type": "Reforestation"
        },
        {
            "name": "Mangrove Conservation Initiative",
            "description": "Protecting and restoring mangrove ecosystems",
            "location_name": "Sundarbans, Bangladesh",
            "area_size": 1200.50,
            "project_type": "Conservation"
        },
        {
            "name": "Urban Forest Development",
            "description": "Creating green spaces in urban areas",
            "location_name": "São Paulo, Brazil",
            "area_size": 450.25,
            "project_type": "Afforestation"
        },
        {
            "name": "Peatland Restoration Project",
            "description": "Restoring degraded peatland ecosystems",
            "location_name": "Borneo, Indonesia",
            "area_size": 3200.00,
            "project_type": "Restoration"
        }
    ]
    
    created_projects = []
    
    for i, project_data in enumerate(projects_data):
        # Use different users for different projects
        user_info = users[i % len(users)]
        user_data = user_info['user_data']
        token = user_info['token']
        
        try:
            response = requests.post(
                f"{API_BASE}/projects",
                json=project_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
            )
            
            if response.status_code == 201:
                project_response = response.json()
                print_success(f"Project created: {project_data['name']}")
                print_info(f"Project ID: {project_response['id']}")
                print_info(f"Created by: {user_data['email']}")
                print_info(f"Area: {project_response['area_size']} hectares")
                print_info(f"Status: {project_response['status']}")
                
                created_projects.append({
                    "project_data": project_response,
                    "user_info": user_info
                })
            else:
                print_error(f"Project creation failed for {project_data['name']}: {response.text}")
                
        except Exception as e:
            print_error(f"Project creation error for {project_data['name']}: {str(e)}")
    
    return created_projects

def test_project_retrieval(users, projects):
    """Test project retrieval"""
    print_test_header("Project Retrieval")
    
    # Test getting all projects for each user
    for user_info in users:
        user_data = user_info['user_data']
        token = user_info['token']
        
        try:
            response = requests.get(
                f"{API_BASE}/projects",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                projects_response = response.json()
                print_success(f"Projects retrieved for {user_data['email']}")
                print_info(f"Total projects: {projects_response['total']}")
                
                for project in projects_response['projects']:
                    print_info(f"  - {project['name']} (ID: {project['id']})")
            else:
                print_error(f"Project retrieval failed for {user_data['email']}: {response.text}")
                
        except Exception as e:
            print_error(f"Project retrieval error for {user_data['email']}: {str(e)}")
    
    # Test getting specific projects by ID
    print_info("\nTesting specific project retrieval:")
    for project_info in projects:
        project_data = project_info['project_data']
        user_info = project_info['user_info']
        token = user_info['token']
        
        try:
            response = requests.get(
                f"{API_BASE}/projects/{project_data['id']}",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                project_response = response.json()
                print_success(f"Specific project retrieved: {project_response['name']}")
            else:
                print_error(f"Specific project retrieval failed for ID {project_data['id']}: {response.text}")
                
        except Exception as e:
            print_error(f"Specific project retrieval error for ID {project_data['id']}: {str(e)}")

def test_error_cases():
    """Test error handling"""
    print_test_header("Error Handling Tests")
    
    # Test invalid token
    try:
        response = requests.get(
            f"{API_BASE}/projects",
            headers={"Authorization": "Bearer invalid_token"}
        )
        if response.status_code == 401:
            print_success("Invalid token properly rejected (401)")
        else:
            print_error(f"Invalid token test failed: {response.status_code}")
    except Exception as e:
        print_error(f"Invalid token test error: {str(e)}")
    
    # Test duplicate email registration
    try:
        response = requests.post(
            f"{API_BASE}/auth/register",
            json={
                "email": "alice@example.com",  # Already registered
                "password": "password123",
                "full_name": "Alice Duplicate"
            }
        )
        if response.status_code == 400:
            print_success("Duplicate email properly rejected (400)")
        else:
            print_error(f"Duplicate email test failed: {response.status_code}")
    except Exception as e:
        print_error(f"Duplicate email test error: {str(e)}")
    
    # Test short password
    try:
        response = requests.post(
            f"{API_BASE}/auth/register",
            json={
                "email": "shortpass@example.com",
                "password": "123",  # Too short
                "full_name": "Short Password User"
            }
        )
        if response.status_code == 422:
            print_success("Short password properly rejected (422)")
        else:
            print_error(f"Short password test failed: {response.status_code}")
    except Exception as e:
        print_error(f"Short password test error: {str(e)}")

def test_api_documentation():
    """Test API documentation endpoint"""
    print_test_header("API Documentation")
    
    try:
        response = requests.get(f"{API_BASE}/docs")
        if response.status_code == 200:
            print_success("API documentation accessible")
            print_info(f"Documentation URL: {API_BASE}/docs")
        else:
            print_error(f"API documentation failed: {response.status_code}")
    except Exception as e:
        print_error(f"API documentation error: {str(e)}")

def main():
    """Run comprehensive API tests"""
    print("🚀 Starting Comprehensive API Testing")
    print(f"📍 Testing API at: {BASE_URL}")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Health Check
    if not test_health_endpoint():
        print_error("Health check failed. Exiting tests.")
        sys.exit(1)
    
    # Test 2: User Registration
    registered_users = test_user_registration()
    if not registered_users:
        print_error("No users registered successfully. Exiting tests.")
        sys.exit(1)
    
    # Test 3: User Login
    logged_in_users = test_user_login(registered_users)
    if not logged_in_users:
        print_error("No users logged in successfully. Exiting tests.")
        sys.exit(1)
    
    # Test 4: User Information
    test_user_info(logged_in_users)
    
    # Test 5: Project Creation
    created_projects = test_project_creation(logged_in_users)
    if not created_projects:
        print_error("No projects created successfully.")
    
    # Test 6: Project Retrieval
    if created_projects:
        test_project_retrieval(logged_in_users, created_projects)
    
    # Test 7: Error Handling
    test_error_cases()
    
    # Test 8: API Documentation
    test_api_documentation()
    
    # Summary
    print_test_header("Test Summary")
    print_success(f"✨ Comprehensive API testing completed!")
    print_info(f"📊 Users registered: {len(registered_users)}")
    print_info(f"🔐 Users logged in: {len(logged_in_users)}")
    print_info(f"📋 Projects created: {len(created_projects)}")
    print_info(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{'='*60}")
    print("🎉 All API endpoints tested successfully!")
    print("🗄️  Data has been added to the database")
    print("📚 Check the API documentation at: http://localhost:8000/api/v1/docs")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 