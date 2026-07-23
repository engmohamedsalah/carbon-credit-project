"""
Comprehensive API Test Suite
Tests for all major API endpoints with edge cases and error handling
"""
import pytest
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the main app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from main import app

client = TestClient(app)

class TestAPIComprehensive:
    """Comprehensive test suite for all API endpoints"""
    
    @pytest.fixture
    def admin_token(self):
        """Get admin authentication token"""
        login_data = {
            "username": "testadmin@example.com",
            "password": "changeme"
        }
        response = client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 200
        return response.json()["access_token"]
    
    @pytest.fixture
    def user_token(self):
        """Get regular user authentication token"""
        login_data = {
            "username": "test@example.com",
            "password": "changeme"
        }
        response = client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 200
        return response.json()["access_token"]
    
    @pytest.fixture
    def test_project_id(self, admin_token):
        """Create a test project"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        project_data = {
            "name": "API Test Project",
            "description": "Test project for comprehensive API testing",
            "location_name": "Test Location",
            "area_hectares": 150.0,
            "project_type": "Reforestation",
            "estimated_carbon_credits": 5000.0
        }
        response = client.post("/api/v1/projects", json=project_data, headers=headers)
        assert response.status_code == 201
        return response.json()["id"]
    
    # Authentication Tests
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_register_user(self):
        """Test user registration"""
        user_data = {
            "email": "newuser@test.com",
            "password": "changeme",
            "full_name": "New Test User",
            "role": "Project Developer"
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        login_data = {
            "username": "nonexistent@test.com",
            "password": "wrongpassword"
        }
        response = client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 401
    
    def test_get_current_user_info(self, admin_token):
        """Test getting current user information"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "email" in data
        assert "full_name" in data
        assert "role" in data
    
    # Project Management Tests
    def test_create_project(self, admin_token):
        """Test project creation"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        project_data = {
            "name": "Test Project Creation",
            "description": "Testing project creation",
            "location_name": "Test Location",
            "area_hectares": 100.0,
            "project_type": "Reforestation"
        }
        response = client.post("/api/v1/projects", json=project_data, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Project Creation"
        assert data["status"] == "Planning"
        assert "id" in data
        assert "created_at" in data
    
    def test_get_projects(self, admin_token):
        """Test getting projects list"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/projects", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_project_by_id(self, admin_token, test_project_id):
        """Test getting specific project"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get(f"/api/v1/projects/{test_project_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_project_id
        assert data["name"] == "API Test Project"
    
    def test_update_project(self, admin_token, test_project_id):
        """Test project update"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        update_data = {
            "name": "Updated Project Name",
            "description": "Updated description",
            "location_name": "Updated Location",
            "area_hectares": 200.0,
            "project_type": "Afforestation"
        }
        response = client.put(f"/api/v1/projects/{test_project_id}", json=update_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Project Name"
        assert data["area_hectares"] == 200.0
    
    def test_update_project_status(self, admin_token, test_project_id):
        """Test project status update"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        status_data = {"status": "Active"}
        response = client.patch(f"/api/v1/projects/{test_project_id}/status", json=status_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Active"
    
    def test_get_project_status_logs(self, admin_token, test_project_id):
        """Test getting project status logs"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get(f"/api/v1/projects/{test_project_id}/status-logs", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    # ML Analysis Tests
    def test_ml_status(self, admin_token):
        """Test ML service status"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/ml/status", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
    
    def test_analyze_location(self, admin_token, test_project_id):
        """Test location analysis"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        analysis_data = {
            "project_id": test_project_id,
            "latitude": 40.7128,
            "longitude": -74.0060,
            "analysis_type": "comprehensive"
        }
        response = client.post("/api/v1/ml/analyze-location", json=analysis_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == test_project_id
        assert data["analysis_type"] == "comprehensive"
        assert "results" in data
        assert "timestamp" in data
    
    # Verification Tests
    def test_create_verification(self, admin_token, test_project_id):
        """Test verification creation"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        verification_data = {
            "project_id": test_project_id,
            "carbon_impact": 2500.0,
            "verification_notes": "Test verification"
        }
        response = client.post("/api/v1/verification", json=verification_data, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert data["project_id"] == test_project_id
        assert data["carbon_impact"] == 2500.0
        assert data["status"] == "Pending"
    
    def test_get_verifications(self, admin_token):
        """Test getting verifications list"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/verification", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    # Analytics Tests
    def test_dashboard_analytics(self, admin_token):
        """Test dashboard analytics"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/analytics/dashboard", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_projects" in data
        assert "total_verifications" in data
        assert "total_carbon_impact" in data
    
    def test_performance_analytics(self, admin_token):
        """Test performance analytics"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/analytics/performance", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "model_accuracy" in data
        assert "processing_times" in data
    
    def test_carbon_impact_analytics(self, admin_token):
        """Test carbon impact analytics"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/analytics/carbon-impact", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_impact" in data
        assert "project_breakdown" in data
    
    # Report Tests
    def test_download_analytics_report(self, admin_token):
        """Test analytics report download"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/reports/analytics", headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
    
    def test_download_project_report(self, admin_token, test_project_id):
        """Test project report download"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get(f"/api/v1/reports/project/{test_project_id}", headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
    
    # XAI Tests
    def test_get_xai_methods(self, admin_token):
        """Test getting available XAI methods"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/xai/methods", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "methods" in data
        assert isinstance(data["methods"], list)
    
    def test_generate_explanation(self, admin_token, test_project_id):
        """Test XAI explanation generation"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        explanation_data = {
            "model_id": "forest_cover_ensemble",
            "instance_data": {
                "project_id": test_project_id,
                "location": {"lat": 40.7128, "lng": -74.0060}
            },
            "explanation_method": "shap",
            "business_friendly": True
        }
        response = client.post("/api/v1/xai/generate-explanation", json=explanation_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "explanation_id" in data
        assert "status" in data
    
    # Error Handling Tests
    def test_invalid_project_id(self, admin_token):
        """Test accessing non-existent project"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/api/v1/projects/99999", headers=headers)
        assert response.status_code == 404
    
    def test_unauthorized_access(self, user_token):
        """Test unauthorized access to admin-only endpoints"""
        headers = {"Authorization": f"Bearer {user_token}"}
        response = client.get("/api/v1/analytics/dashboard", headers=headers)
        # Should either succeed (if user has access) or return 403
        assert response.status_code in [200, 403]
    
    def test_invalid_token(self):
        """Test access with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/projects", headers=headers)
        assert response.status_code == 401
    
    def test_missing_token(self):
        """Test access without token"""
        response = client.get("/api/v1/projects")
        assert response.status_code == 401
    
    # Rate Limiting Tests
    def test_rate_limiting_login(self):
        """Test rate limiting on login endpoint"""
        login_data = {
            "username": "testadmin@example.com",
            "password": "wrongpassword"
        }
        
        # Make multiple requests to trigger rate limiting
        responses = []
        for _ in range(6):  # More than 5/minute limit
            response = client.post("/api/v1/auth/login", data=login_data)
            responses.append(response.status_code)
        
        # At least one should be rate limited (429)
        assert 429 in responses
    
    # Data Validation Tests
    def test_invalid_project_data(self, admin_token):
        """Test project creation with invalid data"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        invalid_project_data = {
            "name": "",  # Empty name should fail
            "location_name": "Test Location",
            "area_hectares": -100  # Negative area should fail
        }
        response = client.post("/api/v1/projects", json=invalid_project_data, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_verification_data(self, admin_token, test_project_id):
        """Test verification creation with invalid data"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        invalid_verification_data = {
            "project_id": 99999,  # Non-existent project
            "carbon_impact": -1000  # Negative carbon impact
        }
        response = client.post("/api/v1/verification", json=invalid_verification_data, headers=headers)
        assert response.status_code in [404, 422]  # Project not found or validation error

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 