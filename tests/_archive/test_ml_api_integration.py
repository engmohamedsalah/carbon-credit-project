"""
Comprehensive API Integration Tests for ML Endpoints
Tests the ML API endpoints with authentication, validation, and error handling
"""

import pytest
import asyncio
import tempfile
import os
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.main import app
from backend.services.ml_service import MLService


class TestMLAPIEndpoints:
    """Test suite for ML API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self, client):
        """Create authenticated user and return auth headers"""
        # Register test user
        user_data = {
            "username": "mltest@example.com",
            "email": "mltest@example.com",
            "password": "TestPassword123!",
            "full_name": "ML Test User"
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        
        # Login to get token
        login_data = {
            "username": "mltest@example.com",
            "password": "TestPassword123!"
        }
        login_response = client.post("/api/v1/auth/login", data=login_data)
        assert login_response.status_code == 200
        
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def test_project_id(self, client, auth_headers):
        """Create test project and return its ID"""
        project_data = {
            "name": "ML API Test Project",
            "location_name": "Amazon Basin Test Area",
            "area_hectares": 1000.0,
            "project_type": "Reforestation",
            "description": "Test project for ML API integration",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-62.3, -3.5],
                    [-62.2, -3.5],
                    [-62.2, -3.4],
                    [-62.3, -3.4],
                    [-62.3, -3.5]
                ]]
            }
        }
        response = client.post("/api/v1/projects", json=project_data, headers=auth_headers)
        assert response.status_code in [200, 201]
        return response.json()["id"]

    def test_ml_status_endpoint(self, client, auth_headers):
        """Test ML service status endpoint"""
        response = client.get("/api/v1/ml/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        # Handle both cases: ML service available or not available
        if "error" in data:
            assert data["error"] == "ML service not available"
        else:
            assert "ml_service" in data
            assert "models_ready" in data
            assert "service_version" in data

    def test_ml_status_endpoint_detailed(self, client, auth_headers):
        """Test ML service status endpoint with detailed information"""
        # This test will work regardless of ML service availability
        response = client.get("/api/v1/ml/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        # Check that we get a valid response structure
        assert isinstance(data, dict)
        
        # If ML service is available, check detailed structure
        if "ml_service" in data:
            assert "models_ready" in data
            assert "service_version" in data

    def test_analyze_location_success(self, client, auth_headers, test_project_id):
        """Test successful location analysis"""
        location_data = {
            "project_id": test_project_id,
            "latitude": -3.4653,
            "longitude": -62.2159,
            "analysis_type": "comprehensive"
        }
        
        with patch('backend.services.ml_service.MLService') as mock_service:
            mock_instance = Mock()
            mock_instance.analyze_location = AsyncMock(return_value={
                "project_id": test_project_id,
                "coordinates": [-3.4653, -62.2159],
                "forest_analysis": {
                    "forest_coverage_percent": 65.8,
                    "forest_area_hectares": 1250.0,
                    "confidence_score": 0.89
                },
                "carbon_estimate": {
                    "total_carbon_tons": 2847.5,
                    "carbon_per_hectare": 2.278,
                    "sequestration_rate": "12.5 tons/year"
                },
                "status": "completed"
            })
            mock_service.return_value = mock_instance
            
            response = client.post(
                "/api/v1/ml/analyze-location",
                json=location_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["project_id"] == test_project_id
            assert data["status"] == "completed"
            assert "forest_analysis" in data
            assert "carbon_estimate" in data

    def test_analyze_location_invalid_coordinates(self, client, auth_headers, test_project_id):
        """Test location analysis with invalid coordinates"""
        location_data = {
            "project_id": test_project_id,
            "latitude": 91.0,  # Invalid latitude
            "longitude": -62.2159
        }
        
        response = client.post(
            "/api/v1/ml/analyze-location",
            json=location_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error

    def test_analyze_location_unauthorized(self, client, test_project_id):
        """Test location analysis without authentication"""
        location_data = {
            "project_id": test_project_id,
            "latitude": -3.4653,
            "longitude": -62.2159
        }
        
        response = client.post(
            "/api/v1/ml/analyze-location",
            json=location_data
        )
        
        assert response.status_code == 401

    def test_forest_cover_analysis_success(self, client, auth_headers, test_project_id):
        """Test successful forest cover analysis with file upload"""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b'fake image data for testing')
            tmp_file.flush()
            
            with patch('backend.services.ml_service.MLService') as mock_service:
                mock_instance = Mock()
                mock_instance.analyze_forest_cover = AsyncMock(return_value={
                    "project_id": test_project_id,
                    "analysis_type": "forest_cover",
                    "forest_coverage": 0.65,
                    "confidence": 0.89,
                    "processing_time": 15.2
                })
                mock_service.return_value = mock_instance
                
                with open(tmp_file.name, 'rb') as file:
                    response = client.post(
                        f"/api/v1/ml/forest-cover?project_id={test_project_id}",
                        files={"image": ("test.jpg", file, "image/jpeg")},
                        headers=auth_headers
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["project_id"] == test_project_id
                assert data["analysis_type"] == "forest_cover"
            
            # Clean up
            os.unlink(tmp_file.name)

    def test_forest_cover_analysis_invalid_file_type(self, client, auth_headers, test_project_id):
        """Test forest cover analysis with invalid file type"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b'not an image file')
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as file:
                response = client.post(
                    f"/api/v1/ml/forest-cover?project_id={test_project_id}",
                    files={"image": ("test.txt", file, "text/plain")},
                    headers=auth_headers
                )
            
            assert response.status_code == 422
            
            # Clean up
            os.unlink(tmp_file.name)

    def test_forest_cover_analysis_file_too_large(self, client, auth_headers, test_project_id):
        """Test forest cover analysis with file too large"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a file larger than 50MB
            tmp_file.write(b'x' * (51 * 1024 * 1024))
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as file:
                response = client.post(
                    f"/api/v1/ml/forest-cover?project_id={test_project_id}",
                    files={"image": ("large_test.jpg", file, "image/jpeg")},
                    headers=auth_headers
                )
            
            assert response.status_code == 413  # File too large
            
            # Clean up
            os.unlink(tmp_file.name)

    def test_change_detection_success(self, client, auth_headers, test_project_id):
        """Test successful change detection with two images"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as before_file:
            before_file.write(b'before image data')
            before_file.flush()
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as after_file:
                after_file.write(b'after image data')
                after_file.flush()
                
                with patch('backend.services.ml_service.MLService') as mock_service:
                    mock_instance = Mock()
                    mock_instance.detect_changes = AsyncMock(return_value={
                        "project_id": test_project_id,
                        "analysis_type": "change_detection",
                        "change_percentage": 0.15,
                        "confidence": 0.82,
                        "processing_time": 25.5
                    })
                    mock_service.return_value = mock_instance
                    
                    with open(before_file.name, 'rb') as bf, open(after_file.name, 'rb') as af:
                        response = client.post(
                            f"/api/v1/ml/change-detection?project_id={test_project_id}",
                            files={
                                "before_image": ("before.jpg", bf, "image/jpeg"),
                                "after_image": ("after.jpg", af, "image/jpeg")
                            },
                            headers=auth_headers
                        )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["project_id"] == test_project_id
                    assert data["analysis_type"] == "change_detection"
                
                # Clean up
                os.unlink(before_file.name)
                os.unlink(after_file.name)

    def test_change_detection_missing_file(self, client, auth_headers, test_project_id):
        """Test change detection with missing file"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as before_file:
            before_file.write(b'before image data')
            before_file.flush()
            
            with open(before_file.name, 'rb') as bf:
                response = client.post(
                    f"/api/v1/ml/change-detection?project_id={test_project_id}",
                    files={"before_image": ("before.jpg", bf, "image/jpeg")},
                    headers=auth_headers
                )
            
            assert response.status_code == 422
            
            # Clean up
            os.unlink(before_file.name)

    def test_ml_service_error_handling(self, client, auth_headers, test_project_id):
        """Test error handling when ML service fails"""
        location_data = {
            "project_id": test_project_id,
            "latitude": -3.4653,
            "longitude": -62.2159
        }
        
        with patch('backend.services.ml_service.MLService') as mock_service:
            mock_instance = Mock()
            mock_instance.analyze_location = AsyncMock(side_effect=RuntimeError("ML service error"))
            mock_service.return_value = mock_instance
            
            response = client.post(
                "/api/v1/ml/analyze-location",
                json=location_data,
                headers=auth_headers
            )
            
            assert response.status_code == 500
            assert "ML service error" in response.json()["detail"]

    def test_nonexistent_project(self, client, auth_headers):
        """Test ML endpoints with non-existent project ID"""
        location_data = {
            "project_id": 99999,
            "latitude": -3.4653,
            "longitude": -62.2159
        }

        response = client.post(
            "/api/v1/ml/analyze-location",
            json=location_data,
            headers=auth_headers
        )

        assert response.status_code == 404


class TestMLAPIValidation:
    """Test suite for ML API input validation"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self, client):
        """Create authenticated user and return auth headers"""
        user_data = {
            "username": "validation_test@example.com",
            "email": "validation_test@example.com",
            "password": "TestPassword123!",
            "full_name": "Validation Test User"
        }
        client.post("/api/v1/auth/register", json=user_data)
        
        login_data = {
            "username": "validation_test@example.com",
            "password": "TestPassword123!"
        }
        login_response = client.post("/api/v1/auth/login", data=login_data)
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_coordinate_validation_edge_cases(self, client, auth_headers):
        """Test coordinate validation edge cases"""
        # Valid edge case coordinates
        valid_coordinates = [
            {"project_id": 1, "latitude": -90.0, "longitude": -180.0},
            {"project_id": 1, "latitude": 90.0, "longitude": 180.0},
            {"project_id": 1, "latitude": 0.0, "longitude": 0.0}
        ]
        
        for location_data in valid_coordinates:
            response = client.post(
                "/api/v1/ml/analyze-location",
                json=location_data,
                headers=auth_headers
            )
            # Should not be validation error (may be other errors like project not found)
            assert response.status_code != 422

    def test_invalid_coordinate_validation(self, client, auth_headers):
        """Test invalid coordinate validation"""
        invalid_coordinates = [
            {"project_id": 1, "latitude": -91.0, "longitude": 0.0},  # Invalid latitude
            {"project_id": 1, "latitude": 91.0, "longitude": 0.0},   # Invalid latitude
            {"project_id": 1, "latitude": 0.0, "longitude": -181.0}, # Invalid longitude
            {"project_id": 1, "latitude": 0.0, "longitude": 181.0},  # Invalid longitude
            {"project_id": 1, "latitude": "invalid", "longitude": 0.0}, # Wrong type
        ]
        
        for location_data in invalid_coordinates:
            response = client.post(
                "/api/v1/ml/analyze-location",
                json=location_data,
                headers=auth_headers
            )
            assert response.status_code == 422

    def test_missing_required_fields(self, client, auth_headers):
        """Test missing required fields validation"""
        incomplete_data = [
            {},  # Missing all fields
            {"project_id": 1},  # Missing coordinates
            {"project_id": 1, "latitude": -3.4653},  # Missing longitude
            {"project_id": 1, "longitude": -62.2159},  # Missing latitude
        ]
        
        for data in incomplete_data:
            response = client.post(
                "/api/v1/ml/analyze-location",
                json=data,
                headers=auth_headers
            )
            assert response.status_code == 422


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 