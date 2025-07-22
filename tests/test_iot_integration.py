"""
IoT Integration Test Suite
Tests for IoT sensor management and real-time data functionality
"""
import pytest
import json
import sqlite3
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the main app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from main import app

client = TestClient(app)

class TestIoTIntegration:
    """Test suite for IoT sensor and data management"""
    
    @pytest.fixture
    def auth_token(self):
        """Get authentication token for testing"""
        # Create test user and get token
        login_data = {
            "username": "testadmin@example.com",
            "password": "password123"
        }
        response = client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 200
        return response.json()["access_token"]
    
    @pytest.fixture
    def test_project_id(self, auth_token):
        """Create a test project for IoT testing"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        project_data = {
            "name": "IoT Test Project",
            "description": "Test project for IoT integration",
            "location_name": "Test Location",
            "area_hectares": 100.0,
            "project_type": "Reforestation"
        }
        response = client.post("/api/v1/projects", json=project_data, headers=headers)
        assert response.status_code == 201
        return response.json()["id"]
    
    def test_create_iot_sensor(self, auth_token, test_project_id):
        """Test creating an IoT sensor"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        sensor_data = {
            "sensor_id": "TEST_SENSOR_001",
            "sensor_type": "soil_moisture",
            "location_lat": 40.7128,
            "location_lng": -74.0060,
            "project_id": test_project_id,
            "installation_date": "2024-01-15",
            "calibration_data": {"calibration_factor": 1.02}
        }
        
        response = client.post("/api/v1/iot/sensors", json=sensor_data, headers=headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["sensor_id"] == "TEST_SENSOR_001"
        assert data["sensor_type"] == "soil_moisture"
        assert data["project_id"] == test_project_id
        assert data["status"] == "active"
        assert "id" in data
        assert "created_at" in data
    
    def test_get_iot_sensors(self, auth_token, test_project_id):
        """Test retrieving IoT sensors"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Create a sensor first
        sensor_data = {
            "sensor_id": "TEST_SENSOR_002",
            "sensor_type": "co2_flux",
            "location_lat": 40.7128,
            "location_lng": -74.0060,
            "project_id": test_project_id
        }
        client.post("/api/v1/iot/sensors", json=sensor_data, headers=headers)
        
        # Get sensors
        response = client.get(f"/api/v1/iot/sensors?project_id={test_project_id}", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        
        # Verify sensor data
        sensor = next((s for s in data if s["sensor_id"] == "TEST_SENSOR_002"), None)
        assert sensor is not None
        assert sensor["sensor_type"] == "co2_flux"
    
    def test_create_sensor_reading(self, auth_token, test_project_id):
        """Test creating sensor readings"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Create a sensor first
        sensor_data = {
            "sensor_id": "TEST_SENSOR_003",
            "sensor_type": "temperature",
            "location_lat": 40.7128,
            "location_lng": -74.0060,
            "project_id": test_project_id
        }
        client.post("/api/v1/iot/sensors", json=sensor_data, headers=headers)
        
        # Create reading
        reading_data = {
            "sensor_id": "TEST_SENSOR_003",
            "reading_type": "temperature",
            "value": 25.5,
            "unit": "°C",
            "metadata": {"accuracy": "±0.1°C"}
        }
        
        response = client.post("/api/v1/iot/readings", json=reading_data, headers=headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["sensor_id"] == "TEST_SENSOR_003"
        assert data["reading_type"] == "temperature"
        assert data["value"] == 25.5
        assert data["unit"] == "°C"
        assert "id" in data
        assert "timestamp" in data
    
    def test_get_sensor_readings(self, auth_token, test_project_id):
        """Test retrieving sensor readings"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Create a sensor and reading
        sensor_data = {
            "sensor_id": "TEST_SENSOR_004",
            "sensor_type": "tree_growth",
            "location_lat": 40.7128,
            "location_lng": -74.0060,
            "project_id": test_project_id
        }
        client.post("/api/v1/iot/sensors", json=sensor_data, headers=headers)
        
        reading_data = {
            "sensor_id": "TEST_SENSOR_004",
            "reading_type": "growth_rate",
            "value": 2.3,
            "unit": "cm/month"
        }
        client.post("/api/v1/iot/readings", json=reading_data, headers=headers)
        
        # Get readings
        response = client.get("/api/v1/iot/readings/TEST_SENSOR_004?limit=10", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        
        reading = data[0]
        assert reading["sensor_id"] == "TEST_SENSOR_004"
        assert reading["reading_type"] == "growth_rate"
        assert reading["value"] == 2.3
    
    def test_iot_analytics(self, auth_token, test_project_id):
        """Test IoT analytics endpoint"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Create multiple sensors and readings
        sensors = [
            {"sensor_id": "ANALYTICS_001", "sensor_type": "soil_moisture", "location_lat": 40.7128, "location_lng": -74.0060, "project_id": test_project_id},
            {"sensor_id": "ANALYTICS_002", "sensor_type": "co2_flux", "location_lat": 40.7128, "location_lng": -74.0060, "project_id": test_project_id}
        ]
        
        for sensor in sensors:
            client.post("/api/v1/iot/sensors", json=sensor, headers=headers)
        
        # Create readings
        readings = [
            {"sensor_id": "ANALYTICS_001", "reading_type": "moisture", "value": 45.2, "unit": "%"},
            {"sensor_id": "ANALYTICS_002", "reading_type": "co2_flux", "value": 12.5, "unit": "μmol/m²/s"}
        ]
        
        for reading in readings:
            client.post("/api/v1/iot/readings", json=reading, headers=headers)
        
        # Get analytics
        response = client.get("/api/v1/iot/analytics", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify analytics structure
        assert "total_sensors" in data
        assert "active_sensors" in data
        assert "total_readings" in data
        assert "sensor_types" in data
        assert "recent_activity" in data
        
        # Verify counts
        assert data["total_sensors"] >= 2
        assert data["total_readings"] >= 2
    
    def test_iot_sensor_authorization(self, auth_token, test_project_id):
        """Test IoT sensor authorization (non-admin users can only access their own projects)"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Try to create sensor for a different project (should fail for non-admin)
        # First, create a project with a different user
        other_project_data = {
            "name": "Other User Project",
            "description": "Project owned by different user",
            "location_name": "Other Location",
            "area_hectares": 50.0,
            "project_type": "Reforestation"
        }
        
        # This would require a different user token, but for now we'll test the structure
        sensor_data = {
            "sensor_id": "UNAUTHORIZED_SENSOR",
            "sensor_type": "soil_moisture",
            "location_lat": 40.7128,
            "location_lng": -74.0060,
            "project_id": 99999  # Non-existent project
        }
        
        response = client.post("/api/v1/iot/sensors", json=sensor_data, headers=headers)
        
        # Should fail with 404 (project not found) or 403 (not authorized)
        assert response.status_code in [403, 404]
    
    def test_iot_data_persistence(self, auth_token, test_project_id):
        """Test that IoT data persists correctly in the database"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Create sensor
        sensor_data = {
            "sensor_id": "PERSISTENCE_TEST",
            "sensor_type": "temperature",
            "location_lat": 40.7128,
            "location_lng": -74.0060,
            "project_id": test_project_id
        }
        
        response = client.post("/api/v1/iot/sensors", json=sensor_data, headers=headers)
        assert response.status_code == 201
        sensor_id = response.json()["id"]
        
        # Create reading
        reading_data = {
            "sensor_id": "PERSISTENCE_TEST",
            "reading_type": "temperature",
            "value": 22.5,
            "unit": "°C"
        }
        
        response = client.post("/api/v1/iot/readings", json=reading_data, headers=headers)
        assert response.status_code == 201
        
        # Verify data persists by retrieving it
        response = client.get("/api/v1/iot/sensors", headers=headers)
        assert response.status_code == 200
        sensors = response.json()
        
        # Find our sensor
        sensor = next((s for s in sensors if s["sensor_id"] == "PERSISTENCE_TEST"), None)
        assert sensor is not None
        assert sensor["id"] == sensor_id
        
        # Verify readings persist
        response = client.get("/api/v1/iot/readings/PERSISTENCE_TEST", headers=headers)
        assert response.status_code == 200
        readings = response.json()
        assert len(readings) >= 1
        
        reading = readings[0]
        assert reading["sensor_id"] == "PERSISTENCE_TEST"
        assert reading["value"] == 22.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 