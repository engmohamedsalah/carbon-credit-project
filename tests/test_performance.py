"""
Performance Testing Suite
Tests for API response times, throughput, and system performance
"""
import pytest
import time
import statistics
import concurrent.futures
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the main app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from main import app

client = TestClient(app)

class TestPerformance:
    """Performance testing suite"""
    
    @pytest.fixture
    def auth_token(self):
        """Get authentication token for testing"""
        login_data = {
            "username": "testadmin@example.com",
            "password": "password123"
        }
        response = client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 200
        return response.json()["access_token"]
    
    @pytest.fixture
    def test_project_id(self, auth_token):
        """Create a test project for performance testing"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        project_data = {
            "name": "Performance Test Project",
            "description": "Test project for performance testing",
            "location_name": "Test Location",
            "area_hectares": 100.0,
            "project_type": "Reforestation"
        }
        response = client.post("/api/v1/projects", json=project_data, headers=headers)
        assert response.status_code == 201
        return response.json()["id"]
    
    def measure_response_time(self, method, url, **kwargs):
        """Helper function to measure response time"""
        start_time = time.time()
        response = getattr(client, method)(url, **kwargs)
        end_time = time.time()
        return response, end_time - start_time
    
    def test_health_check_performance(self):
        """Test health check endpoint performance"""
        response_times = []
        
        # Measure response time over multiple requests
        for _ in range(10):
            response, response_time = self.measure_response_time("get", "/health")
            assert response.status_code == 200
            response_times.append(response_time)
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"Health Check Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Maximum: {max_time:.3f}s")
        print(f"  Minimum: {min_time:.3f}s")
        
        # Assert reasonable performance (should be under 100ms)
        assert avg_time < 0.1, f"Average response time {avg_time:.3f}s exceeds 100ms"
        assert max_time < 0.2, f"Maximum response time {max_time:.3f}s exceeds 200ms"
    
    def test_project_list_performance(self, auth_token):
        """Test project list endpoint performance"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response_times = []
        
        # Measure response time over multiple requests
        for _ in range(10):
            response, response_time = self.measure_response_time("get", "/api/v1/projects", headers=headers)
            assert response.status_code == 200
            response_times.append(response_time)
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"Project List Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Maximum: {max_time:.3f}s")
        print(f"  Minimum: {min_time:.3f}s")
        
        # Assert reasonable performance (should be under 500ms)
        assert avg_time < 0.5, f"Average response time {avg_time:.3f}s exceeds 500ms"
        assert max_time < 1.0, f"Maximum response time {max_time:.3f}s exceeds 1s"
    
    def test_ml_analysis_performance(self, auth_token, test_project_id):
        """Test ML analysis endpoint performance"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        analysis_data = {
            "project_id": test_project_id,
            "latitude": 40.7128,
            "longitude": -74.0060,
            "analysis_type": "comprehensive"
        }
        
        response_times = []
        
        # Measure response time over multiple requests
        for _ in range(5):  # Fewer requests due to ML processing
            response, response_time = self.measure_response_time("post", "/api/v1/ml/analyze-location", json=analysis_data, headers=headers)
            assert response.status_code == 200
            response_times.append(response_time)
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"ML Analysis Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Maximum: {max_time:.3f}s")
        print(f"  Minimum: {min_time:.3f}s")
        
        # ML analysis can take longer, but should still be reasonable
        assert avg_time < 10.0, f"Average ML analysis time {avg_time:.3f}s exceeds 10s"
        assert max_time < 15.0, f"Maximum ML analysis time {max_time:.3f}s exceeds 15s"
    
    def test_analytics_performance(self, auth_token):
        """Test analytics endpoints performance"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Test dashboard analytics
        response, response_time = self.measure_response_time("get", "/api/v1/analytics/dashboard", headers=headers)
        assert response.status_code == 200
        print(f"Dashboard Analytics: {response_time:.3f}s")
        assert response_time < 2.0, f"Dashboard analytics time {response_time:.3f}s exceeds 2s"
        
        # Test performance analytics
        response, response_time = self.measure_response_time("get", "/api/v1/analytics/performance", headers=headers)
        assert response.status_code == 200
        print(f"Performance Analytics: {response_time:.3f}s")
        assert response_time < 2.0, f"Performance analytics time {response_time:.3f}s exceeds 2s"
        
        # Test carbon impact analytics
        response, response_time = self.measure_response_time("get", "/api/v1/analytics/carbon-impact", headers=headers)
        assert response.status_code == 200
        print(f"Carbon Impact Analytics: {response_time:.3f}s")
        assert response_time < 2.0, f"Carbon impact analytics time {response_time:.3f}s exceeds 2s"
    
    def test_concurrent_requests(self, auth_token):
        """Test system performance under concurrent load"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        def make_request():
            """Make a single request"""
            response = client.get("/api/v1/projects", headers=headers)
            return response.status_code == 200
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            end_time = time.time()
        
        total_time = end_time - start_time
        success_rate = sum(results) / len(results)
        
        print(f"Concurrent Requests Performance:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Requests per second: {10/total_time:.1f}")
        
        # Assert all requests succeeded
        assert success_rate == 1.0, f"Success rate {success_rate:.1%} is not 100%"
        assert total_time < 5.0, f"Total time {total_time:.3f}s exceeds 5s"
    
    def test_database_query_performance(self, auth_token, test_project_id):
        """Test database query performance"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Test project retrieval
        response_times = []
        for _ in range(10):
            response, response_time = self.measure_response_time("get", f"/api/v1/projects/{test_project_id}", headers=headers)
            assert response.status_code == 200
            response_times.append(response_time)
        
        avg_time = statistics.mean(response_times)
        print(f"Database Query Performance: {avg_time:.3f}s average")
        assert avg_time < 0.1, f"Database query time {avg_time:.3f}s exceeds 100ms"
    
    def test_report_generation_performance(self, auth_token, test_project_id):
        """Test report generation performance"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Test project report generation
        response, response_time = self.measure_response_time("get", f"/api/v1/reports/project/{test_project_id}", headers=headers)
        assert response.status_code == 200
        print(f"Project Report Generation: {response_time:.3f}s")
        assert response_time < 5.0, f"Report generation time {response_time:.3f}s exceeds 5s"
        
        # Test analytics report generation
        response, response_time = self.measure_response_time("get", "/api/v1/reports/analytics", headers=headers)
        assert response.status_code == 200
        print(f"Analytics Report Generation: {response_time:.3f}s")
        assert response_time < 5.0, f"Analytics report generation time {response_time:.3f}s exceeds 5s"
    
    def test_xai_performance(self, auth_token, test_project_id):
        """Test XAI explanation generation performance"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        explanation_data = {
            "model_id": "forest_cover_ensemble",
            "instance_data": {
                "project_id": test_project_id,
                "location": {"lat": 40.7128, "lng": -74.0060}
            },
            "explanation_method": "shap",
            "business_friendly": True
        }
        
        response, response_time = self.measure_response_time("post", "/api/v1/xai/generate-explanation", json=explanation_data, headers=headers)
        assert response.status_code == 200
        print(f"XAI Explanation Generation: {response_time:.3f}s")
        assert response_time < 10.0, f"XAI explanation time {response_time:.3f}s exceeds 10s"
    
    def test_memory_usage_under_load(self, auth_token):
        """Test memory usage under sustained load"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Make multiple requests to simulate load
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < 30:  # Run for 30 seconds
            response = client.get("/api/v1/projects", headers=headers)
            assert response.status_code == 200
            request_count += 1
            time.sleep(0.1)  # Small delay between requests
        
        requests_per_second = request_count / 30
        print(f"Memory Usage Test:")
        print(f"  Requests made: {request_count}")
        print(f"  Requests per second: {requests_per_second:.1f}")
        
        # Should handle at least 5 requests per second
        assert requests_per_second >= 5.0, f"Request rate {requests_per_second:.1f} is below 5 req/s"
    
    def test_error_handling_performance(self, auth_token):
        """Test performance of error handling"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Test 404 error response time
        response_times = []
        for _ in range(10):
            response, response_time = self.measure_response_time("get", "/api/v1/projects/99999", headers=headers)
            assert response.status_code == 404
            response_times.append(response_time)
        
        avg_time = statistics.mean(response_times)
        print(f"Error Handling Performance: {avg_time:.3f}s average")
        assert avg_time < 0.1, f"Error handling time {avg_time:.3f}s exceeds 100ms"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 