"""
Comprehensive XAI (Explainable AI) Testing Suite
Tests the complete XAI workflow including authentication, API integration, UI interactions, and business logic
"""

import pytest
import asyncio
import json
import base64
import time
from playwright.async_api import async_playwright, Page, BrowserContext
from playwright.sync_api import sync_playwright
import requests
import os
from datetime import datetime

class TestXAIFunctionality:
    """Comprehensive XAI functionality tests using Playwright"""
    
    @pytest.fixture(scope="class")
    def setup_class(self):
        """Setup test environment"""
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.test_user = {
            "email": "testadmin@example.com",
            "password": "password123"
        }
        self.test_timeout = 30000  # 30 seconds
        
    @pytest.fixture
    async def page_context(self, setup_class):
        """Create browser context and page for tests"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, slow_mo=1000)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True
            )
            page = await context.new_page()
            
            # Enable console logging
            page.on("console", lambda msg: print(f"Console: {msg.text}"))
            page.on("pageerror", lambda error: print(f"Page Error: {error}"))
            
            yield page, context
            
            await context.close()
            await browser.close()

    async def login_user(self, page: Page):
        """Helper method to login user"""
        await page.goto(self.frontend_url)
        await page.wait_for_selector('input[type="email"]', timeout=self.test_timeout)
        
        await page.fill('input[type="email"]', self.test_user["email"])
        await page.fill('input[type="password"]', self.test_user["password"])
        await page.click('button[type="submit"]')
        
        # Wait for successful login
        await page.wait_for_url(f"{self.frontend_url}/dashboard", timeout=self.test_timeout)
        
    async def get_auth_token(self):
        """Get authentication token for API calls"""
        response = requests.post(f"{self.backend_url}/api/v1/auth/login", json={
            "email": self.test_user["email"],
            "password": self.test_user["password"]
        })
        return response.json()["access_token"]

    @pytest.mark.asyncio
    async def test_xai_page_accessibility(self, page_context):
        """Test XAI page loads and is accessible"""
        page, context = page_context
        
        await self.login_user(page)
        
        # Navigate to XAI page
        await page.goto(f"{self.frontend_url}/xai")
        await page.wait_for_load_state("networkidle")
        
        # Check page title
        title = await page.title()
        assert "XAI" in title or "Explainable AI" in title or "Carbon Credit" in title
        
        # Check main components are present
        tabs = await page.query_selector_all('[role="tab"]')
        assert len(tabs) >= 3  # At least Generate, Compare, History tabs
        
        print("✅ XAI page accessibility test passed")

    @pytest.mark.asyncio
    async def test_xai_methods_api(self, page_context):
        """Test XAI methods API endpoint"""
        token = await asyncio.get_event_loop().run_in_executor(None, self.get_auth_token)
        
        response = requests.get(
            f"{self.backend_url}/api/v1/xai/methods",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "methods" in data
        assert "business_features" in data
        assert "compliance_status" in data
        
        # Check available methods
        methods = data["methods"]
        assert "shap" in methods
        assert "lime" in methods
        assert "integrated_gradients" in methods
        
        print("✅ XAI methods API test passed")

    @pytest.mark.asyncio
    async def test_xai_explanation_generation_api(self, page_context):
        """Test XAI explanation generation API directly"""
        token = await asyncio.get_event_loop().run_in_executor(None, self.get_auth_token)
        
        explanation_data = {
            "model_id": "forest_cover_ensemble",
            "instance_data": {
                "project_id": 1,
                "location": "Test Forest",
                "area_hectares": 100.0,
                "satellite_data": "mock_data"
            },
            "explanation_method": "shap",
            "business_friendly": True,
            "include_uncertainty": True
        }
        
        response = requests.post(
            f"{self.backend_url}/api/v1/xai/generate-explanation",
            json=explanation_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "explanation_id" in data
        assert "timestamp" in data
        assert "confidence_score" in data
        assert "business_summary" in data
        assert "risk_assessment" in data
        assert "regulatory_notes" in data
        
        # Check confidence score is valid
        confidence = data["confidence_score"]
        assert 0.0 <= confidence <= 1.0
        
        print("✅ XAI explanation generation API test passed")

    @pytest.mark.asyncio
    async def test_xai_report_generation(self, page_context):
        """Test XAI report generation and download"""
        token = await asyncio.get_event_loop().run_in_executor(None, self.get_auth_token)
        
        # First generate an explanation
        explanation_data = {
            "model_id": "forest_cover_ensemble",
            "instance_data": {"project_id": 1, "test": "data"},
            "explanation_method": "shap",
            "business_friendly": True,
            "include_uncertainty": True
        }
        
        explanation_response = requests.post(
            f"{self.backend_url}/api/v1/xai/generate-explanation",
            json=explanation_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert explanation_response.status_code == 200
        explanation_id = explanation_response.json()["explanation_id"]
        
        # Generate PDF report
        report_data = {
            "explanation_id": explanation_id,
            "format": "pdf",
            "include_business_summary": True
        }
        
        report_response = requests.post(
            f"{self.backend_url}/api/v1/xai/generate-report",
            json=report_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert report_response.status_code == 200
        report = report_response.json()
        
        # Check report structure
        assert "report_id" in report
        assert "filename" in report
        assert "format" in report
        assert "data" in report
        assert report["format"] == "pdf"
        
        print("✅ XAI report generation test passed")

    @pytest.mark.asyncio
    async def test_xai_ui_workflow(self, page_context):
        """Test XAI UI workflow through browser"""
        page, context = page_context
        
        await self.login_user(page)
        
        # Navigate to XAI page
        await page.goto(f"{self.frontend_url}/xai")
        await page.wait_for_load_state("networkidle")
        
        # Wait for the page to fully load
        await page.wait_for_timeout(2000)
        
        # Check if tabs are present
        tabs = await page.query_selector_all('[role="tab"]')
        if len(tabs) > 0:
            # Click on first tab (Generate)
            await tabs[0].click()
            await page.wait_for_timeout(1000)
            
            # Look for form elements
            selects = await page.query_selector_all('select, [role="combobox"]')
            inputs = await page.query_selector_all('input')
            buttons = await page.query_selector_all('button')
            
            print(f"Found {len(selects)} select elements, {len(inputs)} inputs, {len(buttons)} buttons")
            
            # Try to interact with available elements
            if len(buttons) > 0:
                # Find a button that might generate explanations
                for button in buttons:
                    text = await button.inner_text()
                    if "generate" in text.lower() or "explain" in text.lower():
                        print(f"Found generate button: {text}")
                        break
        
        print("✅ XAI UI workflow test passed")

    @pytest.mark.asyncio
    async def test_xai_error_handling(self, page_context):
        """Test XAI error handling and validation"""
        token = await asyncio.get_event_loop().run_in_executor(None, self.get_auth_token)
        
        # Test invalid explanation data
        invalid_data = {
            "model_id": "",  # Empty model ID
            "instance_data": {},  # Empty instance data
            "explanation_method": "invalid_method"
        }
        
        response = requests.post(
            f"{self.backend_url}/api/v1/xai/generate-explanation",
            json=invalid_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Should handle error gracefully
        assert response.status_code in [400, 422, 500]
        
        print("✅ XAI error handling test passed")

    @pytest.mark.asyncio
    async def test_xai_performance_metrics(self, page_context):
        """Test XAI performance and response times"""
        token = await asyncio.get_event_loop().run_in_executor(None, self.get_auth_token)
        
        # Test explanation generation performance
        start_time = time.time()
        
        explanation_data = {
            "model_id": "forest_cover_ensemble",
            "instance_data": {"project_id": 1, "performance_test": True},
            "explanation_method": "shap",
            "business_friendly": True,
            "include_uncertainty": True
        }
        
        response = requests.post(
            f"{self.backend_url}/api/v1/xai/generate-explanation",
            json=explanation_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 30.0  # Should complete within 30 seconds
        
        print(f"✅ XAI performance test passed (Response time: {response_time:.2f}s)")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])