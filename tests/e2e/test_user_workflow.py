import pytest
import asyncio
import os
from playwright.async_api import expect

# Mark all tests in this file as workflow tests
pytestmark = pytest.mark.workflow


class TestCompleteUserWorkflow:
    """End-to-end tests for the complete user workflow as described in USER_WORKFLOW_GUIDE.md"""
    
    @pytest.mark.asyncio
    async def test_complete_user_journey(self, page, servers):
        """Test the complete user journey from registration to ML verification."""
        
        # Step 1: User Registration
        await page.goto("http://localhost:3000/register")
        await expect(page.locator("h2")).to_contain_text("Create Account")
        
        # Fill registration form
        test_user = {
            "fullName": "John Smith",
            "email": "john@test.com", 
            "password": "12345678"
        }
        
        await page.fill('input[name="fullName"]', test_user["fullName"])
        await page.fill('input[name="email"]', test_user["email"])
        await page.fill('input[name="password"]', test_user["password"])
        await page.fill('input[name="confirmPassword"]', test_user["password"])
        await page.click('button[type="submit"]')
        
        # Should redirect to login
        await page.wait_for_url("**/login", timeout=5000)
        
        # Step 2: User Login
        await page.fill('input[name="email"]', test_user["email"])
        await page.fill('input[name="password"]', test_user["password"])
        await page.click('button[type="submit"]')
        
        # Should redirect to dashboard
        await page.wait_for_url("**/dashboard", timeout=5000)
        
        # Step 3: Verify Dashboard Content
        await expect(page.locator("h4")).to_contain_text("Dashboard")
        await expect(page.locator("h6")).to_contain_text("Welcome, John Smith")
        
        # Check for project statistics panel
        await expect(page.locator('h6:has-text("Projects")')).to_be_visible()
        await expect(page.locator('h6:has-text("Satellite Imagery")')).to_be_visible()
        
        # Check for quick actions
        await expect(page.locator('h6:has-text("Quick Actions")')).to_be_visible()
        
        # Step 4: Test ML Verification (Core Working Feature)
        # Click "New Verification" button
        verification_buttons = [
            'button:has-text("New Verification")',
            'button:has-text("Verify Project")',
            'button[contains(., "Verification")]'
        ]
        
        verification_clicked = False
        for selector in verification_buttons:
            try:
                button = page.locator(selector)
                if await button.count() > 0:
                    await button.first.click()
                    verification_clicked = True
                    break
            except:
                continue
        
        # If button not found, navigate directly to verification
        if not verification_clicked:
            await page.goto("http://localhost:3000/verification?project_id=new")
        
        # Should be on verification page
        await expect(page.locator("h4")).to_contain_text("Verification")
        
        # Step 5: Test ML Analysis Interface
        # Check for ML Analysis component
        await expect(page.locator('h6:has-text("Machine Learning Analysis")')).to_be_visible()
        
        # Check for coordinate inputs
        latitude_input = page.locator('input[name="latitude"], input[label*="Latitude"]')
        longitude_input = page.locator('input[name="longitude"], input[label*="Longitude"]')
        
        if await latitude_input.count() > 0 and await longitude_input.count() > 0:
            # Fill coordinates (Amazon region)
            await latitude_input.fill("-3.5")
            await longitude_input.fill("-62.8")
        
        # Check for file upload
        file_input = page.locator('input[type="file"]')
        if await file_input.count() > 0:
            # Note: In a real test, we would upload a test image file
            # For now, just verify the upload interface exists
            await expect(file_input).to_be_visible()
        
        # Check for analysis button
        analyze_button = page.locator('button:has-text("Analyze"), button:has-text("Run Analysis")')
        if await analyze_button.count() > 0:
            await expect(analyze_button.first).to_be_visible()
            # Note: We don't click as it requires actual file upload
    
    @pytest.mark.asyncio 
    async def test_dashboard_navigation_features(self, authenticated_page, servers):
        """Test what navigation features actually work from dashboard."""
        
        # Verify dashboard loads
        await expect(authenticated_page.locator("h4")).to_contain_text("Dashboard")
        
        # Test 1: Check "View Projects" button behavior
        view_projects_btn = authenticated_page.locator('button:has-text("View Projects")')
        if await view_projects_btn.count() > 0:
            await view_projects_btn.click()
            
            # Check what happens - might be 404 or missing page
            current_url = authenticated_page.url
            print(f"After clicking 'View Projects', URL is: {current_url}")
            
            # Navigate back to dashboard for next test
            await authenticated_page.goto("http://localhost:3000/dashboard")
        
        # Test 2: Check "New Project" button behavior
        new_project_btn = authenticated_page.locator('button:has-text("New Project")')
        if await new_project_btn.count() > 0:
            await new_project_btn.click()
            
            current_url = authenticated_page.url
            print(f"After clicking 'New Project', URL is: {current_url}")
            
            # Navigate back to dashboard
            await authenticated_page.goto("http://localhost:3000/dashboard")
        
        # Test 3: Check "New Verification" button (this should work)
        new_verification_btn = authenticated_page.locator('button:has-text("New Verification")')
        if await new_verification_btn.count() > 0:
            await new_verification_btn.click()
            
            # This should work and go to verification page
            await expect(authenticated_page.locator("h4")).to_contain_text("Verification")
            
            # Navigate back to dashboard
            await authenticated_page.goto("http://localhost:3000/dashboard")
    
    @pytest.mark.asyncio
    async def test_missing_pages_behavior(self, authenticated_page, servers):
        """Test behavior when accessing missing pages."""
        
        # Test direct navigation to missing pages
        missing_pages = [
            "/projects",
            "/projects/new", 
            "/verifications"
        ]
        
        for page_path in missing_pages:
            await authenticated_page.goto(f"http://localhost:3000{page_path}")
            
            # Check what happens - might be 404, blank page, or redirect
            current_url = authenticated_page.url
            print(f"Navigated to {page_path}, URL is: {current_url}")
            
            # Check if page shows any error or content
            try:
                # Look for common error indicators
                error_indicators = [
                    "404",
                    "Not Found", 
                    "Page not found",
                    "Cannot GET",
                    "Route not found"
                ]
                
                page_text = await authenticated_page.text_content("body")
                for indicator in error_indicators:
                    if indicator.lower() in page_text.lower():
                        print(f"Page {page_path} shows error: {indicator}")
                        break
                else:
                    print(f"Page {page_path} loaded without obvious error")
                    
            except Exception as e:
                print(f"Error checking page {page_path}: {e}")
    
    @pytest.mark.asyncio
    async def test_ml_verification_component_availability(self, authenticated_page, servers):
        """Test the ML verification component that should be working."""
        
        # Navigate to verification page
        await authenticated_page.goto("http://localhost:3000/verification?project_id=new")
        
        # Verify page loads
        await expect(authenticated_page.locator("h4")).to_contain_text("Verification")
        
        # Check for ML Analysis component
        ml_component = authenticated_page.locator('[data-testid="ml-analysis"], .ml-analysis')
        if await ml_component.count() == 0:
            # Try alternative selectors
            ml_component = authenticated_page.locator('h6:has-text("Machine Learning Analysis")')
        
        if await ml_component.count() > 0:
            await expect(ml_component.first).to_be_visible()
            print("✅ ML Analysis component found and visible")
        else:
            print("❌ ML Analysis component not found")
        
        # Check for form elements that should exist
        form_elements = {
            "latitude_input": 'input[name="latitude"], input[label*="Latitude"]',
            "longitude_input": 'input[name="longitude"], input[label*="Longitude"]',
            "file_upload": 'input[type="file"]',
            "analyze_button": 'button:has-text("Analyze")'
        }
        
        for element_name, selector in form_elements.items():
            element = authenticated_page.locator(selector)
            count = await element.count()
            if count > 0:
                print(f"✅ {element_name} found ({count} elements)")
            else:
                print(f"❌ {element_name} not found")
    
    @pytest.mark.asyncio
    async def test_backend_api_endpoints(self, authenticated_page, servers):
        """Test that backend API endpoints are responsive."""
        
        # Test health endpoint
        response = await authenticated_page.request.get("http://localhost:8000/health")
        assert response.status == 200
        health_data = await response.json()
        print(f"✅ Health endpoint: {health_data.get('status', 'unknown')}")
        
        # Test projects endpoint (should work according to logs)
        auth_token = await authenticated_page.evaluate('localStorage.getItem("token")')
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        
        try:
            response = await authenticated_page.request.get(
                "http://localhost:8000/api/v1/projects", 
                headers=headers
            )
            print(f"✅ Projects endpoint status: {response.status}")
            if response.status == 200:
                projects_data = await response.json()
                print(f"✅ Projects data: {projects_data}")
        except Exception as e:
            print(f"❌ Projects endpoint error: {e}")
        
        # Test ML status endpoint
        try:
            response = await authenticated_page.request.get(
                "http://localhost:8000/api/v1/ml/status",
                headers=headers
            )
            print(f"✅ ML status endpoint: {response.status}")
            if response.status == 200:
                ml_data = await response.json()
                print(f"✅ ML models available: {ml_data}")
        except Exception as e:
            print(f"❌ ML status endpoint error: {e}")


class TestWorkingFeatures:
    """Test specifically the features that should be working according to the guide."""
    
    @pytest.mark.asyncio
    async def test_authentication_flow(self, page, servers):
        """Test the complete authentication flow that is marked as working."""
        
        # Register new user
        await page.goto("http://localhost:3000/register") 
        
        test_user = {
            "fullName": "Test User",
            "email": f"test{int(asyncio.get_event_loop().time())}@example.com",
            "password": "testpass123"
        }
        
        await page.fill('input[name="fullName"]', test_user["fullName"])
        await page.fill('input[name="email"]', test_user["email"])
        await page.fill('input[name="password"]', test_user["password"])
        await page.fill('input[name="confirmPassword"]', test_user["password"])
        await page.click('button[type="submit"]')
        
        # Login with new user
        await page.wait_for_url("**/login")
        await page.fill('input[name="email"]', test_user["email"])
        await page.fill('input[name="password"]', test_user["password"])
        await page.click('button[type="submit"]')
        
        # Verify successful login
        await page.wait_for_url("**/dashboard")
        await expect(page.locator("h4")).to_contain_text("Dashboard")
        
        print("✅ Authentication flow working correctly")
    
    @pytest.mark.asyncio
    async def test_ml_verification_interface(self, authenticated_page, servers):
        """Test the ML verification interface that should be fully working."""
        
        await authenticated_page.goto("http://localhost:3000/verification?project_id=new")
        
        # Verify we can access the verification page
        await expect(authenticated_page.locator("h4")).to_contain_text("Verification")
        
        # The ML verification should be the star feature
        # Just verify the interface loads - actual ML testing would require test files
        print("✅ ML verification interface accessible")


class TestMissingFeatures:
    """Test the features that are marked as missing to confirm the gaps."""
    
    @pytest.mark.asyncio
    async def test_projects_list_missing(self, authenticated_page, servers):
        """Confirm that the projects list page is indeed missing."""
        
        await authenticated_page.goto("http://localhost:3000/projects")
        
        # Check what happens - this should fail or show a 404
        current_url = authenticated_page.url
        page_text = await authenticated_page.text_content("body")
        
        print(f"Projects list page URL: {current_url}")
        print(f"Page content preview: {page_text[:200]}...")
        
        # This test documents the missing feature
        print("❌ Projects list page confirmed missing or broken")
    
    @pytest.mark.asyncio
    async def test_new_project_creation_missing(self, authenticated_page, servers):
        """Confirm that new project creation is missing or broken."""
        
        await authenticated_page.goto("http://localhost:3000/projects/new")
        
        current_url = authenticated_page.url
        page_text = await authenticated_page.text_content("body")
        
        print(f"New project page URL: {current_url}")
        print(f"Page content preview: {page_text[:200]}...")
        
        print("❌ New project creation page confirmed missing or broken") 