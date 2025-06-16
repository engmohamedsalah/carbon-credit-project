import pytest
import pytest_asyncio
import asyncio
import subprocess
import time
import requests
import os
from playwright.async_api import async_playwright


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def servers():
    """Start backend and frontend servers for testing."""
    backend_process = None
    frontend_process = None
    
    # Get the project root directory (two levels up from tests/e2e)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    backend_dir = os.path.join(project_root, "backend")
    frontend_dir = os.path.join(project_root, "frontend")
    
    try:
        # Start backend server
        backend_process = subprocess.Popen(
            ["python", "demo_main.py"],
            cwd=backend_dir,
            env={"PYTHONPATH": ".", **dict(subprocess.os.environ)}
        )
        
        # Wait for backend to be ready
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        else:
            raise Exception("Backend server failed to start")
        
        # Start frontend server
        frontend_process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            env={"BROWSER": "none", **dict(subprocess.os.environ)}
        )
        
        # Wait for frontend to be ready
        for _ in range(60):  # Wait up to 60 seconds for React to compile
            try:
                response = requests.get("http://localhost:3000", timeout=1)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        else:
            raise Exception("Frontend server failed to start")
        
        print("✅ Both servers are running and ready for E2E tests")
        yield
        
    finally:
        # Cleanup
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()


@pytest_asyncio.fixture
async def browser():
    """Create a browser instance for testing."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest_asyncio.fixture
async def page(browser):
    """Create a new page for each test."""
    page = await browser.new_page()
    yield page
    await page.close()


@pytest_asyncio.fixture
async def authenticated_page(page):
    """Create a page with an authenticated user."""
    # Navigate to registration page
    await page.goto("http://localhost:3000/register")
    
    # Fill registration form
    test_email = f"test_{int(time.time())}@example.com"
    await page.fill('input[name="fullName"]', "Test User")
    await page.fill('input[name="email"]', test_email)
    await page.fill('input[name="password"]', "testpass123")
    await page.fill('input[name="confirmPassword"]', "testpass123")
    
    # Submit registration
    await page.click('button[type="submit"]')
    
    # Wait for redirect to login page
    await page.wait_for_url("**/login")
    
    # Login with the created user
    await page.fill('input[name="email"]', test_email)
    await page.fill('input[name="password"]', "testpass123")
    await page.click('button[type="submit"]')
    
    # Wait for redirect to dashboard
    await page.wait_for_url("**/dashboard")
    
    yield page


# Test data fixtures
@pytest.fixture
def test_user_data():
    """Provide test user data."""
    return {
        "email": f"test_{int(time.time())}@example.com",
        "password": "testpass123",
        "fullName": "Test User",
        "role": "Project Developer"
    }


@pytest.fixture
def invalid_user_data():
    """Provide invalid user data for negative testing."""
    return {
        "email": "invalid-email",
        "password": "123",  # Too short
        "fullName": "",
        "role": "Project Developer"
    } 