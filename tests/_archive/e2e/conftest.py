import pytest
import pytest_asyncio
import asyncio
import subprocess
import time
import requests
import os
import signal
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

    def kill_port(port: int):
        try:
            p = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            if p.stdout:
                pids = [int(pid) for pid in p.stdout.strip().splitlines() if pid.strip().isdigit()]
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            # Best effort cleanup only
            pass
    
    # Get the project root directory (two levels up from tests/e2e)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    backend_dir = os.path.join(project_root, "backend")
    frontend_dir = os.path.join(project_root, "frontend")
    
    try:
        # Ensure clean ports before start
        kill_port(8000)
        kill_port(3000)
        time.sleep(1)

        # Start backend server
        backend_process = subprocess.Popen(
            ["python", "main.py"],
            cwd=backend_dir,
            env={"PYTHONPATH": ".", **dict(subprocess.os.environ)}
        )
        
        # Wait for backend to be ready
        for _ in range(45):  # Wait up to 45 seconds
            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        else:
            print("⚠️ Backend server not started, continuing with frontend-only tests")
        
        # Start frontend server
        frontend_process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            env={"BROWSER": "none", **dict(subprocess.os.environ)}
        )
        
        # Wait for frontend to be ready
        for _ in range(90):  # Wait up to 90 seconds for React to compile
            try:
                response = requests.get("http://localhost:3000", timeout=1)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        else:
            print("⚠️ Frontend server not started, continuing with basic tests")
        
        print("✅ E2E test environment ready")
        yield
        
    finally:
        # Cleanup
        try:
            if backend_process:
                backend_process.terminate()
                backend_process.wait(timeout=10)
        except Exception:
            pass
        try:
            if frontend_process:
                frontend_process.terminate()
                frontend_process.wait(timeout=10)
        except Exception:
            pass
        # Final port cleanup
        kill_port(8000)
        kill_port(3000)


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