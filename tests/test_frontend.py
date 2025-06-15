from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import sys

def test_frontend():
    """Test the frontend application using Selenium"""
    print("Starting frontend tests...")
    
    # Setup Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        # Initialize the driver
        driver = webdriver.Chrome(options=options)
        
        # Test login page
        print("Testing login page...")
        driver.get("http://localhost:3000/login")
        assert "Carbon Credit Verification" in driver.title
        
        # Find login form elements
        email_input = driver.find_element(By.ID, "email")
        password_input = driver.find_element(By.ID, "password")
        login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Sign In')]")
        
        # Enter credentials and login
        email_input.send_keys("test@example.com")
        password_input.send_keys("password123")
        login_button.click()
        
        # Wait for dashboard to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'Dashboard')]"))
        )
        
        print("✅ Login test passed")
        
        # Test navigation to projects page
        print("Testing navigation to projects page...")
        projects_link = driver.find_element(By.XPATH, "//span[contains(text(), 'Projects')]")
        projects_link.click()
        
        # Wait for projects page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'Projects')]"))
        )
        
        print("✅ Navigation test passed")
        
        # Test creating a new project
        print("Testing project creation...")
        new_project_button = driver.find_element(By.XPATH, "//button[contains(text(), 'New Project')]")
        new_project_button.click()
        
        # Wait for new project form to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'Create New Project')]"))
        )
        
        # Fill out project form
        driver.find_element(By.NAME, "name").send_keys("Selenium Test Project")
        driver.find_element(By.NAME, "location_name").send_keys("Test Location")
        driver.find_element(By.NAME, "description").send_keys("A project created by Selenium tests")
        
        # Submit form
        submit_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Create Project')]")
        submit_button.click()
        
        # Wait for project detail page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'Selenium Test Project')]"))
        )
        
        print("✅ Project creation test passed")
        
        # Test logout
        print("Testing logout...")
        profile_button = driver.find_element(By.XPATH, "//button[@aria-label='account of current user']")
        profile_button.click()
        
        # Wait for menu to appear and click logout
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//li[contains(text(), 'Logout')]"))
        ).click()
        
        # Wait for login page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h5[contains(text(), 'Sign In')]"))
        )
        
        print("✅ Logout test passed")
        
        print("\n🎉 All frontend tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Frontend test failed: {e}")
    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    test_frontend()
