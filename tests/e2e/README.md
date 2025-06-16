# End-to-End (E2E) Tests

This directory contains comprehensive end-to-end tests for the Carbon Credit Verification application using Playwright.

## 🎯 **Test Coverage**

### **Authentication Tests** (`test_authentication.py`)
- ✅ User registration flow
- ✅ Login/logout functionality  
- ✅ Form validation (client-side)
- ✅ Error handling and display
- ✅ Navigation between auth pages
- ✅ API integration testing
- ✅ Loading states
- ✅ Accessibility features

### **Dashboard Tests** (`test_dashboard.py`)
- ✅ Dashboard loading and display
- ✅ Project management functionality
- ✅ Navigation and routing
- ✅ Responsive design
- ✅ Performance testing
- ✅ UI components
- ✅ Accessibility compliance

## 🚀 **Running Tests**

### **Prerequisites**
```bash
# Install dependencies
source .venv/bin/activate
pip install playwright pytest-playwright

# Install browsers
playwright install
```

### **Run All E2E Tests**
```bash
# From project root
source .venv/bin/activate
cd tests/e2e
pytest
```

### **Run Specific Test Categories**
```bash
# Authentication tests only
pytest test_authentication.py

# Authentication failure tests only
pytest test_auth_failures.py

# Dashboard tests only  
pytest test_dashboard.py

# Run with specific markers
pytest -m auth
pytest -m ui
pytest -m accessibility
```

### **Run Tests with Different Options**
```bash
# Run in headed mode (see browser)
pytest --headed

# Run with specific browser
pytest --browser firefox
pytest --browser webkit

# Run with video recording
pytest --video on

# Run with screenshots
pytest --screenshot on

# Run specific test
pytest test_authentication.py::TestAuthentication::test_user_login_success
```

## 🔧 **Test Configuration**

### **Browser Configuration**
- **Default**: Chromium (headless)
- **Available**: Chromium, Firefox, WebKit
- **Viewport**: Responsive testing included

### **Test Data**
- **Dynamic**: Test users created with timestamps
- **Cleanup**: Automatic server startup/shutdown
- **Isolation**: Each test runs in clean state

### **Fixtures Available**
- `servers`: Starts backend + frontend servers
- `browser`: Browser instance
- `page`: Clean page for each test
- `authenticated_page`: Pre-authenticated user session
- `test_user_data`: Valid test user data
- `invalid_user_data`: Invalid data for negative testing

## 📊 **Test Structure**

```
tests/e2e/
├── conftest.py              # Pytest configuration & fixtures
├── test_authentication.py   # Auth flow tests
├── test_auth_failures.py    # Auth failure & security tests
├── test_dashboard.py        # Dashboard & UI tests
├── pytest.ini              # Test configuration
└── README.md               # This file
```

## 🎭 **Test Categories**

### **Functional Tests**
- User registration and login flows
- Form validation and error handling
- Navigation and routing
- CRUD operations

### **Integration Tests**
- Frontend-backend API communication
- Authentication token handling
- Data persistence
- Error propagation

### **UI/UX Tests**
- Responsive design across devices
- Loading states and feedback
- Error message display
- Navigation usability

### **Accessibility Tests**
- Keyboard navigation
- ARIA labels and roles
- Color contrast (basic)
- Screen reader compatibility

### **Security & Failure Tests**
- SQL injection prevention
- XSS attack protection
- Session fixation protection
- Timing attack resistance
- Brute force protection
- Input validation edge cases
- Network error handling
- Server error scenarios

### **Performance Tests**
- Page load times
- Network request optimization
- Resource loading

## 🐛 **Debugging Tests**

### **Run Tests in Debug Mode**
```bash
# Run with browser visible
pytest --headed --slowmo 1000

# Run single test with debugging
pytest test_authentication.py::TestAuthentication::test_user_login_success --headed --slowmo 2000

# Capture screenshots on failure
pytest --screenshot only-on-failure
```

### **Common Issues**

1. **Server Startup Timeout**
   ```bash
   # Increase timeout in conftest.py or start servers manually
   cd backend && python demo_main.py &
   cd frontend && npm start &
   pytest --no-cov
   ```

2. **Element Not Found**
   ```bash
   # Run with slowmo to see what's happening
   pytest --headed --slowmo 1000
   ```

3. **Network Issues**
   ```bash
   # Check if servers are running
   curl http://localhost:8000/health
   curl http://localhost:3000
   ```

## 📈 **Test Reports**

### **Generate HTML Report**
```bash
pytest --html=report.html --self-contained-html
```

### **Generate JUnit XML**
```bash
pytest --junitxml=results.xml
```

### **Coverage with E2E Tests**
```bash
# Note: E2E tests don't provide code coverage
# Use unit tests for coverage metrics
```

## 🔄 **Continuous Integration**

### **GitHub Actions Example**
```yaml
- name: Run E2E Tests
  run: |
    source .venv/bin/activate
    playwright install --with-deps
    cd tests/e2e
    pytest --browser chromium --video retain-on-failure
```

### **Test Parallelization**
```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto
```

## 🎯 **Best Practices**

1. **Test Independence**: Each test should be able to run independently
2. **Data Isolation**: Use unique test data (timestamps, UUIDs)
3. **Explicit Waits**: Use Playwright's built-in waiting mechanisms
4. **Page Object Pattern**: Consider implementing for complex pages
5. **Error Handling**: Test both happy path and error scenarios
6. **Accessibility**: Include accessibility checks in all UI tests

## 🚀 **Next Steps**

1. **Add Visual Regression Tests**: Compare screenshots
2. **API Testing**: Add dedicated API endpoint tests
3. **Performance Monitoring**: Add detailed performance metrics
4. **Cross-browser Testing**: Expand browser coverage
5. **Mobile Testing**: Add mobile-specific test scenarios

## 📚 **Resources**

- [Playwright Documentation](https://playwright.dev/python/)
- [pytest-playwright Plugin](https://github.com/microsoft/playwright-pytest)
- [Accessibility Testing Guide](https://playwright.dev/python/accessibility-testing)
- [Best Practices](https://playwright.dev/python/best-practices) 