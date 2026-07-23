# E2E Testing Setup Guide

## 🎯 **What We've Created**

A comprehensive End-to-End testing framework for the Carbon Credit Verification application using Playwright and pytest.

## 📁 **File Structure Created**

```
tests/e2e/
├── conftest.py              # Pytest fixtures and configuration
├── test_authentication.py   # Authentication flow tests
├── test_dashboard.py        # Dashboard and UI tests
├── pytest.ini              # Test configuration
├── run_tests.sh            # Test runner script (executable)
├── demo_test.py            # Simple demo test
├── README.md               # Comprehensive documentation
└── SETUP.md                # This setup guide
```

## 🔧 **Installation Steps**

### **1. Install Dependencies**
```bash
# From project root
source .venv/bin/activate
pip install playwright pytest-playwright pytest-xdist pytest-html

# Install browsers
playwright install
```

### **2. Verify Installation**
```bash
# Test basic functionality
cd tests/e2e
python -c "import playwright; import pytest; print('✅ Ready for E2E testing!')"
```

### **3. Run Demo Test**
```bash
# Run simple demo test
cd tests/e2e
python -m pytest demo_test.py -v
```

## 🚀 **Running E2E Tests**

### **Option 1: Using the Test Runner Script**
```bash
cd tests/e2e
./run_tests.sh --help                    # Show all options
./run_tests.sh                          # Run all tests
./run_tests.sh --headed                  # Run with visible browser
./run_tests.sh --auth                    # Run only auth tests
./run_tests.sh --browser firefox         # Use Firefox
```

### **Option 2: Direct pytest Commands**
```bash
cd tests/e2e

# Run all tests
pytest

# Run specific test file
pytest test_authentication.py

# Run with browser visible
pytest --headed

# Run with video recording
pytest --video on

# Run in parallel
pytest -n auto
```

## 🎭 **Test Categories**

### **Authentication Tests** (`test_authentication.py`)
- ✅ User registration flow
- ✅ Login/logout functionality
- ✅ Form validation
- ✅ Error handling
- ✅ API integration
- ✅ Loading states
- ✅ Accessibility

### **Dashboard Tests** (`test_dashboard.py`)
- ✅ Dashboard loading
- ✅ Project management
- ✅ Navigation
- ✅ Responsive design
- ✅ Performance
- ✅ UI components

## 🔧 **Configuration**

### **Browser Options**
- **Chromium** (default)
- **Firefox**
- **WebKit**

### **Test Modes**
- **Headless** (default) - Faster, CI-friendly
- **Headed** - Visual debugging
- **Slow motion** - For debugging

### **Reporting**
- **Screenshots** on failure
- **Video recording** on failure
- **HTML reports**
- **JUnit XML** for CI

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Module not found errors**
   ```bash
   # Ensure virtual environment is activated
   source ../../.venv/bin/activate
   pip install playwright pytest-playwright
   ```

2. **Browser not found**
   ```bash
   # Install browsers
   playwright install
   ```

3. **Server connection issues**
   ```bash
   # Start servers manually
   cd ../../backend && python demo_main.py &
   cd ../../frontend && npm start &
   ```

4. **Permission denied on run_tests.sh**
   ```bash
   chmod +x run_tests.sh
   ```

## 🎯 **Test Features**

### **Automatic Server Management**
- Starts backend and frontend servers
- Waits for servers to be ready
- Cleans up after tests

### **Test Isolation**
- Each test runs in clean state
- Unique test data generation
- No test interference

### **Comprehensive Coverage**
- **Functional testing** - User flows
- **Integration testing** - API communication
- **UI testing** - Visual elements
- **Accessibility testing** - ARIA, keyboard nav
- **Performance testing** - Load times

### **Advanced Features**
- **Cross-browser testing**
- **Responsive design testing**
- **Error boundary testing**
- **Network monitoring**
- **Visual regression** (ready to add)

## 📊 **Example Test Run**

```bash
cd tests/e2e
./run_tests.sh --headed --auth

# Output:
[INFO] Starting E2E tests...
[INFO] Browser: chromium
[INFO] Headed: true
[INFO] Video: retain-on-failure
[INFO] Screenshot: only-on-failure

🚀 Running E2E tests...

test_authentication.py::TestAuthentication::test_user_registration_success PASSED
test_authentication.py::TestAuthentication::test_user_login_success PASSED
test_authentication.py::TestAuthentication::test_login_form_validation PASSED

✅ All tests passed!
```

## 🔄 **CI/CD Integration**

### **GitHub Actions Example**
```yaml
- name: Run E2E Tests
  run: |
    source .venv/bin/activate
    playwright install --with-deps
    cd tests/e2e
    pytest --browser chromium --video retain-on-failure
```

## 🚀 **Next Steps**

1. **Fix virtual environment** - Ensure consistent Python version
2. **Run first test** - Start with demo_test.py
3. **Add custom tests** - Extend for your specific needs
4. **Set up CI** - Integrate with your deployment pipeline
5. **Add visual regression** - Screenshot comparison tests

## 📚 **Resources**

- [Playwright Documentation](https://playwright.dev/python/)
- [pytest Documentation](https://docs.pytest.org/)
- [E2E Testing Best Practices](https://playwright.dev/python/best-practices)

## ✅ **What's Ready**

- ✅ Complete test framework structure
- ✅ Authentication test suite
- ✅ Dashboard test suite
- ✅ Test runner script
- ✅ Configuration files
- ✅ Documentation
- ✅ CI/CD examples

The E2E testing framework follows common Playwright/pytest best practices. 