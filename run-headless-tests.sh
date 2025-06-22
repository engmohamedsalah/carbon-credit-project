#!/bin/bash

# Master Headless Test Runner for Carbon Credit Verification
# Runs ALL tests in background without opening browser windows

echo "🚀 Carbon Credit Verification - HEADLESS TEST SUITE"
echo "📍 No browser windows will open during testing"
echo "⏰ Started at: $(date)"
echo "============================================================"

# Set headless environment variables
export HEADLESS=true
export BROWSER=none
export CI=true

# Function to check if servers are running
check_servers() {
    echo "🔍 Checking if servers are running..."
    
    # Check backend
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend server is running on port 8000"
    else
        echo "❌ Backend server is not running. Please start it first:"
        echo "   cd backend && python main.py &"
        exit 1
    fi
    
    # Check frontend
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend server is running on port 3000"
    else
        echo "❌ Frontend server is not running. Please start it first:"
        echo "   cd frontend && npm start &"
        exit 1
    fi
    
    echo ""
}

# Run tests function
run_tests() {
    local test_type=$1
    local exit_code=0
    
    echo "============================================================"
    echo "🧪 Running $test_type Tests (HEADLESS)"
    echo "============================================================"
    
    case $test_type in
        "Backend")
            python tests/test_backend.py
            exit_code=$?
            ;;
        "E2E")
            cd tests/e2e && python run-headless-e2e.py
            exit_code=$?
            cd ../..
            ;;
        "Playwright")
            cd tests/playwright && node run-headless-tests.js
            exit_code=$?
            cd ../..
            ;;
        "Implementation")
            python tests/validate_implementation.py
            exit_code=$?
            ;;
    esac
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_type tests PASSED"
    else
        echo "❌ $test_type tests FAILED (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Main execution
main() {
    check_servers
    
    local overall_exit_code=0
    
    # Run Backend Tests
    run_tests "Backend"
    if [ $? -ne 0 ]; then overall_exit_code=1; fi
    
    # Run Implementation Validation
    run_tests "Implementation" 
    if [ $? -ne 0 ]; then overall_exit_code=1; fi
    
    # Run E2E Tests (headless)
    run_tests "E2E"
    if [ $? -ne 0 ]; then overall_exit_code=1; fi
    
    # Run Playwright Tests (headless)
    run_tests "Playwright"
    if [ $? -ne 0 ]; then overall_exit_code=1; fi
    
    echo ""
    echo "============================================================"
    echo "🏁 FINAL RESULTS"
    echo "============================================================"
    
    if [ $overall_exit_code -eq 0 ]; then
        echo "🎉 ALL TESTS PASSED - Application is production ready!"
    else
        echo "⚠️  Some tests failed - Check logs above for details"
    fi
    
    echo "⏰ Completed at: $(date)"
    echo "📍 All tests ran in headless mode - no browser windows opened"
    
    return $overall_exit_code
}

# Run main function
main "$@" 