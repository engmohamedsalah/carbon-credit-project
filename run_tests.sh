#!/bin/bash

# Script to run all tests for the Carbon Credit Verification SaaS application

echo "Running tests for Carbon Credit Verification SaaS application..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 before continuing."
    exit 1
fi

# Initialize the SQLite database
echo "Initializing test database..."
cd backend
python3 init_db.py
cd ..

# Run backend tests
echo ""
echo "========================================"
echo "Running backend tests..."
echo "========================================"
python3 test_backend.py

BACKEND_RESULT=$?

# Run frontend tests if backend tests pass
if [ $BACKEND_RESULT -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Running frontend tests..."
    echo "========================================"
    python3 test_frontend.py
    FRONTEND_RESULT=$?
else
    echo "Backend tests failed. Skipping frontend tests."
    FRONTEND_RESULT=1
fi

# Run validation if all tests pass
if [ $BACKEND_RESULT -eq 0 ] && [ $FRONTEND_RESULT -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Validating implementation..."
    echo "========================================"
    python3 validate_implementation.py
    VALIDATION_RESULT=$?
else
    echo "Tests failed. Skipping validation."
    VALIDATION_RESULT=1
fi

# Report results
echo ""
echo "========================================"
echo "Test Results:"
echo "========================================"
echo "Backend Tests: $([ $BACKEND_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "Frontend Tests: $([ $FRONTEND_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "Implementation Validation: $([ $VALIDATION_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo ""

# Exit with error if any tests failed
if [ $BACKEND_RESULT -eq 0 ] && [ $FRONTEND_RESULT -eq 0 ] && [ $VALIDATION_RESULT -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed. Please check the output above for details."
    exit 1
fi 