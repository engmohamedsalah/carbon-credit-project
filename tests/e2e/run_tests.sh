#!/bin/bash

# E2E Test Runner Script for Carbon Credit Verification
# Usage: ./run_tests.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BROWSER="chromium"
HEADED=false
VIDEO="retain-on-failure"
SCREENSHOT="only-on-failure"
PARALLEL=false
SPECIFIC_TEST=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -b, --browser BROWSER   Browser to use (chromium, firefox, webkit)"
    echo "  -H, --headed            Run in headed mode (show browser)"
    echo "  -v, --video MODE        Video recording mode (on, off, retain-on-failure)"
    echo "  -s, --screenshot MODE   Screenshot mode (on, off, only-on-failure)"
    echo "  -p, --parallel          Run tests in parallel"
    echo "  -t, --test TEST         Run specific test file or test case"
    echo "  --auth                  Run only authentication tests"
    echo "  --auth-failures         Run only authentication failure tests"
    echo "  --dashboard             Run only dashboard tests"
    echo "  --ui                    Run only UI tests"
    echo "  --accessibility         Run only accessibility tests"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests"
    echo "  $0 --headed --browser firefox         # Run with Firefox in headed mode"
    echo "  $0 --auth                            # Run only auth tests"
    echo "  $0 --auth-failures                   # Run only auth failure tests"
    echo "  $0 -t test_authentication.py         # Run specific test file"
    echo "  $0 --parallel --video on             # Run in parallel with video"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--browser)
            BROWSER="$2"
            shift 2
            ;;
        -H|--headed)
            HEADED=true
            shift
            ;;
        -v|--video)
            VIDEO="$2"
            shift 2
            ;;
        -s|--screenshot)
            SCREENSHOT="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --auth)
            SPECIFIC_TEST="-m auth"
            shift
            ;;
        --auth-failures)
            SPECIFIC_TEST="-m auth_failures"
            shift
            ;;
        --dashboard)
            SPECIFIC_TEST="-m dashboard"
            shift
            ;;
        --ui)
            SPECIFIC_TEST="-m ui"
            shift
            ;;
        --accessibility)
            SPECIFIC_TEST="-m accessibility"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [[ ! -f "conftest.py" ]]; then
    print_error "Please run this script from the tests/e2e directory"
    exit 1
fi

# Check if we're in CI environment
if [[ "$CI" == "true" ]]; then
    print_status "Running in CI environment"
    HEADED=false
    VIDEO="retain-on-failure"
    SCREENSHOT="only-on-failure"
else
    # Check if virtual environment is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Virtual environment not detected. Attempting to activate..."
        if [[ -f "../../.venv/bin/activate" ]]; then
            source ../../.venv/bin/activate
            print_success "Virtual environment activated"
        else
            print_error "Virtual environment not found. Please activate it manually:"
            print_error "  source ../../.venv/bin/activate"
            exit 1
        fi
    fi
fi

# Check if required packages are installed
print_status "Checking dependencies..."
python -c "import playwright, pytest" 2>/dev/null || {
    print_error "Required packages not installed. Installing..."
    pip install playwright pytest-playwright
    playwright install
}

# Check if browsers are installed
print_status "Checking browser installation..."
playwright install --dry-run >/dev/null 2>&1 || {
    print_warning "Browsers not installed. Installing..."
    playwright install
}

# Build pytest command
PYTEST_CMD="pytest"

# Add browser option
PYTEST_CMD="$PYTEST_CMD --browser $BROWSER"

# Add headed option (force headless in CI)
if [[ "$CI" == "true" ]] || [[ "$HEADLESS" == "true" ]]; then
    # Force headless mode in CI or when explicitly requested
    PYTEST_CMD="$PYTEST_CMD --headed=false"
elif [[ "$HEADED" == "true" ]]; then
    PYTEST_CMD="$PYTEST_CMD --headed"
fi

# Add video option
PYTEST_CMD="$PYTEST_CMD --video $VIDEO"

# Add screenshot option
PYTEST_CMD="$PYTEST_CMD --screenshot $SCREENSHOT"

# Add parallel option
if [[ "$PARALLEL" == "true" ]]; then
    # Check if pytest-xdist is installed
    python -c "import xdist" 2>/dev/null || {
        print_warning "pytest-xdist not installed. Installing for parallel execution..."
        pip install pytest-xdist
    }
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add specific test
if [[ -n "$SPECIFIC_TEST" ]]; then
    PYTEST_CMD="$PYTEST_CMD $SPECIFIC_TEST"
fi

# Add verbose output
PYTEST_CMD="$PYTEST_CMD -v"

print_status "Starting E2E tests..."
print_status "Command: $PYTEST_CMD"
print_status "Browser: $BROWSER"
print_status "Headed: $HEADED"
print_status "Video: $VIDEO"
print_status "Screenshot: $SCREENSHOT"
print_status "Parallel: $PARALLEL"

# Run the tests
echo ""
print_status "🚀 Running E2E tests..."
echo ""

if eval $PYTEST_CMD; then
    echo ""
    print_success "✅ All tests passed!"
    
    # Show test artifacts if any
    if [[ -d "test-results" ]]; then
        print_status "Test artifacts saved in: test-results/"
        ls -la test-results/ 2>/dev/null || true
    fi
    
    exit 0
else
    echo ""
    print_error "❌ Some tests failed!"
    
    # Show test artifacts if any
    if [[ -d "test-results" ]]; then
        print_status "Test artifacts (videos, screenshots) saved in: test-results/"
        ls -la test-results/ 2>/dev/null || true
    fi
    
    exit 1
fi 