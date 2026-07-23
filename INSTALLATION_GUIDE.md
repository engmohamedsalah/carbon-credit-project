# Carbon Credit Verification System - Installation & Setup Guide

This guide provides comprehensive instructions for installing and running the Carbon Credit Verification SaaS application.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Project Dependencies](#project-dependencies)
4. [Project Setup](#project-setup)
5. [Running the Application](#running-the-application)
6. [Docker Deployment](#docker-deployment)
7. [Stopping the Application](#stopping-the-application)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 20.04+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Processor**: Intel i5/AMD Ryzen 5 or equivalent
- **Internet Connection**: Required for downloading dependencies and satellite imagery

### Software Requirements
- Python 3.8 or higher
- Node.js 14.x or higher
- npm 6.x or higher
- Git
- Docker (optional, for containerized deployment)

## Prerequisites Installation

### 1. Install Python

#### Windows
1. Download Python from https://www.python.org/downloads/
2. Run the installer and check "Add Python to PATH"
3. Verify installation:
   ```bash
   python --version
   ```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.8

# Verify installation
python3 --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.8 python3-pip python3-venv
python3 --version
```

### 2. Install Node.js and npm

#### Windows/macOS
1. Download Node.js from https://nodejs.org/
2. Run the installer (npm is included)
3. Verify installation:
   ```bash
   node --version
   npm --version
   ```

#### Linux (Ubuntu/Debian)
```bash
curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version
npm --version
```

### 3. Install Git

#### Windows
Download and install from https://git-scm.com/download/windows

#### macOS
```bash
brew install git
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt install git
```

### 4. Install Docker (Optional)

#### All Platforms
Visit https://docs.docker.com/get-docker/ and follow the instructions for your operating system.

## Project Dependencies

### Backend Dependencies (Python)
The following Python packages will be installed automatically via requirements.txt:

**Core Framework:**
- FastAPI (0.104.1) - Modern web framework
- Uvicorn (0.24.0) - ASGI server
- Pydantic (2.5.0) - Data validation

**Security & Authentication:**
- passlib[bcrypt] (1.7.4) - Password hashing
- python-jose[cryptography] (3.3.0) - JWT tokens
- PyJWT (2.8.0) - JSON Web Tokens
- python-multipart (0.0.6) - Form data parsing

**API & Networking:**
- httpx (0.25.2) - Async HTTP client
- requests (2.31.0) - HTTP library
- slowapi (0.1.9) - Rate limiting
- starlette (0.27.0) - ASGI toolkit

**Data & Reporting:**
- reportlab (4.0.4) - PDF generation
- qrcode[pil] (7.4.2) - QR code generation
- email-validator (2.1.0) - Email validation

**Development Tools:**
- pytest (7.4.3) - Testing framework
- pytest-asyncio (0.21.1) - Async test support
- pytest-cov (4.1.0) - Code coverage
- python-dotenv (1.0.0) - Environment variables
- structlog (23.2.0) - Structured logging

### Frontend Dependencies (npm)
The following npm packages will be installed automatically via package.json:

**Core React & Routing:**
- react (18.2.0) - UI library
- react-dom (18.2.0) - DOM rendering
- react-router-dom (6.10.0) - Routing
- react-scripts (5.0.1) - Build tooling

**State Management:**
- @reduxjs/toolkit (1.9.3) - Redux toolkit
- react-redux (8.0.5) - React bindings for Redux

**UI Framework:**
- @mui/material (5.12.0) - Material-UI components
- @mui/icons-material (5.11.16) - Material icons
- @emotion/react (11.10.6) - CSS-in-JS
- @emotion/styled (11.10.6) - Styled components

**Data Visualization:**
- chart.js (4.5.0) - Chart library
- react-chartjs-2 (5.3.0) - React wrapper for Chart.js
- recharts (2.15.4) - React charts
- d3 (7.9.0) - Data visualization
- plotly.js (3.0.1) - Interactive charts
- react-plotly.js (2.6.0) - React Plotly wrapper
- @mui/x-charts (8.9.0) - MUI charts

**Mapping:**
- leaflet (1.9.3) - Interactive maps
- react-leaflet (4.2.1) - React Leaflet wrapper
- leaflet-draw (1.0.4) - Drawing tools
- leaflet-geometryutil (0.10.3) - Geometry utilities

**3D Visualization:**
- three (0.160.0) - 3D library
- @react-three/fiber (8.15.16) - React Three.js
- @react-three/drei (9.105.6) - Three.js helpers

**Utilities:**
- axios (1.3.5) - HTTP client
- lodash (4.17.21) - Utility functions
- date-fns (4.1.0) - Date utilities
- html2canvas (1.4.1) - HTML to canvas
- jspdf (3.0.1) - PDF generation
- react-to-print (3.1.0) - Print components

**Testing (Dev Dependencies):**
- @testing-library/react (16.3.0) - React testing
- @testing-library/jest-dom (6.6.3) - DOM matchers
- @testing-library/user-event (14.6.1) - User interaction simulation

### Additional System Dependencies

**For ML/AI Features (installed separately):**
- PyTorch - Deep learning framework
- torchvision - Computer vision models
- NumPy - Numerical computing
- Pandas - Data manipulation
- scikit-learn - Machine learning utilities
- GDAL - Geospatial data processing (for satellite imagery)

**Optional for Production:**
- PostgreSQL - Production database
- PostGIS - Spatial database extension
- Redis - Caching and session storage
- Nginx - Reverse proxy

## Project Setup

### 1. Clone the Repository

```bash
# Clone the project
git clone [repository-url]
cd carbon_credit_project

# Or if you have a ZIP file
unzip carbon_credit_project.zip
cd carbon_credit_project
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Initialize the database
cd backend
python init_db.py
cd ..
```

### 4. Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install

# Return to root directory
cd ..
```

### 5. Download ML Models

The ML models are large files and need to be downloaded separately:

```bash
# Create models directory if it doesn't exist
mkdir -p ml/models

# Download models (URLs would be provided separately due to file size)
# Place the following files in ml/models/:
# - forest_cover_unet_focal_alpha_0.75_threshold_0.53.pth
# - change_detection_siamese_unet.pth
# - convlstm_fast_final.pth
```

### 6. Environment Configuration

Create a `.env` file in the root directory:

```bash
# Backend Configuration
BACKEND_HOST=localhost
BACKEND_PORT=8000
DATABASE_URL=sqlite:///./database/carbon_credits.db
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_MAPBOX_TOKEN=your-mapbox-token-here

# ML Configuration
ML_MODEL_PATH=./ml/models
SENTINEL_API_KEY=your-sentinel-hub-api-key
```

## Running the Application

### Method 1: Using the Unified Start Script (Recommended)

```bash
# Make the script executable (first time only)
chmod +x run_app.sh

# Start both frontend and backend
./run_app.sh
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Method 2: Running Services Separately

#### Terminal 1 - Backend
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start backend server
cd backend
python main.py
```

#### Terminal 2 - Frontend
```bash
# Start frontend development server
cd frontend
npm start
```

### Method 3: Using Development Setup Script

```bash
# Run the local development setup
./scripts/local_dev_setup.sh

# This will:
# 1. Create virtual environment
# 2. Install all dependencies
# 3. Initialize database
# 4. Start both services
```

## Docker Deployment

### 1. Build and Start Services

```bash
# Navigate to docker directory
cd docker

# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 2. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### 3. View Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs frontend
docker-compose logs backend
```

## Testing the Installation

### 1. Run Health Checks

```bash
# Check backend health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "database": "connected"}
```

### 2. Run Test Suite

```bash
# Run all tests
./scripts/run_tests.sh

# Run backend tests only
python tests/test_backend.py

# Run frontend tests only
cd frontend && npm test

# Run E2E tests
cd tests/e2e && ./run_tests.sh
```

### 3. Verify ML Models

```bash
# Test ML model loading
cd backend
python -c "from services.ml_service import MLService; ml = MLService(); print('ML models loaded successfully')"
```

## Default Credentials

For development/testing purposes:

- **Admin User**: provisioned via the `ADMIN_EMAIL` / `ADMIN_PASSWORD` environment variables

- **Test User**:
  - Username: user@example.com
  - Password: user123

**Important**: Change these credentials before deploying to production!

## Stopping the Application

### Method 1: Using the Stop Script (Recommended)

```bash
# Make the script executable (first time only)
chmod +x stop_app.sh

# Stop all services cleanly
./stop_app.sh
```

The stop script will:
- Gracefully terminate backend server on port 8000
- Gracefully terminate frontend server on port 3000
- Clean up any remaining Python main.py processes
- Clean up any remaining React processes
- Provide status updates during shutdown

### Method 2: Manual Shutdown

#### If Using run_app.sh Script
Press `Ctrl+C` in the terminal where the script is running.

#### If Running Services Separately
1. Stop frontend: Press `Ctrl+C` in the frontend terminal
2. Stop backend: Press `Ctrl+C` in the backend terminal
3. Deactivate virtual environment: `deactivate`

### Method 3: Force Stop (Emergency)

If services don't stop gracefully, use these commands:

```bash
# Kill all processes on specific ports
# Frontend (port 3000)
lsof -ti:3000 | xargs kill -9  # macOS/Linux
taskkill /F /IM node.exe       # Windows

# Backend (port 8000)
lsof -ti:8000 | xargs kill -9  # macOS/Linux
taskkill /F /IM python.exe     # Windows
```

### If Using Docker
```bash
# Stop all containers
cd docker
docker-compose down

# Stop and remove volumes (careful - this deletes data)
docker-compose down -v

# Force stop if containers are stuck
docker-compose kill
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Kill process on port 3000 (frontend)
# On macOS/Linux:
lsof -ti:3000 | xargs kill -9
# On Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Kill process on port 8000 (backend)
# On macOS/Linux:
lsof -ti:8000 | xargs kill -9
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

#### 2. Module Not Found Errors
```bash
# Reinstall dependencies
pip install -r backend/requirements.txt --force-reinstall
cd frontend && npm install
```

#### 3. Database Connection Issues
```bash
# Recreate database
cd backend
rm -f ../database/carbon_credits.db
python init_db.py
```

#### 4. ML Model Loading Errors
- Ensure all model files are present in `ml/models/`
- Check file permissions: `chmod 644 ml/models/*.pth`
- Verify Python torch installation: `pip install torch torchvision`

### Getting Help

If you encounter issues:

1. Check the logs:
   - Backend logs: Check terminal output
   - Frontend logs: Check browser console (F12)
   - Docker logs: `docker-compose logs`

2. Verify all dependencies are installed:
   ```bash
   pip list | grep -E "fastapi|torch|pandas"
   cd frontend && npm list react redux
   ```

3. Ensure all environment variables are set correctly in `.env`

4. Try the Docker deployment as an alternative

## Additional Scripts

### Database Management
```bash
# Backup database
./scripts/backup_db.sh

# Restore database
./scripts/restore_db.sh backup_file.sql
```

### Performance Testing
```bash
# Run performance tests
./scripts/performance_test.sh
```

### Clean Installation
```bash
# Remove all dependencies and start fresh
./scripts/clean_install.sh
```

## System Architecture Notes

- **Frontend**: React app with Redux state management, served on port 3000
- **Backend**: FastAPI REST API served on port 8000
- **Database**: SQLite for development, PostgreSQL for production (Docker)
- **ML Pipeline**: Pre-trained models for forest change detection
- **Authentication**: JWT-based with role-based access control

## Next Steps

1. Access the application at http://localhost:3000
2. Log in with the default credentials
3. Create a new project to verify carbon credits
4. Upload satellite imagery or use the demo data
5. Review the XAI explanations for ML predictions
6. Generate verification reports

For production deployment guidelines, refer to the `DEPLOYMENT.md` file.
