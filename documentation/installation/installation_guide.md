# Carbon Credit Verification SaaS Application
## Installation Guide

This comprehensive installation guide provides step-by-step instructions for setting up and deploying the Carbon Credit Verification SaaS application in various environments.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Database Setup](#database-setup)
6. [Environment Configuration](#environment-configuration)
7. [ML Model Setup](#ml-model-setup)
8. [Blockchain Integration](#blockchain-integration)
9. [Testing the Installation](#testing-the-installation)
10. [Troubleshooting](#troubleshooting)
11. [Upgrading](#upgrading)

## Prerequisites

Before installing the Carbon Credit Verification SaaS application, ensure you have the following prerequisites:

### System Requirements

- **CPU**: 4+ cores recommended (2 cores minimum)
- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 20GB+ free space (more needed for satellite imagery)
- **Operating System**: Ubuntu 20.04+, macOS 10.15+, or Windows 10+ with WSL2

### Software Requirements

- **Docker**: Docker Engine 20.10+ and Docker Compose 2.0+
- **Git**: Git 2.25+
- **Python**: Python 3.10+ (if not using Docker)
- **Node.js**: Node.js 16+ (if not using Docker)
- **PostgreSQL**: PostgreSQL 14+ with PostGIS extension (if not using Docker)

### Access Requirements

- **Sentinel-2 API**: Account with Copernicus Open Access Hub
- **Polygon Network**: Access to Polygon network (testnet for development)
- **Cloud Provider**: Account with AWS, GCP, or Azure (for cloud deployment)

## Local Development Setup

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/carbon-credit-verification.git

# Navigate to the project directory
cd carbon-credit-verification
```

### Step 2: Set Up Environment Variables

```bash
# Copy the example environment file
cp backend/.env.example backend/.env

# Edit the environment file with your settings
nano backend/.env
```

Update the following variables in the `.env` file:

```
# Database settings
DATABASE_URL=postgresql://postgres:password@localhost:5432/carbon_verification

# JWT settings
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Sentinel-2 API credentials
SENTINEL_USERNAME=your-copernicus-username
SENTINEL_PASSWORD=your-copernicus-password

# Blockchain settings
POLYGON_RPC_URL=https://polygon-mumbai.infura.io/v3/your-infura-key
POLYGON_PRIVATE_KEY=your-private-key
```

### Step 3: Install Backend Dependencies

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Step 4: Install Frontend Dependencies

```bash
# Navigate to the frontend directory
cd ../frontend

# Install dependencies
npm install
```

### Step 5: Set Up the Database

```bash
# Install PostgreSQL and PostGIS (if not using Docker)
sudo apt update
sudo apt install postgresql postgresql-contrib postgis

# Create the database
sudo -u postgres createdb carbon_verification

# Enable PostGIS extension
sudo -u postgres psql -d carbon_verification -c "CREATE EXTENSION postgis;"

# Run database migrations
cd ../backend
alembic upgrade head
```

### Step 6: Start the Application

```bash
# Start the backend (from the backend directory)
uvicorn main:app --reload

# In a new terminal, start the frontend (from the frontend directory)
npm start
```

The application should now be running at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API documentation: http://localhost:8000/docs

## Docker Deployment

For a simpler setup, you can use Docker to deploy the entire application stack.

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/carbon-credit-verification.git

# Navigate to the project directory
cd carbon-credit-verification
```

### Step 2: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the environment file
nano .env
```

Update the environment variables as needed.

### Step 3: Build and Start Docker Containers

```bash
# Build and start the containers
docker-compose up -d

# Check the status of the containers
docker-compose ps
```

### Step 4: Initialize the Database

```bash
# Run database migrations
docker-compose exec backend alembic upgrade head

# (Optional) Seed the database with sample data
docker-compose exec backend python -m app.scripts.seed_data
```

The application should now be running at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API documentation: http://localhost:8000/docs

## Cloud Deployment

### AWS Deployment

#### Step 1: Set Up AWS Resources

1. Create an AWS account if you don't have one
2. Install the AWS CLI and configure it with your credentials
3. Create an ECR repository for Docker images
4. Set up an RDS PostgreSQL instance with PostGIS
5. Create an ECS cluster for container deployment

#### Step 2: Build and Push Docker Images

```bash
# Log in to ECR
aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account-id.dkr.ecr.your-region.amazonaws.com

# Build and tag images
docker-compose build
docker tag carbon-verification-backend:latest your-account-id.dkr.ecr.your-region.amazonaws.com/carbon-verification-backend:latest
docker tag carbon-verification-frontend:latest your-account-id.dkr.ecr.your-region.amazonaws.com/carbon-verification-frontend:latest

# Push images to ECR
docker push your-account-id.dkr.ecr.your-region.amazonaws.com/carbon-verification-backend:latest
docker push your-account-id.dkr.ecr.your-region.amazonaws.com/carbon-verification-frontend:latest
```

#### Step 3: Deploy to ECS

1. Create task definitions for backend and frontend
2. Create ECS services using the task definitions
3. Set up an Application Load Balancer
4. Configure security groups and networking
5. Set up environment variables in the task definitions

### Google Cloud Platform Deployment

#### Step 1: Set Up GCP Resources

1. Create a GCP account if you don't have one
2. Install the Google Cloud SDK and configure it
3. Create a Cloud SQL PostgreSQL instance with PostGIS
4. Set up a Google Kubernetes Engine (GKE) cluster

#### Step 2: Build and Push Docker Images

```bash
# Configure Docker to use Google Container Registry
gcloud auth configure-docker

# Build and tag images
docker-compose build
docker tag carbon-verification-backend:latest gcr.io/your-project-id/carbon-verification-backend:latest
docker tag carbon-verification-frontend:latest gcr.io/your-project-id/carbon-verification-frontend:latest

# Push images to GCR
docker push gcr.io/your-project-id/carbon-verification-backend:latest
docker push gcr.io/your-project-id/carbon-verification-frontend:latest
```

#### Step 3: Deploy to GKE

1. Create Kubernetes deployment files
2. Apply the deployment files to your GKE cluster
3. Set up a load balancer
4. Configure environment variables as Kubernetes secrets

## Database Setup

### PostgreSQL with PostGIS

The application requires PostgreSQL with the PostGIS extension for geospatial data handling.

#### Manual Setup

```bash
# Install PostgreSQL and PostGIS
sudo apt update
sudo apt install postgresql postgresql-contrib postgis

# Create a database user
sudo -u postgres createuser --interactive --pwprompt

# Create the database
sudo -u postgres createdb carbon_verification

# Enable PostGIS extension
sudo -u postgres psql -d carbon_verification -c "CREATE EXTENSION postgis;"
```

#### Docker Setup

The Docker Compose configuration includes a PostgreSQL container with PostGIS already configured.

## Environment Configuration

The application uses environment variables for configuration. Key variables include:

### Backend Environment Variables

```
# Database
DATABASE_URL=postgresql://user:password@host:port/dbname

# Authentication
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Sentinel-2 API
SENTINEL_USERNAME=your-copernicus-username
SENTINEL_PASSWORD=your-copernicus-password

# Blockchain
POLYGON_RPC_URL=https://polygon-mumbai.infura.io/v3/your-infura-key
POLYGON_PRIVATE_KEY=your-private-key
CONTRACT_ADDRESS=your-contract-address

# Storage
STORAGE_PATH=/path/to/storage
```

### Frontend Environment Variables

```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_MAPBOX_TOKEN=your-mapbox-token
```

## ML Model Setup

The ML components require additional setup for training and inference.

### Step 1: Install ML Dependencies

```bash
# Navigate to the ML directory
cd ml

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Pre-trained Models

```bash
# Create models directory
mkdir -p models

# Download pre-trained model
python -c "
import torch
import torch.nn as nn
import os
from training.train_forest_change import UNet

# Initialize model
model = UNet(in_channels=4, out_channels=2)

# Save empty model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/forest_change_unet.pth')
print('Created empty model file. Replace with actual trained model when available.')
"
```

### Step 3: Set Up Data Directories

```bash
# Create data directories
mkdir -p data/raw/sentinel
mkdir -p data/raw/hansen
mkdir -p data/prepared
mkdir -p results
```

### Step 4: Configure Sentinel-2 API Access

Ensure your Sentinel-2 API credentials are set in the environment variables:

```bash
export SENTINEL_USERNAME=your-copernicus-username
export SENTINEL_PASSWORD=your-copernicus-password
```

## Blockchain Integration

The application integrates with the Polygon blockchain for certification.

### Step 1: Set Up Polygon Connection

1. Create an account on Polygon (if you don't have one)
2. Get testnet MATIC from a faucet for development
3. Set up an Infura or Alchemy account for RPC access

### Step 2: Deploy Smart Contract

```bash
# Navigate to the blockchain directory
cd blockchain

# Install dependencies
npm install

# Compile contracts
npx hardhat compile

# Deploy to Polygon Mumbai testnet
npx hardhat run scripts/deploy.js --network mumbai
```

### Step 3: Configure Contract Address

Update the `CONTRACT_ADDRESS` environment variable with the deployed contract address.

## Testing the Installation

### Backend Tests

```bash
# Navigate to the backend directory
cd backend

# Run tests
pytest
```

### Frontend Tests

```bash
# Navigate to the frontend directory
cd frontend

# Run tests
npm test
```

### End-to-End Tests

```bash
# Navigate to the project root
cd ..

# Run end-to-end tests
npm run test:e2e
```

### Manual Testing

1. Open the application in your browser
2. Register a new account
3. Create a test project
4. Upload sample satellite imagery
5. Run a forest change detection analysis
6. Verify the results are displayed correctly

## Troubleshooting

### Common Installation Issues

#### Database Connection Errors

**Problem**: Unable to connect to the database.

**Solutions**:
1. Check that PostgreSQL is running: `sudo systemctl status postgresql`
2. Verify database credentials in the `.env` file
3. Ensure the database and user exist: `sudo -u postgres psql -c "\l"`
4. Check network connectivity if using a remote database

#### Docker Compose Errors

**Problem**: Docker Compose fails to start containers.

**Solutions**:
1. Check Docker and Docker Compose versions: `docker --version && docker-compose --version`
2. Verify that Docker daemon is running: `sudo systemctl status docker`
3. Check for port conflicts: `netstat -tuln`
4. Inspect container logs: `docker-compose logs`

#### ML Model Loading Errors

**Problem**: ML model fails to load.

**Solutions**:
1. Verify the model file exists: `ls -la ml/models/`
2. Check PyTorch version compatibility: `python -c "import torch; print(torch.__version__)"`
3. Ensure CUDA is available if using GPU: `python -c "import torch; print(torch.cuda.is_available())"`
4. Try downloading a pre-trained model from the project repository

### Getting Help

If you encounter issues not covered in this guide:

1. Check the project GitHub repository for known issues
2. Join the developer community on Discord
3. Submit an issue on GitHub with detailed information
4. Contact the maintainers directly

## Upgrading

### Upgrading from Previous Versions

```bash
# Pull the latest changes
git pull origin main

# Update dependencies
cd backend && pip install -r requirements.txt
cd ../frontend && npm install

# Run database migrations
cd ../backend && alembic upgrade head

# Rebuild Docker containers (if using Docker)
cd .. && docker-compose build && docker-compose up -d
```

### Version-Specific Upgrade Notes

#### Upgrading to v1.1

- Database schema changes require running migrations
- New environment variables for blockchain integration
- Frontend dependencies updated to React 18

#### Upgrading to v1.2

- ML model format changed, requires redownloading pre-trained models
- API endpoints for verification have changed
- New features for carbon sequestration estimation

---

This installation guide provides comprehensive instructions for setting up and deploying the Carbon Credit Verification SaaS application. For user instructions, please refer to the user guide documentation.
