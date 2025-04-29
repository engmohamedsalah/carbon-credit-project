#!/bin/bash

# Local development setup script for Carbon Credit Verification SaaS

echo "Setting up local development environment for Carbon Credit Verification SaaS..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 before continuing."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js before continuing."
    exit 1
fi

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
python3 -m pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create necessary directories for ML pipeline
echo "Setting up ML directories..."
mkdir -p ml/data/raw
mkdir -p ml/data/processed
mkdir -p ml/models

echo "Local development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Start backend server: cd backend && python3 -m uvicorn main:app --reload"
echo "2. Start frontend server: cd frontend && npm start"
echo ""
echo "Note: For full functionality with database, you'll need to install Docker and PostgreSQL." 