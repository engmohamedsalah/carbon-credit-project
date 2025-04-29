#!/bin/bash

# This script sets up the environment and starts the Carbon Credit Verification SaaS application

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker before continuing."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose before continuing."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js before continuing."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 before continuing."
    exit 1
fi

echo "Starting Carbon Credit Verification SaaS application..."

# Create .env file if it doesn't exist
if [ ! -f ./backend/.env ]; then
    echo "Creating default .env file..."
    cp ./backend/.env.example ./backend/.env
fi

# Start the database
echo "Starting PostgreSQL database..."
docker-compose -f ./docker/docker-compose.yml up -d db

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 10

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
python3 -m pip install -r requirements.txt

# Start the backend in the background
echo "Starting backend server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Go back to root directory
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install

# Start the frontend
echo "Starting frontend server..."
npm start &
FRONTEND_PID=$!

# Go back to root directory
cd ..

echo "Carbon Credit Verification SaaS application is now running!"
echo "Backend API: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Documentation: http://localhost:8000/docs"

# Function to handle script termination
function cleanup {
    echo "Shutting down servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    docker-compose -f ./docker/docker-compose.yml down
    echo "Servers shut down successfully."
}

# Register the cleanup function to be called on exit
trap cleanup EXIT

# Keep the script running
echo "Press Ctrl+C to stop the application"
wait
