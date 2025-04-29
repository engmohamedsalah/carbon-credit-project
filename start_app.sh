#!/bin/bash

# Script to start both frontend and backend servers for Carbon Credit Verification SaaS

echo "Starting Carbon Credit Verification SaaS Application..."
echo ""

# Check prerequisites
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 before continuing."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js before continuing."
    exit 1
fi

# Kill any existing servers on our ports
echo "Checking for existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

# Initialize the SQLite database if it doesn't exist
if [ ! -f ./data/carbon_credits.db ]; then
    echo "Initializing database..."
    python3 simple_init_db.py
fi

# Start the backend server
echo "Starting backend server..."
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a bit for the backend to initialize
sleep 2

# Start the frontend server
echo "Starting frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "Carbon Credit Verification SaaS application is now running!"
echo "Backend API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to handle script termination
function cleanup {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servers shut down successfully."
    exit 0
}

# Register the cleanup function to be called on exit
trap cleanup SIGINT SIGTERM

# Keep the script running
wait 