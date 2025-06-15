#!/bin/bash

# Professional local startup script
# Uses SQLite database and professional backend architecture

echo "Starting Carbon Credit Verification SaaS (Professional Architecture)..."
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

# Using professional backend with SQLite database
echo "Using professional backend architecture with SQLite database..."

# Install backend dependencies if needed
echo "Installing backend dependencies..."
cd backend
python3 -m pip install -r requirements.txt
cd ..

# Start the backend server (Professional Architecture)
echo "Starting professional backend server..."
cd backend
python3 demo_main.py &
BACKEND_PID=$!
cd ..

# Wait a bit for the backend to initialize
sleep 3

# Install frontend dependencies if needed
echo "Installing frontend dependencies..."
cd frontend
npm install

# Start the frontend server
echo "Starting frontend server..."
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "🎉 Carbon Credit Verification SaaS application is now running!"
echo "🔧 Backend API: http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo "📖 API Documentation: http://localhost:8000/api/v1/docs"
echo "❤️  Health Check: http://localhost:8000/health"
echo ""
echo "✅ Professional Features Active:"
echo "   • Clean Architecture (Modular Structure)"
echo "   • Security Best Practices"
echo "   • Input Validation & Error Handling"
echo "   • API Documentation (Swagger/OpenAPI)"
echo "   • Logging & Monitoring"
echo "   • SQLite Database Integration"
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