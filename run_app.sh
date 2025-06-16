#!/bin/bash

# Carbon Credit Verification System - Unified Run Script
# Starts both frontend and backend together with proper cleanup

set -e  # Exit on any error

echo "🌱 Carbon Credit Verification System"
echo "===================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down all servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "   ✅ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "   ✅ Frontend stopped"
    fi
    echo "🏁 All servers shut down successfully"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Initialize database if needed
echo "🗄️  Initializing SQLite database..."
cd backend
if [ ! -f "../database/carbon_credits.db" ]; then
    echo "   Creating new database..."
    python init_db.py
else
    echo "   ✅ Database already exists"
fi

# Start backend
echo "🚀 Starting Backend API (with SQLite database)..."
if [ ! -f "main.py" ]; then
    echo "❌ Backend file main.py not found!"
    exit 1
fi

python main.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ Backend is ready!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "   ❌ Backend failed to start after 15 seconds"
        exit 1
    fi
    sleep 1
done

# Start frontend
echo "🎨 Starting Frontend..."
cd frontend
if [ ! -f "package.json" ]; then
    echo "❌ Frontend package.json not found!"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

npm start &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "   ✅ Frontend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ⚠️  Frontend might still be starting (taking longer than expected)"
        break
    fi
    sleep 1
done

echo ""
echo "🎉 System is ready!"
echo "=================="
echo "🌐 Frontend:    http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📖 API Docs:    http://localhost:8000/api/v1/docs"
echo "❤️  Health:     http://localhost:8000/health"
echo ""
echo "💡 Both servers are running with PIDs:"
echo "   Backend:  $BACKEND_PID"
echo "   Frontend: $FRONTEND_PID"
echo ""
echo "⚠️  Press Ctrl+C to stop BOTH servers together"

# Keep script running and wait for both processes
wait 