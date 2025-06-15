#!/bin/bash

# Carbon Credit Verification System - Demo Startup Script
echo "🌱 Starting Carbon Credit Verification System (Demo)"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers shut down successfully."
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start backend
echo "🚀 Starting Professional Backend API..."
cd backend
python demo_main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Test backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend running at http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/api/v1/docs"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
echo "🎨 Starting React Frontend..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to initialize..."
sleep 10

# Test frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend running at http://localhost:3000"
else
    echo "❌ Frontend failed to start"
fi

echo ""
echo "🎉 System is ready!"
echo "================================"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/api/v1/docs"
echo "❤️  Health Check: http://localhost:8000/health"
echo ""
echo "Professional Features Demonstrated:"
echo "✓ Clean Architecture (Modular Structure)"
echo "✓ Proper Error Handling & Validation"
echo "✓ Security Best Practices"
echo "✓ API Documentation (Swagger/OpenAPI)"
echo "✓ Logging & Monitoring"
echo "✓ CORS Configuration"
echo "✓ Authentication & Authorization"
echo "✓ Database Integration (SQLite)"
echo ""
echo "Press Ctrl+C to stop all servers"

# Keep script running
wait 