#!/bin/bash

# Carbon Credit Verification System - Stop Script
# Cleanly shuts down both frontend and backend servers

echo "🛑 Stopping Carbon Credit Verification System"
echo "=============================================="

# Function to kill processes on specific ports
kill_port() {
    local port=$1
    local service=$2
    
    pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "🔄 Stopping $service on port $port..."
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        remaining_pids=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$remaining_pids" ]; then
            echo "⚡ Force stopping $service..."
            echo "$remaining_pids" | xargs kill -9 2>/dev/null || true
        fi
        echo "   ✅ $service stopped"
    else
        echo "   ℹ️  No $service process found on port $port"
    fi
}

# Kill backend on port 8000
kill_port 8000 "Backend"

# Kill frontend on port 3000  
kill_port 3000 "Frontend"

# Additional cleanup for any main.py processes
main_pids=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$main_pids" ]; then
    echo "🔄 Stopping remaining backend processes..."
    echo "$main_pids" | xargs kill -TERM 2>/dev/null || true
    sleep 1
    echo "$main_pids" | xargs kill -9 2>/dev/null || true
    echo "   ✅ Backend processes stopped"
fi

# Additional cleanup for react-scripts
react_pids=$(ps aux | grep react-scripts | grep -v grep | awk '{print $2}')
if [ ! -z "$react_pids" ]; then
    echo "🔄 Stopping remaining React processes..."
    echo "$react_pids" | xargs kill -TERM 2>/dev/null || true
    sleep 1
    echo "$react_pids" | xargs kill -9 2>/dev/null || true
    echo "   ✅ React processes stopped"
fi

echo ""
echo "🏁 All servers have been stopped successfully!"
echo "" 