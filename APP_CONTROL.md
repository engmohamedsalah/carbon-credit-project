# 🚀 App Control Guide

## ✅ **Fixed Inconsistency Issue**

The previous setup had 3 different start scripts with different approaches, causing inconsistency between frontend and backend startup/shutdown.

**NEW UNIFIED SOLUTION:**

## 🎯 **Quick Start**

### **Start Everything Together**
```bash
./run_app.sh
```

### **Stop Everything Together**  
```bash
./stop_app.sh
```

## 📋 **What the Scripts Do**

### **`run_app.sh`** - Unified Startup
- ✅ Activates virtual environment
- 🗄️ Initializes SQLite database (if needed)
- 🧹 Cleans up any existing processes on ports 3000 & 8000
- 🚀 Starts backend (`main.py`) with SQLite database on port 8000
- 🎨 Starts frontend (React) on port 3000
- ⏳ Waits for both to be ready
- 🎯 **Ctrl+C stops BOTH servers together**

### **`stop_app.sh`** - Clean Shutdown
- 🛑 Stops backend on port 8000
- 🛑 Stops frontend on port 3000
- 🧹 Cleans up any remaining processes
- ✅ Graceful shutdown with fallback force-kill

## 🔗 **URLs When Running**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/v1/docs
- **Health Check**: http://localhost:8000/health

## 🆚 **Old vs New**

| Issue | Old Approach | New Solution |
|-------|-------------|--------------|
| **Multiple Scripts** | 3 different start scripts | 1 unified script |
| **Inconsistent Startup** | Frontend ≠ Backend approach | Both use same method |
| **No Clean Shutdown** | Manual process killing | Proper cleanup script |
| **Port Conflicts** | No cleanup before start | Auto-cleanup existing processes |
| **Process Tracking** | No PID tracking | Tracks both PIDs |

## 🔧 **Troubleshooting**

### **If scripts fail:**
```bash
# Make executable
chmod +x run_app.sh stop_app.sh

# Check virtual environment
source .venv/bin/activate
pip install -r requirements.txt
```

### **If ports are blocked:**
```bash
# Use stop script to clean up
./stop_app.sh

# Or manual cleanup
lsof -ti:3000,8000 | xargs kill -9
```

## ✨ **Benefits**
- 🎯 **Consistent**: Same approach for both services
- 🔄 **Reliable**: Proper startup/shutdown sequence  
- 🧹 **Clean**: Auto-cleanup before starting
- 📊 **Monitored**: Health checks and PID tracking
- 🗄️ **Persistent**: Uses real SQLite database (no more in-memory data loss!)
- 🚀 **Simple**: One command to rule them all!

## 🗄️ **Database Information**
- **Type**: SQLite (persistent storage)
- **Location**: `database/carbon_credits.db`
- **Auto-initialization**: Database is created automatically on first run
- **Benefits**: All users, projects, and authentication tokens are saved permanently 