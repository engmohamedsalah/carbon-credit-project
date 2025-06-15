# Professional API Architecture

This directory contains the professional API service layer for the Carbon Credit Verification frontend.

## 🏗️ **Architecture Overview**

### **1. Centralized Configuration (`../config/api.js`)**
- All API endpoints defined in one place
- Environment-based configuration
- Easy to maintain and update API versions

### **2. Professional API Service (`apiService.js`)**
- Singleton pattern for consistent API client
- Built-in error handling and retries
- Automatic authentication token management
- Interceptors for request/response processing

### **3. Environment Configuration**
Create a `.env` file in the frontend root with:
```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_VERSION=v1
REACT_APP_API_TIMEOUT=30000
```

## 🎯 **Usage Examples**

### **Authentication**
```javascript
import apiService from '../services/apiService';

// Register user
const response = await apiService.auth.register(userData);

// Login user
const response = await apiService.auth.login(credentials);

// Get current user
const response = await apiService.auth.getCurrentUser();
```

### **Projects**
```javascript
// Get all projects
const response = await apiService.projects.list();

// Create project
const response = await apiService.projects.create(projectData);

// Get project by ID
const response = await apiService.projects.getById(id);

// Update project
const response = await apiService.projects.update(id, projectData);
```

### **Custom Requests**
```javascript
// Generic HTTP methods available
const response = await apiService.get('/custom/endpoint');
const response = await apiService.post('/custom/endpoint', data);
```

## ✅ **Benefits**

1. **Maintainability**: All API endpoints in one place
2. **Flexibility**: Easy to change API versions or base URLs
3. **Error Handling**: Consistent error handling across the app
4. **Retries**: Automatic retry logic for network failures
5. **Authentication**: Automatic token management
6. **Type Safety**: Clear method signatures for all API calls
7. **Environment Support**: Different configurations for dev/staging/prod

## 🔧 **Configuration**

### **API Endpoints** (`../config/api.js`)
```javascript
export const API_ENDPOINTS = {
  auth: {
    register: `${API_BASE}/auth/register`,
    login: `${API_BASE}/auth/login`,
    // ... more endpoints
  },
  projects: {
    list: `${API_BASE}/projects`,
    create: `${API_BASE}/projects`,
    // ... more endpoints
  }
};
```

### **API Service** (`apiService.js`)
```javascript
class ApiService {
  auth = {
    register: (userData) => this.client.post(API_ENDPOINTS.auth.register, userData),
    login: (credentials) => this.client.post(API_ENDPOINTS.auth.login, credentials),
    // ... more methods
  };
}
```

## 🚀 **Migration from Old API**

**Before (Hardcoded):**
```javascript
const response = await api.post('/api/v1/auth/register', userData);
```

**After (Professional):**
```javascript
const response = await apiService.auth.register(userData);
```

## 🔒 **Security Features**

- Automatic token injection in requests
- Token refresh handling
- Automatic logout on 401 errors
- Secure token storage management

This architecture provides a professional, maintainable, and scalable foundation for all API interactions in the application. 