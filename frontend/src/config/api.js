/**
 * API Configuration
 * Centralized configuration for all API endpoints and settings
 */

// Environment-based configuration
const config = {
  // Base API URL from environment or default
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
  
  // API version - can be changed globally
  apiVersion: process.env.REACT_APP_API_VERSION || 'v1',
  
  // Timeout settings
  timeout: parseInt(process.env.REACT_APP_API_TIMEOUT) || 30000,
  
  // Other API settings
  retryAttempts: 3,
  retryDelay: 1000,
};

/**
 * API Endpoints
 * All API endpoints defined in one place for easy maintenance
 * Using relative URLs that will be appended to the baseURL
 */
export const API_ENDPOINTS = {
  // Authentication endpoints
  auth: {
    register: '/auth/register',
    login: '/auth/login',
    me: '/auth/me',
    refresh: '/auth/refresh',
    logout: '/auth/logout',
  },
  
  // Project endpoints
  projects: {
    list: '/projects',
    create: '/projects',
    detail: (id) => `/projects/${id}`,
    update: (id) => `/projects/${id}`,
    delete: (id) => `/projects/${id}`,
    updateStatus: (id) => `/projects/${id}/status`,
  },
  
  // Verification endpoints
  verification: {
    list: '/verification',
    create: '/verification',
    detail: (id) => `/verification/${id}`,
    verify: (projectId) => `/verification/verify/${projectId}`,
  },
  
  // Satellite imagery endpoints
  satellite: {
    upload: '/satellite/images',
    list: (projectId) => `/satellite/images?project_id=${projectId}`,
    detail: (id) => `/satellite/images/${id}`,
  },
  
  // Health and system endpoints (these need to be absolute since they don't use /api/v1)
  system: {
    health: 'http://localhost:8000/health',
    docs: 'http://localhost:8000/api/v1/docs',
    openapi: 'http://localhost:8000/openapi.json',
  },
};

/**
 * API Configuration object
 */
export const API_CONFIG = {
  baseURL: config.baseURL,
  apiVersion: config.apiVersion,
  timeout: config.timeout,
  retryAttempts: config.retryAttempts,
  retryDelay: config.retryDelay,
  
  // Headers
  defaultHeaders: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  
  // Authentication header
  getAuthHeader: (token) => ({
    'Authorization': `Bearer ${token}`,
  }),
};

export default API_ENDPOINTS; 