/**
 * API Configuration
 * Centralized configuration for all API endpoints and settings
 */

// Environment-based configuration
const config = {
  // Base API URL from environment or default
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  
  // API version - can be changed globally
  apiVersion: process.env.REACT_APP_API_VERSION || 'v1',
  
  // Timeout settings
  timeout: parseInt(process.env.REACT_APP_API_TIMEOUT) || 30000,
  
  // Other API settings
  retryAttempts: 3,
  retryDelay: 1000,
};

// Construct base API path
const API_BASE = `${config.baseURL}/api/${config.apiVersion}`;

/**
 * API Endpoints
 * All API endpoints defined in one place for easy maintenance
 */
export const API_ENDPOINTS = {
  // Authentication endpoints
  auth: {
    register: `${API_BASE}/auth/register`,
    login: `${API_BASE}/auth/login`,
    me: `${API_BASE}/auth/me`,
    refresh: `${API_BASE}/auth/refresh`,
    logout: `${API_BASE}/auth/logout`,
  },
  
  // Project endpoints
  projects: {
    list: `${API_BASE}/projects`,
    create: `${API_BASE}/projects`,
    detail: (id) => `${API_BASE}/projects/${id}`,
    update: (id) => `${API_BASE}/projects/${id}`,
    delete: (id) => `${API_BASE}/projects/${id}`,
    updateStatus: (id) => `${API_BASE}/projects/${id}/status`,
  },
  
  // Verification endpoints
  verification: {
    list: `${API_BASE}/verification`,
    create: `${API_BASE}/verification`,
    detail: (id) => `${API_BASE}/verification/${id}`,
    verify: (projectId) => `${API_BASE}/verification/verify/${projectId}`,
  },
  
  // Satellite imagery endpoints
  satellite: {
    upload: `${API_BASE}/satellite/images`,
    list: (projectId) => `${API_BASE}/satellite/images?project_id=${projectId}`,
    detail: (id) => `${API_BASE}/satellite/images/${id}`,
  },
  
  // Health and system endpoints
  system: {
    health: `${config.baseURL}/health`,
    docs: `${API_BASE}/docs`,
    openapi: `${config.baseURL}/openapi.json`,
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