/**
 * Professional API Service
 * Centralized API client with error handling, retries, and interceptors
 */
import axios from 'axios';
import { API_ENDPOINTS, API_CONFIG } from '../config/api';

class ApiService {
  constructor() {
    // Create axios instance with base configuration
    this.client = axios.create({
      baseURL: API_CONFIG.baseURL,
      timeout: API_CONFIG.timeout,
      headers: API_CONFIG.defaultHeaders,
    });

    // Setup interceptors
    this.setupInterceptors();
  }

  /**
   * Setup request and response interceptors
   */
  setupInterceptors() {
    // Request interceptor - add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('token');
        if (token) {
          config.headers = {
            ...config.headers,
            ...API_CONFIG.getAuthHeader(token),
          };
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor - handle common errors
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        // Handle 401 Unauthorized
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          localStorage.removeItem('token');
          window.location.href = '/login';
          return Promise.reject(error);
        }

        // Handle network errors with retry
        if (!error.response && originalRequest._retryCount < API_CONFIG.retryAttempts) {
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;
          
          // Wait before retry
          await new Promise(resolve => 
            setTimeout(resolve, API_CONFIG.retryDelay * originalRequest._retryCount)
          );
          
          return this.client(originalRequest);
        }

        return Promise.reject(error);
      }
    );
  }

  /**
   * Authentication API methods
   */
  auth = {
    register: (userData) => this.client.post(API_ENDPOINTS.auth.register, userData),
    login: (credentials) => this.client.post(API_ENDPOINTS.auth.login, credentials, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    }),
    getCurrentUser: () => this.client.get(API_ENDPOINTS.auth.me),
    logout: () => this.client.post(API_ENDPOINTS.auth.logout),
  };

  /**
   * Project API methods
   */
  projects = {
    list: (params = {}) => this.client.get(API_ENDPOINTS.projects.list, { params }),
    create: (projectData) => this.client.post(API_ENDPOINTS.projects.create, projectData),
    getById: (id) => this.client.get(API_ENDPOINTS.projects.detail(id)),
    update: (id, projectData) => this.client.put(API_ENDPOINTS.projects.update(id), projectData),
    delete: (id) => this.client.delete(API_ENDPOINTS.projects.delete(id)),
    updateStatus: (id, status) => this.client.patch(API_ENDPOINTS.projects.updateStatus(id), { status }),
  };

  /**
   * Verification API methods
   */
  verification = {
    list: (params = {}) => this.client.get(API_ENDPOINTS.verification.list, { params }),
    create: (verificationData) => this.client.post(API_ENDPOINTS.verification.create, verificationData),
    getById: (id) => this.client.get(API_ENDPOINTS.verification.detail(id)),
    verify: (projectId) => this.client.post(API_ENDPOINTS.verification.verify(projectId)),
    submitHumanReview: (id, reviewData) => this.client.post(`${API_ENDPOINTS.verification.detail(id)}/human-review`, reviewData),
    certify: (verificationId) => this.client.post(`/blockchain/certify/${verificationId}`),
  };

  /**
   * Satellite imagery API methods
   */
  satellite = {
    upload: (formData) => this.client.post(API_ENDPOINTS.satellite.upload, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    }),
    list: (projectId) => this.client.get(API_ENDPOINTS.satellite.list(projectId)),
    getById: (id) => this.client.get(API_ENDPOINTS.satellite.detail(id)),
  };

  /**
   * System API methods
   */
  system = {
    health: () => this.client.get(API_ENDPOINTS.system.health),
    getDocs: () => this.client.get(API_ENDPOINTS.system.docs),
  };

  /**
   * Generic HTTP methods for custom requests
   */
  get = (url, config = {}) => this.client.get(url, config);
  post = (url, data = {}, config = {}) => this.client.post(url, data, config);
  put = (url, data = {}, config = {}) => this.client.put(url, data, config);
  delete = (url, config = {}) => this.client.delete(url, config);
  patch = (url, data = {}, config = {}) => this.client.patch(url, data, config);

  /**
   * Utility methods
   */
  setAuthToken = (token) => {
    if (token) {
      localStorage.setItem('token', token);
      this.client.defaults.headers.common = {
        ...this.client.defaults.headers.common,
        ...API_CONFIG.getAuthHeader(token),
      };
    } else {
      localStorage.removeItem('token');
      delete this.client.defaults.headers.common['Authorization'];
    }
  };

  clearAuthToken = () => {
    this.setAuthToken(null);
  };
}

// Create and export singleton instance
const apiService = new ApiService();
export default apiService; 