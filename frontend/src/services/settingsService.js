/**
 * Settings Service
 * Handles all settings-related API calls
 */
import axios from 'axios';
import { API_ENDPOINTS, API_CONFIG } from '../config/api';

class SettingsService {
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
      (response) => response.data,
      async (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('token');
          window.location.href = '/login';
          return Promise.reject(error);
        }
        return Promise.reject(error);
      }
    );
  }

  /**
   * Get user settings
   * @returns {Promise} User settings data
   */
  async getSettings() {
    try {
      const response = await this.client.get(API_ENDPOINTS.settings.get);
      return response;
    } catch (error) {
      console.error('Error fetching user settings:', error);
      throw error;
    }
  }

  /**
   * Update user settings
   * @param {Object} settingsData - Settings data to update
   * @returns {Promise} Updated settings data
   */
  async updateSettings(settingsData) {
    try {
      const response = await this.client.patch(API_ENDPOINTS.settings.update, settingsData);
      return response;
    } catch (error) {
      console.error('Error updating user settings:', error);
      throw error;
    }
  }

  /**
   * Change user password
   * @param {Object} passwordData - Password change data
   * @param {string} passwordData.current_password - Current password
   * @param {string} passwordData.new_password - New password
   * @returns {Promise} Success message
   */
  async changePassword(passwordData) {
    try {
      const response = await this.client.post(API_ENDPOINTS.settings.changePassword, passwordData);
      return response;
    } catch (error) {
      console.error('Error changing password:', error);
      throw error;
    }
  }

  /**
   * Update notification settings
   * @param {Object} notificationSettings - Notification settings
   * @returns {Promise} Updated settings
   */
  async updateNotifications(notificationSettings) {
    return this.updateSettings(notificationSettings);
  }

  /**
   * Update theme settings
   * @param {string} theme - Theme name ('light' or 'dark')
   * @returns {Promise} Updated settings
   */
  async updateTheme(theme) {
    return this.updateSettings({ theme });
  }

  /**
   * Update language settings
   * @param {string} language - Language code
   * @returns {Promise} Updated settings
   */
  async updateLanguage(language) {
    return this.updateSettings({ language });
  }

  /**
   * Toggle API key
   * @param {boolean} enabled - Whether API key should be enabled
   * @returns {Promise} Updated settings
   */
  async toggleApiKey(enabled) {
    return this.updateSettings({ api_key_enabled: enabled });
  }
}

// Create and export a singleton instance
const settingsService = new SettingsService();
export default settingsService;