/**
 * Error handling utilities for the Carbon Credit Verification application
 * Includes filtering for Chrome extension errors and proper error logging
 */

// Immediately setup basic error suppression
if (typeof window !== 'undefined') {
  const originalConsoleError = console.error;
  console.error = (...args) => {
    const message = args[0];
    if (typeof message === 'string' && message.includes('runtime.lastError')) {
      return; // Suppress immediately
    }
    originalConsoleError.apply(console, args);
  };
}

// List of Chrome extension error patterns to filter out
const EXTENSION_ERROR_PATTERNS = [
  'runtime.lastError',
  'Extension context invalidated',
  'message channel closed',
  'Could not establish connection',
  'The message port closed before a response was received',
  'A listener indicated an asynchronous response by returning true, but the message channel closed',
  'Unchecked runtime.lastError',
  'chrome-extension://',
  'moz-extension://',
  'safari-extension://',
  'chrome.runtime',
  'browser.runtime'
];

/**
 * Check if an error is related to Chrome extensions
 * @param {Error|string} error - The error to check
 * @returns {boolean} - True if it's an extension error
 */
export const isExtensionError = (error) => {
  const message = typeof error === 'string' ? error : error?.message || '';
  return EXTENSION_ERROR_PATTERNS.some(pattern => 
    message.toLowerCase().includes(pattern.toLowerCase())
  );
};

/**
 * Enhanced console.error that filters out extension errors
 * @param {...any} args - Arguments to log
 */
export const logError = (...args) => {
  const message = args[0];
  if (!isExtensionError(message)) {
    console.error(...args);
  }
};

/**
 * Setup global error handlers to suppress extension errors
 */
export const setupErrorHandlers = () => {
  // Handle unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    if (isExtensionError(event.reason)) {
      event.preventDefault(); // Suppress extension errors
      return;
    }
    console.error('Unhandled promise rejection:', event.reason);
  });

  // Handle global errors
  window.addEventListener('error', (event) => {
    if (isExtensionError(event.error || event.message)) {
      event.preventDefault(); // Suppress extension errors
      return;
    }
    console.error('Global error:', event.error || event.message);
  });

  // Override console.error to filter extension errors
  const originalConsoleError = console.error;
  console.error = (...args) => {
    // Check all arguments for extension errors
    const hasExtensionError = args.some(arg => isExtensionError(arg));
    if (!hasExtensionError) {
      originalConsoleError.apply(console, args);
    }
  };

  // Override console.warn to filter extension warnings
  const originalConsoleWarn = console.warn;
  console.warn = (...args) => {
    const hasExtensionError = args.some(arg => isExtensionError(arg));
    if (!hasExtensionError) {
      originalConsoleWarn.apply(console, args);
    }
  };
};

/**
 * Format error messages for user display
 * @param {Error|string} error - The error to format
 * @returns {string} - User-friendly error message
 */
export const formatErrorMessage = (error) => {
  if (isExtensionError(error)) {
    return null; // Don't show extension errors to users
  }

  if (typeof error === 'string') {
    return error;
  }

  if (error?.response?.data?.detail) {
    return error.response.data.detail;
  }

  if (error?.message) {
    return error.message;
  }

  return 'An unexpected error occurred. Please try again.';
};

/**
 * Format API error response for display in UI (backward compatibility)
 * @param {Object} action - Redux action with payload and error
 * @param {string} defaultMessage - Default error message if parsing fails
 * @returns {string} Formatted error message
 */
export const formatApiError = (action, defaultMessage = 'An error occurred') => {
  // Handle different error response formats
  if (action.payload?.detail) {
    if (Array.isArray(action.payload.detail)) {
      // Validation errors array from FastAPI
      return action.payload.detail.map(err => err.msg).join(', ');
    } else if (typeof action.payload.detail === 'string') {
      // Simple string error
      return action.payload.detail;
    } else {
      return defaultMessage;
    }
  } else if (action.error?.message) {
    // Network or other errors
    return action.error.message;
  } else {
    return defaultMessage;
  }
};

/**
 * Extract field-specific errors from validation error array
 * @param {Array} errors - Array of validation errors
 * @returns {Object} Object with field names as keys and error messages as values
 */
export const extractFieldErrors = (errors) => {
  if (!Array.isArray(errors)) return {};
  
  const fieldErrors = {};
  errors.forEach(error => {
    if (error.loc && error.loc.length > 0) {
      const fieldName = error.loc[error.loc.length - 1]; // Get the last part of the location path
      fieldErrors[fieldName] = error.msg;
    }
  });
  
  return fieldErrors;
};

/**
 * Handle async operations with proper error catching
 * @param {Function} asyncFn - The async function to execute
 * @param {Function} onError - Error handler function
 * @returns {Promise} - The promise result
 */
export const handleAsync = async (asyncFn, onError = null) => {
  try {
    return await asyncFn();
  } catch (error) {
    const errorMessage = formatErrorMessage(error);
    if (errorMessage && onError) {
      onError(errorMessage);
    } else if (errorMessage) {
      console.error('Async operation failed:', errorMessage);
    }
    throw error;
  }
};

/**
 * Debounced error logger to prevent spam
 */
const errorLogHistory = new Set();
const ERROR_LOG_TIMEOUT = 5000; // 5 seconds

export const logErrorOnce = (error) => {
  const message = formatErrorMessage(error);
  if (message && !errorLogHistory.has(message)) {
    errorLogHistory.add(message);
    console.error(message);
    
    // Clear from history after timeout
    setTimeout(() => {
      errorLogHistory.delete(message);
    }, ERROR_LOG_TIMEOUT);
  }
};

// Default export with all utilities
const errorUtils = {
  isExtensionError,
  logError,
  setupErrorHandlers,
  formatErrorMessage,
  formatApiError,
  extractFieldErrors,
  handleAsync,
  logErrorOnce
};

export default errorUtils; 