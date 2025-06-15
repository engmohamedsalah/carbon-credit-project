/**
 * Utility functions for handling API errors consistently across the application
 */

/**
 * Format API error response for display in UI
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