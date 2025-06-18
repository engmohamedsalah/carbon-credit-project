/**
 * Date formatting utilities
 * Eliminates DRY violation from multiple formatDate implementations
 */

/**
 * Format date string with fallback handling
 * @param {string} dateString - ISO date string
 * @param {Object} options - Formatting options
 * @returns {string} Formatted date string
 */
export const formatDate = (dateString, options = {}) => {
  if (!dateString) return options.fallback || 'Not set';
  
  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
      console.warn('Invalid date:', dateString);
      return options.fallback || 'Invalid date';
    }
    
    return date.toLocaleDateString(
      options.locale || 'en-US',
      options.format || {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      }
    );
  } catch (error) {
    console.warn('Date formatting error:', error, dateString);
    return options.fallback || 'Invalid date';
  }
};

/**
 * Format date range with proper handling
 * @param {string} startDate - Start date string
 * @param {string} endDate - End date string
 * @param {string} separator - Separator between dates
 * @returns {string} Formatted date range
 */
export const formatDateRange = (startDate, endDate, separator = ' to ') => {
  const start = formatDate(startDate);
  const end = formatDate(endDate);
  return `${start}${separator}${end}`;
};

/**
 * Check if date string is valid
 * @param {string} dateString - Date string to validate
 * @returns {boolean} True if valid date
 */
export const isValidDate = (dateString) => {
  if (!dateString) return false;
  return !isNaN(new Date(dateString).getTime());
};

/**
 * Format date for form inputs (YYYY-MM-DD)
 * @param {string} dateString - Date string
 * @returns {string} Date formatted for input[type="date"]
 */
export const formatDateForInput = (dateString) => {
  if (!dateString) return '';
  try {
    const date = new Date(dateString);
    return date.toISOString().split('T')[0];
  } catch (error) {
    console.warn('Date input formatting error:', error);
    return '';
  }
};

/**
 * Get relative time (e.g., "2 days ago")
 * @param {string} dateString - Date string
 * @returns {string} Relative time string
 */
export const getRelativeTime = (dateString) => {
  if (!dateString) return 'Unknown';
  
  try {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    
    return `${Math.floor(diffDays / 365)} years ago`;
  } catch (error) {
    console.warn('Relative time calculation error:', error);
    return 'Unknown';
  }
};

/**
 * Common date formats
 */
export const DATE_FORMATS = {
  SHORT: { month: 'short', day: 'numeric', year: 'numeric' },
  LONG: { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' },
  COMPACT: { month: 'numeric', day: 'numeric', year: '2-digit' },
  ISO: 'iso'
}; 