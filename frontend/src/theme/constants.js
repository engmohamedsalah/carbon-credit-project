/**
 * Theme constants for consistent styling across the application
 * Eliminates hardcoded values and provides centralized design tokens
 */

// Theme constants for the Carbon Credit Verification System

export const THEME_COLORS = {
  primary: '#2e7d32',      // Forest green - represents sustainability
  secondary: '#4caf50',    // Lighter green
  accent: '#81c784',       // Light green accent
  error: '#d32f2f',        // Red for errors
  warning: '#ff9800',      // Orange for warnings
  info: '#2196f3',         // Blue for information
  success: '#4caf50',      // Green for success
  
  // Role-specific colors
  roles: {
    admin: '#d32f2f',        // Red for admin
    verifier: '#2196f3',     // Blue for verifiers
    scientist: '#9c27b0',    // Purple for scientists
    developer: '#2e7d32',    // Green for developers
    investor: '#ff9800',     // Orange for investors
    regulatory: '#795548',   // Brown for regulatory
    default: '#757575'       // Grey for default/unknown roles
  },
  
  // Status colors
  status: {
    pending: '#ff9800',
    verified: '#4caf50',
    rejected: '#d32f2f',
    draft: '#757575',
    reviewing: '#2196f3'
  }
};

// Spacing scale based on Material-UI theme units
export const SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48
};

// Layout dimensions
export const LAYOUT = {
  DRAWER_WIDTH: 240,
  HEADER_HEIGHT: 64,
  CONTAINER_MAX_WIDTH: 'lg',
  FOOTER_HEIGHT: 60
};

// Component dimensions
export const DIMENSIONS = {
  MAP_HEIGHT: 400,
  DASHBOARD_CARD_HEIGHT: 240,
  TABLE_MIN_HEIGHT: 500,
  SEARCH_MIN_WIDTH: 300,
  MODAL_MAX_WIDTH: 600,
  SIDEBAR_WIDTH: 280,
  CHART_HEIGHT: 300
};

// Common spacing patterns
export const SPACING_PATTERNS = {
  CONTAINER_MARGIN: { mt: SPACING.lg, mb: SPACING.lg },
  SECTION_MARGIN: { mb: SPACING.md },
  SMALL_MARGIN: { mb: SPACING.sm },
  LARGE_MARGIN: { mb: SPACING.xl }
};

// Common style objects
export const COMMON_STYLES = {
  CONTAINER: {
    mt: SPACING.lg,
    mb: SPACING.lg
  },
  LOADING_CONTAINER: {
    mt: SPACING.lg,
    mb: SPACING.lg,
    textAlign: 'center'
  },
  PAPER: {
    p: SPACING.md
  },
  PAPER_SMALL: {
    p: SPACING.sm
  },
  CARD: {
    p: SPACING.sm,
    textAlign: 'center'
  },
  FULL_WIDTH: {
    width: '100%'
  },
  FULL_HEIGHT: {
    height: '100%'
  },
  FLEX_CENTER: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
  },
  FLEX_BETWEEN: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  FLEX_COLUMN: {
    display: 'flex',
    flexDirection: 'column'
  },
  FLEX_GROW: {
    flexGrow: 1
  }
};

// Border radius values
export const BORDER_RADIUS = {
  small: 4,
  medium: 8,
  large: 12,
  circle: '50%'
};

// Shadow levels
export const SHADOWS = {
  light: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
  medium: '0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23)',
  heavy: '0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23)'
};

// Z-index values
export const Z_INDEX = {
  modal: 1300,
  drawer: 1200,
  appBar: 1100,
  tooltip: 1500,
  snackbar: 1400
};

// Breakpoint helpers
export const BREAKPOINTS = {
  xs: 0,
  sm: 600,
  md: 960,
  lg: 1280,
  xl: 1920
};

// Animation durations
export const ANIMATION = {
  fast: '150ms',
  normal: '250ms',
  slow: '350ms',
  ease: 'cubic-bezier(0.4, 0, 0.2, 1)'
};

// Typography scale
export const TYPOGRAPHY = {
  fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  sizes: {
    h1: '2.5rem',
    h2: '2rem',
    h3: '1.75rem',
    h4: '1.5rem',
    h5: '1.25rem',
    h6: '1rem',
    body1: '1rem',
    body2: '0.875rem',
    caption: '0.75rem'
  }
};

// Professional role-based styling configuration
export const ROLE_STYLES = {
  adminBadge: {
    backgroundColor: THEME_COLORS.roles.admin,
    color: 'white',
    fontWeight: 600,
    fontSize: '0.75rem'
  },
  
  roleIndicator: {
    borderRadius: '4px',
    padding: '2px 6px',
    fontSize: '0.7rem',
    fontWeight: 500,
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  },
  
  menuItemActive: {
    backgroundColor: 'rgba(46, 125, 50, 0.1)',
    borderLeft: '3px solid #2e7d32',
    '&:hover': {
      backgroundColor: 'rgba(46, 125, 50, 0.2)'
    }
  }
};

const themeConstants = {
  THEME_COLORS,
  SPACING,
  LAYOUT,
  DIMENSIONS,
  SPACING_PATTERNS,
  COMMON_STYLES,
  BORDER_RADIUS,
  Z_INDEX,
  ANIMATION,
  BREAKPOINTS,
  TYPOGRAPHY,
  SHADOWS,
  ROLE_STYLES
};

export default themeConstants; 