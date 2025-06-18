/**
 * Theme constants for consistent styling across the application
 * Eliminates hardcoded values and provides centralized design tokens
 */

// Spacing scale based on Material-UI theme units
export const SPACING = {
  xs: 1,    // 8px
  sm: 2,    // 16px
  md: 3,    // 24px
  lg: 4,    // 32px
  xl: 5,    // 40px
  xxl: 6    // 48px
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
  small: '0 1px 3px rgba(0,0,0,0.12)',
  medium: '0 2px 6px rgba(0,0,0,0.15)',
  large: '0 4px 12px rgba(0,0,0,0.18)'
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
  xs: '(max-width: 599px)',
  sm: '(min-width: 600px)',
  md: '(min-width: 960px)',
  lg: '(min-width: 1280px)',
  xl: '(min-width: 1920px)'
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
  fontSize: {
    small: '0.875rem',
    normal: '1rem',
    large: '1.125rem',
    xlarge: '1.25rem'
  },
  lineHeight: {
    tight: 1.25,
    normal: 1.5,
    loose: 1.75
  }
};

// Color palette extensions
export const COLORS = {
  gradients: {
    primary: 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)',
    secondary: 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)',
    success: 'linear-gradient(135deg, #4caf50 0%, #2e7d32 100%)',
    warning: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
    error: 'linear-gradient(135deg, #f44336 0%, #d32f2f 100%)'
  },
  overlay: {
    light: 'rgba(255, 255, 255, 0.8)',
    medium: 'rgba(255, 255, 255, 0.9)',
    dark: 'rgba(0, 0, 0, 0.5)'
  }
}; 