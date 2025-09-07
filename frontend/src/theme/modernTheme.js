/**
 * Modern Glassmorphism Theme for Carbon Credit Verification System
 * "Environmental Mission Control" Design System
 */
import { createTheme } from '@mui/material/styles';

// Modern Color Palette - Environmental Mission Control
export const MODERN_COLORS = {
  // Primary Theme - Deep Space Blue with Electric Green
  background: {
    primary: '#0A0E27',        // Deep Space Blue
    secondary: '#151B3B',      // Lighter Space Blue  
    tertiary: '#1E2449',       // Card backgrounds
    surface: 'rgba(255, 255, 255, 0.05)', // Glass surfaces
  },
  
  // Accent Colors
  accent: {
    primary: '#00FF88',        // Electric Green - main CTA
    secondary: '#64FFDA',      // Light Teal - secondary actions
    warning: '#FFB74D',        // Warm orange
    error: '#F48FB1',          // Soft pink-red
    success: '#81C784',        // Soft green
    info: '#64B5F6',           // Soft blue
  },
  
  // Text Colors
  text: {
    primary: '#FFFFFF',        // Pure white
    secondary: '#B0BEC5',      // Light grey
    tertiary: '#78909C',       // Darker grey
    disabled: '#546E7A',       // Disabled text
  },
  
  // Glassmorphism
  glass: {
    background: 'rgba(255, 255, 255, 0.1)',
    border: 'rgba(255, 255, 255, 0.2)',
    backdrop: 'blur(20px)',
    shadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
  },
  
  // Status Colors (modern versions)
  status: {
    pending: '#FFB74D',        // Warm amber
    verified: '#00FF88',       // Electric green
    rejected: '#F48FB1',       // Soft pink
    draft: '#78909C',          // Grey
    reviewing: '#64B5F6',      // Soft blue
    processing: '#64FFDA',     // Teal
  },
  
  // Interactive Elements
  interactive: {
    hover: 'rgba(255, 255, 255, 0.08)',
    pressed: 'rgba(255, 255, 255, 0.12)',
    focus: 'rgba(0, 255, 136, 0.2)',
    disabled: 'rgba(255, 255, 255, 0.05)',
  }
};

// Modern Typography System
export const MODERN_TYPOGRAPHY = {
  fontFamily: {
    primary: [
      'Inter',
      '-apple-system', 
      'BlinkMacSystemFont',
      'Segoe UI',
      'Roboto',
      'sans-serif'
    ].join(','),
    
    code: [
      'JetBrains Mono',
      'Fira Code', 
      'Monaco',
      'Consolas',
      'monospace'
    ].join(','),
    
    display: [
      'Inter',
      'system-ui',
      'sans-serif'
    ].join(',')
  },
  
  // Modern type scale
  scale: {
    h1: { fontSize: '3rem', fontWeight: 700, lineHeight: 1.2, letterSpacing: '-0.02em' },
    h2: { fontSize: '2.25rem', fontWeight: 600, lineHeight: 1.3, letterSpacing: '-0.01em' },
    h3: { fontSize: '1.875rem', fontWeight: 600, lineHeight: 1.4 },
    h4: { fontSize: '1.5rem', fontWeight: 500, lineHeight: 1.4 },
    h5: { fontSize: '1.25rem', fontWeight: 500, lineHeight: 1.5 },
    h6: { fontSize: '1.125rem', fontWeight: 500, lineHeight: 1.5 },
    body1: { fontSize: '1rem', fontWeight: 400, lineHeight: 1.6 },
    body2: { fontSize: '0.875rem', fontWeight: 400, lineHeight: 1.6 },
    caption: { fontSize: '0.75rem', fontWeight: 400, lineHeight: 1.5 },
    overline: { fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' },
  }
};

// Modern Spacing System (8pt grid)
export const MODERN_SPACING = {
  xs: 4,    // 4px
  sm: 8,    // 8px
  md: 16,   // 16px
  lg: 24,   // 24px
  xl: 32,   // 32px
  xxl: 48,  // 48px
  xxxl: 64, // 64px
};

// Modern Shadows with Glassmorphism
export const MODERN_SHADOWS = {
  none: 'none',
  glass: '0 8px 32px rgba(0, 0, 0, 0.3)',
  soft: '0 4px 20px rgba(0, 0, 0, 0.15)',
  medium: '0 8px 25px rgba(0, 0, 0, 0.25)',
  hard: '0 15px 35px rgba(0, 0, 0, 0.35)',
  glow: `0 0 20px rgba(0, 255, 136, 0.3)`,
  glowSoft: `0 0 10px rgba(0, 255, 136, 0.2)`,
};

// Modern Border Radius
export const MODERN_RADIUS = {
  xs: 4,
  sm: 8, 
  md: 12,
  lg: 16,
  xl: 24,
  round: '50%',
  pill: '9999px'
};

// Animation System
export const MODERN_ANIMATIONS = {
  duration: {
    fast: '150ms',
    normal: '250ms',
    slow: '350ms',
    slower: '500ms',
  },
  
  easing: {
    standard: 'cubic-bezier(0.4, 0, 0.2, 1)',
    decelerate: 'cubic-bezier(0, 0, 0.2, 1)',
    accelerate: 'cubic-bezier(0.4, 0, 1, 1)',
    sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',
  },
  
  keyframes: {
    fadeIn: {
      from: { opacity: 0 },
      to: { opacity: 1 }
    },
    slideUp: {
      from: { transform: 'translateY(20px)', opacity: 0 },
      to: { transform: 'translateY(0)', opacity: 1 }
    },
    pulse: {
      '0%': { transform: 'scale(1)' },
      '50%': { transform: 'scale(1.05)' },
      '100%': { transform: 'scale(1)' }
    },
    glow: {
      '0%': { boxShadow: '0 0 5px rgba(0, 255, 136, 0.2)' },
      '50%': { boxShadow: '0 0 20px rgba(0, 255, 136, 0.4)' },
      '100%': { boxShadow: '0 0 5px rgba(0, 255, 136, 0.2)' }
    }
  }
};

// Glassmorphism Component Styles
export const GLASS_COMPONENTS = {
  card: {
    background: MODERN_COLORS.glass.background,
    backdropFilter: MODERN_COLORS.glass.backdrop,
    border: `1px solid ${MODERN_COLORS.glass.border}`,
    borderRadius: MODERN_RADIUS.lg,
    boxShadow: MODERN_SHADOWS.glass,
    transition: `all ${MODERN_ANIMATIONS.duration.normal} ${MODERN_ANIMATIONS.easing.standard}`,
    
    '&:hover': {
      background: 'rgba(255, 255, 255, 0.15)',
      transform: 'translateY(-2px)',
      boxShadow: '0 12px 40px rgba(0, 0, 0, 0.4)',
    }
  },
  
  panel: {
    background: 'rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(15px)',
    border: `1px solid ${MODERN_COLORS.glass.border}`,
    borderRadius: MODERN_RADIUS.md,
    boxShadow: MODERN_SHADOWS.soft,
  },
  
  button: {
    background: MODERN_COLORS.accent.primary,
    color: MODERN_COLORS.background.primary,
    border: 'none',
    borderRadius: MODERN_RADIUS.sm,
    padding: `${MODERN_SPACING.sm}px ${MODERN_SPACING.md}px`,
    fontWeight: 600,
    fontSize: '0.875rem',
    textTransform: 'none',
    letterSpacing: '0.02em',
    boxShadow: MODERN_SHADOWS.glowSoft,
    transition: `all ${MODERN_ANIMATIONS.duration.fast} ${MODERN_ANIMATIONS.easing.standard}`,
    
    '&:hover': {
      background: '#00E07A',
      transform: 'translateY(-1px)',
      boxShadow: MODERN_SHADOWS.glow,
    }
  },
  
  input: {
    background: 'rgba(255, 255, 255, 0.08)',
    border: `1px solid ${MODERN_COLORS.glass.border}`,
    borderRadius: MODERN_RADIUS.sm,
    color: MODERN_COLORS.text.primary,
    backdropFilter: 'blur(10px)',
    
    '&:focus': {
      outline: 'none',
      borderColor: MODERN_COLORS.accent.primary,
      boxShadow: `0 0 0 2px rgba(0, 255, 136, 0.2)`,
    }
  }
};

// Create the Modern Material-UI Theme
const modernTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: MODERN_COLORS.accent.primary,
      light: MODERN_COLORS.accent.secondary,
      dark: '#00CC6A',
      contrastText: MODERN_COLORS.background.primary,
    },
    secondary: {
      main: MODERN_COLORS.accent.secondary,
      light: '#7DFFEA',
      dark: '#4DD0E1',
      contrastText: MODERN_COLORS.background.primary,
    },
    background: {
      default: MODERN_COLORS.background.primary,
      paper: MODERN_COLORS.background.secondary,
    },
    text: {
      primary: MODERN_COLORS.text.primary,
      secondary: MODERN_COLORS.text.secondary,
    },
    error: {
      main: MODERN_COLORS.accent.error,
    },
    warning: {
      main: MODERN_COLORS.accent.warning,
    },
    info: {
      main: MODERN_COLORS.accent.info,
    },
    success: {
      main: MODERN_COLORS.accent.success,
    },
  },
  
  typography: {
    fontFamily: MODERN_TYPOGRAPHY.fontFamily.primary,
    h1: MODERN_TYPOGRAPHY.scale.h1,
    h2: MODERN_TYPOGRAPHY.scale.h2,
    h3: MODERN_TYPOGRAPHY.scale.h3,
    h4: MODERN_TYPOGRAPHY.scale.h4,
    h5: MODERN_TYPOGRAPHY.scale.h5,
    h6: MODERN_TYPOGRAPHY.scale.h6,
    body1: MODERN_TYPOGRAPHY.scale.body1,
    body2: MODERN_TYPOGRAPHY.scale.body2,
    caption: MODERN_TYPOGRAPHY.scale.caption,
    overline: MODERN_TYPOGRAPHY.scale.overline,
  },
  
  shape: {
    borderRadius: MODERN_RADIUS.sm,
  },
  
  shadows: [
    'none',
    MODERN_SHADOWS.soft,
    MODERN_SHADOWS.medium,
    MODERN_SHADOWS.hard,
    MODERN_SHADOWS.glass,
    ...Array(20).fill(MODERN_SHADOWS.glass), // Fill remaining shadow levels
  ],
  
  components: {
    MuiCard: {
      styleOverrides: {
        root: GLASS_COMPONENTS.card,
      },
    },
    
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          ...GLASS_COMPONENTS.panel,
        },
      },
    },
    
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: MODERN_RADIUS.sm,
          ...MODERN_ANIMATIONS,
        },
        contained: GLASS_COMPONENTS.button,
      },
    },
    
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': GLASS_COMPONENTS.input,
        },
      },
    },
    
    MuiAppBar: {
      styleOverrides: {
        root: {
          ...GLASS_COMPONENTS.panel,
          background: 'rgba(10, 14, 39, 0.9)',
          backdropFilter: 'blur(20px)',
        },
      },
    },
    
    MuiDrawer: {
      styleOverrides: {
        paper: {
          ...GLASS_COMPONENTS.panel,
          background: 'rgba(10, 14, 39, 0.95)',
          backdropFilter: 'blur(25px)',
        },
      },
    },
  },
});

export default modernTheme;

// Export individual design tokens for use in components
export {
  MODERN_COLORS as colors,
  MODERN_TYPOGRAPHY as typography,
  MODERN_SPACING as spacing,
  MODERN_SHADOWS as shadows,
  MODERN_RADIUS as radius,
  MODERN_ANIMATIONS as animations,
  GLASS_COMPONENTS as glass,
};