/**
 * GlassCard Component - Modern Glassmorphism Card
 * Part of the Environmental Mission Control design system
 */
import React from 'react';
import { Card, CardContent, Box, styled } from '@mui/material';
import { colors, glass, radius, animations, shadows } from '../../theme/modernTheme';

const StyledGlassCard = styled(Card, {
  shouldForwardProp: (prop) => !['variant', 'interactive'].includes(prop),
})(({ theme, variant = 'default', interactive = true }) => ({
  background: variant === 'elevated' 
    ? 'rgba(255, 255, 255, 0.15)' 
    : glass.background,
  backdropFilter: glass.backdrop,
  border: `1px solid ${glass.border}`,
  borderRadius: radius.lg,
  boxShadow: variant === 'elevated' ? shadows.hard : shadows.glass,
  overflow: 'hidden',
  position: 'relative',
  transition: `all ${animations.duration.normal} ${animations.easing.standard}`,
  
  // Glass effect enhancement
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '1px',
    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
    opacity: 0.5,
  },
  
  ...(interactive && {
    cursor: 'pointer',
    '&:hover': {
      background: variant === 'elevated' 
        ? 'rgba(255, 255, 255, 0.2)' 
        : 'rgba(255, 255, 255, 0.15)',
      transform: 'translateY(-4px) scale(1.01)',
      boxShadow: variant === 'elevated' 
        ? '0 20px 50px rgba(0, 0, 0, 0.5)' 
        : '0 15px 45px rgba(0, 0, 0, 0.4)',
      
      '&::before': {
        opacity: 1,
      }
    },
    
    '&:active': {
      transform: 'translateY(-2px) scale(1.005)',
    }
  }),
  
  // Variant styles
  ...(variant === 'subtle' && {
    background: 'rgba(255, 255, 255, 0.05)',
    border: `1px solid rgba(255, 255, 255, 0.1)`,
    boxShadow: shadows.soft,
  }),
  
  ...(variant === 'glow' && {
    boxShadow: `${shadows.glass}, 0 0 20px rgba(0, 255, 136, 0.1)`,
    border: `1px solid rgba(0, 255, 136, 0.2)`,
    
    '&:hover': {
      boxShadow: `${shadows.hard}, 0 0 30px rgba(0, 255, 136, 0.2)`,
    }
  }),
}));

const StyledCardContent = styled(CardContent)(({ theme, padding = 'normal' }) => ({
  padding: padding === 'compact' 
    ? `${theme.spacing(2)} ${theme.spacing(3)}` 
    : padding === 'spacious'
    ? `${theme.spacing(4)} ${theme.spacing(5)}`
    : `${theme.spacing(3)} ${theme.spacing(4)}`,
    
  '&:last-child': {
    paddingBottom: padding === 'compact' 
      ? theme.spacing(2) 
      : padding === 'spacious'
      ? theme.spacing(4)
      : theme.spacing(3),
  }
}));

const GlassCard = ({ 
  children, 
  variant = 'default', 
  interactive = true, 
  padding = 'normal',
  onClick,
  sx = {},
  ...props 
}) => {
  return (
    <StyledGlassCard
      variant={variant}
      interactive={interactive}
      onClick={onClick}
      sx={sx}
      {...props}
    >
      <StyledCardContent padding={padding}>
        {children}
      </StyledCardContent>
    </StyledGlassCard>
  );
};

// Specialized glass card variants
export const GlassCardElevated = (props) => <GlassCard variant="elevated" {...props} />;
export const GlassCardSubtle = (props) => <GlassCard variant="subtle" {...props} />;
export const GlassCardGlow = (props) => <GlassCard variant="glow" {...props} />;
export const GlassCardStatic = (props) => <GlassCard interactive={false} {...props} />;

export default GlassCard;