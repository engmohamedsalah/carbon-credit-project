/**
 * GlassPanel Component - Modern Glassmorphism Panel for Dashboard Layout
 * Part of the Environmental Mission Control design system
 */
import React from 'react';
import { Box, Typography, IconButton, styled } from '@mui/material';
import { colors, glass, radius, animations, shadows, spacing } from '../../theme/modernTheme';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

const StyledGlassPanel = styled(Box)(({ theme, size = 'medium', blur = 'normal' }) => ({
  background: glass.background,
  backdropFilter: blur === 'heavy' ? 'blur(25px)' : blur === 'light' ? 'blur(10px)' : glass.backdrop,
  border: `1px solid ${glass.border}`,
  borderRadius: size === 'large' ? radius.xl : size === 'small' ? radius.sm : radius.lg,
  boxShadow: shadows.glass,
  position: 'relative',
  overflow: 'hidden',
  transition: `all ${animations.duration.normal} ${animations.easing.standard}`,
  
  // Subtle glass reflection effect
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: '-100%',
    width: '100%',
    height: '100%',
    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)',
    transition: `left ${animations.duration.slower} ${animations.easing.standard}`,
  },
  
  '&:hover::after': {
    left: '100%',
  },
  
  // Size variants
  ...(size === 'small' && {
    padding: `${spacing.sm}px ${spacing.md}px`,
  }),
  
  ...(size === 'medium' && {
    padding: `${spacing.md}px ${spacing.lg}px`,
  }),
  
  ...(size === 'large' && {
    padding: `${spacing.lg}px ${spacing.xl}px`,
  }),
}));

const PanelHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  marginBottom: spacing.md,
  paddingBottom: spacing.sm,
  borderBottom: `1px solid ${glass.border}`,
  position: 'relative',
  zIndex: 1,
}));

const PanelTitle = styled(Typography)(({ theme }) => ({
  color: colors.text.primary,
  fontWeight: 600,
  fontSize: '1.125rem',
  letterSpacing: '0.01em',
  display: 'flex',
  alignItems: 'center',
  gap: spacing.sm,
}));

const PanelSubtitle = styled(Typography)(({ theme }) => ({
  color: colors.text.secondary,
  fontSize: '0.875rem',
  marginTop: 2,
  fontWeight: 400,
}));

const PanelContent = styled(Box, {
  shouldForwardProp: (prop) => !['collapsible', 'collapsed'].includes(prop),
})(({ theme, collapsible, collapsed }) => ({
  position: 'relative',
  zIndex: 1,
  ...(collapsible && {
    overflow: 'hidden',
    transition: `all ${animations.duration.normal} ${animations.easing.standard}`,
    maxHeight: collapsed ? 0 : '1000px',
    opacity: collapsed ? 0 : 1,
  }),
}));

const StatusIndicator = styled(Box)(({ status }) => ({
  width: 8,
  height: 8,
  borderRadius: '50%',
  backgroundColor: status === 'active' ? colors.accent.primary :
                   status === 'warning' ? colors.accent.warning :
                   status === 'error' ? colors.accent.error :
                   status === 'info' ? colors.accent.info :
                   colors.text.tertiary,
  boxShadow: `0 0 10px ${status === 'active' ? colors.accent.primary + '60' :
                         status === 'warning' ? colors.accent.warning + '60' :
                         status === 'error' ? colors.accent.error + '60' :
                         status === 'info' ? colors.accent.info + '60' :
                         'transparent'}`,
  animation: status === 'active' ? 'pulse 2s infinite' : 'none',
  
  '@keyframes pulse': {
    '0%': {
      boxShadow: `0 0 10px ${colors.accent.primary}60`,
    },
    '50%': {
      boxShadow: `0 0 20px ${colors.accent.primary}80`,
    },
    '100%': {
      boxShadow: `0 0 10px ${colors.accent.primary}60`,
    },
  }
}));

const GlassPanel = ({ 
  title, 
  subtitle,
  icon,
  children, 
  size = 'medium',
  blur = 'normal',
  status,
  collapsible = false,
  defaultExpanded = true,
  onToggle,
  actions,
  sx = {},
  ...props 
}) => {
  const [collapsed, setCollapsed] = React.useState(!defaultExpanded);
  
  const handleToggle = () => {
    setCollapsed(!collapsed);
    if (onToggle) {
      onToggle(!collapsed);
    }
  };

  return (
    <StyledGlassPanel
      size={size}
      blur={blur}
      sx={sx}
      {...props}
    >
      {(title || collapsible || actions) && (
        <PanelHeader>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {status && <StatusIndicator status={status} />}
            <Box>
              <PanelTitle>
                {icon && icon}
                {title}
              </PanelTitle>
              {subtitle && <PanelSubtitle>{subtitle}</PanelSubtitle>}
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {actions}
            {collapsible && (
              <IconButton
                onClick={handleToggle}
                size="small"
                sx={{ 
                  color: colors.text.secondary,
                  '&:hover': { 
                    color: colors.text.primary,
                    backgroundColor: colors.interactive.hover,
                  }
                }}
              >
                {collapsed ? <ExpandMoreIcon /> : <ExpandLessIcon />}
              </IconButton>
            )}
          </Box>
        </PanelHeader>
      )}
      
      <PanelContent 
        collapsible={collapsible} 
        collapsed={collapsible && collapsed}
      >
        {children}
      </PanelContent>
    </StyledGlassPanel>
  );
};

// Specialized panel variants
export const GlassPanelCompact = (props) => <GlassPanel size="small" {...props} />;
export const GlassPanelLarge = (props) => <GlassPanel size="large" {...props} />;
export const GlassPanelBlur = (props) => <GlassPanel blur="heavy" {...props} />;

export default GlassPanel;