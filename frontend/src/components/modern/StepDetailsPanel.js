/**
 * StepDetailsPanel - Fancy details panel for pipeline steps
 * Polished glass panel with header, features, and CTA.
 */
import React from 'react';
import { Box, Typography, Button, Chip, Grid, styled } from '@mui/material';
import { colors, spacing, radius, glass, shadows, animations } from '../../theme/modernTheme';

const Panel = styled(Box)(({ theme }) => ({
  marginTop: spacing.sm,
  padding: spacing.lg,
  borderRadius: radius.xl,
  position: 'relative',
  overflow: 'hidden',
  background: `linear-gradient(135deg, ${colors.background.tertiary}90, ${colors.background.secondary}95)`,
  border: `1px solid ${colors.accent.secondary}30`,
  boxShadow: `0 20px 40px ${colors.accent.primary}10`,
  transition: `transform ${animations.duration.normal} ${animations.easing.standard}, box-shadow ${animations.duration.normal} ${animations.easing.standard}`,

  // Ambient animated layers
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: '-120%',
    right: 0,
    bottom: 0,
    background: `linear-gradient(90deg, transparent, ${colors.accent.primary}14, transparent)`,
    animation: 'sweep 6s ease-in-out infinite',
    pointerEvents: 'none',
  },
  '&::after': {
    content: '""',
    position: 'absolute',
    inset: -2,
    borderRadius: radius.xl,
    background: `linear-gradient(45deg, ${colors.accent.primary}18, ${colors.accent.secondary}12, transparent)`,
    zIndex: -1,
    animation: 'borderPulse 5s ease-in-out infinite',
  },

  '@keyframes sweep': {
    '0%': { left: '-120%' },
    '50%': { left: '120%' },
    '100%': { left: '120%' }
  },
  '@keyframes borderPulse': {
    '0%, 100%': { opacity: 0.25 },
    '50%': { opacity: 0.6 }
  },

  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: `0 25px 50px ${colors.accent.primary}18`,
  },

  '@media (prefers-reduced-motion: reduce)': {
    transition: 'none',
    '&::before, &::after': { animation: 'none' },
  }
}));

const Header = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: spacing.md,
  marginBottom: spacing.md,
}));

const IconBadge = styled(Box)(({ theme }) => ({
  width: 52,
  height: 52,
  borderRadius: 14,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: `linear-gradient(135deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
  color: colors.background.primary,
  boxShadow: `0 8px 22px ${colors.accent.primary}35`,
  flexShrink: 0,
}));

const RightChips = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: spacing.sm,
  marginLeft: 'auto',
}));

const AccentBar = styled(Box)(({ theme }) => ({
  height: 2,
  width: 0,
  background: `linear-gradient(90deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
  borderRadius: 2,
  marginBottom: spacing.md,
  animation: 'growBar 800ms ease-out forwards',
  '@keyframes growBar': {
    'to': { width: 140 }
  },
  '@media (prefers-reduced-motion: reduce)': {
    animation: 'none',
    width: 140,
  }
}));

const FeatureTile = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: spacing.sm,
  padding: `${spacing.sm}px ${spacing.md}px`,
  borderRadius: radius.lg,
  border: `1px solid ${colors.accent.secondary}25`,
  background: `linear-gradient(135deg, ${colors.background.tertiary}60, ${colors.background.tertiary}40)`,
  transform: 'translateY(8px)',
  opacity: 0,
  animation: 'tileIn 600ms ease-out forwards',
  transition: 'transform 200ms ease, box-shadow 200ms ease',
  '&:hover': {
    transform: 'translateY(-2px) scale(1.02)',
    boxShadow: `0 8px 18px ${colors.accent.primary}18`,
  },
  '@keyframes tileIn': {
    'to': { transform: 'translateY(0)', opacity: 1 }
  },
  '@media (prefers-reduced-motion: reduce)': {
    animation: 'none',
    opacity: 1,
    transform: 'none',
  }
}));

const ShimmerButton = styled(Button)(({ theme }) => ({
  position: 'relative',
  overflow: 'hidden',
  background: `linear-gradient(135deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
  color: colors.background.primary,
  fontWeight: 800,
  boxShadow: `0 10px 24px ${colors.accent.primary}26`,
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: '-100%',
    width: '120%',
    height: '100%',
    background: `linear-gradient(90deg, transparent, ${colors.background.primary}40, transparent)`,
    transform: 'skewX(-20deg)',
    animation: 'shine 3s ease-in-out infinite',
  },
  '@keyframes shine': {
    '0%': { left: '-120%' },
    '60%': { left: '120%' },
    '100%': { left: '120%' }
  },
  '@media (prefers-reduced-motion: reduce)': {
    '&::after': { animation: 'none' },
  }
}));

const ContentRow = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: spacing.lg,
  alignItems: 'stretch',
  [theme.breakpoints.down('sm')]: {
    flexDirection: 'column',
  }
}));

const MediaBox = styled(Box)(({ theme }) => ({
  flex: '0 0 360px',
  maxWidth: 360,
  borderRadius: radius.lg,
  border: `1px solid ${colors.accent.secondary}25`,
  background: `linear-gradient(135deg, ${colors.background.secondary}70, ${colors.background.tertiary}60)`,
  boxShadow: `0 10px 24px ${colors.accent.primary}12`,
  overflow: 'hidden',
  transform: 'translateY(8px)',
  opacity: 0,
  animation: 'mediaIn 700ms ease-out forwards',
  '@keyframes mediaIn': { 'to': { transform: 'translateY(0)', opacity: 1 } },
  '@media (prefers-reduced-motion: reduce)': { animation: 'none', opacity: 1, transform: 'none' }
}));

const StepDetailsPanel = ({
  icon,
  title,
  explanation,
  features = [],
  stepNumber,
  totalSteps,
  status = 'selected',
  primaryLabel = 'Open',
  onPrimaryAction,
  media = null,
  mediaLabel = 'Preview',
}) => {
  const statusColor = status === 'completed' ? colors.accent.success : status === 'selected' ? colors.accent.primary : colors.text.secondary;

  return (
    <Panel aria-live="polite">
      <ContentRow>
        {/* Main Left Content */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          {/* Header */}
          <Header>
            <IconBadge>{icon}</IconBadge>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="h6" sx={{ color: colors.text.primary, fontWeight: 800, letterSpacing: '-0.01em' }}>
                {title}
              </Typography>
              <Typography variant="body2" sx={{ color: colors.text.secondary, mt: 0.5, lineHeight: 1.5 }}>
                {explanation}
              </Typography>
            </Box>
            <RightChips>
              <Chip size="small" label={`Step ${stepNumber} of ${totalSteps}`} sx={{ color: colors.text.secondary, backgroundColor: colors.interactive.hover }} />
              <Chip size="small" label={status.toUpperCase()} sx={{ color: colors.background.primary, backgroundColor: statusColor }} />
            </RightChips>
          </Header>

          {/* Animated underline */}
          <AccentBar />

          {/* Features with stagger */}
          <Grid container spacing={spacing.sm} sx={{ mb: spacing.md }}>
            {features.slice(0, 3).map((f, idx) => (
              <Grid key={`${f.text || f}-${idx}`} item xs={12} sm={4}>
                <FeatureTile sx={{ animationDelay: `${150 + idx * 120}ms` }}>
                  <Typography sx={{ fontSize: '1rem' }}>{f.icon || '•'}</Typography>
                  <Typography variant="body2" sx={{ color: colors.text.primary, fontWeight: 600 }}>
                    {f.text || f}
                  </Typography>
                </FeatureTile>
              </Grid>
            ))}
          </Grid>

          {/* Actions */}
          <Box sx={{ display: 'flex', gap: spacing.sm, flexWrap: 'wrap' }}>
            <ShimmerButton
              variant="contained"
              onClick={onPrimaryAction}
            >
              {primaryLabel}
            </ShimmerButton>
            <Button
              variant="outlined"
              sx={{
                borderColor: colors.accent.secondary,
                color: colors.accent.secondary,
                '&:hover': { borderColor: colors.accent.secondary }
              }}
            >
              Learn more
            </Button>
          </Box>
        </Box>

        {/* Right Media Preview */}
        {media && (
          <MediaBox aria-label={mediaLabel} role="img">
            {media}
          </MediaBox>
        )}
      </ContentRow>

      {/* Ambient orbs */}
      {[...Array(3)].map((_, i) => (
        <Box key={i} sx={{
          position: 'absolute',
          width: 6,
          height: 6,
          borderRadius: '50%',
          backgroundColor: colors.accent.secondary + '45',
          bottom: `${10 + i * 10}%`,
          right: `${8 + i * 14}%`,
          filter: 'blur(0.2px)',
          animation: `${3 + i}s floating ease-in-out infinite`,
          '@keyframes floating': {
            '0%, 100%': { transform: 'translateY(0)' },
            '50%': { transform: 'translateY(-6px)' }
          }
        }} />
      ))}
    </Panel>
  );
};

export default StepDetailsPanel;

