/**
 * Flip Pipeline Cards - Medium cards that flip to reveal details
 * Accessible, responsive, and respects reduced motion.
 */
import React, { useMemo, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Grid,
  Chip,
  styled
} from '@mui/material';
import { colors, glass, radius, animations, spacing, shadows } from '../../theme/modernTheme';

// Icons
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import PsychologyIcon from '@mui/icons-material/Psychology';
import SensorsIcon from '@mui/icons-material/Sensors';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import PersonIcon from '@mui/icons-material/Person';
import SecurityIcon from '@mui/icons-material/Security';

const CardWrap = styled(Box)(({ theme }) => ({
  perspective: '1200px',
  width: '100%',
  height: 210,
}));

const CardInner = styled(Box, {
  shouldForwardProp: (prop) => !['flipped', 'status'].includes(prop),
})(({ theme, flipped, status }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  transformStyle: 'preserve-3d',
  transition: `transform ${animations.duration.slow} ${animations.easing.standard}`,
  transform: flipped ? 'rotateY(180deg)' : 'none',
  borderRadius: radius.lg,
  boxShadow: shadows.small,
  background: `linear-gradient(135deg, ${colors.background.secondary}70, ${colors.background.tertiary}70)`,
  border: `1px solid ${glass.border}`,
  overflow: 'hidden',

  // Status border cues
  ...(status === 'completed' && {
    borderColor: colors.accent.success + '60',
  }),
  ...(status === 'active' && {
    borderColor: colors.accent.primary + '80',
    boxShadow: `0 10px 30px ${colors.accent.primary}25`,
  }),

  '@media (prefers-reduced-motion: reduce)': {
    transition: 'none',
  }
}));

const Face = styled(Box)(({ theme }) => ({
  position: 'absolute',
  inset: 0,
  backfaceVisibility: 'hidden',
  WebkitBackfaceVisibility: 'hidden',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: spacing.md,
}));

const FaceBack = styled(Face)(({ theme }) => ({
  transform: 'rotateY(180deg)',
  background: `linear-gradient(135deg, ${colors.background.tertiary}90, ${colors.background.secondary}90)`,
}));

const IconBadge = styled(Box)(({ theme }) => ({
  width: 48,
  height: 48,
  borderRadius: 12,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: `linear-gradient(135deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
  color: colors.background.primary,
  boxShadow: `0 8px 20px ${colors.accent.primary}40`,
  marginBottom: spacing.sm,
}));

const FeaturePill = styled(Box)(({ theme }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '4px 10px',
  borderRadius: 999,
  border: `1px solid ${colors.accent.secondary}30`,
  color: colors.text.secondary,
  background: colors.background.tertiary + '40',
  fontSize: '0.75rem',
}));

const Container = styled(Box)(({ theme }) => ({
  borderRadius: radius.xl,
  padding: spacing.lg,
  border: `1px solid ${glass.border}`,
  background: `linear-gradient(135deg, ${colors.background.tertiary}40, ${colors.background.secondary}50)`,
}));

const FlipPipelineCards = ({ currentStep = 0, onGoToStep }) => {
  const [flippedIndex, setFlippedIndex] = useState(-1);

  const steps = useMemo(() => ([
    { id: 'create', label: 'Create Project', desc: 'Initialize project parameters', icon: <AccountTreeIcon />, features: ['Smart Forms','GPS Mapping','Auto Validation','Baseline Calc'] },
    { id: 'satellite', label: 'Satellite Data', desc: 'Download & process imagery', icon: <SatelliteAltIcon />, features: ['Sentinel-2','Global Coverage','Real-time','Time Series'] },
    { id: 'ai', label: 'AI Analysis', desc: 'ML model processing', icon: <PsychologyIcon />, features: ['Deep Learning','Change Detection','U-Net','Ensemble AI'] },
    { id: 'iot', label: 'IoT Validation', desc: 'Ground truth sensors', icon: <SensorsIcon />, features: ['Sensor Network','Climate Data','Real-time Sync','Ground Truth'] },
    { id: 'xai', label: 'XAI Insights', desc: 'Model explanations', icon: <AutoAwesomeIcon />, features: ['SHAP','LIME','Audit Trail','Transparency'] },
    { id: 'human', label: 'Human Review', desc: 'Expert validation', icon: <PersonIcon />, features: ['Expert Review','Compliance','Documentation','Quality Control'] },
    { id: 'blockchain', label: 'Blockchain', desc: 'NFT certification', icon: <SecurityIcon />, features: ['NFT Minting','Immutable','Tradeable','Global Market'] },
  ]), []);

  const statusOf = (index) => {
    if (index < currentStep) return 'completed';
    if (index === currentStep) return 'active';
    return 'upcoming';
  };

  const handleFlip = (index) => setFlippedIndex((prev) => (prev === index ? -1 : index));

  const handleKey = (e, index) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleFlip(index);
    }
  };

  return (
    <Container>
      <Typography variant="h6" sx={{ color: colors.text.primary, fontWeight: 700, mb: spacing.md }}>
        Verification Pipeline
      </Typography>
      <Grid container spacing={spacing.md}>
        {steps.map((step, index) => (
          <Grid key={step.id} item xs={12} sm={6} md={4}>
            <CardWrap>
              <CardInner
                flipped={flippedIndex === index}
                status={statusOf(index)}
                role="button"
                tabIndex={0}
                aria-pressed={flippedIndex === index}
                aria-label={`View details for ${step.label}`}
                onClick={() => handleFlip(index)}
                onKeyDown={(e) => handleKey(e, index)}
                sx={{ cursor: 'pointer' }}
              >
                {/* Front face */}
                <Face>
                  <IconBadge>{step.icon}</IconBadge>
                  <Typography variant="subtitle1" sx={{ color: colors.text.primary, fontWeight: 700 }}>
                    {step.label}
                  </Typography>
                  <Typography variant="body2" sx={{ color: colors.text.secondary, mt: 0.5, textAlign: 'center' }}>
                    {step.desc}
                  </Typography>
                  <Box sx={{ position: 'absolute', top: 10, right: 10 }}>
                    {statusOf(index) === 'completed' && (
                      <Chip size="small" label="Done" sx={{ backgroundColor: colors.accent.success, color: colors.background.primary }} />
                    )}
                    {statusOf(index) === 'active' && (
                      <Chip size="small" label="Now" sx={{ backgroundColor: colors.accent.primary, color: colors.background.primary }} />
                    )}
                  </Box>
                </Face>

                {/* Back face */}
                <FaceBack>
                  <Typography variant="subtitle1" sx={{ color: colors.accent.secondary, fontWeight: 700, mb: spacing.sm }}>
                    {step.label}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', justifyContent: 'center', mb: spacing.md }}>
                    {step.features.slice(0, 4).map((f) => (
                      <FeaturePill key={f}>{f}</FeaturePill>
                    ))}
                  </Box>
                  <Button
                    variant="contained"
                    onClick={(e) => { e.stopPropagation(); onGoToStep && onGoToStep(step); }}
                    sx={{
                      backgroundColor: colors.accent.primary,
                      color: colors.background.primary,
                      fontWeight: 700,
                    }}
                  >
                    Go to {step.label}
                  </Button>
                </FaceBack>
              </CardInner>
            </CardWrap>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default FlipPipelineCards;

