/**
 * Compact Pipeline Visualization - Stepper with Expandable Details
 * Replaces the overwhelming 7-card layout with a clean, progressive disclosure approach
 */
import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Chip, 
  IconButton, 
  Collapse, 
  Stepper, 
  Step, 
  StepLabel, 
  StepContent,
  Card,
  CardContent,
  Grid,
  Fade,
  styled 
} from '@mui/material';
import { colors, glass, radius, animations, spacing } from '../../theme/modernTheme';

// Icons
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import PsychologyIcon from '@mui/icons-material/Psychology';
import SensorsIcon from '@mui/icons-material/Sensors';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import PersonIcon from '@mui/icons-material/Person';
import SecurityIcon from '@mui/icons-material/Security';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const PipelineContainer = styled(Box)(({ theme }) => ({
  background: `linear-gradient(135deg, ${colors.background.tertiary}40, ${colors.background.secondary}60)`,
  backdropFilter: 'blur(20px)',
  border: `1px solid ${glass.border}`,
  borderRadius: radius.xl,
  padding: spacing.lg,
  position: 'relative',
  overflow: 'hidden',
}));

const StepperStyled = styled(Stepper)(({ theme }) => ({
  '& .MuiStepLabel-root': {
    cursor: 'pointer',
    padding: spacing.sm,
    borderRadius: radius.md,
    transition: `all ${animations.duration.normal}`,
    
    '&:hover': {
      backgroundColor: colors.interactive.hover,
    }
  },
  
  '& .MuiStepLabel-label': {
    color: colors.text.secondary,
    fontWeight: 600,
    fontSize: '0.9rem',
    
    '&.Mui-active': {
      color: colors.accent.primary,
      fontWeight: 700,
    },
    
    '&.Mui-completed': {
      color: colors.text.primary,
    }
  },
  
  '& .MuiStepIcon-root': {
    color: colors.text.tertiary,
    
    '&.Mui-active': {
      color: colors.accent.primary,
      animation: 'pulse 2s infinite',
    },
    
    '&.Mui-completed': {
      color: colors.accent.success,
    }
  },
  
  '& .MuiStepConnector-line': {
    borderColor: colors.text.tertiary + '40',
  },
  
  '& .MuiStepConnector-root.Mui-active .MuiStepConnector-line': {
    borderColor: colors.accent.primary,
    borderWidth: 2,
    background: `linear-gradient(90deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
    height: 2,
  },
  
  '@keyframes pulse': {
    '0%': {
      boxShadow: `0 0 0 0 ${colors.accent.primary}40`,
    },
    '70%': {
      boxShadow: `0 0 0 10px transparent`,
    },
    '100%': {
      boxShadow: `0 0 0 0 transparent`,
    },
  }
}));

const DetailCard = styled(Card)(({ theme }) => ({
  background: `linear-gradient(135deg, ${colors.background.secondary}80, ${colors.background.tertiary}60)`,
  backdropFilter: 'blur(10px)',
  border: `1px solid ${colors.accent.primary}30`,
  borderRadius: radius.lg,
  marginTop: spacing.md,
  overflow: 'hidden',
  
  '&:hover': {
    border: `1px solid ${colors.accent.primary}50`,
    transform: 'translateY(-2px)',
    boxShadow: `0 8px 25px ${colors.accent.primary}20`,
  }
}));

const FeatureChip = styled(Chip)(({ theme }) => ({
  backgroundColor: colors.background.tertiary + '60',
  color: colors.text.primary,
  border: `1px solid ${colors.accent.secondary}30`,
  fontSize: '0.75rem',
  height: 28,
  
  '&:hover': {
    backgroundColor: colors.accent.secondary + '20',
    border: `1px solid ${colors.accent.secondary}60`,
  },
  
  '& .MuiChip-icon': {
    fontSize: '1rem',
  }
}));

const CompactPipelineVisualization = ({ 
  currentStep = 0,
  onStepClick,
  className,
  ...props 
}) => {
  const [activeStep, setActiveStep] = useState(currentStep);
  const [expandedStep, setExpandedStep] = useState(currentStep);

  const steps = [
    {
      id: 'create',
      label: 'Create Project',
      description: 'Initialize project parameters',
      icon: <AccountTreeIcon />,
      features: [
        { icon: '📋', label: 'Smart Forms' },
        { icon: '📍', label: 'GPS Mapping' },
        { icon: '⚡', label: 'Auto Validation' },
        { icon: '📊', label: 'Baseline Calc' }
      ],
      details: 'Transform carbon sequestration concepts into verified, trackable projects with automated parameter validation and smart baseline calculations.'
    },
    {
      id: 'satellite',
      label: 'Satellite Data',
      description: 'Download & process imagery',
      icon: <SatelliteAltIcon />,
      features: [
        { icon: '🛰️', label: 'Sentinel-2' },
        { icon: '🌍', label: 'Global Coverage' },
        { icon: '🔄', label: 'Real-time' },
        { icon: '📈', label: 'Time Series' }
      ],
      details: 'Harness cutting-edge Sentinel-2 satellite imagery with AI-powered land cover analysis to monitor forest changes in real-time.'
    },
    {
      id: 'ai',
      label: 'AI Analysis',
      description: 'ML model processing',
      icon: <PsychologyIcon />,
      features: [
        { icon: '🧠', label: 'Deep Learning' },
        { icon: '🎯', label: 'Change Detection' },
        { icon: '📊', label: 'U-Net Models' },
        { icon: '⚡', label: 'Ensemble AI' }
      ],
      details: 'Deploy advanced machine learning models including U-Net, ConvLSTM, and ensemble algorithms for precise carbon quantification.'
    },
    {
      id: 'iot',
      label: 'IoT Validation',
      description: 'Ground truth sensors',
      icon: <SensorsIcon />,
      features: [
        { icon: '📡', label: 'Sensor Network' },
        { icon: '🌡️', label: 'Climate Data' },
        { icon: '📱', label: 'Real-time Sync' },
        { icon: '✅', label: 'Ground Truth' }
      ],
      details: 'Integrate ground-based IoT sensors to validate satellite data with real-world measurements and comprehensive verification.'
    },
    {
      id: 'xai',
      label: 'XAI Insights',
      description: 'Model explanations',
      icon: <AutoAwesomeIcon />,
      features: [
        { icon: '🔍', label: 'SHAP Analysis' },
        { icon: '💡', label: 'LIME Insights' },
        { icon: '📋', label: 'Audit Trail' },
        { icon: '🎯', label: 'Transparency' }
      ],
      details: 'Leverage explainable AI with SHAP, LIME, and gradient analysis for transparent, auditable decision-making.'
    },
    {
      id: 'human',
      label: 'Human Review',
      description: 'Expert validation',
      icon: <PersonIcon />,
      features: [
        { icon: '👨‍💼', label: 'Expert Review' },
        { icon: '✅', label: 'Compliance' },
        { icon: '📝', label: 'Documentation' },
        { icon: '🔒', label: 'Quality Control' }
      ],
      details: 'Expert verification workflow where certified carbon analysts review AI recommendations through professional oversight.'
    },
    {
      id: 'blockchain',
      label: 'Blockchain',
      description: 'NFT certification',
      icon: <SecurityIcon />,
      features: [
        { icon: '🔗', label: 'NFT Minting' },
        { icon: '🛡️', label: 'Immutable' },
        { icon: '💳', label: 'Tradeable' },
        { icon: '🌐', label: 'Global Market' }
      ],
      details: 'Mint verified carbon credits as immutable NFT certificates on blockchain, creating transparent, tradeable digital assets.'
    }
  ];

  const handleStepClick = (stepIndex) => {
    setExpandedStep(expandedStep === stepIndex ? -1 : stepIndex);
    if (onStepClick) {
      onStepClick(steps[stepIndex], stepIndex);
    }
  };

  const getStepStatus = (stepIndex) => {
    if (stepIndex < activeStep) return 'completed';
    if (stepIndex === activeStep) return 'active';
    return 'upcoming';
  };

  return (
    <PipelineContainer className={className} {...props}>
      {/* Header */}
      <Box sx={{ mb: spacing.lg, textAlign: 'center' }}>
        <Typography variant="h6" sx={{
          color: colors.accent.primary,
          fontWeight: 700,
          mb: spacing.xs,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: spacing.sm
        }}>
          <PlayArrowIcon />
          Verification Pipeline
        </Typography>
        <Typography variant="body2" sx={{ color: colors.text.secondary }}>
          Step {activeStep + 1} of {steps.length} • Click any step to view details
        </Typography>
      </Box>

      {/* Compact Stepper */}
      <StepperStyled 
        activeStep={activeStep} 
        orientation="horizontal" 
        alternativeLabel
        sx={{ mb: spacing.lg }}
      >
        {steps.map((step, index) => (
          <Step key={step.id} completed={getStepStatus(index) === 'completed'}>
            <StepLabel 
              onClick={() => handleStepClick(index)}
              StepIconComponent={() => (
                <Box sx={{
                  width: 32,
                  height: 32,
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: getStepStatus(index) === 'completed' 
                    ? colors.accent.success 
                    : getStepStatus(index) === 'active'
                    ? colors.accent.primary
                    : colors.text.tertiary + '40',
                  color: colors.background.primary,
                  fontSize: '0.875rem',
                  transition: `all ${animations.duration.normal}`,
                }}>
                  {getStepStatus(index) === 'completed' ? (
                    <CheckCircleIcon sx={{ fontSize: '1.2rem' }} />
                  ) : (
                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 'bold' }}>
                      {index + 1}
                    </Typography>
                  )}
                </Box>
              )}
            >
              {step.label}
            </StepLabel>
          </Step>
        ))}
      </StepperStyled>

      {/* Expandable Details */}
      <Collapse in={expandedStep !== -1} timeout={500}>
        <Fade in={expandedStep !== -1} timeout={700}>
          <DetailCard>
            <CardContent sx={{ p: spacing.lg }}>
              {expandedStep !== -1 && steps[expandedStep] && (
                <>
                  {/* Step Header */}
                  <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: spacing.md,
                    mb: spacing.lg 
                  }}>
                    <Box sx={{
                      width: 48,
                      height: 48,
                      borderRadius: radius.md,
                      background: `linear-gradient(135deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: colors.background.primary,
                    }}>
                      {steps[expandedStep].icon}
                    </Box>
                    
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="h6" sx={{
                        color: colors.text.primary,
                        fontWeight: 700,
                        mb: spacing.xs / 2
                      }}>
                        {steps[expandedStep].label}
                      </Typography>
                      <Typography variant="body2" sx={{ color: colors.text.secondary }}>
                        {steps[expandedStep].description}
                      </Typography>
                    </Box>

                    <Chip 
                      label={getStepStatus(expandedStep).toUpperCase()}
                      size="small"
                      color={getStepStatus(expandedStep) === 'completed' ? 'success' : 'primary'}
                      sx={{ fontWeight: 600 }}
                    />
                  </Box>

                  {/* Description */}
                  <Typography 
                    variant="body1" 
                    sx={{ 
                      color: colors.text.primary,
                      lineHeight: 1.6,
                      mb: spacing.lg,
                      fontStyle: 'italic'
                    }}
                  >
                    {steps[expandedStep].details}
                  </Typography>

                  {/* Features Grid */}
                  <Typography variant="subtitle2" sx={{
                    color: colors.accent.secondary,
                    fontWeight: 700,
                    mb: spacing.md,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em'
                  }}>
                    Key Features
                  </Typography>
                  
                  <Grid container spacing={spacing.sm}>
                    {steps[expandedStep].features.map((feature, featureIndex) => (
                      <Grid item xs={6} sm={3} key={featureIndex}>
                        <FeatureChip
                          icon={<span>{feature.icon}</span>}
                          label={feature.label}
                          variant="outlined"
                          size="small"
                        />
                      </Grid>
                    ))}
                  </Grid>
                </>
              )}
            </CardContent>
          </DetailCard>
        </Fade>
      </Collapse>

      {/* Progress Bar */}
      <Box sx={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        height: 3,
        width: `${((activeStep + 1) / steps.length) * 100}%`,
        background: `linear-gradient(90deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
        transition: `width ${animations.duration.slow}`,
        borderRadius: `0 0 ${radius.xl}px ${radius.xl}px`,
      }} />
    </PipelineContainer>
  );
};

export default CompactPipelineVisualization;
