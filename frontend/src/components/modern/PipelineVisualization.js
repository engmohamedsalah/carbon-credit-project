/**
 * Pipeline Visualization Component - Interactive Workflow Display
 * Part of the Environmental Mission Control design system
 */
import React, { useState, useEffect } from 'react';
import { Box, Typography, Chip, IconButton, styled, Tooltip } from '@mui/material';
import { colors, glass, radius, animations, shadows, spacing } from '../../theme/modernTheme';

// Pipeline icons
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import PsychologyIcon from '@mui/icons-material/Psychology';
import SensorsIcon from '@mui/icons-material/Sensors';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import PersonIcon from '@mui/icons-material/Person';
import SecurityIcon from '@mui/icons-material/Security';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import InfoIcon from '@mui/icons-material/Info';
import StepDetailsPanel from './StepDetailsPanel';

const PipelineContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'row',
  alignItems: 'center',
  gap: spacing.sm,
  padding: spacing.lg,
  background: 'rgba(255, 255, 255, 0.02)',
  borderRadius: radius.xl,
  border: `1px solid ${glass.border}`,
  backdropFilter: 'blur(15px)',
  position: 'relative',
  overflow: 'hidden',
  
  // Flowing animation background
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.05), transparent)',
    animation: 'flow 3s ease-in-out infinite',
    zIndex: 0,
  },
  
  '@keyframes flow': {
    '0%': { transform: 'translateX(-100%)' },
    '100%': { transform: 'translateX(100%)' }
  },
  
  // Responsive design
  [theme.breakpoints.down('md')]: {
    flexDirection: 'column',
    gap: spacing.md,
  },
}));

const PipelineStep = styled(Box, {
  shouldForwardProp: (prop) => !['status', 'interactive'].includes(prop),
})(({ theme, status, interactive }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: spacing.md,
  borderRadius: radius.lg,
  background: status === 'completed' 
    ? 'rgba(0, 255, 136, 0.1)' 
    : status === 'active' 
    ? 'rgba(100, 255, 218, 0.1)'
    : status === 'error'
    ? 'rgba(244, 143, 177, 0.1)'
    : 'rgba(255, 255, 255, 0.05)',
  border: `1px solid ${
    status === 'completed' 
      ? colors.accent.primary + '40'
      : status === 'active' 
      ? colors.accent.secondary + '40'
      : status === 'error'
      ? colors.accent.error + '40'
      : glass.border
  }`,
  transition: `all ${animations.duration.normal} ${animations.easing.standard}`,
  position: 'relative',
  zIndex: 1,
  minWidth: 120,
  
  ...(interactive && {
    cursor: 'pointer',
    '&:hover': {
      transform: 'translateY(-4px)',
      background: status === 'completed' 
        ? 'rgba(0, 255, 136, 0.15)' 
        : status === 'active' 
        ? 'rgba(100, 255, 218, 0.15)'
        : 'rgba(255, 255, 255, 0.1)',
      boxShadow: shadows.medium,
    }
  }),
  
  // Glow effect for active/completed steps
  ...(status === 'active' && {
    boxShadow: `0 0 20px ${colors.accent.secondary}30`,
  }),
  
  ...(status === 'completed' && {
    boxShadow: `0 0 15px ${colors.accent.primary}20`,
  }),
}));

const StepIcon = styled(Box)(({ theme, status }) => ({
  width: 48,
  height: 48,
  borderRadius: '50%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: status === 'selected' 
    ? colors.accent.secondary  // Selected gets teal highlight
    : colors.accent.primary,   // All others get green (completed look)
  color: colors.background.primary,  // Always white text/icons
  marginBottom: spacing.sm,
  transition: `all ${animations.duration.normal} ${animations.easing.standard}`,
  
  '& .MuiSvgIcon-root': {
    fontSize: '1.5rem',
  },
  
  // Selected step gets pulse animation
  ...(status === 'selected' && {
    animation: 'pulse 2s infinite',
    boxShadow: `0 0 20px ${colors.accent.secondary}40`,
  }),
  
  // Completed steps get subtle glow
  ...(status === 'completed' && {
    boxShadow: `0 0 15px ${colors.accent.primary}20`,
  }),
  
  '@keyframes pulse': {
    '0%': {
      boxShadow: `0 0 15px ${colors.accent.secondary}40`,
    },
    '50%': {
      boxShadow: `0 0 25px ${colors.accent.secondary}60`,
    },
    '100%': {
      boxShadow: `0 0 15px ${colors.accent.secondary}40`,
    },
  }
}));

const StepLabel = styled(Typography)(({ theme, status }) => ({
  fontSize: '0.875rem',
  fontWeight: 600,
  color: status === 'selected' ? colors.accent.secondary : colors.text.primary,  // Selected gets teal text
  textAlign: 'center',
  marginBottom: spacing.xs,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
}));

const StepDescription = styled(Typography)(({ theme }) => ({
  fontSize: '0.75rem',
  color: colors.text.tertiary,
  textAlign: 'center',
  lineHeight: 1.4,
}));

const ConnectorLine = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'active',
})(({ theme, active }) => ({
  height: 2,
  flex: 1,
  background: active 
    ? `linear-gradient(90deg, ${colors.accent.primary}, ${colors.accent.secondary})`
    : `linear-gradient(90deg, ${colors.text.tertiary}40, ${colors.text.tertiary}20)`,
  borderRadius: 1,
  position: 'relative',
  transition: `all ${animations.duration.normal} ${animations.easing.standard}`,
  
  ...(active && {
    '&::after': {
      content: '""',
      position: 'absolute',
      top: -1,
      left: 0,
      height: 4,
      width: '100%',
      background: `linear-gradient(90deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
      borderRadius: 2,
      animation: 'flow-line 2s ease-in-out infinite',
    }
  }),
  
  '@keyframes flow-line': {
    '0%': { transform: 'scaleX(0)', transformOrigin: 'left' },
    '50%': { transform: 'scaleX(1)', transformOrigin: 'left' },
    '100%': { transform: 'scaleX(0)', transformOrigin: 'right' }
  },
  
  [theme.breakpoints.down('md')]: {
    height: 30,
    width: 2,
    flex: 'none',
  },
}));

const ProgressBar = styled(Box)(({ theme, progress }) => ({
  position: 'absolute',
  bottom: 0,
  left: 0,
  height: 3,
  width: `${progress}%`,
  background: `linear-gradient(90deg, ${colors.accent.primary}, ${colors.accent.secondary})`,
  transition: `width ${animations.duration.slow} ${animations.easing.standard}`,
  borderRadius: `0 0 ${radius.xl}px ${radius.xl}px`,
}));

const PipelineVisualization = ({ 
  steps = [],
  currentStep = 0,
  onStepClick,
  showProgress = true,
  interactive = true,
  autoPlay = false,
  autoPlayInterval = 5000,
  pauseOnHover = true,
  loop = true,
  syncProgressWithSelection = false,
  showGeneratedMedia = false,
  ...props 
}) => {
  const [selectedStep, setSelectedStep] = useState(currentStep || 0); // initialize from currentStep if provided
  const [isPaused, setIsPaused] = useState(false);
  
  const defaultSteps = [
    {
      id: 'create',
      label: 'Create Project',
      description: 'Initialize project parameters',
      icon: <AccountTreeIcon />,
    },
    {
      id: 'satellite',
      label: 'Satellite Data',
      description: 'Download & process imagery',
      icon: <SatelliteAltIcon />,
    },
    {
      id: 'ai',
      label: 'AI Analysis',
      description: 'ML model processing',
      icon: <PsychologyIcon />,
    },
    {
      id: 'iot',
      label: 'IoT Validation',
      description: 'Ground truth sensors',
      icon: <SensorsIcon />,
    },
    {
      id: 'xai',
      label: 'XAI Insights',
      description: 'Model explanations',
      icon: <AutoAwesomeIcon />,
    },
    {
      id: 'review',
      label: 'Human Review',
      description: 'Expert validation',
      icon: <PersonIcon />,
    },
    {
      id: 'blockchain',
      label: 'Blockchain',
      description: 'NFT certification',
      icon: <SecurityIcon />,
    },
  ];
  
  const pipelineSteps = steps.length > 0 ? steps : defaultSteps;
  const effectiveStep = syncProgressWithSelection ? selectedStep : currentStep;
  const progress = ((effectiveStep + 1) / pipelineSteps.length) * 100;
  
  // Auto-play slider behavior
  useEffect(() => {
    if (!autoPlay) return;
    const id = setInterval(() => {
      if (pauseOnHover && isPaused) return;
      setSelectedStep(prev => {
        const next = prev + 1;
        if (next >= pipelineSteps.length) {
          return loop ? 0 : prev;
        }
        return next;
      });
    }, autoPlayInterval);
    return () => clearInterval(id);
  }, [autoPlay, autoPlayInterval, isPaused, pauseOnHover, loop, pipelineSteps.length]);
  
  // Rich content for each step
  const getStepExplanation = (stepId) => {
    const explanations = {
      create: "Transform carbon sequestration concepts into verified, trackable projects with automated parameter validation and smart baseline calculations.",
      satellite: "Harness cutting-edge Sentinel-2 satellite imagery with AI-powered land cover analysis to monitor forest changes in real-time across vast territories.",
      ai: "Deploy advanced machine learning models including U-Net, ConvLSTM, and ensemble algorithms to precisely quantify carbon sequestration and forest change detection.",
      iot: "Integrate ground-based IoT sensors and environmental monitors to validate satellite data with real-world measurements and create a comprehensive verification network.",
      xai: "Leverage explainable AI with SHAP, LIME, and Integrated Gradients to ensure transparent, auditable decision-making that regulators and stakeholders can trust.",
      review: "Expert verification workflow where certified carbon analysts review AI recommendations and validate critical decisions through professional oversight.",
      blockchain: "Mint verified carbon credits as immutable NFT certificates on blockchain, creating transparent, tradeable, and fraud-proof digital assets."
    };
    return explanations[stepId] || "Advanced carbon credit verification process.";
  };
  
  const getStepFeatures = (stepId) => {
    const features = {
      create: [
        { icon: "📋", text: "Smart Forms" },
        { icon: "📍", text: "GPS Mapping" },
        { icon: "⚡", text: "Auto Validation" },
        { icon: "📊", text: "Baseline Calc" }
      ],
      satellite: [
        { icon: "🛰️", text: "Sentinel-2" },
        { icon: "🌍", text: "Global Coverage" },
        { icon: "🔄", text: "Real-time" },
        { icon: "📈", text: "Time Series" }
      ],
      ai: [
        { icon: "🧠", text: "Deep Learning" },
        { icon: "🎯", text: "Change Detection" },
        { icon: "📊", text: "U-Net Models" },
        { icon: "⚡", text: "Ensemble AI" }
      ],
      iot: [
        { icon: "📡", text: "Sensor Network" },
        { icon: "🌡️", text: "Climate Data" },
        { icon: "📱", text: "Real-time Sync" },
        { icon: "✅", text: "Ground Truth" }
      ],
      xai: [
        { icon: "🔍", text: "SHAP Analysis" },
        { icon: "💡", text: "LIME Insights" },
        { icon: "🌈", text: "Integrated Gradients" },
        { icon: "📋", text: "Audit Trail" }
      ],
      review: [
        { icon: "👨‍💼", text: "Expert Review" },
        { icon: "✅", text: "Compliance" },
        { icon: "📝", text: "Documentation" },
        { icon: "🔒", text: "Quality Control" }
      ],
      blockchain: [
        { icon: "🔗", text: "NFT Minting" },
        { icon: "🛡️", text: "Immutable" },
        { icon: "💳", text: "Tradeable" },
        { icon: "🌐", text: "Global Market" }
      ]
    };
    return features[stepId] || [
      { icon: "⚡", text: "Fast Processing" },
      { icon: "🔒", text: "Secure" },
      { icon: "📊", text: "Analytics" },
      { icon: "✅", text: "Verified" }
    ];
  };

  // Lightweight per-step media previews
  const getStepMedia = (stepId) => {
    const baseBox = (children, extra = {}) => (
      <Box sx={{
        width: '100%',
        height: 180,
        borderRadius: radius.lg,
        position: 'relative',
        overflow: 'hidden',
        p: spacing.md,
        background: `linear-gradient(135deg, ${colors.background.tertiary}80, ${colors.background.secondary}70)`,
        border: `1px solid ${colors.accent.secondary}30`,
        ...extra,
      }}>
        {children}
      </Box>
    );

    switch (stepId) {
      case 'create':
        return baseBox(
          <>
            {[...Array(5)].map((_, i) => (
              <Box key={i} sx={{
                position: 'absolute',
                top: 20 + i * 28,
                left: 20,
                right: 20,
                height: 10,
                borderRadius: 6,
                background: colors.interactive.hover,
              }} />
            ))}
            <AccountTreeIcon sx={{ position: 'absolute', bottom: 12, right: 12, color: colors.accent.primary }} />
          </>
        );
      case 'satellite':
        return baseBox(
          <>
            {[...Array(6)].map((_, r) => (
              [...Array(10)].map((_, c) => (
                <Box key={`${r}-${c}`} sx={{
                  position: 'absolute',
                  top: 12 + r * 26,
                  left: 12 + c * 26,
                  width: 20,
                  height: 20,
                  borderRadius: 3,
                  backgroundColor: (r + c) % 2 === 0 ? colors.accent.primary + '25' : colors.accent.secondary + '20',
                }} />
              ))
            ))}
            <SatelliteAltIcon sx={{ position: 'absolute', bottom: 12, right: 12, color: colors.accent.secondary }} />
          </>
        );
      case 'ai':
        return baseBox(
          <Box sx={{ display: 'flex', alignItems: 'flex-end', height: '100%', gap: 1 }}>
            {[40, 75, 55, 90, 65, 80].map((h, i) => (
              <Box key={i} sx={{ width: 16, height: `${h}%`, background: `linear-gradient(180deg, ${colors.accent.info}99, ${colors.accent.info}33)`, borderRadius: 2 }} />
            ))}
            <PsychologyIcon sx={{ position: 'absolute', bottom: 12, right: 12, color: colors.accent.info }} />
          </Box>
        );
      case 'iot':
        return baseBox(
          <>
            {[...Array(4)].map((_, i) => (
              <Box key={i} sx={{ position: 'absolute', left: 16, right: 16, top: 30 + i * 28, height: 2, background: colors.accent.secondary + '60' }} />
            ))}
            <SensorsIcon sx={{ position: 'absolute', bottom: 12, right: 12, color: colors.accent.secondary }} />
          </>
        );
      case 'xai':
        return baseBox(
          <>
            {[...Array(4)].map((_, r) => (
              [...Array(6)].map((_, c) => (
                <Box key={`${r}-${c}`} sx={{
                  position: 'absolute',
                  top: 18 + r * 36,
                  left: 18 + c * 36,
                  width: 22,
                  height: 22,
                  borderRadius: 4,
                  background: `linear-gradient(135deg, ${colors.accent.primary}${(30 + (r + c) * 10).toString()} , ${colors.accent.secondary}30)`,
                  opacity: 0.7,
                }} />
              ))
            ))}
            <AutoAwesomeIcon sx={{ position: 'absolute', bottom: 12, right: 12, color: colors.accent.info }} />
          </>
        );
      case 'review':
        return baseBox(
          <>
            {[...Array(3)].map((_, i) => (
              <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1, position: 'absolute', left: 16, top: 24 + i * 40 }}>
                <CheckCircleIcon sx={{ fontSize: 18, color: colors.accent.success }} />
                <Box sx={{ width: 220, height: 8, borderRadius: 6, background: colors.interactive.hover }} />
              </Box>
            ))}
            <PersonIcon sx={{ position: 'absolute', bottom: 12, right: 12, color: colors.text.secondary }} />
          </>
        );
      case 'blockchain':
        return baseBox(
          <>
            {[...Array(8)].map((_, i) => (
              <Box key={i} sx={{ position: 'absolute', left: 16 + i * 32, top: 80 + (i % 2 ? 8 : -8), width: 18, height: 18, borderRadius: '50%', border: `2px solid ${colors.accent.success}55` }} />
            ))}
            <SecurityIcon sx={{ position: 'absolute', bottom: 12, right: 12, color: colors.accent.success }} />
          </>
        );
      default:
        return baseBox(<Box />);
    }
  };
  
  const getStepStatus = (index) => {
    // All steps now show as completed, only selection state differs
    return selectedStep === index ? 'selected' : 'completed';
  };
  
  const handleStepClick = (step, index) => {
    if (!interactive) return;
    
    setSelectedStep(index); // Always set selected step
    if (onStepClick) {
      onStepClick(step, index);
    }
  };

  return (
    <Box position="relative" {...props}>
      <PipelineContainer
        onMouseEnter={pauseOnHover ? () => setIsPaused(true) : undefined}
        onMouseLeave={pauseOnHover ? () => setIsPaused(false) : undefined}
      >
        {pipelineSteps.map((step, index) => (
          <React.Fragment key={step.id}>
            <Tooltip 
              title={`${step.label}: ${step.description}`}
              placement="top"
            >
              <PipelineStep
                status={getStepStatus(index)}
                interactive={interactive}
                onClick={() => handleStepClick(step, index)}
              >
                <StepIcon status={getStepStatus(index)}>
                  {step.icon}
                </StepIcon>
                
                <StepLabel status={getStepStatus(index)}>
                  {step.label}
                </StepLabel>
                
                <StepDescription>
                  {step.description}
                </StepDescription>
                
                {step.status && (
                  <Chip
                    label={step.status}
                    size="small"
                    sx={{
                      mt: 0.5,
                      fontSize: '0.65rem',
                      height: 20,
                      backgroundColor: colors.interactive.hover,
                      color: colors.text.secondary,
                    }}
                  />
                )}
              </PipelineStep>
            </Tooltip>
            
            {index < pipelineSteps.length - 1 && (
              <ConnectorLine active={index < effectiveStep} />
            )}
          </React.Fragment>
        ))}
        
        {showProgress && (
          <ProgressBar progress={progress} />
        )}
      </PipelineContainer>
      
      {/* Fancy Step Details Panel */}
      <StepDetailsPanel
        key={selectedStep} // Force re-mount to replay animations
        icon={pipelineSteps[selectedStep].icon}
        title={pipelineSteps[selectedStep].label}
        explanation={getStepExplanation(pipelineSteps[selectedStep].id)}
        features={getStepFeatures(pipelineSteps[selectedStep].id)}
        stepNumber={selectedStep + 1}
        totalSteps={pipelineSteps.length}
        status={getStepStatus(selectedStep)}
        primaryLabel={`Go to ${pipelineSteps[selectedStep].label}`}
        onPrimaryAction={() => {
          if (onStepClick) onStepClick(pipelineSteps[selectedStep], selectedStep);
        }}
        media={pipelineSteps[selectedStep].media ? pipelineSteps[selectedStep].media : (showGeneratedMedia ? getStepMedia(pipelineSteps[selectedStep].id) : null)}
        mediaLabel={`${pipelineSteps[selectedStep].label} preview`}
      />
    </Box>
  );
};

export default PipelineVisualization;