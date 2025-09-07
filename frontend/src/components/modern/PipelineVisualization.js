/**
 * Pipeline Visualization Component - Interactive Workflow Display
 * Part of the Environmental Mission Control design system
 */
import React, { useState } from 'react';
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
  ...props 
}) => {
  const [selectedStep, setSelectedStep] = useState(0); // First box selected by default
  
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
  const progress = ((currentStep + 1) / pipelineSteps.length) * 100;
  
  // Rich content for each step
  const getStepExplanation = (stepId) => {
    const explanations = {
      create: "Transform carbon sequestration concepts into verified, trackable projects with automated parameter validation and smart baseline calculations.",
      satellite: "Harness cutting-edge Sentinel-2 satellite imagery with AI-powered land cover analysis to monitor forest changes in real-time across vast territories.",
      ai: "Deploy advanced machine learning models including U-Net, ConvLSTM, and ensemble algorithms to precisely quantify carbon sequestration and forest change detection.",
      iot: "Integrate ground-based IoT sensors and environmental monitors to validate satellite data with real-world measurements and create a comprehensive verification network.",
      xai: "Leverage explainable AI with SHAP, LIME, and gradient analysis to ensure transparent, auditable decision-making that regulators and stakeholders can trust.",
      human: "Expert verification workflow where certified carbon analysts review AI recommendations and validate critical decisions through professional oversight.",
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
        { icon: "📋", text: "Audit Trail" },
        { icon: "🎯", text: "Transparency" }
      ],
      human: [
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
      <PipelineContainer>
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
              <ConnectorLine active={index < currentStep} />
            )}
          </React.Fragment>
        ))}
        
        {showProgress && (
          <ProgressBar progress={progress} />
        )}
      </PipelineContainer>
      
      {/* Ultra-Compact Animated Showcase */}
      <Box
        sx={{
          mt: spacing.xs,
          mx: -spacing.sm,
          background: `linear-gradient(135deg, ${colors.background.tertiary}90, ${colors.background.secondary}95)`,
          backdropFilter: 'blur(25px)',
          border: `1px solid ${colors.accent.secondary}30`,
          borderRadius: radius.xl,
          position: 'relative',
          overflow: 'hidden',
          minHeight: 80,
          transition: 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)',
          
          '&:hover': {
            transform: 'translateY(-4px) scale(1.02)',
            boxShadow: `0 20px 40px ${colors.accent.primary}25`,
            border: `1px solid ${colors.accent.secondary}50`,
          },
          
          // Multiple animated layers
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            right: 0,
            bottom: 0,
            background: `linear-gradient(90deg, transparent, ${colors.accent.primary}15, transparent)`,
            animation: 'sweepGlow 3s ease-in-out infinite',
            zIndex: 1,
          },
          
          '&::after': {
            content: '""',
            position: 'absolute',
            top: -2,
            left: -2,
            right: -2,
            bottom: -2,
            background: `linear-gradient(45deg, ${colors.accent.primary}20, ${colors.accent.secondary}20, ${colors.accent.primary}20)`,
            borderRadius: radius.xl,
            zIndex: -1,
            animation: 'borderPulse 4s ease-in-out infinite',
          },
          
          '@keyframes sweepGlow': {
            '0%': { left: '-100%' },
            '50%': { left: '100%' },
            '100%': { left: '100%' }
          },
          
          '@keyframes borderPulse': {
            '0%, 100%': { opacity: 0.3 },
            '50%': { opacity: 0.8 }
          },
          
          '@keyframes iconBounce': {
            '0%, 20%, 50%, 80%, 100%': { transform: 'translateY(0) scale(1)' },
            '40%': { transform: 'translateY(-6px) scale(1.1)' },
            '60%': { transform: 'translateY(-3px) scale(1.05)' }
          },
          
          '@keyframes textSlide': {
            '0%': { opacity: 0, transform: 'translateX(-20px)' },
            '100%': { opacity: 1, transform: 'translateX(0)' }
          },
          
          '@keyframes badgeFloat': {
            '0%, 100%': { transform: 'translateY(0)' },
            '50%': { transform: 'translateY(-2px)' }
          }
        }}
      >
        <Box sx={{ 
          position: 'relative', 
          zIndex: 2,
          p: spacing.md,
        }}>
          {/* Compact Header Row */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: spacing.sm, mb: spacing.sm }}>
            {/* Animated Icon */}
            <Box sx={{ 
              width: 36,
              height: 36,
              borderRadius: '10px',
              background: `linear-gradient(135deg, ${colors.accent.secondary}, ${colors.accent.primary})`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              animation: 'iconBounce 4s ease-in-out infinite',
              boxShadow: `0 4px 15px ${colors.accent.primary}40`,
              
              '& .MuiSvgIcon-root': {
                fontSize: '1.2rem',
                color: colors.background.primary,
              }
            }}>
              {pipelineSteps[selectedStep].icon}
            </Box>
            
            {/* Title & Description - Horizontal */}
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography sx={{ 
                color: colors.accent.secondary, 
                fontWeight: 700,
                fontSize: '1rem',
                mb: spacing.xs / 2,
                animation: 'textSlide 0.5s ease-out',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}>
                {pipelineSteps[selectedStep].label}
              </Typography>
              <Typography sx={{ 
                color: colors.text.secondary,
                fontSize: '0.8rem',
                lineHeight: 1.3,
                opacity: 0.85,
                animation: 'textSlide 0.6s ease-out',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
              }}>
                {getStepExplanation(pipelineSteps[selectedStep].id)}
              </Typography>
            </Box>
            
            {/* Pulsing Status */}
            <Box sx={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              backgroundColor: colors.accent.primary,
              animation: 'pulse 2s infinite',
              boxShadow: `0 0 8px ${colors.accent.primary}80`,
              flexShrink: 0,
            }} />
          </Box>
          
          {/* Compact Feature Pills */}
          <Box sx={{ 
            display: 'flex',
            flexWrap: 'wrap',
            gap: spacing.xs,
            justifyContent: 'space-between',
          }}>
            {getStepFeatures(pipelineSteps[selectedStep].id).map((feature, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: spacing.xs / 2,
                  px: spacing.sm,
                  py: spacing.xs / 2,
                  borderRadius: '20px',
                  background: `linear-gradient(90deg, ${colors.background.tertiary}60, ${colors.background.tertiary}40)`,
                  border: `1px solid ${colors.accent.secondary}20`,
                  minWidth: '22%',
                  animation: `badgeFloat ${3 + index * 0.5}s ease-in-out infinite`,
                  animationDelay: `${index * 0.1}s`,
                  
                  '&:hover': {
                    background: `linear-gradient(90deg, ${colors.accent.secondary}20, ${colors.accent.primary}15)`,
                    border: `1px solid ${colors.accent.secondary}40`,
                    transform: 'translateY(-1px) scale(1.05)',
                    boxShadow: `0 2px 8px ${colors.accent.primary}30`,
                  }
                }}
              >
                <Typography sx={{ fontSize: '0.65rem', lineHeight: 1 }}>
                  {feature.icon}
                </Typography>
                <Typography sx={{ 
                  color: colors.text.secondary,
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                }}>
                  {feature.text}
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
        
        {/* Animated Particles */}
        {[...Array(3)].map((_, i) => (
          <Box key={i} sx={{
            position: 'absolute',
            width: 4,
            height: 4,
            borderRadius: '50%',
            backgroundColor: colors.accent.secondary + '40',
            top: `${20 + i * 30}%`,
            right: `${10 + i * 15}%`,
            animation: `floatingOrb ${4 + i}s ease-in-out infinite`,
            animationDelay: `${i * 0.8}s`,
            zIndex: 0,
          }} />
        ))}
        
      </Box>
    </Box>
  );
};

export default PipelineVisualization;