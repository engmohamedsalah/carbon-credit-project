import React from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Stack
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  AutoAwesome as SparkleIcon,
  TrendingUp as ChartIcon
} from '@mui/icons-material';

const ModernEmptyState = ({ 
  variant = 'generate', // 'generate', 'compare', 'history'
  onAction,
  actionText = 'Get Started'
}) => {
  const getEmptyStateConfig = () => {
    switch (variant) {
      case 'generate':
        return {
          icon: <PsychologyIcon sx={{ fontSize: 64, color: 'primary.main' }} />,
          title: 'Ready to explain AI decisions?',
          description: 'Generate your first explanation to understand how our models make predictions about carbon credit verification.',
          features: [
            'SHAP feature importance analysis',
            'LIME local explanations', 
            'Business-friendly summaries',
            'Regulatory compliance notes'
          ]
        };
      case 'compare':
        return {
          icon: <ChartIcon sx={{ fontSize: 64, color: 'secondary.main' }} />,
          title: 'Compare explanations side-by-side',
          description: 'Select multiple explanations to analyze differences and validate consistency across methods.',
          features: [
            'Method comparison',
            'Confidence analysis',
            'Business impact assessment'
          ]
        };
      case 'history':
        return {
          icon: <SparkleIcon sx={{ fontSize: 64, color: 'success.main' }} />,
          title: 'Your explanation history',
          description: 'Generate explanations to build your audit trail and track AI decision patterns over time.',
          features: [
            'Complete audit trail',
            'Regulatory compliance',
            'Export capabilities'
          ]
        };
      default:
        return getEmptyStateConfig('generate');
    }
  };

  const config = getEmptyStateConfig();

  return (
    <Paper 
      sx={{ 
        p: 6, 
        textAlign: 'center',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        border: '1px solid',
        borderColor: 'divider'
      }}
    >
      <Stack spacing={3} alignItems="center">
        {config.icon}
        
        <Box>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
            {config.title}
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3, maxWidth: 500 }}>
            {config.description}
          </Typography>
        </Box>

        <Box sx={{ textAlign: 'left', maxWidth: 400 }}>
          <Typography variant="subtitle2" gutterBottom sx={{ textAlign: 'center', mb: 2 }}>
            What you'll get:
          </Typography>
          <Stack spacing={1}>
            {config.features.map((feature, index) => (
              <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ 
                  width: 6, 
                  height: 6, 
                  borderRadius: '50%', 
                  bgcolor: 'primary.main' 
                }} />
                <Typography variant="body2" color="text.secondary">
                  {feature}
                </Typography>
              </Box>
            ))}
          </Stack>
        </Box>

        <Button
          variant="contained"
          size="large"
          onClick={onAction}
          sx={{ 
            mt: 2,
            px: 4,
            py: 1.5,
            borderRadius: 3,
            textTransform: 'none',
            fontSize: '1.1rem'
          }}
        >
          {actionText}
        </Button>
      </Stack>
    </Paper>
  );
};

export default ModernEmptyState; 