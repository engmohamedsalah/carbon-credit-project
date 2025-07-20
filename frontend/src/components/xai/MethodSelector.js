import React from 'react';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Chip,
  Paper,
  useTheme,
  useMediaQuery,
  Tooltip
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  Lightbulb as LightbulbIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material';

const MethodSelector = ({ 
  methods = [], 
  selectedMethod, 
  onMethodChange, 
  loading = false,
  variant = 'dropdown', // 'dropdown' or 'cards'
  sx = {} // Allow custom styling
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const getMethodIcon = (methodName) => {
    switch (methodName?.toLowerCase()) {
      case 'shap':
        return <AnalyticsIcon />;
      case 'lime':
        return <LightbulbIcon />;
      case 'integrated_gradients':
        return <PsychologyIcon />;
      default:
        return <PsychologyIcon />;
    }
  };

  const getMethodDisplayName = (method) => {
    const displayNames = {
      shap: 'SHAP',
      lime: 'LIME', 
      integrated_gradients: 'Integrated Gradients'
    };
    return displayNames[method.name] || method.display_name || method.name.toUpperCase();
  };

  const getMethodDescription = (method) => {
    const descriptions = {
      shap: 'Global feature importance analysis',
      lime: 'Local interpretable explanations',
      integrated_gradients: 'Deep learning attribution method'
    };
    return descriptions[method.name] || method.description || 'AI explanation method';
  };

  const getMethodFullDescription = (method) => {
    const fullDescriptions = {
      shap: 'SHAP (SHapley Additive exPlanations) uses game theory to explain individual predictions by computing the contribution of each feature',
      lime: 'LIME (Local Interpretable Model-agnostic Explanations) explains predictions by learning an interpretable model locally around the prediction',
      integrated_gradients: 'Integrated Gradients computes feature attributions by integrating gradients along a path from a baseline to the input'
    };
    return fullDescriptions[method.name] || getMethodDescription(method);
  };

  if (variant === 'cards' && !isMobile) {
    return (
      <Box sx={{ mb: 3, ...sx }}>
        <Typography variant="subtitle2" gutterBottom>
          XAI Method
        </Typography>
        <Grid container spacing={2}>
          {methods.map((method) => (
            <Grid item xs={12} sm={6} md={4} key={method.name}>
              <Tooltip title={getMethodFullDescription(method)} arrow placement="top">
                <Paper
                  sx={{
                    p: 2,
                    cursor: 'pointer',
                    border: selectedMethod === method.name ? 2 : 1,
                    borderColor: selectedMethod === method.name ? 'primary.main' : 'divider',
                    '&:hover': {
                      borderColor: 'primary.main',
                      boxShadow: 1
                    },
                    transition: 'all 0.2s'
                  }}
                  onClick={() => !loading && onMethodChange(method.name)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    {getMethodIcon(method.name)}
                    <Typography variant="subtitle2">
                      {getMethodDisplayName(method)}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    {getMethodDescription(method)}
                  </Typography>
                  {selectedMethod === method.name && (
                    <Chip 
                      label="Selected" 
                      color="primary" 
                      size="small" 
                      sx={{ mt: 1 }}
                    />
                  )}
                </Paper>
              </Tooltip>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  // Dropdown variant (default and mobile) - Fixed styling and text display
  return (
    <FormControl fullWidth sx={{ ...sx }}>
      <InputLabel id="method-select-label" sx={{ fontSize: '0.875rem' }}>
        XAI Method
      </InputLabel>
      <Select
        labelId="method-select-label"
        id="method-select"
        value={selectedMethod}
        label="XAI Method"
        onChange={(e) => onMethodChange(e.target.value)}
        disabled={loading}
        size="small"
        sx={{
          '& .MuiSelect-select': {
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            pr: 4 // Space for dropdown arrow
          }
        }}
        MenuProps={{
          PaperProps: {
            style: {
              maxHeight: 280,
              minWidth: '320px' // Ensure enough width for full text
            }
          },
          anchorOrigin: {
            vertical: 'bottom',
            horizontal: 'left'
          },
          transformOrigin: {
            vertical: 'top', 
            horizontal: 'left'
          }
        }}
        renderValue={(selected) => {
          const method = methods.find(m => m.name === selected);
          if (!method) return '';
          
          return (
            <Tooltip title={getMethodFullDescription(method)} arrow>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                {getMethodIcon(method.name)}
                <Typography variant="body2" noWrap sx={{ fontWeight: 500 }}>
                  {getMethodDisplayName(method)}
                </Typography>
              </Box>
            </Tooltip>
          );
        }}
      >
        {methods.map((method) => (
          <MenuItem 
            key={method.name} 
            value={method.name}
            sx={{ 
              minHeight: 60,
              py: 1.5,
              px: 2,
              '&:hover': {
                backgroundColor: 'action.hover'
              }
            }}
          >
            <Tooltip title={getMethodFullDescription(method)} arrow placement="right">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, width: '100%' }}>
                <Box sx={{ color: 'primary.main', display: 'flex' }}>
                  {getMethodIcon(method.name)}
                </Box>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500, mb: 0.5 }}>
                    {getMethodDisplayName(method)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.2 }}>
                    {getMethodDescription(method)}
                  </Typography>
                </Box>
              </Box>
            </Tooltip>
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};

export default React.memo(MethodSelector); 