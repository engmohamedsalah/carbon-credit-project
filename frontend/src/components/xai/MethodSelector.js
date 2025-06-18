import React from 'react';
import {
  Box,
  Typography,
  Button,
  ButtonGroup,
  Card,
  CardContent,
  Chip,
  Tooltip,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  TrendingUp as SHAPIcon,
  Visibility as LIMEIcon,
  Science as IGIcon,
  SelectAll as AllIcon
} from '@mui/icons-material';

const MethodSelector = ({ 
  methods = [], 
  selectedMethod, 
  onMethodChange, 
  loading = false,
  variant = 'buttons' // 'buttons' or 'cards'
}) => {
  
  // Method icon mapping
  const getMethodIcon = (methodName) => {
    switch (methodName) {
      case 'shap': return <SHAPIcon />;
      case 'lime': return <LIMEIcon />;
      case 'integrated_gradients': return <IGIcon />;
      case 'all': return <AllIcon />;
      default: return <IGIcon />;
    }
  };

  // Method color mapping
  const getMethodColor = (methodName) => {
    switch (methodName) {
      case 'shap': return 'primary';
      case 'lime': return 'secondary';
      case 'integrated_gradients': return 'success';
      case 'all': return 'warning';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2 }}>
        <CircularProgress size={20} />
        <Typography variant="body2">Loading XAI methods...</Typography>
      </Box>
    );
  }

  if (methods.length === 0) {
    return (
      <Alert severity="warning">
        No XAI methods available. Please check service status.
      </Alert>
    );
  }

  if (variant === 'cards') {
    return (
      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Select Explanation Method
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {methods.map((method) => (
            <Card
              key={method.name}
              variant={selectedMethod === method.name ? "elevation" : "outlined"}
              sx={{
                cursor: 'pointer',
                border: selectedMethod === method.name ? 2 : 1,
                borderColor: selectedMethod === method.name ? `${getMethodColor(method.name)}.main` : 'divider',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  elevation: 3,
                  borderColor: `${getMethodColor(method.name)}.main`
                }
              }}
              onClick={() => onMethodChange(method.name)}
            >
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                  <Box sx={{ color: `${getMethodColor(method.name)}.main`, mt: 0.5 }}>
                    {getMethodIcon(method.name)}
                  </Box>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'medium', mb: 0.5 }}>
                      {method.display_name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {method.description}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                      Best for: {method.bestFor}
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {method.visualization_types?.map((viz, index) => (
                        <Chip
                          key={index}
                          label={viz}
                          size="small"
                          variant="outlined"
                          color={getMethodColor(method.name)}
                        />
                      ))}
                    </Box>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          ))}
        </Box>
      </Box>
    );
  }

  // Button variant (default)
  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Explanation Method
      </Typography>
      <ButtonGroup
        orientation="vertical"
        variant="outlined"
        fullWidth
        sx={{ mb: 2 }}
      >
        {methods.map((method) => (
          <Tooltip
            key={method.name}
            title={
              <Box>
                <Typography variant="subtitle2">{method.display_name}</Typography>
                <Typography variant="body2">{method.description}</Typography>
                <Typography variant="caption">Best for: {method.bestFor}</Typography>
              </Box>
            }
            placement="right"
            arrow
          >
            <Button
              variant={selectedMethod === method.name ? "contained" : "outlined"}
              color={getMethodColor(method.name)}
              startIcon={getMethodIcon(method.name)}
              onClick={() => onMethodChange(method.name)}
              sx={{
                justifyContent: 'flex-start',
                textAlign: 'left',
                px: 2,
                py: 1.5
              }}
            >
              <Box sx={{ textAlign: 'left' }}>
                <Typography variant="button" sx={{ fontWeight: 'medium' }}>
                  {method.display_name}
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', textTransform: 'none' }}>
                  {method.name === 'all' ? 'Generate all explanations' : method.description.substring(0, 40) + '...'}
                </Typography>
              </Box>
            </Button>
          </Tooltip>
        ))}
      </ButtonGroup>

      {/* Quick method selector */}
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        {['shap', 'lime', 'integrated_gradients', 'all'].map((methodName) => {
          const method = methods.find(m => m.name === methodName);
          if (!method) return null;
          
          return (
            <Chip
              key={methodName}
              label={method.name.toUpperCase()}
              icon={getMethodIcon(methodName)}
              color={selectedMethod === methodName ? getMethodColor(methodName) : 'default'}
              variant={selectedMethod === methodName ? 'filled' : 'outlined'}
              onClick={() => onMethodChange(methodName)}
              sx={{ cursor: 'pointer' }}
            />
          );
        })}
      </Box>
    </Box>
  );
};

export default MethodSelector; 