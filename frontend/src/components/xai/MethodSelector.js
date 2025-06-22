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
  useMediaQuery
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

  const getMethodDescription = (method) => {
    const descriptions = {
      shap: 'Global feature importance with game theory foundations',
      lime: 'Local interpretable model-agnostic explanations',
      integrated_gradients: 'Attribution method for deep neural networks'
    };
    return descriptions[method.name] || method.description || 'AI explanation method';
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
                    {method.display_name || method.name.toUpperCase()}
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
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  // Dropdown variant (default and mobile)
  return (
    <FormControl fullWidth sx={{ ...sx }}>
      <InputLabel id="method-select-label">XAI Method</InputLabel>
      <Select
        labelId="method-select-label"
        id="method-select"
        value={selectedMethod}
        label="XAI Method"
        onChange={(e) => onMethodChange(e.target.value)}
        disabled={loading}
        size="small"
      >
        {methods.map((method) => (
          <MenuItem key={method.name} value={method.name}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {getMethodIcon(method.name)}
              <Box>
                <Typography variant="body2">
                  {method.display_name || method.name.toUpperCase()}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {getMethodDescription(method)}
                </Typography>
              </Box>
            </Box>
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};

export default React.memo(MethodSelector); 