import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  IconButton,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  Science as ScienceIcon,
  Info as InfoIcon
} from '@mui/icons-material';

const IntegratedGradientsVisualization = ({ data }) => {
  if (!data) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          No Integrated Gradients data available
        </Typography>
      </Box>
    );
  }

  const { attributionStats, pathIntegration, sensitivity } = data;

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ScienceIcon />
          Integrated Gradients Explanation
          <Tooltip title="Integrated Gradients provides attribution method for deep learning models">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>
      </Box>

      {/* Attribution Statistics */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Attribution Statistics
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {attributionStats?.min?.toFixed(3)}
                </Typography>
                <Typography variant="caption">Min Attribution</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="secondary">
                  {attributionStats?.max?.toFixed(3)}
                </Typography>
                <Typography variant="caption">Max Attribution</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="success.main">
                  {attributionStats?.mean?.toFixed(3)}
                </Typography>
                <Typography variant="caption">Mean Attribution</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="warning.main">
                  {attributionStats?.std?.toFixed(3)}
                </Typography>
                <Typography variant="caption">Std Deviation</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {/* Path Integration */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Path Integration Analysis
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Integration Steps
              </Typography>
              <Typography variant="h6">{pathIntegration?.steps}</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Baseline
              </Typography>
              <Typography variant="h6">{pathIntegration?.baseline}</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Convergence
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <LinearProgress 
                  variant="determinate" 
                  value={(pathIntegration?.convergence || 0) * 100} 
                  sx={{ flexGrow: 1 }}
                />
                <Typography variant="body2">
                  {((pathIntegration?.convergence || 0) * 100).toFixed(1)}%
                </Typography>
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Sensitivity Analysis */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Sensitivity Analysis
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Input Sensitivity
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={(sensitivity?.inputSensitivity || 0) * 100} 
                color="primary"
                sx={{ mb: 1 }}
              />
              <Typography variant="body2">
                {((sensitivity?.inputSensitivity || 0) * 100).toFixed(1)}%
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Noise Robustness
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={(sensitivity?.noiseRobustness || 0) * 100} 
                color="secondary"
                sx={{ mb: 1 }}
              />
              <Typography variant="body2">
                {((sensitivity?.noiseRobustness || 0) * 100).toFixed(1)}%
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Spatial Coherence
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={(sensitivity?.spatialCoherence || 0) * 100} 
                color="success"
                sx={{ mb: 1 }}
              />
              <Typography variant="body2">
                {((sensitivity?.spatialCoherence || 0) * 100).toFixed(1)}%
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default IntegratedGradientsVisualization; 