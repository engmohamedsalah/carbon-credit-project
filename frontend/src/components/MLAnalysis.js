/**
 * ML Analysis Component
 * Provides comprehensive machine learning analysis interface for carbon credit verification
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  IconButton
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CloudUpload as UploadIcon,
  LocationOn as LocationIcon,
  Analytics as AnalyticsIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import mlService from '../services/mlService';

const MLAnalysis = ({ projectId, projectData, onAnalysisComplete }) => {
  const [analysisStep, setAnalysisStep] = useState('input'); // input, running, results
  const [loading, setLoading] = useState(false);
  const [mlStatus, setMLStatus] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [eligibilityResults, setEligibilityResults] = useState(null);
  const [error, setError] = useState(null);

  // Form state
  const [coordinates, setCoordinates] = useState({ latitude: '', longitude: '' });
  const [forestCoverImage, setForestCoverImage] = useState(null);
  const [beforeImage, setBeforeImage] = useState(null);
  const [afterImage, setAfterImage] = useState(null);

  useEffect(() => {
    checkMLStatus();
  }, []);

  const checkMLStatus = async () => {
    try {
      const status = await mlService.getMLStatus();
      setMLStatus(status);
    } catch (error) {
      console.error('Failed to check ML status:', error);
      setError('ML service is currently unavailable');
    }
  };

  const handleFileUpload = (file, type) => {
    if (!file) return;
    
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/geotiff'];
    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.tif')) {
      setError('Please upload a valid image file (JPEG, PNG, TIFF, or GeoTIFF)');
      return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      setError('File size must be less than 50MB');
      return;
    }

    switch (type) {
      case 'forestCover':
        setForestCoverImage(file);
        break;
      case 'before':
        setBeforeImage(file);
        break;
      case 'after':
        setAfterImage(file);
        break;
      default:
        break;
    }
    setError(null);
  };

  const runAnalysis = async () => {
    setLoading(true);
    setAnalysisStep('running');
    setError(null);

    try {
      const analysisData = {};

      // Add coordinates if provided
      if (coordinates.latitude && coordinates.longitude) {
        analysisData.coordinates = {
          latitude: parseFloat(coordinates.latitude),
          longitude: parseFloat(coordinates.longitude)
        };
      }

      // Add images if provided
      if (forestCoverImage) {
        analysisData.forestCoverImage = forestCoverImage;
      }
      if (beforeImage && afterImage) {
        analysisData.beforeImage = beforeImage;
        analysisData.afterImage = afterImage;
      }

      // Run comprehensive analysis
      const results = await mlService.runComprehensiveAnalysis(projectId, analysisData);
      
      // Format results for display
      const formattedResults = mlService.formatAnalysisResults(results);
      
      // Calculate eligibility
      const eligibility = mlService.calculateEligibility(formattedResults);

      setAnalysisResults(formattedResults);
      setEligibilityResults(eligibility);
      setAnalysisStep('results');

      // Notify parent component
      if (onAnalysisComplete) {
        onAnalysisComplete({
          results: formattedResults,
          eligibility: eligibility
        });
      }

    } catch (error) {
      console.error('Analysis failed:', error);
      setError(error.response?.data?.detail || 'Analysis failed. Please try again.');
      setAnalysisStep('input');
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setAnalysisStep('input');
    setAnalysisResults(null);
    setEligibilityResults(null);
    setError(null);
    setCoordinates({ latitude: '', longitude: '' });
    setForestCoverImage(null);
    setBeforeImage(null);
    setAfterImage(null);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'excellent':
        return <CheckIcon color="success" />;
      case 'good':
        return <CheckIcon color="primary" />;
      case 'fair':
        return <WarningIcon color="warning" />;
      case 'poor':
        return <ErrorIcon color="error" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'excellent':
        return 'success';
      case 'good':
        return 'primary';
      case 'fair':
        return 'warning';
      case 'poor':
        return 'error';
      default:
        return 'default';
    }
  };

  if (!mlStatus || !mlStatus.models_ready) {
    return (
      <Card>
        <CardContent>
          <Alert severity="warning" action={
            <IconButton onClick={checkMLStatus}>
              <RefreshIcon />
            </IconButton>
          }>
            ML service is not available. Please try again later.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Analysis Input Step */}
      {analysisStep === 'input' && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              Machine Learning Analysis
            </Typography>
            
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <Grid container spacing={3}>
              {/* Location Analysis */}
              <Grid item xs={12}>
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">
                      <LocationIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Location Analysis
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <TextField
                          label="Latitude"
                          type="number"
                          fullWidth
                          value={coordinates.latitude}
                          onChange={(e) => setCoordinates(prev => ({ ...prev, latitude: e.target.value }))}
                          placeholder="-3.4653"
                          helperText="Decimal degrees (e.g., -3.4653)"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField
                          label="Longitude"
                          type="number"
                          fullWidth
                          value={coordinates.longitude}
                          onChange={(e) => setCoordinates(prev => ({ ...prev, longitude: e.target.value }))}
                          placeholder="-62.2159"
                          helperText="Decimal degrees (e.g., -62.2159)"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>

              {/* Forest Cover Analysis */}
              <Grid item xs={12}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">
                      <UploadIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Forest Cover Analysis
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      <input
                        accept="image/*,.tif,.tiff"
                        style={{ display: 'none' }}
                        id="forest-cover-upload"
                        type="file"
                        onChange={(e) => handleFileUpload(e.target.files[0], 'forestCover')}
                      />
                      <label htmlFor="forest-cover-upload">
                        <Button variant="outlined" component="span" startIcon={<UploadIcon />}>
                          Upload Satellite Image
                        </Button>
                      </label>
                      {forestCoverImage && (
                        <Chip 
                          label={`✓ ${forestCoverImage.name}`} 
                          color="success" 
                          sx={{ ml: 2 }}
                          onDelete={() => setForestCoverImage(null)}
                        />
                      )}
                      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                        Supported formats: JPEG, PNG, TIFF, GeoTIFF (max 50MB)
                      </Typography>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Grid>

              {/* Change Detection */}
              <Grid item xs={12}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">
                      Change Detection Analysis
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Box>
                          <Typography variant="body2" gutterBottom>Before Image</Typography>
                          <input
                            accept="image/*,.tif,.tiff"
                            style={{ display: 'none' }}
                            id="before-image-upload"
                            type="file"
                            onChange={(e) => handleFileUpload(e.target.files[0], 'before')}
                          />
                          <label htmlFor="before-image-upload">
                            <Button variant="outlined" component="span" startIcon={<UploadIcon />} size="small">
                              Upload Before
                            </Button>
                          </label>
                          {beforeImage && (
                            <Chip 
                              label={`✓ ${beforeImage.name}`} 
                              color="success" 
                              size="small"
                              sx={{ ml: 1 }}
                              onDelete={() => setBeforeImage(null)}
                            />
                          )}
                        </Box>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Box>
                          <Typography variant="body2" gutterBottom>After Image</Typography>
                          <input
                            accept="image/*,.tif,.tiff"
                            style={{ display: 'none' }}
                            id="after-image-upload"
                            type="file"
                            onChange={(e) => handleFileUpload(e.target.files[0], 'after')}
                          />
                          <label htmlFor="after-image-upload">
                            <Button variant="outlined" component="span" startIcon={<UploadIcon />} size="small">
                              Upload After
                            </Button>
                          </label>
                          {afterImage && (
                            <Chip 
                              label={`✓ ${afterImage.name}`} 
                              color="success" 
                              size="small"
                              sx={{ ml: 1 }}
                              onDelete={() => setAfterImage(null)}
                            />
                          )}
                        </Box>
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>

              {/* Analysis Button */}
              <Grid item xs={12}>
                <Box sx={{ mt: 2, textAlign: 'center' }}>
                  <Button 
                    variant="contained" 
                    color="primary" 
                    size="large"
                    onClick={runAnalysis}
                    disabled={loading || (!coordinates.latitude && !forestCoverImage && (!beforeImage || !afterImage))}
                    startIcon={loading ? <CircularProgress size={20} /> : <AnalyticsIcon />}
                  >
                    {loading ? 'Analyzing...' : 'Run ML Analysis'}
                  </Button>
                  <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                    Provide at least coordinates or images to run analysis
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Analysis Running Step */}
      {analysisStep === 'running' && (
        <Card>
          <CardContent>
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <CircularProgress size={60} sx={{ mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Running ML Analysis...
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Processing your data with our advanced machine learning models
              </Typography>
              <LinearProgress sx={{ mt: 2, maxWidth: 400, mx: 'auto' }} />
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Analysis Results Step */}
      {analysisStep === 'results' && analysisResults && eligibilityResults && (
        <Box>
          {/* Eligibility Overview */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Carbon Credit Eligibility Assessment
                </Typography>
                <Chip 
                  icon={getStatusIcon(eligibilityResults.status)}
                  label={`${eligibilityResults.percentage}% Eligible`}
                  color={getStatusColor(eligibilityResults.status)}
                  size="large"
                />
              </Box>
              
              <Typography variant="h5" color={getStatusColor(eligibilityResults.status)} gutterBottom>
                {eligibilityResults.recommendation}
              </Typography>
              
              <LinearProgress 
                variant="determinate" 
                value={eligibilityResults.percentage} 
                color={getStatusColor(eligibilityResults.status)}
                sx={{ height: 10, borderRadius: 5, mb: 2 }}
              />
              
              <Typography variant="body2" color="text.secondary">
                Score: {eligibilityResults.eligibilityScore} / {eligibilityResults.maxScore} points
              </Typography>
            </CardContent>
          </Card>

          {/* Detailed Analysis Results */}
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Detailed Analysis Results</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                {analysisResults.summary.forestCoverage && (
                  <Grid item xs={12} sm={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        {analysisResults.summary.forestCoverage.toFixed(1)}%
                      </Typography>
                      <Typography variant="body2">Forest Coverage</Typography>
                    </Paper>
                  </Grid>
                )}
                
                {analysisResults.summary.forestArea && (
                  <Grid item xs={12} sm={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main">
                        {analysisResults.summary.forestArea.toFixed(0)}
                      </Typography>
                      <Typography variant="body2">Hectares</Typography>
                    </Paper>
                  </Grid>
                )}
                
                {analysisResults.summary.carbonEstimate && (
                  <Grid item xs={12} sm={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="secondary.main">
                        {analysisResults.summary.carbonEstimate.toFixed(0)}
                      </Typography>
                      <Typography variant="body2">Carbon Tons</Typography>
                    </Paper>
                  </Grid>
                )}
                
                {analysisResults.summary.confidenceScore && (
                  <Grid item xs={12} sm={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="warning.main">
                        {(analysisResults.summary.confidenceScore * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="body2">Confidence</Typography>
                    </Paper>
                  </Grid>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* Eligibility Factors */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Eligibility Factors</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List>
                {eligibilityResults.factors.map((factor, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      {getStatusIcon(factor.status)}
                    </ListItemIcon>
                    <ListItemText 
                      primary={factor.factor}
                      secondary={`${factor.score} points`}
                    />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>

          {/* Next Steps */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Recommended Next Steps</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List>
                {eligibilityResults.nextSteps.map((step, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <InfoIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={step} />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>

          {/* Actions */}
          <Box sx={{ mt: 3, display: 'flex', gap: 2, justifyContent: 'center' }}>
            <Button 
              variant="outlined" 
              onClick={resetAnalysis}
              startIcon={<RefreshIcon />}
            >
              Run New Analysis
            </Button>
            <Button 
              variant="contained" 
              color="primary"
              startIcon={<DownloadIcon />}
              onClick={() => {
                // Export results as JSON
                const dataStr = JSON.stringify({ analysisResults, eligibilityResults }, null, 2);
                const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
                const exportFileDefaultName = `ml-analysis-project-${projectId}.json`;
                const linkElement = document.createElement('a');
                linkElement.setAttribute('href', dataUri);
                linkElement.setAttribute('download', exportFileDefaultName);
                linkElement.click();
              }}
            >
              Export Results
            </Button>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default MLAnalysis; 