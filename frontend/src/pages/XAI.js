import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useSelector } from 'react-redux';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Tab,
  Tabs,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Paper,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Checkbox,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Switch,
  FormControlLabel,
  Snackbar,
  useMediaQuery,
  useTheme,
  CircularProgress,
  Tooltip
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  Assessment as AssessmentIcon,
  Compare as CompareIcon,
  PictureAsPdf as PdfIcon,
  History as HistoryIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import xaiService from '../services/xaiService';
import { canAccessFeature, FEATURE_ACCESS } from '../utils/roleUtils';
import ModernEmptyState from '../components/xai/ModernEmptyState';
import ExplanationSkeleton from '../components/xai/ExplanationSkeleton';
import MobileComparisonTable from '../components/xai/MobileComparisonTable';
import MethodSelector from '../components/xai/MethodSelector';

const XAI = () => {
  const { user } = useSelector(state => state.auth);
  const { projects } = useSelector(state => state.projects);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // Consolidated state management
  const [state, setState] = useState({
    // UI State
    tabValue: 0,
    loading: false,
    error: null,
    success: null,
    
    // Generation state
    selectedProject: '',
    selectedMethod: 'shap',
    businessFriendly: true,
    includeUncertainty: true,
    currentExplanation: null,
    
    // Data state
    availableMethods: null,
    explanationHistory: [],
    selectedExplanations: [],
    comparisonResult: null,
    
    // Dialog state
    reportDialogOpen: false,
    reportFormat: 'pdf',
    includeBusinessSummary: true
  });

  // Memoized selectors
  const filteredProjects = useMemo(() => 
    projects?.filter(project => canAccessFeature(user?.role, 'XAI_FEATURES')) || []
  , [projects, user?.role]);

  const canAccessXAI = useCallback(() => {
    if (!user) {
      console.log('🚫 XAI Access Denied: No user logged in');
      return false;
    }
    
    const hasAccess = canAccessFeature(user.role, 'XAI_FEATURES');
    console.log(`🔐 XAI Access Check: User role="${user.role}", hasAccess=${hasAccess}`);
    
    if (!hasAccess) {
      console.log('🚫 XAI Access Denied: User role not in XAI_FEATURES');
      console.log('Available roles for XAI:', FEATURE_ACCESS.XAI_FEATURES);
    }
    
    return hasAccess;
  }, [user]);

  // Update state helper
  const updateState = useCallback((updates) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  // Load explanation history function
  const loadHistory = useCallback(async (projectId) => {
    try {
      const history = await xaiService.getExplanationHistory(projectId);
      updateState({ explanationHistory: history.explanations || [] });
      console.log('📚 Explanation history loaded:', history);
    } catch (error) {
      console.error('Failed to load explanation history:', error);
      updateState({ explanationHistory: [] });
    }
  }, [updateState]);

  // Load available methods on component mount
  useEffect(() => {
    const loadMethods = async () => {
      try {
        const methods = await xaiService.getAvailableMethods();
        updateState({ availableMethods: methods });
        console.log('📋 Available methods loaded:', methods);
      } catch (error) {
        console.error('Failed to load XAI methods:', error);
        updateState({ error: 'Failed to load XAI methods' });
      }
    };
    loadMethods();
  }, [updateState]);

  // Load explanation history when project is selected
  useEffect(() => {
    if (state.selectedProject) {
      loadHistory(state.selectedProject);
    }
  }, [state.selectedProject, loadHistory]);

  // Auto-clear success messages
  useEffect(() => {
    if (state.success) {
      const timer = setTimeout(() => {
        updateState({ success: null });
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [state.success, updateState]);

  const handleGenerateExplanation = async () => {
    if (!state.selectedProject) {
      updateState({ error: 'Please select a project first' });
      return;
    }

    updateState({ loading: true, error: null, success: null });

    try {
      console.log('🔍 Generating explanation for project:', state.selectedProject);
      
      const explanationData = xaiService.createSampleExplanationData(
        parseInt(state.selectedProject)
      );
      explanationData.method = state.selectedMethod;
      explanationData.businessFriendly = state.businessFriendly;
      explanationData.includeUncertainty = state.includeUncertainty;

      console.log('📤 Sending explanation request:', explanationData);

      const explanation = await xaiService.generateExplanation(explanationData);
      
      console.log('📥 Received explanation:', explanation);
      
      updateState({ 
        currentExplanation: explanation,
        success: 'Explanation generated successfully!',
        tabValue: 0 // Stay on generate tab to show results
      });
      
      // Refresh history
      await loadHistory(state.selectedProject);
      
    } catch (error) {
      console.error('❌ Failed to generate explanation:', error);
      updateState({ error: error.message || 'Failed to generate explanation' });
    } finally {
      updateState({ loading: false });
    }
  };

  const handleCompareExplanations = async () => {
    if (state.selectedExplanations.length < 2) {
      updateState({ error: 'Please select at least 2 explanations to compare' });
      return;
    }

    updateState({ loading: true, error: null });

    try {
      const comparison = await xaiService.compareExplanations(state.selectedExplanations);
      updateState({ 
        comparisonResult: comparison,
        success: 'Explanations compared successfully!' 
      });
    } catch (error) {
      updateState({ error: error.message });
    } finally {
      updateState({ loading: false });
    }
  };

  const handleGenerateReport = async () => {
    if (!state.currentExplanation) {
      updateState({ error: 'No explanation available to generate report' });
      return;
    }

    updateState({ loading: true, error: null, reportDialogOpen: false });

    try {
      const report = await xaiService.generateReport(
        state.currentExplanation.explanation_id,
        state.reportFormat,
        state.includeBusinessSummary
      );

      if (report && report.data) {
        const filename = `xai_explanation_${state.currentExplanation.explanation_id}.${state.reportFormat}`;
        xaiService.downloadReport(report, filename);
        updateState({ success: `Report downloaded successfully as ${filename}` });
      }
    } catch (error) {
      updateState({ error: error.message });
    } finally {
      updateState({ loading: false });
    }
  };

  const handleExplanationSelect = (explanationId, isSelected) => {
    const newSelected = isSelected 
      ? [...state.selectedExplanations, explanationId]
      : state.selectedExplanations.filter(id => id !== explanationId);
    
    updateState({ selectedExplanations: newSelected });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getRiskLevelIcon = (level) => {
    switch (level) {
      case 'Low': return <CheckCircleIcon color="success" />;
      case 'Medium': return <WarningIcon color="warning" />;
      case 'High': return <ErrorIcon color="error" />;
      default: return <InfoIcon />;
    }
  };

  // Error boundary effect
  useEffect(() => {
    const handleError = (event) => {
      console.error('XAI Page Error:', event.error);
      updateState({ error: 'An unexpected error occurred. Please try again.' });
    };

    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, [updateState]);

  if (!canAccessXAI()) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="warning">
          You don't have permission to access the XAI features. Please contact your administrator.
        </Alert>
      </Container>
    );
  }

  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`xai-tabpanel-${index}`}
      aria-labelledby={`xai-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <PsychologyIcon sx={{ fontSize: '2rem', color: 'primary.main' }} />
          Explainable AI (XAI)
          <Tooltip title="AI transparency and model interpretability for regulatory compliance">
            <IconButton size="small">
              <InfoIcon />
            </IconButton>
          </Tooltip>
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Understand how AI models make decisions about carbon credit verification
        </Typography>
      </Box>

      {/* Global Loading Bar */}
      {state.loading && (
        <Box sx={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1300 }}>
          <LinearProgress />
        </Box>
      )}

      {/* Navigation Tabs */}
      <Card sx={{ mb: 3 }}>
        <Tabs 
          value={state.tabValue} 
          onChange={(e, v) => updateState({ tabValue: v })}
          variant={isMobile ? "scrollable" : "fullWidth"}
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
          aria-label="XAI navigation tabs"
        >
          <Tab 
            label="Generate & Analyze" 
            icon={<PsychologyIcon />} 
            id="xai-tab-0"
            aria-controls="xai-tabpanel-0"
          />
          <Tab 
            label="Compare Methods" 
            icon={<CompareIcon />} 
            id="xai-tab-1"
            aria-controls="xai-tabpanel-1"
          />
          <Tab 
            label="History & Reports" 
            icon={<HistoryIcon />} 
            id="xai-tab-2"
            aria-controls="xai-tabpanel-2"
          />
        </Tabs>
      </Card>

      {/* Tab 1: Generate & Analyze */}
      <TabPanel value={state.tabValue} index={0}>
        {/* Configuration Panel - Now Horizontal at Top */}
        <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
              <PsychologyIcon color="primary" />
              Configuration & Settings
              <Chip 
                label={
                  state.currentExplanation ? "✅ Complete" :
                  state.selectedProject && state.selectedMethod ? "Ready" :
                  state.selectedProject ? "Step 2" :
                  "Step 1"
                } 
                size="small" 
                color={
                  state.currentExplanation ? "success" :
                  state.selectedProject && state.selectedMethod ? "primary" :
                  "default"
                }
                variant={state.currentExplanation ? "filled" : "outlined"}
                sx={{ ml: 'auto' }}
              />
            </Typography>
            
            <Grid container spacing={3} alignItems="center">
              {/* Project Selection */}
              <Grid item xs={12} md={4}>
                <Box sx={{ position: 'relative' }}>
                  <Typography variant="caption" color="primary" sx={{ mb: 1, display: 'block', fontWeight: 600 }}>
                    1. Select Project
                  </Typography>
                  <FormControl fullWidth>
                    <InputLabel id="project-select-label">Choose a project to analyze</InputLabel>
                    <Select
                      labelId="project-select-label"
                      id="project-select"
                      value={state.selectedProject}
                      label="Choose a project to analyze"
                      onChange={(e) => updateState({ selectedProject: e.target.value })}
                      disabled={state.loading}
                      size="small"
                      sx={{ 
                        bgcolor: 'background.paper',
                        '& .MuiOutlinedInput-root': {
                          '&:hover fieldset': {
                            borderColor: 'primary.main',
                          },
                        }
                      }}
                    >
                      {filteredProjects.map((project) => (
                        <MenuItem key={project.id} value={project.id.toString()}>
                          <Box>
                            <Typography variant="body2" fontWeight="500">
                              {project.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {project.location_name} • {project.area_hectares} hectares
                            </Typography>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  {state.selectedProject && (
                    <CheckCircleIcon 
                      sx={{ 
                        position: 'absolute', 
                        top: 25, 
                        right: 8, 
                        color: 'success.main',
                        fontSize: 16
                      }} 
                    />
                  )}
                </Box>
              </Grid>

              {/* Method Selection */}
              <Grid item xs={12} md={4}>
                <Box sx={{ position: 'relative' }}>
                  <Typography variant="caption" color="primary" sx={{ mb: 1, display: 'block', fontWeight: 600 }}>
                    2. Choose AI Method
                  </Typography>
                  {state.availableMethods && (
                    <MethodSelector
                      methods={state.availableMethods.methods || []}
                      selectedMethod={state.selectedMethod}
                      onMethodChange={(method) => updateState({ selectedMethod: method })}
                      loading={state.loading}
                      variant="dropdown"
                      sx={{ 
                        '& .MuiOutlinedInput-root': {
                          bgcolor: 'background.paper',
                          '&:hover fieldset': {
                            borderColor: 'primary.main',
                          },
                        }
                      }}
                    />
                  )}
                  {state.selectedMethod && (
                    <CheckCircleIcon 
                      sx={{ 
                        position: 'absolute', 
                        top: 25, 
                        right: 8, 
                        color: 'success.main',
                        fontSize: 16
                      }} 
                    />
                  )}
                </Box>
              </Grid>

              {/* Generate Button */}
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography variant="caption" color="primary" sx={{ mb: 1, display: 'block', fontWeight: 600 }}>
                    3. Generate Analysis
                  </Typography>
                  <Button
                    variant="contained"
                    size="large"
                    fullWidth
                    onClick={handleGenerateExplanation}
                    disabled={!state.selectedProject || state.loading}
                    startIcon={state.loading ? <CircularProgress size={20} /> : <PsychologyIcon />}
                    sx={{ 
                      height: 40,
                      background: state.selectedProject ? 
                        'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)' : 
                        'linear-gradient(45deg, #bdbdbd 30%, #e0e0e0 90%)',
                      boxShadow: state.selectedProject ? 
                        '0 3px 5px 2px rgba(33, 203, 243, .3)' : 
                        '0 3px 5px 2px rgba(189, 189, 189, .3)',
                      '&:hover': {
                        background: state.selectedProject ? 
                          'linear-gradient(45deg, #1976D2 30%, #1BA3D6 90%)' :
                          'linear-gradient(45deg, #bdbdbd 30%, #e0e0e0 90%)',
                      },
                      '&:disabled': {
                        color: 'rgba(0, 0, 0, 0.26)',
                      }
                    }}
                  >
                    {state.loading ? 'Generating...' : 'Generate Explanation'}
                  </Button>
                  {!state.selectedProject && (
                    <Typography variant="caption" color="error" sx={{ mt: 0.5, display: 'block' }}>
                      Please select a project first
                    </Typography>
                  )}
                </Box>
              </Grid>
            </Grid>

            {/* Progress Indicator */}
            <Box sx={{ mt: 3, mb: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Configuration Progress
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {state.selectedProject && state.selectedMethod ? '100%' : 
                   state.selectedProject ? '50%' : '0%'}
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={
                  state.selectedProject && state.selectedMethod ? 100 : 
                  state.selectedProject ? 50 : 0
                }
                sx={{ 
                  height: 6, 
                  borderRadius: 3,
                  backgroundColor: 'rgba(0,0,0,0.1)',
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 3,
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)'
                  }
                }}
              />
            </Box>

            {/* Advanced Options - Collapsible */}
            <Accordion sx={{ mt: 2, boxShadow: 'none', border: '1px solid', borderColor: 'divider' }}>
              <AccordionSummary 
                expandIcon={<ExpandMoreIcon />}
                sx={{ minHeight: 48, '&.Mui-expanded': { minHeight: 48 } }}
              >
                <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <InfoIcon fontSize="small" />
                  Advanced Options
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 0 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={state.businessFriendly}
                          onChange={(e) => updateState({ businessFriendly: e.target.checked })}
                          disabled={state.loading}
                          color="primary"
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2">Business-friendly explanations</Typography>
                          <Typography variant="caption" color="text.secondary">
                            Generate explanations in business language
                          </Typography>
                        </Box>
                      }
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={state.includeUncertainty}
                          onChange={(e) => updateState({ includeUncertainty: e.target.checked })}
                          disabled={state.loading}
                          color="primary"
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2">Include uncertainty analysis</Typography>
                          <Typography variant="caption" color="text.secondary">
                            Add confidence intervals and risk analysis
                          </Typography>
                        </Box>
                      }
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </CardContent>
        </Card>

        {/* Results Panel - Now Full Width */}
        <Box>
          {state.loading ? (
            <ExplanationSkeleton />
          ) : state.currentExplanation ? (
            <Card>
              <CardContent>
                {/* Header with metadata */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3, flexWrap: 'wrap' }}>
                  <Typography variant="h6">
                    {state.selectedMethod?.toUpperCase()} Explanation
                  </Typography>
                  <Chip 
                    label={`${(state.currentExplanation.confidence_score * 100).toFixed(1)}% Confidence`}
                    color={getConfidenceColor(state.currentExplanation.confidence_score)}
                    size="small"
                  />
                  <Box sx={{ ml: 'auto', display: 'flex', gap: 1 }}>
                    <Button
                      size="small"
                      onClick={() => updateState({ reportDialogOpen: true })}
                      startIcon={<PdfIcon />}
                    >
                      Export
                    </Button>
                    <IconButton
                      size="small"
                      onClick={handleGenerateExplanation}
                      disabled={state.loading}
                    >
                      <RefreshIcon />
                    </IconButton>
                  </Box>
                </Box>

                {/* Business Summary */}
                {state.currentExplanation.business_summary && (
                  <Accordion defaultExpanded>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Executive Summary</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>
                        {state.currentExplanation.business_summary}
                      </Typography>
                    </AccordionDetails>
                  </Accordion>
                )}

                {/* Risk Assessment */}
                {state.currentExplanation.risk_assessment && (
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {getRiskLevelIcon(state.currentExplanation.risk_assessment.level)}
                        <Typography variant="subtitle1">
                          Risk Assessment - {state.currentExplanation.risk_assessment.level}
                        </Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="body2" gutterBottom>
                        {state.currentExplanation.risk_assessment.description}
                      </Typography>
                      {state.currentExplanation.risk_assessment.mitigation_recommendations && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="body2" fontWeight="bold">Recommendations:</Typography>
                          <List dense>
                            {state.currentExplanation.risk_assessment.mitigation_recommendations.map((rec, index) => (
                              <ListItem key={index}>
                                <ListItemText primary={rec} />
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}
                    </AccordionDetails>
                  </Accordion>
                )}

                {/* Visualizations */}
                {state.currentExplanation.visualizations && Object.keys(state.currentExplanation.visualizations).length > 0 && (
                  <Accordion defaultExpanded data-testid="visualizations-section">
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Visualizations & Charts</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        {Object.entries(state.currentExplanation.visualizations).map(([key, imageData]) => (
                          <Grid item xs={12} md={6} key={key}>
                            <Paper sx={{ p: 2, textAlign: 'center' }} data-testid={`chart-${key}`}>
                              <Typography variant="subtitle2" gutterBottom>
                                {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </Typography>
                              <img 
                                src={imageData} 
                                alt={key}
                                style={{ maxWidth: '100%', height: 'auto', minHeight: '200px' }}
                                data-testid={`chart-image-${key}`}
                              />
                            </Paper>
                          </Grid>
                        ))}
                      </Grid>
                      <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          📊 {Object.keys(state.currentExplanation.visualizations).length} real-time visualization(s) generated from production ML models
                        </Typography>
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                )}

                {/* Regulatory Compliance */}
                {state.currentExplanation.regulatory_notes && (
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Regulatory Compliance</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {Object.entries(state.currentExplanation.regulatory_notes).map(([key, value]) => (
                        <Box key={key} sx={{ mb: 1 }}>
                          <Typography variant="body2" fontWeight="bold">
                            {key.replace('_', ' ').toUpperCase()}:
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {value}
                          </Typography>
                        </Box>
                      ))}
                    </AccordionDetails>
                  </Accordion>
                )}
              </CardContent>
            </Card>
          ) : (
            <ModernEmptyState 
              variant="generate"
              onAction={() => {
                if (!state.selectedProject && filteredProjects.length > 0) {
                  updateState({ selectedProject: filteredProjects[0].id.toString() });
                }
                handleGenerateExplanation();
              }}
              actionText={state.selectedProject ? "Generate Explanation" : "Select Project First"}
            />
          )}
        </Box>
      </TabPanel>

      {/* Tab 2: Compare & Analyze */}
      <TabPanel value={state.tabValue} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Compare Explanations
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Select multiple explanations to compare their results and business impact
                </Typography>
                
                {state.explanationHistory.length >= 2 ? (
                  <>
                    {/* Mobile-optimized comparison table */}
                    {isMobile ? (
                      <MobileComparisonTable
                        explanations={state.explanationHistory}
                        selectedExplanations={state.selectedExplanations}
                        onExplanationSelect={handleExplanationSelect}
                        onCompare={handleCompareExplanations}
                      />
                    ) : (
                      // Desktop table
                      <Box>
                        <Typography variant="subtitle1" gutterBottom>
                          Available Explanations for Comparison:
                        </Typography>
                        <TableContainer component={Paper} sx={{ mb: 3 }}>
                          <Table>
                            <TableHead>
                              <TableRow>
                                <TableCell padding="checkbox">Select</TableCell>
                                <TableCell>Method</TableCell>
                                <TableCell>Timestamp</TableCell>
                                <TableCell>Confidence</TableCell>
                                <TableCell>Summary</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {state.explanationHistory.map((explanation) => (
                                <TableRow key={explanation.explanation_id}>
                                  <TableCell padding="checkbox">
                                    <Checkbox
                                      checked={state.selectedExplanations.includes(explanation.explanation_id)}
                                      onChange={(e) => handleExplanationSelect(explanation.explanation_id, e.target.checked)}
                                    />
                                  </TableCell>
                                  <TableCell>{explanation.method?.toUpperCase()}</TableCell>
                                  <TableCell>{new Date(explanation.timestamp).toLocaleString()}</TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={`${(explanation.confidence_score * 100).toFixed(1)}%`}
                                      color={getConfidenceColor(explanation.confidence_score)}
                                      size="small"
                                    />
                                  </TableCell>
                                  <TableCell>{explanation.business_summary?.substring(0, 100)}...</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                        
                        <Button
                          variant="contained"
                          onClick={handleCompareExplanations}
                          disabled={state.selectedExplanations.length < 2 || state.loading}
                          startIcon={<CompareIcon />}
                        >
                          Compare Selected Explanations ({state.selectedExplanations.length})
                        </Button>
                      </Box>
                    )}

                    {state.comparisonResult && (
                      <Box sx={{ mt: 3 }}>
                        <Typography variant="h6" gutterBottom>
                          Comparison Results
                        </Typography>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="body2">
                            {JSON.stringify(state.comparisonResult, null, 2)}
                          </Typography>
                        </Paper>
                      </Box>
                    )}
                  </>
                ) : (
                  <ModernEmptyState 
                    variant="compare"
                    onAction={() => updateState({ tabValue: 0 })}
                    actionText="Generate More Explanations"
                  />
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tab 3: History & Reports */}
      <TabPanel value={state.tabValue} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6">
                    Explanation History
                  </Typography>
                  <Button
                    startIcon={<RefreshIcon />}
                    onClick={() => state.selectedProject && loadHistory(state.selectedProject)}
                    disabled={!state.selectedProject || state.loading}
                  >
                    Refresh
                  </Button>
                </Box>
                
                {state.explanationHistory.length > 0 ? (
                  <List>
                    {state.explanationHistory.map((explanation, index) => (
                      <React.Fragment key={explanation.explanation_id}>
                        <ListItem>
                          <ListItemIcon>
                            <AssessmentIcon />
                          </ListItemIcon>
                          <ListItemText
                            primary={`${explanation.method?.toUpperCase()} - ${new Date(explanation.timestamp).toLocaleString()}`}
                            secondary={explanation.business_summary}
                          />
                          <Chip 
                            label={`${(explanation.confidence_score * 100).toFixed(1)}%`}
                            color={getConfidenceColor(explanation.confidence_score)}
                            size="small"
                          />
                        </ListItem>
                        {index < state.explanationHistory.length - 1 && <Divider />}
                      </React.Fragment>
                    ))}
                  </List>
                ) : (
                  <ModernEmptyState 
                    variant="history"
                    onAction={() => updateState({ tabValue: 0 })}
                    actionText="Generate First Explanation"
                  />
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Report Generation Dialog */}
      <Dialog open={state.reportDialogOpen} onClose={() => updateState({ reportDialogOpen: false })}>
        <DialogTitle>Generate Report</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Format</InputLabel>
            <Select
              value={state.reportFormat}
              label="Format"
              onChange={(e) => updateState({ reportFormat: e.target.value })}
            >
              <MenuItem value="pdf">PDF</MenuItem>
              <MenuItem value="json">JSON</MenuItem>
            </Select>
          </FormControl>
          <FormControlLabel
            control={
              <Switch
                checked={state.includeBusinessSummary}
                onChange={(e) => updateState({ includeBusinessSummary: e.target.checked })}
              />
            }
            label="Include business summary"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => updateState({ reportDialogOpen: false })}>Cancel</Button>
          <Button onClick={handleGenerateReport} variant="contained" disabled={state.loading}>
            Generate
          </Button>
        </DialogActions>
      </Dialog>

      {/* Error/Success Snackbar */}
      <Snackbar
        open={!!state.error}
        autoHideDuration={6000}
        onClose={() => updateState({ error: null })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => updateState({ error: null })} 
          severity="error" 
          sx={{ width: '100%' }}
        >
          {state.error}
        </Alert>
      </Snackbar>

      <Snackbar
        open={!!state.success}
        autoHideDuration={5000}
        onClose={() => updateState({ success: null })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => updateState({ success: null })} 
          severity="success" 
          sx={{ width: '100%' }}
        >
          {state.success}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default React.memo(XAI); 