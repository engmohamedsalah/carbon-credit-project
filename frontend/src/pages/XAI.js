import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Tabs,
  Tab,
  Button,
  Card,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  AlertTitle,
  CircularProgress,
  Divider,
  ButtonGroup
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  Insights as InsightsIcon,
  Compare as CompareIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  Visibility as VisibilityIcon,
  Science as ScienceIcon
} from '@mui/icons-material';
import { 
  generateExplanation, 
  getAvailableMethods,
  setSelectedMethod,
  setSelectedProjectId,
  clearError,
  exportExplanation,
  selectXAI,
  selectFormattedCurrentExplanation
} from '../store/xaiSlice';
import { selectProjects } from '../store/projectSlice';

// Component imports for specific XAI visualizations
import SHAPVisualization from '../components/xai/SHAPVisualization';
import LIMEVisualization from '../components/xai/LIMEVisualization';
import IntegratedGradientsVisualization from '../components/xai/IntegratedGradientsVisualization';
import ExplanationComparison from '../components/xai/ExplanationComparison';
import MethodSelector from '../components/xai/MethodSelector';
import ExplanationHistory from '../components/xai/ExplanationHistory';

const XAI = () => {
  const dispatch = useDispatch();
  const xaiState = useSelector(selectXAI);
  const formattedExplanation = useSelector(selectFormattedCurrentExplanation);
  const allProjects = useSelector(selectProjects);

  const [activeTab, setActiveTab] = useState(0);
  const [selectedExplanationIds, setSelectedExplanationIds] = useState([]);

  // Component state
  const {
    currentExplanation,
    availableMethods,
    selectedMethod,
    selectedProjectId,
    loading,
    error,
    serviceStatus,
    explanations
  } = xaiState;

  // Initialize XAI methods on component mount
  useEffect(() => {
    dispatch(getAvailableMethods());
  }, [dispatch]);

  // Auto-select first project if none selected
  useEffect(() => {
    if (allProjects.length > 0 && !selectedProjectId) {
      dispatch(setSelectedProjectId(allProjects[0].id));
    }
  }, [allProjects, selectedProjectId, dispatch]);

  // Handle explanation generation
  const handleGenerateExplanation = async () => {
    if (!selectedProjectId || !selectedMethod) {
      return;
    }

    try {
      await dispatch(generateExplanation({
        project_id: selectedProjectId,
        explanation_method: selectedMethod
      })).unwrap();
    } catch (error) {
      console.error('Failed to generate explanation:', error);
    }
  };

  // Handle method selection
  const handleMethodChange = (method) => {
    dispatch(setSelectedMethod(method));
  };

  // Handle project selection
  const handleProjectChange = (event) => {
    dispatch(setSelectedProjectId(event.target.value));
  };

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Handle export
  const handleExport = (format = 'json') => {
    if (currentExplanation?.explanation_id) {
      dispatch(exportExplanation({
        explanationId: currentExplanation.explanation_id,
        format
      }));
    }
  };

  // Clear error handler
  const handleClearError = () => {
    dispatch(clearError());
  };

  // Service status indicator
  const getServiceStatusColor = () => {
    switch (serviceStatus) {
      case 'operational': return 'success';
      case 'unavailable': return 'error';
      default: return 'warning';
    }
  };

  const getServiceStatusText = () => {
    switch (serviceStatus) {
      case 'operational': return 'XAI Service Online';
      case 'unavailable': return 'XAI Service Unavailable';
      default: return 'XAI Service Status Unknown';
    }
  };

  // Tab panel component
  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`xai-tabpanel-${index}`}
      aria-labelledby={`xai-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PsychologyIcon fontSize="large" />
          Explainable AI (XAI)
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          Understand how AI models make predictions with comprehensive explanations and visualizations
        </Typography>
        
        {/* Service Status */}
        <Chip 
          icon={<ScienceIcon />}
          label={getServiceStatusText()}
          color={getServiceStatusColor()}
          variant="outlined"
          size="small"
        />
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" onClose={handleClearError} sx={{ mb: 3 }}>
          <AlertTitle>XAI Error</AlertTitle>
          {error}
        </Alert>
      )}

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Left Panel - Controls */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: 'fit-content' }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <InsightsIcon />
              Generate Explanation
            </Typography>

            {/* Project Selection */}
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Select Project</InputLabel>
              <Select
                value={selectedProjectId || ''}
                onChange={handleProjectChange}
                label="Select Project"
              >
                {allProjects.map((project) => (
                  <MenuItem key={project.id} value={project.id}>
                    {project.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Method Selection */}
            <MethodSelector
              methods={availableMethods}
              selectedMethod={selectedMethod}
              onMethodChange={handleMethodChange}
              loading={loading.methods}
            />

            {/* Generate Button */}
            <Button
              variant="contained"
              fullWidth
              size="large"
              onClick={handleGenerateExplanation}
              disabled={!selectedProjectId || !selectedMethod || loading.generating}
              startIcon={loading.generating ? <CircularProgress size={20} /> : <PsychologyIcon />}
              sx={{ mt: 2, mb: 2 }}
            >
              {loading.generating ? 'Generating...' : 'Generate Explanation'}
            </Button>

            {/* Export Options */}
            {currentExplanation && (
              <Box>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Export Results
                </Typography>
                <ButtonGroup variant="outlined" size="small" fullWidth>
                  <Button 
                    onClick={() => handleExport('json')}
                    disabled={loading.exporting}
                    startIcon={<DownloadIcon />}
                  >
                    JSON
                  </Button>
                  <Button 
                    onClick={() => handleExport('png')}
                    disabled={loading.exporting}
                    startIcon={<DownloadIcon />}
                  >
                    PNG
                  </Button>
                  <Button 
                    onClick={() => handleExport('pdf')}
                    disabled={loading.exporting}
                    startIcon={<DownloadIcon />}
                  >
                    PDF
                  </Button>
                </ButtonGroup>
              </Box>
            )}

            {/* Quick Stats */}
            {explanations.length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Quick Stats
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Card variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                      <Typography variant="h6">{explanations.length}</Typography>
                      <Typography variant="caption">Total</Typography>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                      <Typography variant="h6">
                        {new Set(explanations.map(e => e.project_id)).size}
                      </Typography>
                      <Typography variant="caption">Projects</Typography>
                    </Card>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Right Panel - Results */}
        <Grid item xs={12} md={8}>
          {currentExplanation ? (
            <Paper sx={{ height: '800px', display: 'flex', flexDirection: 'column' }}>
              {/* Tabs */}
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={activeTab} onChange={handleTabChange} aria-label="XAI explanation tabs">
                  <Tab 
                    label="SHAP" 
                    icon={<TrendingUpIcon />}
                    disabled={!formattedExplanation?.formattedAnalysis?.shap}
                  />
                  <Tab 
                    label="LIME" 
                    icon={<VisibilityIcon />}
                    disabled={!formattedExplanation?.formattedAnalysis?.lime}
                  />
                  <Tab 
                    label="Integrated Gradients" 
                    icon={<ScienceIcon />}
                    disabled={!formattedExplanation?.formattedAnalysis?.integrated_gradients}
                  />
                  <Tab 
                    label="Comparison" 
                    icon={<CompareIcon />}
                  />
                </Tabs>
              </Box>

              {/* Tab Panels */}
              <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                <TabPanel value={activeTab} index={0}>
                  {formattedExplanation?.formattedAnalysis?.shap ? (
                    <SHAPVisualization data={formattedExplanation.formattedAnalysis.shap} />
                  ) : (
                    <Alert severity="info">
                      <AlertTitle>SHAP Explanation Not Available</AlertTitle>
                      Generate a SHAP explanation to view feature importance analysis.
                    </Alert>
                  )}
                </TabPanel>

                <TabPanel value={activeTab} index={1}>
                  {formattedExplanation?.formattedAnalysis?.lime ? (
                    <LIMEVisualization data={formattedExplanation.formattedAnalysis.lime} />
                  ) : (
                    <Alert severity="info">
                      <AlertTitle>LIME Explanation Not Available</AlertTitle>
                      Generate a LIME explanation to view local interpretable results.
                    </Alert>
                  )}
                </TabPanel>

                <TabPanel value={activeTab} index={2}>
                  {formattedExplanation?.formattedAnalysis?.integrated_gradients ? (
                    <IntegratedGradientsVisualization 
                      data={formattedExplanation.formattedAnalysis.integrated_gradients} 
                    />
                  ) : (
                    <Alert severity="info">
                      <AlertTitle>Integrated Gradients Explanation Not Available</AlertTitle>
                      Generate an Integrated Gradients explanation to view attribution analysis.
                    </Alert>
                  )}
                </TabPanel>

                <TabPanel value={activeTab} index={3}>
                  <ExplanationComparison 
                    currentExplanation={currentExplanation}
                    allExplanations={explanations}
                    selectedIds={selectedExplanationIds}
                    onSelectionChange={setSelectedExplanationIds}
                  />
                </TabPanel>
              </Box>
            </Paper>
          ) : (
            <Paper sx={{ p: 6, textAlign: 'center', height: '400px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <PsychologyIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2, mx: 'auto' }} />
              <Typography variant="h5" gutterBottom color="text.secondary">
                No Explanation Generated
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                Select a project and explanation method, then click "Generate Explanation" to get started.
              </Typography>
              <Box>
                <Button 
                  variant="outlined" 
                  startIcon={<InfoIcon />}
                  onClick={() => dispatch(getAvailableMethods())}
                  disabled={loading.methods}
                >
                  {loading.methods ? 'Loading...' : 'Check Available Methods'}
                </Button>
              </Box>
            </Paper>
          )}
        </Grid>
      </Grid>

      {/* Explanation History */}
      {explanations.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <ExplanationHistory 
            explanations={explanations}
            onExplanationSelect={(explanation) => {
              // This would load the selected explanation
              console.log('Selected explanation:', explanation);
            }}
          />
        </Box>
      )}
    </Box>
  );
};

export default XAI; 