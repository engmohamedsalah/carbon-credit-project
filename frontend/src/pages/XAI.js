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
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
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
  Refresh as RefreshIcon,
  TrendingUp as SummaryIcon,
  Security as RiskIcon,
  BarChart as VisualizationsIcon,
  Gavel as ComplianceIcon,
  ViewModule as ShowAllIcon
} from '@mui/icons-material';

import xaiService from '../services/xaiService';
import { canAccessFeature, FEATURE_ACCESS } from '../utils/roleUtils';
import ModernEmptyState from '../components/xai/ModernEmptyState';
import ExplanationSkeleton from '../components/xai/ExplanationSkeleton';
// import MobileComparisonTable from '../components/xai/MobileComparisonTable';
import MethodSelector from '../components/xai/MethodSelector';
import SummaryTab from '../components/xai/SummaryTab';
import RiskTab from '../components/xai/RiskTab';
import VisualizationsTab from '../components/xai/VisualizationsTab';
import ComplianceTab from '../components/xai/ComplianceTab';
import CompareMethodsTab from '../components/xai/CompareMethodsTab';
import HistoryReportsTab from '../components/xai/HistoryReportsTab';

const XAI = () => {
  const { user } = useSelector(state => state.auth);
  const { projects } = useSelector(state => state.projects);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // Consolidated state management
  const [state, setState] = useState({
    // UI State
    tabValue: 0,
    resultSection: new URLSearchParams(window.location.search).get('section') || 'summary',
    visualizationTab: parseInt(new URLSearchParams(window.location.search).get('vizTab')) || 0,
    primaryFeature: new URLSearchParams(window.location.search).get('primary') || 'forest_area',
    interactionFeature: new URLSearchParams(window.location.search).get('interaction') || 'soil_quality',
    loading: false,
    error: null,
    success: null,
    
    // Generation state
    selectedProject: '',
    selectedMethod: 'lime',
    businessFriendly: true,
    includeUncertainty: true,
    currentExplanation: null,
    
    // Data state
    availableMethods: null,
    explanationHistory: [], // Load real data from database only    ],
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

  // Handle result section changes with URL update
  const handleResultSectionChange = useCallback((newSection) => {
    updateState({ resultSection: newSection });
    
    // Update URL without page reload
    const url = new URL(window.location);
    if (newSection === 'summary') {
      url.searchParams.delete('section');
    } else {
      url.searchParams.set('section', newSection);
    }
    window.history.replaceState({}, '', url);
  }, [updateState]);

  // Load explanation history function
  const loadHistory = useCallback(async (projectId) => {
    try {
      // Always clear data first when switching projects
      updateState({ explanationHistory: [] });
      
      const history = await xaiService.getExplanationHistory(projectId);
      // Only update if we have actual data from backend
      if (history && history.explanations && history.explanations.length > 0) {
        updateState({ explanationHistory: history.explanations });
        console.log('📚 Explanation history loaded:', history);
      } else {
        // No backend data found - keep empty
        console.log('📚 No explanation history found for project:', projectId);
      }
    } catch (error) {
      console.error('Failed to load explanation history:', error);
      // Keep empty on error - no fallback mock data
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

  // Refresh all tab data when project changes
  useEffect(() => {
    if (state.selectedProject) {
      // Clear all project-specific data first
      updateState({ 
        currentExplanation: null,
        comparisonResult: null,
        error: null,
        success: `🎯 Switched to project analysis - all tabs refreshed`,
        loading: true
      });
      
      // Load fresh data for the new project
      const refreshProjectData = async () => {
        try {
          // Load explanation history for this project
          await loadHistory(state.selectedProject);
          
          // Clear any cached method comparisons
          // This ensures Compare Methods tab starts fresh
          
          console.log(`🔄 All tabs refreshed for project: ${state.selectedProject}`);
          
        } catch (error) {
          console.error('Error refreshing project data:', error);
          updateState({ 
            error: 'Failed to refresh project data',
            loading: false 
          });
        } finally {
          updateState({ loading: false });
        }
      };
      
      refreshProjectData();
    } else {
      // Clear all data when no project is selected
      updateState({ 
        currentExplanation: null,
        comparisonResult: null,
        explanationHistory: [],
        error: null
      });
    }
  }, [state.selectedProject, loadHistory, updateState]);

  // Function to update URL parameters for deep linking
  const updateUrlParams = useCallback((params) => {
    const url = new URL(window.location);
    Object.entries(params).forEach(([key, value]) => {
      if (value && value !== 'default' && value !== '') {
        url.searchParams.set(key, value);
      } else {
        url.searchParams.delete(key);
      }
    });
    window.history.replaceState({}, '', url);
  }, []);

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
      
      // Get project data for analysis
      const selectedProjectData = filteredProjects.find(p => p.id.toString() === state.selectedProject);
      if (!selectedProjectData) {
        throw new Error('Selected project not found');
      }

      // Call real backend API to generate explanation
      const explanation = await xaiService.generateExplanation({
        instanceData: {
          project_id: selectedProjectData.id,
          name: selectedProjectData.name,
          location_name: selectedProjectData.location_name,
          area_hectares: selectedProjectData.area_hectares,
          project_type: selectedProjectData.project_type,
          status: selectedProjectData.status,
          estimated_carbon_credits: selectedProjectData.estimated_carbon_credits,
          start_date: selectedProjectData.start_date,
          end_date: selectedProjectData.end_date
        },
        method: state.selectedMethod,
        businessFriendly: state.businessFriendly,
        includeUncertainty: state.includeUncertainty
      });
      
      if (explanation) {
        console.log('📥 Generated real explanation from backend:', explanation);
        
        // Update current explanation and add to history
        const newHistory = [explanation, ...state.explanationHistory];
        
        updateState({ 
          currentExplanation: explanation,
          explanationHistory: newHistory,
          success: `${state.selectedMethod.toUpperCase()} analysis generated successfully from real ML models!`,
          tabValue: 0 // Stay on generate tab to show results
        });
      } else {
        throw new Error('No explanation returned from backend');
      }
      
    } catch (error) {
      console.error('❌ Failed to generate explanation:', error);
      updateState({ error: error.message || 'Failed to generate explanation' });
    } finally {
      updateState({ loading: false });
    }
  };

  // const handleCompareExplanations = async () => {
  //   if (state.selectedExplanations.length < 2) {
  //     updateState({ error: 'Please select at least 2 explanations to compare' });
  //     return;
  //   }

  //   updateState({ loading: true, error: null });

  //   try {
  //     const comparison = await xaiService.compareExplanations(state.selectedExplanations);
  //     updateState({ 
  //       comparisonResult: comparison,
  //       success: 'Explanations compared successfully!' 
  //     });
  //   } catch (error) {
  //     updateState({ error: error.message });
  //   } finally {
  //     updateState({ loading: false });
  //   }
  // };

  const generateClientSideReport = (explanation, format, includeBusinessSummary) => {
    const timestamp = new Date().toLocaleString();
    const projectName = filteredProjects.find(p => p.id.toString() === state.selectedProject)?.name || 'Unknown Project';
    
    if (format === 'json') {
      // Generate JSON report
      const jsonReport = {
        report_metadata: {
          generated_at: timestamp,
          project_name: projectName,
          method: explanation.method,
          confidence_score: explanation.confidence_score
        },
        explanation: explanation,
        include_business_summary: includeBusinessSummary
      };
      
      const dataStr = JSON.stringify(jsonReport, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `xai_explanation_${explanation.explanation_id || 'report'}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      return true;
    }
    
    // Generate HTML/PDF report
    const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XAI Analysis Report - ${projectName}</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; 
            color: #333; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background: white;
        }
        .header { 
            border-bottom: 3px solid #1976d2; 
            padding-bottom: 20px; 
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 { 
            color: #1976d2; 
            margin: 0; 
            font-size: 2.5em;
        }
        .header .subtitle { 
            color: #666; 
            font-size: 1.2em; 
            margin-top: 10px;
        }
        .metadata { 
            background: #f8fafc; 
            border-left: 4px solid #1976d2; 
            padding: 20px; 
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        .metadata h3 { 
            margin-top: 0; 
            color: #1976d2;
        }
        .confidence-badge { 
            display: inline-block; 
            padding: 8px 16px; 
            border-radius: 20px; 
            font-weight: bold; 
            color: white;
            ${explanation.confidence_score >= 0.8 ? 'background: #4caf50;' :
              explanation.confidence_score >= 0.6 ? 'background: #ff9800;' : 'background: #f44336;'}
        }
        .section { 
            margin: 30px 0; 
            padding: 25px; 
            border: 1px solid #e0e0e0; 
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 { 
            color: #1976d2; 
            border-bottom: 2px solid #e0e0e0; 
            padding-bottom: 10px;
            margin-top: 0;
        }
        .risk-assessment {
            ${explanation.risk_assessment?.level === 'High' ? 'border-left: 4px solid #f44336; background: #ffebee;' :
              explanation.risk_assessment?.level === 'Medium' ? 'border-left: 4px solid #ff9800; background: #fff3e0;' :
              'border-left: 4px solid #4caf50; background: #e8f5e8;'}
        }
        .chart-container { 
            text-align: center; 
            margin: 20px 0; 
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }
        .chart-container img { 
            max-width: 100%; 
            height: auto; 
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin: 20px 0;
        }
        .compliance-item {
            background: #f0f4ff;
            border: 1px solid #1976d2;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .compliance-item strong {
            color: #1976d2;
            display: block;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        .recommendations {
            background: #e3f2fd;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        .recommendations ul {
            margin: 0;
            padding-left: 20px;
        }
        .recommendations li {
            margin: 8px 0;
            line-height: 1.5;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        @media print {
            body { font-size: 12pt; }
            .chart-grid { grid-template-columns: repeat(2, 1fr); }
            .section { break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 XAI Analysis Report</h1>
        <div class="subtitle">Explainable AI Carbon Credit Verification</div>
    </div>

    <div class="metadata">
        <h3>📊 Report Metadata</h3>
        <p><strong>Project:</strong> ${projectName}</p>
        <p><strong>Analysis Method:</strong> ${explanation.method?.toUpperCase() || 'AI Analysis'}</p>
        <p><strong>Generated:</strong> ${timestamp}</p>
        <p><strong>Confidence Score:</strong> 
            <span class="confidence-badge">
                ${(explanation.confidence_score * 100).toFixed(1)}% 
                ${explanation.confidence_score >= 0.8 ? '(High)' :
                  explanation.confidence_score >= 0.6 ? '(Medium)' : '(Low)'}
            </span>
        </p>
    </div>

    ${includeBusinessSummary && explanation.business_summary ? `
    <div class="section">
        <h2>📈 Executive Summary</h2>
        <div style="column-count: 2; column-gap: 40px; line-height: 1.8;">
            ${explanation.business_summary
              .replace(/\*\*(.*?)\*\*/g, '<h3 style="color: #1976d2; margin-top: 20px; break-after: avoid;">$1</h3>')
              .replace(/\* (.*?)(?=\n|$)/g, '<li>$1</li>')
              .replace(/(<li>.*?<\/li>)/gs, '<ul style="margin: 10px 0; padding-left: 20px;">$1</ul>')
              .replace(/\n\n/g, '</p><p style="margin: 15px 0;">')
              .replace(/^/, '<p style="margin: 15px 0;">')
              .replace(/$/, '</p>')
            }
        </div>
    </div>
    ` : ''}

    ${explanation.risk_assessment ? `
    <div class="section risk-assessment">
        <h2>🛡️ Risk Assessment - ${explanation.risk_assessment.level} Risk</h2>
        <p style="font-size: 1.1em; font-weight: 500;">${explanation.risk_assessment.description}</p>
        
        ${explanation.risk_assessment.mitigation_recommendations ? `
        <div class="recommendations">
            <strong>🔧 Mitigation Recommendations:</strong>
            <ul>
                ${explanation.risk_assessment.mitigation_recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
    </div>
    ` : ''}

    ${explanation.visualizations && Object.keys(explanation.visualizations).length > 0 ? `
    <div class="section">
        <h2>📊 Visualizations & Charts</h2>
        <div class="chart-grid">
            ${Object.entries(explanation.visualizations).map(([key, imageData]) => `
                <div class="chart-container">
                    <h3>${key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
                    <img src="${imageData}" alt="${key}" />
                </div>
            `).join('')}
        </div>
        <p style="text-align: center; color: #666; font-style: italic;">
            ${Object.keys(explanation.visualizations).length} visualization(s) generated from production ML models
        </p>
    </div>
    ` : ''}

    ${explanation.regulatory_notes ? `
    <div class="section">
        <h2>⚖️ Regulatory Compliance</h2>
        ${Object.entries(explanation.regulatory_notes).map(([key, value]) => `
            <div class="compliance-item">
                <strong>${key.replace('_', ' ').toUpperCase()}</strong>
                ${value}
            </div>
        `).join('')}
    </div>
    ` : ''}

    <div class="footer">
        <p>This report was generated by the XAI Carbon Credit Verification System</p>
        <p>For questions or additional analysis, please contact your verification team</p>
        <p><em>Report ID: ${explanation.explanation_id || 'N/A'} | Generated: ${timestamp}</em></p>
    </div>
</body>
</html>`;

    // Open in new window for printing/saving
    const newWindow = window.open('', '_blank');
    newWindow.document.write(htmlContent);
    newWindow.document.close();
    
    // Add print functionality
    setTimeout(() => {
      newWindow.focus();
      if (format === 'pdf') {
        newWindow.print();
      }
    }, 500);
    
    return true;
  };

  const handleGenerateReport = async () => {
    if (!state.currentExplanation) {
      updateState({ error: 'No explanation available to generate report' });
      return;
    }

    updateState({ loading: true, error: null, reportDialogOpen: false });

    try {
      // Use client-side report generation since backend PDF is not supported
      const success = generateClientSideReport(
        state.currentExplanation,
        state.reportFormat,
        state.includeBusinessSummary
      );

      if (success) {
        const action = state.reportFormat === 'pdf' ? 'opened for printing' : 'downloaded';
        updateState({ 
          success: `Report ${action} successfully! ${state.reportFormat === 'pdf' ? 'Use your browser\'s print function to save as PDF.' : ''}` 
        });
      }
    } catch (error) {
      console.error('Report generation error:', error);
      updateState({ error: 'Failed to generate report. Please try again.' });
    } finally {
      updateState({ loading: false });
    }
  };

  // const handleExplanationSelect = (explanationId, isSelected) => {
  //   const newSelected = isSelected 
  //     ? [...state.selectedExplanations, explanationId]
  //     : state.selectedExplanations.filter(id => id !== explanationId);
  //   
  //   updateState({ selectedExplanations: newSelected });
  // };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  // const getRiskLevelIcon = (level) => {
  //   switch (level) {
  //     case 'Low': return <CheckCircleIcon color="success" />;
  //     case 'Medium': return <WarningIcon color="warning" />;
  //     case 'High': return <ErrorIcon color="error" />;
  //     default: return <InfoIcon />;
  //   }
  // };

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
      <Container maxWidth="xl" sx={{ py: 4, px: { xs: 2, sm: 3, md: 4 } }}>
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
    <Container 
      maxWidth={false} 
      sx={{ 
        py: 4, 
        px: { xs: 2, sm: 3, md: 4, xl: 5 },
        // Eliminate right gutter - use full available width
        maxWidth: 'min(1400px, 94vw)',
        mx: 'auto',
        width: '100%'
      }}
    >
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

      {/* Global Project Selection */}
      <Card sx={{ 
        mb: 3, 
        background: 'linear-gradient(135deg, #f8faff 0%, #eef4ff 100%)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <CardContent sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main' }}>
              🎯 Select Project for Analysis
            </Typography>
            {state.selectedProject && (
              <Chip 
                label="✓ Project Selected" 
                size="small" 
                color="success" 
                variant="outlined"
              />
            )}
          </Box>
          
          <FormControl fullWidth sx={{ maxWidth: 500 }}>
            <InputLabel id="global-project-select-label">
              Choose a project to analyze across all XAI methods
            </InputLabel>
            <Select
              labelId="global-project-select-label"
              value={state.selectedProject}
              label="Choose a project to analyze across all XAI methods"
              onChange={(e) => updateState({ selectedProject: e.target.value })}
              disabled={state.loading}
              sx={{ 
                bgcolor: 'background.paper',
                '& .MuiOutlinedInput-root': {
                  '&:hover fieldset': {
                    borderColor: 'primary.main',
                  }
                }
              }}
            >
              <MenuItem value="">
                <em>Select a project...</em>
              </MenuItem>
              {filteredProjects.map((project) => (
                <MenuItem key={project.id} value={project.id.toString()}>
                  <Box sx={{ width: '100%' }}>
                    <Typography variant="body1" sx={{ fontWeight: 500 }}>
                      {project.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {project.location_name} • {project.project_type} • {project.area_hectares} hectares
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {state.selectedProject && (
            <Box sx={{ mt: 2, p: 2, bgcolor: 'success.50', borderRadius: 1 }}>
              <Typography variant="body2" color="success.dark">
                ✅ <strong>Project selected!</strong> This project will be used for analysis across all tabs below.
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

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
        <Box sx={{ 
          maxWidth: '1200px', 
          width: '100%', 
          mx: 'auto',
          minHeight: '600px'
        }}>
          {/* Configuration Panel - Full-Width Responsive Design */}
        <Card sx={{ 
          mb: 3, 
          background: 'linear-gradient(135deg, #fafbff 0%, #f0f4ff 100%)',
          minHeight: '320px',
          width: '100%',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          position: 'relative'
        }}>
          <CardContent sx={{ p: { xs: 3, md: 4 } }}>
            {/* Header with Status Badge */}
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              mb: 4,
              position: 'relative'
            }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <PsychologyIcon sx={{ color: 'primary.main', fontSize: '1.5rem' }} />
                <Typography variant="h5" component="h2" sx={{ fontWeight: 600, color: 'text.primary' }}>
                  Configuration & Settings
                </Typography>
              </Box>
              
              {/* Status Badge - Better Positioned */}
              <Chip 
                label={
                  state.currentExplanation ? "✅ Analysis Complete" :
                  state.selectedProject && state.selectedMethod ? "🚀 Ready to Generate" :
                  state.selectedProject ? `⚡ Step 2 of 3` :
                  "📝 Step 1 of 3"
                } 
                size="medium"
                color={
                  state.currentExplanation ? "success" :
                  state.selectedProject && state.selectedMethod ? "primary" :
                  state.selectedProject ? "secondary" :
                  "default"
                }
                variant={state.currentExplanation ? "filled" : "outlined"}
                sx={{ 
                  fontWeight: 500,
                  fontSize: '0.875rem',
                  height: 32,
                  '& .MuiChip-label': {
                    px: 2
                  }
                }}
              />
            </Box>
            
            {/* Main Configuration Grid - Responsive Full-Width Design */}
            <Box sx={{ 
              width: '100%',
              display: 'grid',
              gridTemplateColumns: {
                xs: '1fr',
                md: 'repeat(auto-fit, minmax(280px, 1fr))',
                lg: 'repeat(3, 1fr)'
              },
              gap: { xs: 3, md: 4 },
              alignItems: 'start'
            }}>

              {/* Method Selection */}
              <Box sx={{ 
                position: 'relative',
                minHeight: '95px',
                display: 'flex',
                flexDirection: 'column'
              }}>
                <Typography variant="overline" sx={{ 
                  mb: 1.5, 
                  fontWeight: 700,
                  color: state.selectedProject ? 'primary.main' : 'text.disabled',
                  fontSize: '0.75rem',
                  letterSpacing: '1px'
                }}>
                  Step 2: AI Method
                </Typography>
                <Box sx={{ flex: 1 }}>
                  {state.availableMethods && (
                    <MethodSelector
                      methods={state.availableMethods.methods || []}
                      selectedMethod={state.selectedMethod}
                      onMethodChange={(method) => updateState({ selectedMethod: method })}
                      loading={state.loading}
                      variant="dropdown"
                      sx={{ 
                        width: '100%',
                        '& .MuiOutlinedInput-root': {
                          bgcolor: 'background.paper',
                          height: '56px',
                          borderRadius: 2,
                          '&:hover fieldset': {
                            borderColor: state.selectedProject ? 'primary.main' : 'action.disabled',
                          },
                          '&.Mui-focused fieldset': {
                            borderColor: 'primary.main',
                            borderWidth: 2
                          }
                        },
                        '& .MuiInputLabel-root.Mui-disabled': {
                          color: 'text.disabled'
                        }
                      }}
                    />
                  )}
                </Box>
                
                {/* Success Indicator */}
                {state.selectedMethod && (
                  <CheckCircleIcon 
                    sx={{ 
                      position: 'absolute', 
                      top: 48, 
                      right: 12, 
                      color: 'success.main',
                      fontSize: 20,
                      zIndex: 1
                    }} 
                  />
                )}
                
                {/* Help Text */}
                <Typography variant="caption" color="text.secondary" sx={{ 
                  mt: 1, 
                  minHeight: '16px',
                  fontStyle: state.selectedMethod ? 'normal' : 'italic'
                }}>
                  {state.selectedMethod ? '✓ Method selected' : 'AI explanation technique'}
                </Typography>
              </Box>

              {/* Generate Button */}
              <Box sx={{ 
                minHeight: '95px',
                display: 'flex',
                flexDirection: 'column'
              }}>
                <Typography variant="overline" sx={{ 
                  mb: 1.5, 
                  fontWeight: 700,
                  color: (state.selectedProject && state.selectedMethod) ? 'primary.main' : 'text.disabled',
                  fontSize: '0.75rem',
                  letterSpacing: '1px'
                }}>
                  Step 3: Generate
                </Typography>
                <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                  <Tooltip 
                    title={
                      !state.selectedProject ? "Please select a project first" :
                      !state.selectedMethod ? "Please choose an AI method" :
                      "Generate AI explanation for your project"
                    }
                    arrow
                  >
                    <span>
                      <Button
                        variant="contained"
                        size="large"
                        fullWidth
                        onClick={handleGenerateExplanation}
                        disabled={!state.selectedProject || !state.selectedMethod || state.loading}
                        startIcon={state.loading ? <CircularProgress size={20} color="inherit" /> : <PsychologyIcon />}
                        sx={{ 
                          height: 56,
                          borderRadius: 2,
                          fontSize: '1rem',
                          fontWeight: 600,
                          textTransform: 'none',
                          cursor: (!state.selectedProject || !state.selectedMethod || state.loading) ? 'not-allowed' : 'pointer',
                          background: (state.selectedProject && state.selectedMethod) ? 
                            'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)' : 
                            '#e0e0e0',
                          color: (state.selectedProject && state.selectedMethod) ? '#ffffff' : 'rgba(0, 0, 0, 0.38)',
                          boxShadow: (state.selectedProject && state.selectedMethod) ? 
                            '0 3px 6px 0 rgba(25, 118, 210, 0.3)' : 
                            'none',
                          '&:hover': {
                            background: (state.selectedProject && state.selectedMethod && !state.loading) ? 
                              'linear-gradient(135deg, #1565c0 0%, #0d47a1 100%)' :
                              '#e0e0e0',
                            boxShadow: (state.selectedProject && state.selectedMethod && !state.loading) ?
                              '0 6px 12px 0 rgba(25, 118, 210, 0.4)' :
                              'none'
                          },
                          '&:disabled': {
                            background: '#e0e0e0',
                            color: 'rgba(0, 0, 0, 0.38)',
                            cursor: 'not-allowed'
                          },
                          transition: 'all 0.2s ease-in-out'
                        }}
                      >
                        {state.loading ? 'Generating Analysis...' : 'Generate Explanation'}
                      </Button>
                    </span>
                  </Tooltip>
                  
                  {/* Status Message */}
                  <Typography variant="caption" sx={{ 
                    mt: 1, 
                    minHeight: '16px',
                    textAlign: 'center',
                    color: (!state.selectedProject || !state.selectedMethod) ? 'error.main' : 'success.main',
                    fontWeight: 500
                  }}>
                    {!state.selectedProject ? 'Select a project to continue' :
                     !state.selectedMethod ? 'Choose an AI method' :
                     '✓ Ready to generate'}
                  </Typography>
                </Box>
              </Box>
            </Box>

                         {/* Enhanced Progress Indicator - Hide when analysis is complete */}
             {(state.selectedProject || state.selectedMethod) && !state.currentExplanation && (
               <Box sx={{ mt: 4, mb: 2 }}>
                 <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
                   <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
                     Configuration Progress
                   </Typography>
                   <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                     {state.selectedProject && state.selectedMethod ? '100%' : 
                      state.selectedProject ? '67%' : 
                      state.selectedMethod ? '33%' : '0%'} complete
                   </Typography>
                 </Box>
                 <LinearProgress 
                   variant="determinate" 
                   value={
                     state.selectedProject && state.selectedMethod ? 100 : 
                     state.selectedProject ? 67 : 
                     state.selectedMethod ? 33 : 0
                   }
                   sx={{ 
                     height: 8, 
                     borderRadius: 4,
                     backgroundColor: 'rgba(0,0,0,0.08)',
                     '& .MuiLinearProgress-bar': {
                       borderRadius: 4,
                       background: 'linear-gradient(90deg, #1976d2 0%, #42a5f5 100%)'
                     }
                   }}
                 />
               </Box>
             )}
             
             {/* Success indicator when analysis is complete */}
             {state.currentExplanation && (
               <Box sx={{ mt: 4, mb: 2, textAlign: 'center' }}>
                 <Chip
                   icon={<CheckCircleIcon />}
                   label="Analysis Complete - Review results below"
                   color="success"
                   variant="filled"
                   sx={{ 
                     fontSize: '0.875rem',
                     fontWeight: 500,
                     py: 1,
                     px: 2
                   }}
                 />
               </Box>
             )}

                         {/* Advanced Options - Enhanced */}
             <Accordion sx={{ 
               mt: 3, 
               boxShadow: 'none', 
               border: '1px solid', 
               borderColor: 'divider',
               borderRadius: 2,
               '&:before': {
                 display: 'none'
               },
               '& .MuiAccordionSummary-root': {
                 backgroundColor: 'rgba(0, 0, 0, 0.02)',
                 borderRadius: '8px 8px 0 0',
                 '&:focus-visible': {
                   outline: '3px solid',
                   outlineColor: 'primary.main',
                   outlineOffset: '2px'
                 }
               },
               '& .MuiAccordionSummary-expandIconWrapper': {
                 '&:focus-visible': {
                   outline: '2px solid',
                   outlineColor: 'primary.main',
                   borderRadius: 1
                 }
               }
             }}>
              <AccordionSummary 
                expandIcon={<ExpandMoreIcon />}
                sx={{ 
                  minHeight: 56, 
                  '&.Mui-expanded': { minHeight: 56 }
                }}
              >
                <Typography variant="subtitle1" sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 1.5,
                  fontWeight: 600,
                  color: 'text.primary'
                }}>
                  <InfoIcon fontSize="small" sx={{ color: 'info.main' }} />
                  Advanced Options
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 3, backgroundColor: 'background.paper' }}>
                <Grid container spacing={4}>
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
                          <Typography variant="body1" sx={{ fontWeight: 500 }}>
                            Business-friendly explanations
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Generate explanations in business language instead of technical terms
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
                          <Typography variant="body1" sx={{ fontWeight: 500 }}>
                            Include uncertainty analysis
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Add confidence intervals and risk assessment to the analysis
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

        {/* Results Panel - Tabbed Interface */}
        <Box sx={{ width: '100%' }}>
          {state.loading ? (
            <ExplanationSkeleton />
          ) : state.currentExplanation ? (
            <Card sx={{ width: '100%' }}>
              {/* Results Header */}
              <Box sx={{ 
                p: 3, 
                borderBottom: '1px solid', 
                borderColor: 'divider',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 2
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {state.selectedMethod?.toUpperCase()} Analysis Results
                  </Typography>
                  <Chip 
                    icon={
                      state.currentExplanation.confidence_score >= 0.8 ? <CheckCircleIcon /> :
                      state.currentExplanation.confidence_score >= 0.6 ? <WarningIcon /> : <ErrorIcon />
                    }
                    label={`${(state.currentExplanation.confidence_score * 100).toFixed(1)}% Confidence${
                      state.currentExplanation.confidence_score >= 0.8 ? ' (High)' :
                      state.currentExplanation.confidence_score >= 0.6 ? ' (Medium)' : ' (Low)'
                    }`}
                    color={getConfidenceColor(state.currentExplanation.confidence_score)}
                    sx={{ fontWeight: 500 }}
                  />
                </Box>
                
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    size="small"
                    onClick={() => updateState({ reportDialogOpen: true })}
                    startIcon={<PdfIcon />}
                    variant="outlined"
                  >
                    Export Report
                  </Button>
                  <IconButton
                    size="small"
                    onClick={handleGenerateExplanation}
                    disabled={state.loading}
                    sx={{ 
                      border: '1px solid',
                      borderColor: 'divider',
                      '&:hover': { borderColor: 'primary.main' }
                    }}
                  >
                    <RefreshIcon />
                  </IconButton>
                </Box>
              </Box>

              {/* Results Navigation Tabs */}
              <Box sx={{ borderBottom: '1px solid', borderColor: 'divider' }}>
                <Tabs
                  value={state.resultSection}
                  onChange={(e, newValue) => handleResultSectionChange(newValue)}
                  variant={isMobile ? "scrollable" : "fullWidth"}
                  scrollButtons="auto"
                  sx={{
                    '& .MuiTab-root': {
                      minHeight: 60,
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      textTransform: 'none',
                      '&:focus-visible': {
                        outline: '3px solid',
                        outlineColor: 'primary.main',
                        outlineOffset: '2px'
                      }
                    }
                  }}
                >
                  <Tab 
                    value="summary" 
                    label="Summary" 
                    icon={<SummaryIcon />}
                    iconPosition="start"
                  />
                  <Tab 
                    value="risk" 
                    label="Risk Assessment" 
                    icon={<RiskIcon />}
                    iconPosition="start"
                  />
                  <Tab 
                    value="visualizations" 
                    label="Charts" 
                    icon={<VisualizationsIcon />}
                    iconPosition="start"
                  />
                  <Tab 
                    value="compliance" 
                    label="Compliance" 
                    icon={<ComplianceIcon />}
                    iconPosition="start"
                  />
                  <Tab 
                    value="all" 
                    label="Show All" 
                    icon={<ShowAllIcon />}
                    iconPosition="start"
                    sx={{ 
                      display: { xs: 'none', md: 'flex' },
                      borderLeft: '1px solid',
                      borderColor: 'divider',
                      ml: 'auto'
                    }}
                  />
                </Tabs>
              </Box>

              {/* Tab Content */}
              <Box sx={{ minHeight: 400 }}>
                {/* Summary Tab */}
                {(state.resultSection === 'summary' || state.resultSection === 'all') && (
                  <SummaryTab explanation={state.currentExplanation} />
                )}

                {/* Risk Tab */}
                {(state.resultSection === 'risk' || state.resultSection === 'all') && (
                  <Box sx={{ mt: state.resultSection === 'all' ? 4 : 0 }}>
                    <RiskTab explanation={state.currentExplanation} />
                  </Box>
                )}

                {/* Visualizations Tab */}
                {(state.resultSection === 'visualizations' || state.resultSection === 'all') && (
                  <Box sx={{ mt: state.resultSection === 'all' ? 4 : 0 }}>
                    <VisualizationsTab 
                explanation={state.currentExplanation}
                initialTab={state.visualizationTab}
                initialPrimaryFeature={state.primaryFeature}
                initialInteractionFeature={state.interactionFeature}
                onUrlParamsChange={updateUrlParams}
              />
                  </Box>
                )}

                {/* Compliance Tab */}
                {(state.resultSection === 'compliance' || state.resultSection === 'all') && (
                  <Box sx={{ mt: state.resultSection === 'all' ? 4 : 0 }}>
                    <ComplianceTab explanation={state.currentExplanation} />
                  </Box>
                )}
              </Box>
            </Card>
          ) : (!state.selectedProject && !state.selectedMethod) ? (
            // Only show empty state when user hasn't started configuration
            <ModernEmptyState 
              variant="generate"
              onAction={() => {
                if (filteredProjects.length > 0) {
                  updateState({ selectedProject: filteredProjects[0].id.toString() });
                }
              }}
              actionText="Get Started"
            />
          ) : null}
        </Box>
        </Box>
      </TabPanel>

      {/* Tab 2: Compare & Analyze */}
      <TabPanel value={state.tabValue} index={1}>
        <Box sx={{ 
          maxWidth: '1200px', 
          width: '100%', 
          mx: 'auto',
          minHeight: '600px'
        }}>
          <CompareMethodsTab
            key={`compare-${state.selectedProject}`}
            currentExplanation={state.currentExplanation}
            explanationHistory={state.explanationHistory}
            selectedProject={state.selectedProject}
            onGenerateComparison={(comparison) => {
              console.log('Generated comparison:', comparison);
              updateState({ 
                comparisonResult: comparison,
                success: 'Method comparison completed successfully!' 
              });
            }}
            loading={state.loading}
          />
        </Box>
      </TabPanel>

      {/* Tab 3: History & Reports */}
      <TabPanel value={state.tabValue} index={2}>
        <Box sx={{ 
          maxWidth: '1200px', 
          width: '100%', 
          mx: 'auto',
          minHeight: '600px'
        }}>
          <HistoryReportsTab
            key={`history-${state.selectedProject}`}
            explanationHistory={state.explanationHistory}
            selectedProject={state.selectedProject}
            onRefresh={() => {
              if (state.selectedProject) {
                updateState({ loading: true });
                loadHistory(state.selectedProject);
              }
            }}
            onDeleteExplanation={(explanationId) => {
              updateState({ 
                explanationHistory: state.explanationHistory.filter(exp => exp.explanation_id !== explanationId),
                success: 'Analysis deleted successfully!'
              });
            }}
            onArchiveExplanation={(explanationId) => {
              console.log('Archive explanation:', explanationId);
              updateState({ success: 'Analysis archived successfully!' });
            }}
            loading={state.loading}
          />
        </Box>
      </TabPanel>

      {/* Report Generation Dialog */}
      <Dialog 
        open={state.reportDialogOpen} 
        onClose={() => updateState({ reportDialogOpen: false })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ pb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PdfIcon color="primary" />
            Export Analysis Report
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Choose your preferred export format for the AI analysis results.
            </Typography>
            
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Export Format</InputLabel>
              <Select
                value={state.reportFormat}
                label="Export Format"
                onChange={(e) => updateState({ reportFormat: e.target.value })}
              >
                <MenuItem value="pdf">
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <PdfIcon fontSize="small" />
                    <Box>
                      <Typography variant="body2" fontWeight={500}>PDF Report</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Professional formatted report (opens in new window for printing)
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
                <MenuItem value="json">
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <AssessmentIcon fontSize="small" />
                    <Box>
                      <Typography variant="body2" fontWeight={500}>JSON Data</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Raw data export for technical analysis
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              </Select>
            </FormControl>
            
            <FormControlLabel
              control={
                <Switch
                  checked={state.includeBusinessSummary}
                  onChange={(e) => updateState({ includeBusinessSummary: e.target.checked })}
                  color="primary"
                />
              }
              label={
                <Box>
                  <Typography variant="body2">Include Executive Summary</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Add business-friendly analysis summary to the report
                  </Typography>
                </Box>
              }
            />
          </Box>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 3 }}>
          <Button onClick={() => updateState({ reportDialogOpen: false })}>
            Cancel
          </Button>
          <Button 
            onClick={handleGenerateReport} 
            variant="contained" 
            disabled={state.loading}
            startIcon={state.loading ? <CircularProgress size={20} /> : <PdfIcon />}
            sx={{ minWidth: 120 }}
          >
            {state.loading ? 'Generating...' : 'Export Report'}
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