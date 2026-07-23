import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  Alert,
  Grid,
  Checkbox
} from '@mui/material';
import {
  Compare as CompareIcon,
  Assessment as AssessmentIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';

import xaiService from '../../services/xaiService';

const CompareMethodsTab = ({
  currentExplanation,
  explanationHistory,
  onGenerateComparison,
  loading = false
}) => {
  const [selectedExplanations, setSelectedExplanations] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [compareError, setCompareError] = useState(null);

  // Available XAI methods information (qualitative descriptions only — no
  // fabricated benchmark numbers)
  const methodInfo = {
    shap: {
      name: 'SHAP',
      fullName: 'SHapley Additive exPlanations',
      description: 'Game theory-based feature attribution',
      strengths: ['Theoretically grounded', 'Consistent', 'Global insights'],
      weaknesses: ['Computationally expensive', 'Complex interpretation'],
      useCase: 'Comprehensive feature importance analysis'
    },
    lime: {
      name: 'LIME',
      fullName: 'Local Interpretable Model-agnostic Explanations',
      description: 'Local surrogate model explanations',
      strengths: ['Model-agnostic', 'Fast execution', 'Local focus'],
      weaknesses: ['Instance-specific only', 'Unstable results'],
      useCase: 'Quick local explanations for individual predictions'
    },
    integrated_gradients: {
      name: 'Integrated Gradients',
      fullName: 'Integrated Gradients Attribution',
      description: 'Path-based gradient attribution method',
      strengths: ['Deep learning friendly', 'Smooth attributions', 'Axiomatically sound'],
      weaknesses: ['Requires baseline selection', 'Neural network specific'],
      useCase: 'Deep learning model explanations with smooth attributions'
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceIcon = (confidence) => {
    if (confidence >= 0.8) return <CheckCircleIcon />;
    if (confidence >= 0.6) return <WarningIcon />;
    return <ErrorIcon />;
  };

  const handleExplanationSelect = (explanationId, isSelected) => {
    if (isSelected) {
      setSelectedExplanations(prev => [...prev, explanationId]);
    } else {
      setSelectedExplanations(prev => prev.filter(id => id !== explanationId));
    }
  };

  const handleCompare = async () => {
    if (selectedExplanations.length < 2) {
      return;
    }

    setComparisonResult(null);
    setCompareError(null);

    // Comparison is computed by the backend from real explanations. We never
    // fabricate results client-side; a 503 surfaces an honest disabled message.
    try {
      const comparison = await xaiService.compareExplanations(selectedExplanations);
      setComparisonResult(comparison);
      if (onGenerateComparison) {
        onGenerateComparison(comparison);
      }
    } catch (error) {
      setCompareError(
        error.disabled
          ? 'Method comparison is not available in this build.'
          : error.message || 'Failed to compare explanations'
      );
    }
  };

  if (!explanationHistory || explanationHistory.length === 0) {
    return (
      <Box sx={{ p: { xs: 2, md: 3 }, width: '100%', maxWidth: 'none' }}>
        <Box sx={{ textAlign: 'center', py: 6 }}>
          <CompareIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No Methods to Compare
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Generate explanations using different methods to compare their results
          </Typography>
          <Alert severity="info" sx={{ maxWidth: 500, mx: 'auto' }}>
            <Typography variant="body2">
              To compare methods, generate analyses using different XAI techniques (SHAP, LIME, Integrated Gradients) 
              for the same project, then return to this tab to compare their insights.
            </Typography>
          </Alert>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 2, md: 3 }, width: '100%', maxWidth: 'none' }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <CompareIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Compare XAI Methods
          </Typography>
          <Chip 
            label={`${explanationHistory.length} Available`}
            size="small"
            color="primary"
            variant="outlined"
            sx={{ fontWeight: 500 }}
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          Compare different explainable AI methods to understand their strengths and differences
        </Typography>
      </Box>

      {/* Method Information Cards */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Available XAI Methods
        </Typography>
        <Grid container spacing={3}>
          {Object.entries(methodInfo).map(([key, info]) => (
            <Grid item xs={12} sm={6} md={4} key={key}>
              <Card sx={{ height: '100%', position: 'relative' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <PsychologyIcon color="primary" />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {info.name}
                    </Typography>
                  </Box>
                  
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                    {info.fullName}
                  </Typography>
                  
                  <Typography variant="body2" sx={{ mb: 2, lineHeight: 1.5 }}>
                    {info.description}
                  </Typography>

                  <Typography variant="caption" color="primary" sx={{ fontWeight: 600 }}>
                    Best for: {info.useCase}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Explanation Selection */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Select Explanations to Compare
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Choose 2 or more explanations from your analysis history to compare their results
        </Typography>

        <TableContainer>
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
              {explanationHistory.map((explanation) => (
                <TableRow key={explanation.explanation_id} hover>
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedExplanations.includes(explanation.explanation_id)}
                      onChange={(e) => handleExplanationSelect(explanation.explanation_id, e.target.checked)}
                      color="primary"
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <PsychologyIcon fontSize="small" color="primary" />
                      <Typography variant="body2" fontWeight={500}>
                        {methodInfo[explanation.method]?.name || explanation.method?.toUpperCase()}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {new Date(explanation.timestamp).toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip 
                      icon={getConfidenceIcon(explanation.confidence_score)}
                      label={`${(explanation.confidence_score * 100).toFixed(1)}%`}
                      color={getConfidenceColor(explanation.confidence_score)}
                      size="small"
                      sx={{ fontWeight: 500 }}
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                      {explanation.business_summary?.substring(0, 100) || 'No summary available'}...
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        <Box sx={{ mt: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
          <Button
            variant="contained"
            onClick={handleCompare}
            disabled={selectedExplanations.length < 2 || loading}
            startIcon={loading ? <AssessmentIcon /> : <CompareIcon />}
            sx={{ minWidth: 180 }}
          >
            {loading ? 'Comparing...' : `Compare Selected (${selectedExplanations.length})`}
          </Button>
          
          {selectedExplanations.length < 2 && (
            <Typography variant="caption" color="text.secondary">
              Select at least 2 explanations to compare
            </Typography>
          )}
        </Box>

        {compareError && (
          <Alert severity="info" sx={{ mt: 3 }}>
            {compareError}
          </Alert>
        )}
      </Paper>

      {/* Comparison Results */}
      {comparisonResult && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
            Comparison Results
          </Typography>

          {/* Methods Compared */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
              Methods Compared:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {comparisonResult.methods_compared?.map((method, index) => (
                <Chip 
                  key={index}
                  label={methodInfo[method]?.name || method.toUpperCase()}
                  color="primary"
                  variant="outlined"
                  size="small"
                />
              ))}
            </Box>
          </Box>

          {/* Confidence Comparison */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
              Confidence Comparison:
            </Typography>
            <Grid container spacing={2}>
              {comparisonResult.confidence_comparison?.map((item, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" fontWeight={600}>
                      {methodInfo[item.method]?.name || item.method.toUpperCase()}
                    </Typography>
                    <Typography variant="h4" color="primary" sx={{ fontWeight: 700, my: 1 }}>
                      {(item.confidence * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(item.timestamp).toLocaleDateString()}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Box>

          {/* Recommendation */}
          {comparisonResult.recommendation && (
            <Alert severity="info" sx={{ mb: 3 }}>
              <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
                Recommended Method:
              </Typography>
              <Typography variant="body2">
                <strong>{methodInfo[comparisonResult.recommendation.method]?.name}</strong> shows
                the highest confidence score ({(comparisonResult.recommendation.confidence_score * 100).toFixed(1)}%)
                for this analysis.
              </Typography>
            </Alert>
          )}

          {/* Key Differences */}
          {comparisonResult.differences?.length > 0 && (
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
                Key Differences:
              </Typography>
              <Box component="ul" sx={{ pl: 2, m: 0 }}>
                {comparisonResult.differences.map((diff, index) => (
                  <Typography component="li" variant="body2" key={index} sx={{ mb: 1 }}>
                    {diff}
                  </Typography>
                ))}
              </Box>
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default React.memo(CompareMethodsTab); 