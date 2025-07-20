import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  LinearProgress
} from '@mui/material';
import {
  Security as SecurityIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Lightbulb as RecommendationIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';

const RiskTab = ({ explanation }) => {
  const getRiskLevelIcon = (level) => {
    switch (level) {
      case 'Low': return <CheckCircleIcon color="success" />;
      case 'Medium': return <WarningIcon color="warning" />;
      case 'High': return <ErrorIcon color="error" />;
      default: return <SecurityIcon />;
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'Low': return 'success';
      case 'Medium': return 'warning';
      case 'High': return 'error';
      default: return 'default';
    }
  };

  const getRiskScore = (level) => {
    switch (level) {
      case 'Low': return 85;
      case 'Medium': return 60;
      case 'High': return 30;
      default: return 50;
    }
  };

  if (!explanation?.risk_assessment) {
    return (
      <Box sx={{ textAlign: 'center', py: 6 }}>
        <SecurityIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No risk assessment available
        </Typography>
      </Box>
    );
  }

  const { risk_assessment } = explanation;
  const riskScore = getRiskScore(risk_assessment.level);

  return (
    <Box sx={{ p: { xs: 2, md: 3 } }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <SecurityIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Risk Assessment
          </Typography>
          <Chip
            icon={getRiskLevelIcon(risk_assessment.level)}
            label={`${risk_assessment.level} Risk`}
            color={getRiskColor(risk_assessment.level)}
            sx={{ fontWeight: 500 }}
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          Comprehensive risk analysis for carbon credit verification
        </Typography>
      </Box>

      {/* Risk Overview */}
      <Box sx={{ 
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, 1fr)' },
        gap: 4,
        mb: 4
      }}>
        {/* Risk Level Card */}
        <Paper sx={{ p: 3, bgcolor: `${getRiskColor(risk_assessment.level)}.50` }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            {getRiskLevelIcon(risk_assessment.level)}
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {risk_assessment.level} Risk Level
            </Typography>
          </Box>
          
          <Typography variant="body1" sx={{ mb: 3, lineHeight: 1.6 }}>
            {risk_assessment.description}
          </Typography>

          {/* Risk Score Visualization */}
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2" fontWeight={500}>
                Risk Score
              </Typography>
              <Typography variant="body2" fontWeight={600}>
                {riskScore}/100
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={riskScore}
              sx={{
                height: 8,
                borderRadius: 4,
                bgcolor: 'rgba(0,0,0,0.1)',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 4,
                  bgcolor: risk_assessment.level === 'Low' ? 'success.main' :
                           risk_assessment.level === 'Medium' ? 'warning.main' : 'error.main'
                }
              }}
            />
          </Box>
        </Paper>

        {/* Risk Factors */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            Risk Factors
          </Typography>
          
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Assessment based on:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <AssessmentIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Data quality and completeness" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <AssessmentIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Model confidence levels" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <AssessmentIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Verification methodology" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <AssessmentIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="External validation results" />
              </ListItem>
            </List>
          </Box>
        </Paper>
      </Box>

      {/* Risk Alert */}
      {risk_assessment.level !== 'Low' && (
        <Alert 
          severity={risk_assessment.level === 'Medium' ? 'warning' : 'error'}
          sx={{ mb: 4 }}
          icon={getRiskLevelIcon(risk_assessment.level)}
        >
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            {risk_assessment.level} Risk Detected
          </Typography>
          <Typography variant="body2">
            {risk_assessment.level === 'Medium' 
              ? 'Additional verification steps may be recommended before final approval.'
              : 'This analysis requires careful review and additional validation before proceeding.'
            }
          </Typography>
        </Alert>
      )}

      {/* Mitigation Recommendations */}
      {risk_assessment.mitigation_recommendations && (
        <Paper sx={{ p: 3, bgcolor: 'info.50' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
            <RecommendationIcon sx={{ color: 'info.main' }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Mitigation Recommendations
            </Typography>
          </Box>
          
          <List>
            {risk_assessment.mitigation_recommendations.map((recommendation, index) => (
              <ListItem key={index} sx={{ px: 0, py: 1 }}>
                <ListItemIcon>
                  <CheckCircleIcon color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText 
                  primary={recommendation}
                  sx={{
                    '& .MuiListItemText-primary': {
                      fontSize: '1rem',
                      lineHeight: 1.5
                    }
                  }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Additional Information */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
          Risk assessment performed using AI analysis at {new Date(explanation.timestamp).toLocaleString()}.
          This assessment should be reviewed by qualified personnel before making final decisions.
        </Typography>
      </Box>
    </Box>
  );
};

export default React.memo(RiskTab); 