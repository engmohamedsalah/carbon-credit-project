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
  Divider
} from '@mui/material';
import {
  Gavel as ComplianceIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Assignment as DocumentIcon,
  Verified as VerifiedIcon
} from '@mui/icons-material';

const ComplianceTab = ({ explanation }) => {
  if (!explanation?.regulatory_notes) {
    return (
      <Box sx={{ textAlign: 'center', py: 6 }}>
        <ComplianceIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No compliance information available
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Regulatory compliance details will appear here when analysis is complete
        </Typography>
      </Box>
    );
  }

  const complianceItems = Object.entries(explanation.regulatory_notes);
  const getComplianceStatus = (key, value) => {
    if (value.toLowerCase().includes('compliant') || value.toLowerCase().includes('meets')) {
      return 'success';
    }
    if (value.toLowerCase().includes('partial') || value.toLowerCase().includes('review')) {
      return 'warning';
    }
    return 'info';
  };

  return (
    <Box sx={{ p: { xs: 2, md: 3 } }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <ComplianceIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Regulatory Compliance
          </Typography>
          <Chip 
            label="Standards Review"
            size="small"
            color="primary"
            variant="outlined"
            sx={{ fontWeight: 500 }}
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          Compliance assessment against international carbon credit standards
        </Typography>
      </Box>

      {/* Compliance Overview */}
      <Paper sx={{ p: 3, mb: 4, bgcolor: 'info.50' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <VerifiedIcon sx={{ color: 'info.main' }} />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Compliance Status Overview
          </Typography>
        </Box>
        <Typography variant="body1" sx={{ lineHeight: 1.6 }}>
          This analysis has been evaluated against major carbon credit verification standards 
          including VCS (Verified Carbon Standard), CDM (Clean Development Mechanism), 
          and Gold Standard requirements.
        </Typography>
      </Paper>

      {/* Compliance Items */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
          Detailed Compliance Assessment
        </Typography>
        
        <Box sx={{ 
          display: 'grid',
          gap: 3
        }}>
          {complianceItems.map(([key, value], index) => (
            <Paper key={key} elevation={1} sx={{ overflow: 'hidden' }}>
              {/* Header */}
              <Box sx={{ 
                p: 2, 
                bgcolor: `${getComplianceStatus(key, value)}.100`,
                borderBottom: '1px solid',
                borderColor: 'divider'
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                  <CheckCircleIcon 
                    sx={{ 
                      color: `${getComplianceStatus(key, value)}.main`,
                      fontSize: 20 
                    }} 
                  />
                  <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.1rem' }}>
                    {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Typography>
                  <Chip
                    label={
                      getComplianceStatus(key, value) === 'success' ? 'Compliant' :
                      getComplianceStatus(key, value) === 'warning' ? 'Needs Review' : 'Info'
                    }
                    size="small"
                    color={getComplianceStatus(key, value)}
                    sx={{ fontWeight: 500 }}
                  />
                </Box>
              </Box>
              
              {/* Content */}
              <Box sx={{ p: 3 }}>
                <Typography variant="body1" sx={{ lineHeight: 1.6 }}>
                  {value}
                </Typography>
              </Box>
            </Paper>
          ))}
        </Box>
      </Box>

      {/* Standards Information */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
          Applicable Standards & Frameworks
        </Typography>
        
        <List>
          <ListItem>
            <ListItemIcon>
              <DocumentIcon color="primary" />
            </ListItemIcon>
            <ListItemText
              primary="Verified Carbon Standard (VCS)"
              secondary="World's most used voluntary GHG program with robust verification requirements"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <DocumentIcon color="primary" />
            </ListItemIcon>
            <ListItemText
              primary="Clean Development Mechanism (CDM)"
              secondary="UN framework for emission reduction projects in developing countries"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <DocumentIcon color="primary" />
            </ListItemIcon>
            <ListItemText
              primary="Gold Standard"
              secondary="Premium certification ensuring projects deliver sustainable development benefits"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <DocumentIcon color="primary" />
            </ListItemIcon>
            <ListItemText
              primary="ISO 14064"
              secondary="International standard for greenhouse gas accounting and verification"
            />
          </ListItem>
        </List>
      </Paper>

      {/* Important Notice */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
          Important Compliance Notice
        </Typography>
        <Typography variant="body2">
          This automated compliance assessment is provided for initial guidance only. 
          Final certification requires formal review by accredited verification bodies 
          and may involve additional documentation and on-site verification procedures.
        </Typography>
      </Alert>

      {/* Next Steps */}
      <Paper sx={{ p: 3, bgcolor: 'primary.50' }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Recommended Next Steps
        </Typography>
        
        <List dense>
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon color="success" fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Review all compliance items and address any flagged issues" />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon color="success" fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Compile supporting documentation for verification body review" />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon color="success" fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Engage with accredited verification body for formal assessment" />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon color="success" fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Prepare for potential on-site verification if required" />
          </ListItem>
        </List>
      </Paper>

      {/* Footer */}
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
          Compliance assessment generated on {new Date(explanation.timestamp).toLocaleString()}.
          For questions about specific standards or requirements, consult with qualified verification professionals.
        </Typography>
      </Box>
    </Box>
  );
};

export default React.memo(ComplianceTab); 