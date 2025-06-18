import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Alert,
  AlertTitle
} from '@mui/material';
import {
  Compare as CompareIcon
} from '@mui/icons-material';

const ExplanationComparison = ({ currentExplanation, allExplanations, selectedIds, onSelectionChange }) => {
  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CompareIcon />
          Explanation Comparison
        </Typography>
      </Box>

      {/* Placeholder for comparison functionality */}
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Alert severity="info">
          <AlertTitle>Comparison Feature Coming Soon</AlertTitle>
          <Typography variant="body2">
            The explanation comparison feature will allow you to compare different XAI methods side-by-side 
            to understand how different algorithms interpret the same prediction.
          </Typography>
        </Alert>
        
        {currentExplanation && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="body2" color="text.secondary">
              Current explanation: {currentExplanation.explanation_id}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Method: {currentExplanation.method}
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default ExplanationComparison; 