import React from 'react';
import {
  Box,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Divider
} from '@mui/material';
import {
  History as HistoryIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';

const ExplanationHistory = ({ explanations, onExplanationSelect }) => {
  if (!explanations || explanations.length === 0) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <HistoryIcon />
          Explanation History
        </Typography>
        <Typography variant="body2" color="text.secondary">
          No explanations generated yet. Create your first explanation above.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <HistoryIcon />
        Explanation History ({explanations.length})
      </Typography>
      
      <List>
        {explanations.slice(0, 10).map((explanation, index) => (
          <React.Fragment key={explanation.explanation_id}>
            <ListItem>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2">
                      Project {explanation.project_id}
                    </Typography>
                    <Chip 
                      label={explanation.method.toUpperCase()} 
                      size="small" 
                      color="primary"
                    />
                  </Box>
                }
                secondary={
                  <Typography variant="body2" color="text.secondary">
                    {new Date(explanation.timestamp).toLocaleString()}
                  </Typography>
                }
              />
              <ListItemSecondaryAction>
                <IconButton 
                  edge="end" 
                  onClick={() => onExplanationSelect(explanation)}
                  size="small"
                >
                  <VisibilityIcon />
                </IconButton>
              </ListItemSecondaryAction>
            </ListItem>
            {index < explanations.length - 1 && <Divider />}
          </React.Fragment>
        ))}
      </List>
      
      {explanations.length > 10 && (
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 2 }}>
          Showing 10 of {explanations.length} explanations
        </Typography>
      )}
    </Paper>
  );
};

export default ExplanationHistory;