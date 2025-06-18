import React from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Card,
  CardContent,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip
} from '@mui/material';
import PsychologyIcon from '@mui/icons-material/Psychology';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ExploreIcon from '@mui/icons-material/Explore';
import { COMMON_STYLES } from '../theme/constants';

const XAI = () => {
  const xaiFeatures = [
    {
      title: 'SHAP Explanations',
      description: 'Shapley values for model feature importance',
      status: 'implemented',
      icon: <TrendingUpIcon />
    },
    {
      title: 'LIME Visualizations',
      description: 'Local interpretable model-agnostic explanations',
      status: 'implemented',
      icon: <ExploreIcon />
    },
    {
      title: 'Integrated Gradients',
      description: 'Attribution maps for deep learning models',
      status: 'implemented',
      icon: <VisibilityIcon />
    },
    {
      title: 'Class Activation Maps',
      description: 'Visual attention maps for forest change detection',
      status: 'implemented',
      icon: <PsychologyIcon />
    }
  ];

  return (
    <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <PsychologyIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Typography variant="h4" gutterBottom>
          Explainable AI (XAI)
        </Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Feature Status:</strong> XAI components are implemented in the ML pipeline but the visualization interface is under development. 
          Advanced explanations are available through the verification process.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Model Transparency Features
            </Typography>
            <Typography variant="body1" paragraph>
              Our Explainable AI system provides transparent insights into how machine learning models 
              make decisions for carbon credit verification. This ensures accountability and builds trust 
              in the verification process.
            </Typography>

            <List>
              {xaiFeatures.map((feature, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {feature.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={feature.title}
                    secondary={feature.description}
                  />
                  <Chip 
                    label={feature.status} 
                    color="success" 
                    size="small"
                    sx={{ textTransform: 'capitalize' }}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                XAI in Verification
              </Typography>
              <Typography variant="body2">
                Explainable AI components are integrated into the verification workflow 
                to provide:
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="• Visual feature importance maps" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Confidence score explanations" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Decision boundary analysis" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Model uncertainty quantification" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          How to Access XAI Features
        </Typography>
        <Typography variant="body1">
          XAI explanations are currently available through the <strong>AI Verification</strong> page when analyzing projects. 
          The dedicated XAI dashboard with interactive visualizations will be available in the next release.
        </Typography>
      </Paper>
    </Container>
  );
};

export default XAI; 