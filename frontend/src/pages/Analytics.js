import React from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Card,
  CardContent,
  Alert
} from '@mui/material';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AssessmentIcon from '@mui/icons-material/Assessment';
import BarChartIcon from '@mui/icons-material/BarChart';
import { COMMON_STYLES } from '../theme/constants';

const Analytics = () => {
  const analyticsFeatures = [
    {
      title: 'Model Performance Metrics',
      description: 'Track accuracy, precision, and recall of ML models',
      icon: <TrendingUpIcon />
    },
    {
      title: 'Verification Trends',
      description: 'Analysis of verification patterns and success rates',
      icon: <AssessmentIcon />
    },
    {
      title: 'Carbon Impact Analytics',
      description: 'Aggregate carbon sequestration and emission trends',
      icon: <BarChartIcon />
    }
  ];

  return (
    <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <AnalyticsIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Typography variant="h4" gutterBottom>
          Analytics & Insights
        </Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Coming Soon:</strong> Advanced analytics dashboard with performance metrics, 
          verification trends, and carbon impact insights will be available in the next release.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {analyticsFeatures.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {feature.icon}
                  <Typography variant="h6" sx={{ ml: 2 }}>
                    {feature.title}
                  </Typography>
                </Box>
                <Typography variant="body2">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Planned Analytics Features
        </Typography>
        <Typography variant="body1">
          The analytics dashboard will provide comprehensive insights into system performance, 
          verification success rates, and carbon impact trends to help optimize the verification process.
        </Typography>
      </Paper>
    </Container>
  );
};

export default Analytics; 