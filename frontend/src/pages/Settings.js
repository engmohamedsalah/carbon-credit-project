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
import SettingsIcon from '@mui/icons-material/Settings';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import NotificationsIcon from '@mui/icons-material/Notifications';
import SecurityIcon from '@mui/icons-material/Security';
import ApiIcon from '@mui/icons-material/Api';
import { COMMON_STYLES } from '../theme/constants';

const Settings = () => {
  const settingsCategories = [
    {
      title: 'Profile Settings',
      description: 'Manage your account information and preferences',
      icon: <AccountCircleIcon />
    },
    {
      title: 'Notifications',
      description: 'Configure email and system notifications',
      icon: <NotificationsIcon />
    },
    {
      title: 'Security',
      description: 'Password, two-factor authentication, and access logs',
      icon: <SecurityIcon />
    },
    {
      title: 'API Configuration',
      description: 'API keys and integration settings',
      icon: <ApiIcon />
    }
  ];

  return (
    <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <SettingsIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Settings Interface:</strong> Advanced settings and configuration options 
          are under development. Basic account management is available through the profile menu.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {settingsCategories.map((category, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {category.icon}
                  <Typography variant="h6" sx={{ ml: 2 }}>
                    {category.title}
                  </Typography>
                </Box>
                <Typography variant="body2">
                  {category.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Current Account Information
        </Typography>
        <Typography variant="body1">
          You can access basic account settings through the profile menu in the top-right corner. 
          Advanced settings and system configuration will be available in future releases.
        </Typography>
      </Paper>
    </Container>
  );
};

export default Settings; 