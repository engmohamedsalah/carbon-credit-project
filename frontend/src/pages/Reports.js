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
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import AssessmentIcon from '@mui/icons-material/Assessment';
import DownloadIcon from '@mui/icons-material/Download';
import DescriptionIcon from '@mui/icons-material/Description';
import HistoryIcon from '@mui/icons-material/History';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import { COMMON_STYLES } from '../theme/constants';

const Reports = () => {
  const reportTypes = [
    {
      title: 'Verification Certificates',
      description: 'Official blockchain-backed carbon credit certificates',
      icon: <VerifiedUserIcon />
    },
    {
      title: 'Audit Trail Reports',
      description: 'Complete verification history and decision logs',
      icon: <HistoryIcon />
    },
    {
      title: 'Technical Analysis',
      description: 'Detailed ML model outputs and explanations',
      icon: <DescriptionIcon />
    }
  ];

  return (
    <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <AssessmentIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Typography variant="h4" gutterBottom>
          Reports & Certificates
        </Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Report Generation:</strong> Automated report generation and certificate creation 
          are available through the verification workflow. Advanced reporting features coming soon.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Available Report Types
            </Typography>
            <List>
              {reportTypes.map((report, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {report.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={report.title}
                    secondary={report.description}
                  />
                  <Button 
                    variant="outlined" 
                    startIcon={<DownloadIcon />}
                    size="small"
                    disabled
                  >
                    Generate
                  </Button>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Report Features
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="• PDF certificate generation" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Blockchain verification links" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• QR codes for validation" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Multi-language support" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Digital signatures" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          How to Generate Reports
        </Typography>
        <Typography variant="body1">
          Reports and certificates are automatically generated when verification processes are completed. 
          You can access them through the verification details page or project history.
        </Typography>
      </Paper>
    </Container>
  );
};

export default Reports; 