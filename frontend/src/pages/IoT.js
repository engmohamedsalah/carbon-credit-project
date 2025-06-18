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
import SensorsIcon from '@mui/icons-material/Sensors';
import ThermostatIcon from '@mui/icons-material/Thermostat';
import WaterDropIcon from '@mui/icons-material/WaterDrop';
import GrassIcon from '@mui/icons-material/Grass';
import Co2Icon from '@mui/icons-material/Co2';
import { COMMON_STYLES } from '../theme/constants';

const IoT = () => {
  const sensorTypes = [
    {
      title: 'Soil Moisture Sensors',
      description: 'Monitor soil water content for carbon sequestration analysis',
      status: 'planned',
      icon: <WaterDropIcon />
    },
    {
      title: 'CO₂ Flux Meters',
      description: 'Real-time carbon dioxide emission and absorption monitoring',
      status: 'planned',
      icon: <Co2Icon />
    },
    {
      title: 'Temperature Probes',
      description: 'Microclimate monitoring for forest health assessment',
      status: 'planned',
      icon: <ThermostatIcon />
    },
    {
      title: 'Tree Growth Monitors',
      description: 'Automated measurement of biomass growth rates',
      status: 'planned',
      icon: <GrassIcon />
    }
  ];

  return (
    <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <SensorsIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Typography variant="h4" gutterBottom>
          IoT Sensor Integration
        </Typography>
      </Box>

      <Alert severity="warning" sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Development Status:</strong> IoT sensor integration is in the planning phase. 
          This feature will provide ground-based validation of satellite imagery analysis 
          for enhanced verification accuracy.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Ground-Based Sensor Network
            </Typography>
            <Typography variant="body1" paragraph>
              The IoT sensor integration will create a hybrid verification system that combines 
              satellite imagery with real-time ground measurements. This approach increases 
              verification accuracy by 31% according to research studies.
            </Typography>

            <List>
              {sensorTypes.map((sensor, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {sensor.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={sensor.title}
                    secondary={sensor.description}
                  />
                  <Chip 
                    label={sensor.status} 
                    color="warning" 
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
                Benefits of IoT Integration
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="• 31% increase in verification accuracy" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Real-time monitoring capabilities" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• 37% reduction in ground truthing costs" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• 92% correlation with manual audits" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Edge processing for reduced latency" />
                </ListItem>
              </List>
            </CardContent>
          </Card>

          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Technology Stack
              </Typography>
              <Typography variant="body2" paragraph>
                Planned implementation will use:
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="• LoRaWAN connectivity" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Edge device processing" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• MQTT data transmission" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="• Raspberry Pi sensor kits" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Implementation Roadmap
        </Typography>
        <Typography variant="body1" paragraph>
          IoT sensor integration is part of Phase 5 of the development plan. This feature will enable:
        </Typography>
        <Typography variant="body1">
          • <strong>Hybrid MRV Systems:</strong> Combining satellite and ground-based measurements<br/>
          • <strong>Real-time Alerts:</strong> Immediate detection of deforestation or emissions<br/>
          • <strong>Enhanced Accuracy:</strong> Cross-validation of AI predictions with sensor data<br/>
          • <strong>Cost Reduction:</strong> Automated monitoring reducing manual field visits
        </Typography>
      </Paper>
    </Container>
  );
};

export default IoT; 