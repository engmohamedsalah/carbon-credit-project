import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Chip,
  Box,
  IconButton,
  Tooltip,
  LinearProgress,
  Avatar
} from '@mui/material';
import {
  Sensors as SensorIcon,
  LocationOn,
  Battery4Bar,
  SignalWifi4Bar,
  MoreVert,
  Warning,
  CheckCircle,
  Error as ErrorIcon
} from '@mui/icons-material';
import { iotService } from '../../services/iotService';

const SensorCard = ({ sensor, onSelect, onMenuClick, showLastReading = true, projectName }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'success';
      case 'maintenance': return 'warning';
      case 'offline': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active': return <CheckCircle />;
      case 'maintenance': return <Warning />;
      case 'offline': return <ErrorIcon />;
      default: return <SensorIcon />;
    }
  };

  const getBatteryLevel = (sensor) => {
    return sensor.last_reading?.battery_level || 0;
  };

  const getSignalStrength = (sensor) => {
    return sensor.last_reading?.signal_strength || 0;
  };

  const getBatteryColor = (level) => {
    if (level > 60) return 'success';
    if (level > 30) return 'warning';
    return 'error';
  };

  const getSignalColor = (strength) => {
    if (strength > 70) return 'success';
    if (strength > 40) return 'warning';
    return 'error';
  };

  const formatLastReading = (sensor) => {
    if (!sensor.last_reading) return 'No data';
    
    const { value, timestamp } = sensor.last_reading;
    const sensorConfig = iotService.getSensorTypeConfig(sensor.sensor_type);
    const formattedValue = iotService.formatReadingValue(value, sensor.sensor_type);
    const timeAgo = getTimeAgo(new Date(timestamp));
    
    return `${formattedValue} ${sensorConfig.unit} (${timeAgo})`;
  };

  const getTimeAgo = (date) => {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  const getReadingPercentage = (sensor) => {
    if (!sensor.last_reading) return 0;
    
    const config = iotService.getSensorTypeConfig(sensor.sensor_type);
    const { value } = sensor.last_reading;
    const { min, max } = config.range;
    
    return Math.min(Math.max(((value - min) / (max - min)) * 100, 0), 100);
  };

  const sensorConfig = iotService.getSensorTypeConfig(sensor.sensor_type);

  // Safely format coordinates to avoid render crashes if values are not numeric
  const lat = Number(sensor?.location_lat);
  const lng = Number(sensor?.location_lng);
  const latText = Number.isFinite(lat) ? lat.toFixed(4) : 'N/A';
  const lngText = Number.isFinite(lng) ? lng.toFixed(4) : 'N/A';

  return (
    <Card 
      sx={{ 
        cursor: 'pointer',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        '&:hover': { 
          transform: 'translateY(-2px)', 
          boxShadow: 3,
          borderColor: 'primary.main'
        },
        transition: 'all 0.2s ease-in-out',
        border: '1px solid',
        borderColor: 'divider'
      }}
      onClick={() => onSelect && onSelect(sensor)}
    >
      <CardContent sx={{ flex: 1, pb: 2 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Avatar 
              sx={{ 
                bgcolor: sensorConfig.color + '20',
                color: sensorConfig.color,
                width: 40,
                height: 40,
                mr: 1
              }}
            >
              <SensorIcon />
            </Avatar>
            <Box>
              <Typography variant="h6" component="h3" sx={{ fontSize: '1rem', fontWeight: 600 }}>
                {sensor.sensor_id}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {sensorConfig.label}
              </Typography>
              {projectName && (
                <Typography variant="caption" color="primary.main" sx={{ fontWeight: 500 }}>
                  Project: {projectName}
                </Typography>
              )}
            </Box>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Chip 
              label={sensor.status} 
              color={getStatusColor(sensor.status)}
              size="small"
              icon={getStatusIcon(sensor.status)}
              sx={{ textTransform: 'capitalize' }}
            />
            <IconButton 
              size="small" 
              onClick={(e) => {
                e.stopPropagation();
                onMenuClick && onMenuClick(e, sensor);
              }}
            >
              <MoreVert />
            </IconButton>
          </Box>
        </Box>

        {/* Location */}
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <LocationOn sx={{ fontSize: 16, mr: 0.5, color: 'text.secondary' }} />
          <Typography variant="body2" color="text.secondary">
            {latText}, {lngText}
          </Typography>
        </Box>

        {/* Last Reading */}
        {showLastReading && sensor.last_reading && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Latest Reading:
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="h5" color={sensorConfig.color} sx={{ fontWeight: 600 }}>
                {iotService.formatReadingValue(sensor.last_reading.value, sensor.sensor_type)}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                {sensorConfig.unit}
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={getReadingPercentage(sensor)}
              sx={{ 
                height: 6, 
                borderRadius: 3,
                backgroundColor: sensorConfig.color + '20',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: sensorConfig.color
                }
              }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
              {getTimeAgo(new Date(sensor.last_reading.timestamp))}
            </Typography>
          </Box>
        )}

        {/* System Status */}
        {sensor.last_reading && (
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 'auto' }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Battery4Bar 
                sx={{ 
                  fontSize: 16, 
                  mr: 0.5, 
                  color: getBatteryColor(getBatteryLevel(sensor)) + '.main'
                }} 
              />
              <Typography variant="caption">
                {getBatteryLevel(sensor)}%
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <SignalWifi4Bar 
                sx={{ 
                  fontSize: 16, 
                  mr: 0.5,
                  color: getSignalColor(getSignalStrength(sensor)) + '.main'
                }} 
              />
              <Typography variant="caption">
                {getSignalStrength(sensor)}%
              </Typography>
            </Box>
          </Box>
        )}

        {!sensor.last_reading && (
          <Box sx={{ textAlign: 'center', py: 2 }}>
            <Typography variant="body2" color="text.secondary">
              No recent data
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Sensor may be offline or not yet configured
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SensorCard;