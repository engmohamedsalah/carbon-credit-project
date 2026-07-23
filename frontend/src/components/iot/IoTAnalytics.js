import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Paper,
  Chip,
  LinearProgress,
  Alert,
  CircularProgress,
  Divider
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import {
  Sensors as SensorsIcon,
  TrendingUp,
  Battery4Bar,
  SignalWifi4Bar,
  CheckCircle,
  Warning,
  Error as ErrorIcon
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { fetchAnalytics } from '../../store/iotSlice';
import { iotService } from '../../services/iotService';

const COLORS = ['#4caf50', '#ff9800', '#f44336', '#2196f3', '#9c27b0', '#00bcd4'];

const IoTAnalytics = ({ projectId }) => {
  const dispatch = useDispatch();
  const { analytics, sensors, loading } = useSelector(state => state.iot);
  const [summaryStats, setSummaryStats] = useState(null);

  useEffect(() => {
    dispatch(fetchAnalytics(projectId));
  }, [dispatch, projectId]);

  useEffect(() => {
    if (sensors.length > 0) {
      calculateSummaryStats();
    }
  }, [sensors]);

  const calculateSummaryStats = () => {
    const total = sensors.length;
    const active = sensors.filter(s => s.status === 'active').length;
    const offline = sensors.filter(s => s.status === 'offline').length;
    const maintenance = sensors.filter(s => s.status === 'maintenance').length;

    // Calculate average battery and signal levels
    const activeSensors = sensors.filter(s => s.status === 'active' && s.last_reading);
    const avgBattery = activeSensors.length > 0 
      ? activeSensors.reduce((sum, s) => sum + (s.last_reading?.battery_level || 0), 0) / activeSensors.length 
      : 0;
    const avgSignal = activeSensors.length > 0
      ? activeSensors.reduce((sum, s) => sum + (s.last_reading?.signal_strength || 0), 0) / activeSensors.length
      : 0;

    // Group sensors by type
    const sensorTypes = sensors.reduce((acc, sensor) => {
      acc[sensor.sensor_type] = (acc[sensor.sensor_type] || 0) + 1;
      return acc;
    }, {});

    // Calculate health score based on various factors
    const healthScore = Math.round(
      (active / total) * 40 + // 40% weight for active sensors
      (avgBattery / 100) * 30 + // 30% weight for battery levels
      (avgSignal / 100) * 30     // 30% weight for signal strength
    );

    setSummaryStats({
      total,
      active,
      offline,
      maintenance,
      avgBattery: Math.round(avgBattery),
      avgSignal: Math.round(avgSignal),
      sensorTypes,
      healthScore,
      uptime: active / total * 100,
      lastUpdated: new Date()
    });
  };

  const getSensorTypeData = () => {
    if (!summaryStats) return [];
    
    return Object.entries(summaryStats.sensorTypes).map(([type, count]) => ({
      name: iotService.getSensorTypeConfig(type).label,
      value: count,
      color: iotService.getSensorTypeConfig(type).color
    }));
  };

  const getStatusData = () => {
    if (!summaryStats) return [];
    
    return [
      { name: 'Active', value: summaryStats.active, color: '#4caf50' },
      { name: 'Maintenance', value: summaryStats.maintenance, color: '#ff9800' },
      { name: 'Offline', value: summaryStats.offline, color: '#f44336' }
    ];
  };

  const getHealthColor = (score) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  if (loading.analytics) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!summaryStats) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info">No sensor readings yet.</Alert>
      </Box>
    );
  }

  const sensorTypeData = getSensorTypeData();
  const statusData = getStatusData();

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <SensorsIcon sx={{ mr: 1, color: 'primary.main' }} />
        IoT Network Analytics
      </Typography>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Total Sensors
                  </Typography>
                  <Typography variant="h4" color="primary.main">
                    {summaryStats.total}
                  </Typography>
                </Box>
                <SensorsIcon sx={{ fontSize: 40, color: 'primary.main', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Network Health
                  </Typography>
                  <Typography variant="h4" color={`${getHealthColor(summaryStats.healthScore)}.main`}>
                    {summaryStats.healthScore}%
                  </Typography>
                </Box>
                <CheckCircle sx={{ 
                  fontSize: 40, 
                  color: `${getHealthColor(summaryStats.healthScore)}.main`, 
                  opacity: 0.3 
                }} />
              </Box>
              <LinearProgress
                variant="determinate"
                value={summaryStats.healthScore}
                color={getHealthColor(summaryStats.healthScore)}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Avg Battery
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {summaryStats.avgBattery}%
                  </Typography>
                </Box>
                <Battery4Bar sx={{ fontSize: 40, color: 'success.main', opacity: 0.3 }} />
              </Box>
              <LinearProgress
                variant="determinate"
                value={summaryStats.avgBattery}
                color="success"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Avg Signal
                  </Typography>
                  <Typography variant="h4" color="info.main">
                    {summaryStats.avgSignal}%
                  </Typography>
                </Box>
                <SignalWifi4Bar sx={{ fontSize: 40, color: 'info.main', opacity: 0.3 }} />
              </Box>
              <LinearProgress
                variant="determinate"
                value={summaryStats.avgSignal}
                color="info"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Sensor Status Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Sensor Status Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {statusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, mt: 2 }}>
                <Chip icon={<CheckCircle />} label={`${summaryStats.active} Active`} color="success" size="small" />
                <Chip icon={<Warning />} label={`${summaryStats.maintenance} Maintenance`} color="warning" size="small" />
                <Chip icon={<ErrorIcon />} label={`${summaryStats.offline} Offline`} color="error" size="small" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Sensor Types Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Sensor Types Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={sensorTypeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="name" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                  />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#2196f3">
                    {sensorTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* System Status */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom color="success.main">
                      ✓ Operational Status
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      • {summaryStats.active} sensors actively reporting<br/>
                      • Network uptime: {summaryStats.uptime.toFixed(1)}%<br/>
                      • Last data received: {summaryStats.lastUpdated.toLocaleString()}
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom color="warning.main">
                      ⚠ Attention Required
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      • {summaryStats.maintenance} sensors in maintenance mode<br/>
                      • {summaryStats.offline} sensors offline<br/>
                      • Average battery level: {summaryStats.avgBattery}%
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default IoTAnalytics;