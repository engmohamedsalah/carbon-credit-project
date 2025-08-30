import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Chip,
  Grid,
  Paper
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine
} from 'recharts';
import { useSelector, useDispatch } from 'react-redux';
import { fetchSensorReadings } from '../../store/iotSlice';
import { iotService } from '../../services/iotService';
import { format, parseISO, subHours, subDays } from 'date-fns';

const ReadingsChart = ({ sensorId, sensorType, sensor }) => {
  const dispatch = useDispatch();
  const { readings, loading } = useSelector(state => state.iot);
  const [timeRange, setTimeRange] = useState('24h');
  const [chartData, setChartData] = useState([]);
  const [stats, setStats] = useState(null);

  const timeRangeOptions = [
    { value: '1h', label: 'Last Hour' },
    { value: '6h', label: 'Last 6 Hours' },
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' }
  ];

  useEffect(() => {
    if (sensorId) {
      dispatch(fetchSensorReadings({ 
        sensorId, 
        timeRange: { period: timeRange } 
      }));
    }
  }, [dispatch, sensorId, timeRange]);

  useEffect(() => {
    const sensorReadings = readings[sensorId] || [];
    
    if (sensorReadings.length === 0) {
      // Generate sample data for demo purposes
      setChartData(generateSampleData(sensorType, timeRange));
      setStats(calculateStats(generateSampleData(sensorType, timeRange)));
      return;
    }

    const formattedData = sensorReadings
      .map(reading => ({
        timestamp: parseISO(reading.timestamp),
        value: parseFloat(reading.value || 0),
        battery: reading.battery_level || 0,
        signal: reading.signal_strength || 0,
        formattedTime: format(parseISO(reading.timestamp), getTimeFormat(timeRange))
      }))
      .sort((a, b) => a.timestamp - b.timestamp);

    setChartData(formattedData);
    setStats(calculateStats(formattedData));
  }, [readings, sensorId, sensorType, timeRange]);

  const generateSampleData = (sensorType, timeRange) => {
    const now = new Date();
    const points = getDataPointsCount(timeRange);
    const interval = getTimeInterval(timeRange);
    const config = iotService.getSensorTypeConfig(sensorType);
    
    const data = [];
    for (let i = points; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - (i * interval));
      const baseValue = (config.range.min + config.range.max) / 2;
      const variation = (config.range.max - config.range.min) * 0.3;
      const value = baseValue + (Math.random() - 0.5) * variation + Math.sin(i / 10) * variation * 0.3;
      
      data.push({
        timestamp,
        value: Math.max(config.range.min, Math.min(config.range.max, value)),
        battery: Math.max(60, 100 - Math.random() * 30),
        signal: Math.max(70, 100 - Math.random() * 20),
        formattedTime: format(timestamp, getTimeFormat(timeRange))
      });
    }
    return data;
  };

  const getDataPointsCount = (range) => {
    switch (range) {
      case '1h': return 12; // Every 5 minutes
      case '6h': return 24; // Every 15 minutes  
      case '24h': return 24; // Every hour
      case '7d': return 28; // Every 6 hours
      case '30d': return 30; // Every day
      default: return 24;
    }
  };

  const getTimeInterval = (range) => {
    switch (range) {
      case '1h': return 5 * 60 * 1000; // 5 minutes
      case '6h': return 15 * 60 * 1000; // 15 minutes
      case '24h': return 60 * 60 * 1000; // 1 hour
      case '7d': return 6 * 60 * 60 * 1000; // 6 hours
      case '30d': return 24 * 60 * 60 * 1000; // 1 day
      default: return 60 * 60 * 1000;
    }
  };

  const getTimeFormat = (range) => {
    switch (range) {
      case '1h':
      case '6h': 
        return 'HH:mm';
      case '24h': 
        return 'HH:mm';
      case '7d': 
        return 'MMM dd HH:mm';
      case '30d': 
        return 'MMM dd';
      default: 
        return 'HH:mm';
    }
  };

  const calculateStats = (data) => {
    if (data.length === 0) return null;

    const values = data.map(d => d.value);
    const latest = values[values.length - 1];
    const previous = values.length > 1 ? values[values.length - 2] : latest;
    const average = values.reduce((sum, val) => sum + val, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const change = latest - previous;
    const changePercent = previous !== 0 ? ((change / previous) * 100) : 0;

    return {
      latest: parseFloat(latest.toFixed(2)),
      average: parseFloat(average.toFixed(2)),
      min: parseFloat(min.toFixed(2)),
      max: parseFloat(max.toFixed(2)),
      change: parseFloat(change.toFixed(2)),
      changePercent: parseFloat(changePercent.toFixed(1)),
      trend: change > 0 ? 'up' : change < 0 ? 'down' : 'stable'
    };
  };

  const getYAxisLabel = () => {
    const config = iotService.getSensorTypeConfig(sensorType);
    return `${config.label} (${config.unit})`;
  };

  const getLineColor = () => {
    const config = iotService.getSensorTypeConfig(sensorType);
    return config.color;
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'up': return 'success';
      case 'down': return 'error';
      default: return 'info';
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return '↗';
      case 'down': return '↘';
      default: return '→';
    }
  };

  if (loading.readings) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (chartData.length === 0) {
    return (
      <Card>
        <CardContent>
          <Alert severity="info">
            No sensor readings available for the selected time period.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const config = iotService.getSensorTypeConfig(sensorType);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Sensor Readings
          </Typography>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              label="Time Range"
            >
              {timeRangeOptions.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        {/* Stats Summary */}
        {stats && (
          <Grid container spacing={1} sx={{ mb: 2 }}>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'primary.main', color: 'white' }}>
                <Typography variant="caption">Latest</Typography>
                <Typography variant="h6">{stats.latest}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 1, textAlign: 'center' }}>
                <Typography variant="caption">Average</Typography>
                <Typography variant="h6">{stats.average}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 1, textAlign: 'center' }}>
                <Typography variant="caption">Min/Max</Typography>
                <Typography variant="body2">{stats.min} / {stats.max}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 1, textAlign: 'center' }}>
                <Typography variant="caption">Change</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Chip
                    size="small"
                    color={getTrendColor(stats.trend)}
                    label={`${getTrendIcon(stats.trend)} ${stats.changePercent}%`}
                  />
                </Box>
              </Paper>
            </Grid>
          </Grid>
        )}

        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="formattedTime" 
              tick={{ fontSize: 12 }}
              angle={timeRange === '7d' || timeRange === '30d' ? -45 : 0}
              textAnchor={timeRange === '7d' || timeRange === '30d' ? 'end' : 'middle'}
              height={timeRange === '7d' || timeRange === '30d' ? 80 : 30}
            />
            <YAxis 
              label={{ 
                value: getYAxisLabel(), 
                angle: -90, 
                position: 'insideLeft',
                style: { textAnchor: 'middle' }
              }}
              domain={[config.range.min * 0.9, config.range.max * 1.1]}
            />
            <Tooltip 
              formatter={(value, name) => [
                `${parseFloat(value).toFixed(2)} ${config.unit}`,
                config.label
              ]}
              labelFormatter={(label) => `Time: ${label}`}
              contentStyle={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: `1px solid ${getLineColor()}`,
                borderRadius: '4px'
              }}
            />
            
            {/* Reference lines for optimal range */}
            {config.range.min !== undefined && (
              <ReferenceLine 
                y={config.range.min} 
                stroke="#ff0000" 
                strokeDasharray="5 5" 
                strokeOpacity={0.5}
              />
            )}
            {config.range.max !== undefined && (
              <ReferenceLine 
                y={config.range.max} 
                stroke="#ff0000" 
                strokeDasharray="5 5" 
                strokeOpacity={0.5}
              />
            )}
            
            <Area
              type="monotone"
              dataKey="value"
              stroke={getLineColor()}
              fill={getLineColor()}
              fillOpacity={0.3}
              strokeWidth={2}
              dot={{ fill: getLineColor(), strokeWidth: 2, r: 3 }}
              activeDot={{ r: 5, stroke: getLineColor(), strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>

        {/* System Status */}
        {chartData.length > 0 && (
          <Box sx={{ mt: 2, p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Data Points:</strong> {chartData.length} readings
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Battery:</strong> {Math.round(chartData[chartData.length - 1]?.battery || 0)}%
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Signal:</strong> {Math.round(chartData[chartData.length - 1]?.signal || 0)}%
                </Typography>
              </Grid>
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ReadingsChart;