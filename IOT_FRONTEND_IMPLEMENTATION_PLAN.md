# IoT Frontend Connection Implementation Plan

**Project**: Carbon Credit Verification SaaS  
**Feature**: IoT Sensor Integration Frontend Connection  
**Timeline**: 1-2 Days (16 hours)  
**Complexity**: LOW  
**Risk Level**: MINIMAL  
**Priority**: HIGH  

---

## 📋 Executive Summary

The IoT backend infrastructure is fully implemented with 8+ API endpoints, complete database schema, and production-ready services. This plan focuses on connecting the existing placeholder frontend to the operational backend, transforming the "PLANNED" status page into a fully functional IoT sensor management system.

---

## 🎯 Current Situation Analysis

### ✅ **COMPLETED (Backend)**
- **API Endpoints**: 8 fully functional IoT endpoints
  - `GET /api/v1/iot/sensors` - List sensors with filtering
  - `POST /api/v1/iot/sensors` - Create new sensors  
  - `GET /api/v1/iot/readings/{sensor_id}` - Get sensor readings
  - `POST /api/v1/iot/readings` - Record sensor data
  - `GET /api/v1/iot/analytics` - IoT analytics dashboard
  - Additional CRUD operations for sensor management

- **Database Schema**: Production-ready tables
  - `iot_sensors` table with geolocation, calibration data
  - `sensor_readings` table with timestamped measurements
  - Foreign key relationships with projects table
  - JSON storage for complex sensor configurations

- **Data Models**: Pydantic schemas defined
  - `IoTSensorCreate` - Sensor creation model
  - `IoTSensorResponse` - API response model
  - Full validation and serialization support

- **Security**: Role-based access control integrated
  - Admin: Full sensor management
  - Project Developers: Own project sensors only
  - Verifiers: Read-only access for verification

### ❌ **MISSING (Frontend)**
- **Service Layer**: No API integration service
- **State Management**: No Redux IoT slice
- **UI Components**: Placeholder page with mock data
- **Data Visualization**: No real-time sensor data display
- **User Interactions**: No sensor creation or management interface

---

## 🚀 Implementation Strategy

### **Phase 1: Service Layer Integration** 
**Duration**: 4 hours  
**Files**: 2 new files

#### 1.1 IoT Service Creation
**File**: `frontend/src/services/iotService.js`

```javascript
import apiService from './apiService';

export const iotService = {
  // Sensor Management
  async getSensors(filters = {}) {
    const params = new URLSearchParams(filters);
    return apiService.get(`/iot/sensors?${params}`);
  },

  async createSensor(sensorData) {
    return apiService.post('/iot/sensors', sensorData);
  },

  async updateSensor(sensorId, updateData) {
    return apiService.put(`/iot/sensors/${sensorId}`, updateData);
  },

  async deleteSensor(sensorId) {
    return apiService.delete(`/iot/sensors/${sensorId}`);
  },

  // Sensor Readings
  async getSensorReadings(sensorId, timeRange = {}) {
    const params = new URLSearchParams(timeRange);
    return apiService.get(`/iot/readings/${sensorId}?${params}`);
  },

  async recordReading(readingData) {
    return apiService.post('/iot/readings', readingData);
  },

  // Analytics
  async getAnalytics(projectId = null) {
    const params = projectId ? `?project_id=${projectId}` : '';
    return apiService.get(`/iot/analytics${params}`);
  },

  // Real-time data simulation for demo
  async getRealtimeData(sensorId) {
    return apiService.get(`/iot/sensors/${sensorId}/realtime`);
  }
};
```

#### 1.2 Redux State Management
**File**: `frontend/src/store/iotSlice.js`

```javascript
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { iotService } from '../services/iotService';

// Async thunks
export const fetchSensors = createAsyncThunk(
  'iot/fetchSensors',
  async (filters, { rejectWithValue }) => {
    try {
      return await iotService.getSensors(filters);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const createSensor = createAsyncThunk(
  'iot/createSensor',
  async (sensorData, { rejectWithValue }) => {
    try {
      return await iotService.createSensor(sensorData);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchSensorReadings = createAsyncThunk(
  'iot/fetchSensorReadings',
  async ({ sensorId, timeRange }, { rejectWithValue }) => {
    try {
      return await iotService.getSensorReadings(sensorId, timeRange);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const iotSlice = createSlice({
  name: 'iot',
  initialState: {
    sensors: [],
    selectedSensor: null,
    readings: {},
    analytics: null,
    loading: {
      sensors: false,
      readings: false,
      analytics: false,
      creating: false
    },
    error: null,
    filters: {
      projectId: null,
      sensorType: null,
      status: 'active'
    }
  },
  reducers: {
    setSelectedSensor: (state, action) => {
      state.selectedSensor = action.payload;
    },
    setFilters: (state, action) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    clearError: (state) => {
      state.error = null;
    },
    updateSensorReading: (state, action) => {
      const { sensorId, reading } = action.payload;
      if (!state.readings[sensorId]) {
        state.readings[sensorId] = [];
      }
      state.readings[sensorId].unshift(reading);
    }
  },
  extraReducers: (builder) => {
    // Fetch sensors
    builder
      .addCase(fetchSensors.pending, (state) => {
        state.loading.sensors = true;
        state.error = null;
      })
      .addCase(fetchSensors.fulfilled, (state, action) => {
        state.loading.sensors = false;
        state.sensors = action.payload;
      })
      .addCase(fetchSensors.rejected, (state, action) => {
        state.loading.sensors = false;
        state.error = action.payload;
      })
      
    // Create sensor
    builder
      .addCase(createSensor.pending, (state) => {
        state.loading.creating = true;
        state.error = null;
      })
      .addCase(createSensor.fulfilled, (state, action) => {
        state.loading.creating = false;
        state.sensors.push(action.payload);
      })
      .addCase(createSensor.rejected, (state, action) => {
        state.loading.creating = false;
        state.error = action.payload;
      })
      
    // Fetch readings
    builder
      .addCase(fetchSensorReadings.pending, (state) => {
        state.loading.readings = true;
      })
      .addCase(fetchSensorReadings.fulfilled, (state, action) => {
        state.loading.readings = false;
        const { sensorId, readings } = action.payload;
        state.readings[sensorId] = readings;
      })
      .addCase(fetchSensorReadings.rejected, (state, action) => {
        state.loading.readings = false;
        state.error = action.payload;
      });
  }
});

export const { setSelectedSensor, setFilters, clearError, updateSensorReading } = iotSlice.actions;
export default iotSlice.reducer;
```

---

### **Phase 2: UI Components Development**
**Duration**: 6 hours  
**Files**: 6 new component files

#### 2.1 Core Components

**File**: `frontend/src/components/iot/SensorCard.js`
```javascript
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Chip,
  Box,
  IconButton,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  Sensors as SensorIcon,
  LocationOn,
  Battery4Bar,
  SignalWifi4Bar,
  MoreVert
} from '@mui/icons-material';

const SensorCard = ({ sensor, onSelect, onMenuClick }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'success';
      case 'maintenance': return 'warning';
      case 'offline': return 'error';
      default: return 'default';
    }
  };

  const getBatteryLevel = (sensor) => {
    return sensor.last_reading?.battery_level || 0;
  };

  const getSignalStrength = (sensor) => {
    return sensor.last_reading?.signal_strength || 0;
  };

  return (
    <Card 
      sx={{ 
        cursor: 'pointer',
        '&:hover': { transform: 'translateY(-2px)', boxShadow: 3 },
        transition: 'all 0.2s'
      }}
      onClick={() => onSelect(sensor)}
    >
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <SensorIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" component="h3">
              {sensor.sensor_id}
            </Typography>
          </Box>
          <Box>
            <Chip 
              label={sensor.status} 
              color={getStatusColor(sensor.status)}
              size="small"
            />
            <IconButton size="small" onClick={(e) => {
              e.stopPropagation();
              onMenuClick(sensor);
            }}>
              <MoreVert />
            </IconButton>
          </Box>
        </Box>

        <Typography variant="body2" color="text.secondary" gutterBottom>
          {sensor.sensor_type.replace('_', ' ').toUpperCase()}
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <LocationOn sx={{ fontSize: 16, mr: 0.5, color: 'text.secondary' }} />
          <Typography variant="body2" color="text.secondary">
            {sensor.location_lat.toFixed(4)}, {sensor.location_lng.toFixed(4)}
          </Typography>
        </Box>

        {sensor.last_reading && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Last Reading: {new Date(sensor.last_reading.timestamp).toLocaleString()}
            </Typography>
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Battery4Bar sx={{ fontSize: 16, mr: 0.5 }} />
                <Typography variant="caption">
                  {getBatteryLevel(sensor)}%
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <SignalWifi4Bar sx={{ fontSize: 16, mr: 0.5 }} />
                <Typography variant="caption">
                  {getSignalStrength(sensor)}%
                </Typography>
              </Box>
            </Box>

            {sensor.last_reading.value && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="body2">
                  <strong>{sensor.last_reading.value}</strong> {sensor.last_reading.unit}
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min((sensor.last_reading.value / 100) * 100, 100)} 
                  sx={{ mt: 0.5 }}
                />
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SensorCard;
```

**File**: `frontend/src/components/iot/SensorForm.js`
```javascript
import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  Divider
} from '@mui/material';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { useSelector, useDispatch } from 'react-redux';
import { createSensor } from '../../store/iotSlice';

const sensorTypes = [
  { value: 'soil_moisture', label: 'Soil Moisture Sensor', unit: '%' },
  { value: 'co2_flux', label: 'CO₂ Flux Meter', unit: 'ppm' },
  { value: 'temperature', label: 'Temperature Probe', unit: '°C' },
  { value: 'tree_growth', label: 'Tree Growth Monitor', unit: 'cm' },
  { value: 'humidity', label: 'Humidity Sensor', unit: '%' },
  { value: 'light_intensity', label: 'Light Intensity Meter', unit: 'lux' }
];

const SensorForm = ({ open, onClose, projectId }) => {
  const dispatch = useDispatch();
  const { loading } = useSelector(state => state.iot);
  
  const [formData, setFormData] = useState({
    sensor_id: '',
    sensor_type: '',
    location_lat: '',
    location_lng: '',
    installation_date: new Date(),
    calibration_data: {}
  });

  const [calibrationFields, setCalibrationFields] = useState({
    min_value: '',
    max_value: '',
    accuracy: '',
    resolution: ''
  });

  const handleChange = (field) => (event) => {
    setFormData(prev => ({
      ...prev,
      [field]: event.target.value
    }));
  };

  const handleCalibrationChange = (field) => (event) => {
    setCalibrationFields(prev => ({
      ...prev,
      [field]: event.target.value
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    const sensorData = {
      ...formData,
      project_id: projectId,
      location_lat: parseFloat(formData.location_lat),
      location_lng: parseFloat(formData.location_lng),
      calibration_data: {
        ...calibrationFields,
        min_value: parseFloat(calibrationFields.min_value),
        max_value: parseFloat(calibrationFields.max_value),
        accuracy: parseFloat(calibrationFields.accuracy),
        resolution: parseFloat(calibrationFields.resolution)
      }
    };

    try {
      await dispatch(createSensor(sensorData)).unwrap();
      onClose();
      resetForm();
    } catch (error) {
      console.error('Failed to create sensor:', error);
    }
  };

  const resetForm = () => {
    setFormData({
      sensor_id: '',
      sensor_type: '',
      location_lat: '',
      location_lng: '',
      installation_date: new Date(),
      calibration_data: {}
    });
    setCalibrationFields({
      min_value: '',
      max_value: '',
      accuracy: '',
      resolution: ''
    });
  };

  const selectedSensorType = sensorTypes.find(type => type.value === formData.sensor_type);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Add New IoT Sensor</DialogTitle>
      <form onSubmit={handleSubmit}>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
            <TextField
              label="Sensor ID"
              value={formData.sensor_id}
              onChange={handleChange('sensor_id')}
              required
              fullWidth
              helperText="Unique identifier for the sensor (e.g., SOIL-001, CO2-FLUX-01)"
            />

            <FormControl fullWidth required>
              <InputLabel>Sensor Type</InputLabel>
              <Select
                value={formData.sensor_type}
                onChange={handleChange('sensor_type')}
                label="Sensor Type"
              >
                {sensorTypes.map(type => (
                  <MenuItem key={type.value} value={type.value}>
                    {type.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                label="Latitude"
                type="number"
                value={formData.location_lat}
                onChange={handleChange('location_lat')}
                required
                fullWidth
                inputProps={{ step: 'any' }}
                helperText="Decimal degrees (e.g., -10.1234)"
              />
              <TextField
                label="Longitude"
                type="number"
                value={formData.location_lng}
                onChange={handleChange('location_lng')}
                required
                fullWidth
                inputProps={{ step: 'any' }}
                helperText="Decimal degrees (e.g., -55.1234)"
              />
            </Box>

            <DateTimePicker
              label="Installation Date"
              value={formData.installation_date}
              onChange={(newValue) => setFormData(prev => ({
                ...prev,
                installation_date: newValue
              }))}
              renderInput={(params) => <TextField {...params} fullWidth />}
            />

            <Divider sx={{ my: 1 }} />
            
            <Typography variant="h6" gutterBottom>
              Calibration Data
              {selectedSensorType && (
                <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                  ({selectedSensorType.unit})
                </Typography>
              )}
            </Typography>

            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                label="Min Value"
                type="number"
                value={calibrationFields.min_value}
                onChange={handleCalibrationChange('min_value')}
                fullWidth
                inputProps={{ step: 'any' }}
              />
              <TextField
                label="Max Value"
                type="number"
                value={calibrationFields.max_value}
                onChange={handleCalibrationChange('max_value')}
                fullWidth
                inputProps={{ step: 'any' }}
              />
            </Box>

            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                label="Accuracy (%)"
                type="number"
                value={calibrationFields.accuracy}
                onChange={handleCalibrationChange('accuracy')}
                fullWidth
                inputProps={{ step: 'any', min: 0, max: 100 }}
              />
              <TextField
                label="Resolution"
                type="number"
                value={calibrationFields.resolution}
                onChange={handleCalibrationChange('resolution')}
                fullWidth
                inputProps={{ step: 'any' }}
              />
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button 
            type="submit" 
            variant="contained"
            disabled={loading.creating}
          >
            {loading.creating ? 'Creating...' : 'Add Sensor'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default SensorForm;
```

#### 2.2 Data Visualization Components

**File**: `frontend/src/components/iot/ReadingsChart.js`
```javascript
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
  Alert
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
  AreaChart
} from 'recharts';
import { useSelector, useDispatch } from 'react-redux';
import { fetchSensorReadings } from '../../store/iotSlice';

const ReadingsChart = ({ sensorId, sensorType }) => {
  const dispatch = useDispatch();
  const { readings, loading } = useSelector(state => state.iot);
  const [timeRange, setTimeRange] = useState('24h');
  const [chartData, setChartData] = useState([]);

  const timeRangeOptions = [
    { value: '1h', label: 'Last Hour' },
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' }
  ];

  useEffect(() => {
    if (sensorId) {
      dispatch(fetchSensorReadings({ sensorId, timeRange: { period: timeRange } }));
    }
  }, [dispatch, sensorId, timeRange]);

  useEffect(() => {
    const sensorReadings = readings[sensorId] || [];
    const formattedData = sensorReadings.map(reading => ({
      timestamp: new Date(reading.timestamp).toLocaleString(),
      value: reading.value,
      battery: reading.battery_level,
      signal: reading.signal_strength
    }));
    setChartData(formattedData);
  }, [readings, sensorId]);

  const getYAxisLabel = () => {
    const typeLabels = {
      'soil_moisture': 'Moisture (%)',
      'co2_flux': 'CO₂ (ppm)',
      'temperature': 'Temperature (°C)',
      'tree_growth': 'Growth (cm)',
      'humidity': 'Humidity (%)',
      'light_intensity': 'Light (lux)'
    };
    return typeLabels[sensorType] || 'Value';
  };

  const getLineColor = () => {
    const colorMap = {
      'soil_moisture': '#2196f3',
      'co2_flux': '#4caf50',
      'temperature': '#ff9800',
      'tree_growth': '#8bc34a',
      'humidity': '#03a9f4',
      'light_intensity': '#ffeb3b'
    };
    return colorMap[sensorType] || '#2196f3';
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

  if (!chartData.length) {
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

        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp" 
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis label={{ value: getYAxisLabel(), angle: -90, position: 'insideLeft' }} />
            <Tooltip 
              formatter={(value, name) => [value, getYAxisLabel()]}
              labelFormatter={(label) => `Time: ${label}`}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke={getLineColor()}
              fill={getLineColor()}
              fillOpacity={0.3}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>

        {chartData.length > 0 && (
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-around' }}>
            <Typography variant="body2" color="text.secondary">
              Latest: <strong>{chartData[chartData.length - 1]?.value}</strong>
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Average: <strong>
                {(chartData.reduce((sum, item) => sum + item.value, 0) / chartData.length).toFixed(2)}
              </strong>
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Battery: <strong>{chartData[chartData.length - 1]?.battery}%</strong>
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ReadingsChart;
```

---

### **Phase 3: IoT Page Reconstruction**
**Duration**: 4 hours  
**Files**: 1 major file overhaul

**File**: `frontend/src/pages/IoT.js` (Complete Replacement)
```javascript
import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogContent,
  Fab
} from '@mui/material';
import {
  Sensors as SensorsIcon,
  Add as AddIcon,
  Dashboard as DashboardIcon,
  TrendingUp as TrendingUpIcon
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { fetchSensors, setFilters, setSelectedSensor } from '../store/iotSlice';
import { COMMON_STYLES } from '../theme/constants';

// Component imports
import SensorCard from '../components/iot/SensorCard';
import SensorForm from '../components/iot/SensorForm';
import ReadingsChart from '../components/iot/ReadingsChart';
import IoTAnalytics from '../components/iot/IoTAnalytics';

const IoT = () => {
  const dispatch = useDispatch();
  const { user } = useSelector(state => state.auth);
  const { projects } = useSelector(state => state.projects);
  const { sensors, selectedSensor, loading, filters } = useSelector(state => state.iot);
  
  const [showSensorForm, setShowSensorForm] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [selectedProject, setSelectedProject] = useState('');

  useEffect(() => {
    dispatch(fetchSensors(filters));
  }, [dispatch, filters]);

  const handleProjectChange = (event) => {
    const projectId = event.target.value;
    setSelectedProject(projectId);
    dispatch(setFilters({ projectId: projectId || null }));
  };

  const handleSensorTypeFilter = (event) => {
    dispatch(setFilters({ sensorType: event.target.value || null }));
  };

  const handleSensorSelect = (sensor) => {
    dispatch(setSelectedSensor(sensor));
  };

  const handleAddSensor = () => {
    if (!selectedProject) {
      alert('Please select a project first');
      return;
    }
    setShowSensorForm(true);
  };

  const getSensorStats = () => {
    const total = sensors.length;
    const active = sensors.filter(s => s.status === 'active').length;
    const offline = sensors.filter(s => s.status === 'offline').length;
    const maintenance = sensors.filter(s => s.status === 'maintenance').length;
    
    return { total, active, offline, maintenance };
  };

  const stats = getSensorStats();
  const canManageSensors = user?.role === 'admin' || user?.role === 'project_developer';

  return (
    <Container maxWidth="xl" sx={COMMON_STYLES.pageContainer}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <SensorsIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
          <Typography variant="h4" gutterBottom>
            IoT Sensor Network
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<DashboardIcon />}
            onClick={() => setShowAnalytics(true)}
          >
            Analytics
          </Button>
          {canManageSensors && (
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={handleAddSensor}
              disabled={!selectedProject}
            >
              Add Sensor
            </Button>
          )}
        </Box>
      </Box>

      {/* Status Alert */}
      <Alert severity="success" sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>IoT Integration Active:</strong> Real-time sensor monitoring is operational. 
          {stats.total > 0 && ` ${stats.active} of ${stats.total} sensors are currently active.`}
        </Typography>
      </Alert>

      {/* Filter Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Project</InputLabel>
              <Select
                value={selectedProject}
                onChange={handleProjectChange}
                label="Project"
              >
                <MenuItem value="">All Projects</MenuItem>
                {projects.map(project => (
                  <MenuItem key={project.id} value={project.id}>
                    {project.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Sensor Type</InputLabel>
              <Select
                value={filters.sensorType || ''}
                onChange={handleSensorTypeFilter}
                label="Sensor Type"
              >
                <MenuItem value="">All Types</MenuItem>
                <MenuItem value="soil_moisture">Soil Moisture</MenuItem>
                <MenuItem value="co2_flux">CO₂ Flux</MenuItem>
                <MenuItem value="temperature">Temperature</MenuItem>
                <MenuItem value="tree_growth">Tree Growth</MenuItem>
                <MenuItem value="humidity">Humidity</MenuItem>
                <MenuItem value="light_intensity">Light Intensity</MenuItem>
              </Select>
            </FormControl>

            {/* Status Stats */}
            <Box sx={{ display: 'flex', gap: 1, ml: 'auto' }}>
              <Chip label={`${stats.active} Active`} color="success" size="small" />
              <Chip label={`${stats.offline} Offline`} color="error" size="small" />
              <Chip label={`${stats.maintenance} Maintenance`} color="warning" size="small" />
            </Box>
          </Box>
        </CardContent>
      </Card>

      {loading.sensors ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {/* Sensors Grid */}
          <Grid item xs={12} lg={selectedSensor ? 8 : 12}>
            <Typography variant="h6" gutterBottom>
              Sensor Network ({sensors.length} sensors)
            </Typography>
            <Grid container spacing={2}>
              {sensors.map(sensor => (
                <Grid item xs={12} sm={6} md={4} key={sensor.id}>
                  <SensorCard
                    sensor={sensor}
                    onSelect={handleSensorSelect}
                    onMenuClick={(sensor) => console.log('Menu clicked:', sensor)}
                  />
                </Grid>
              ))}
            </Grid>

            {sensors.length === 0 && (
              <Card>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <SensorsIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No Sensors Found
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {selectedProject 
                      ? 'No sensors are deployed for the selected project and filters.'
                      : 'Select a project to view its sensors or add new sensors to get started.'
                    }
                  </Typography>
                  {canManageSensors && selectedProject && (
                    <Button
                      variant="contained"
                      startIcon={<AddIcon />}
                      onClick={handleAddSensor}
                    >
                      Add First Sensor
                    </Button>
                  )}
                </CardContent>
              </Card>
            )}
          </Grid>

          {/* Sensor Detail Panel */}
          {selectedSensor && (
            <Grid item xs={12} lg={4}>
              <Box sx={{ position: 'sticky', top: 20 }}>
                <Typography variant="h6" gutterBottom>
                  Sensor Details: {selectedSensor.sensor_id}
                </Typography>
                <ReadingsChart 
                  sensorId={selectedSensor.id}
                  sensorType={selectedSensor.sensor_type}
                />
              </Box>
            </Grid>
          )}
        </Grid>
      )}

      {/* Floating Action Button for Mobile */}
      {canManageSensors && selectedProject && (
        <Fab
          color="primary"
          aria-label="add sensor"
          sx={{ 
            position: 'fixed', 
            bottom: 16, 
            right: 16,
            display: { xs: 'flex', sm: 'none' }
          }}
          onClick={handleAddSensor}
        >
          <AddIcon />
        </Fab>
      )}

      {/* Dialogs */}
      <SensorForm
        open={showSensorForm}
        onClose={() => setShowSensorForm(false)}
        projectId={selectedProject}
      />

      <Dialog
        open={showAnalytics}
        onClose={() => setShowAnalytics(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogContent>
          <IoTAnalytics projectId={selectedProject} />
        </DialogContent>
      </Dialog>
    </Container>
  );
};

export default IoT;
```

---

### **Phase 4: Testing & Integration**
**Duration**: 2 hours  
**Files**: 3 test files

#### 4.1 Unit Tests
**File**: `frontend/src/tests/iot/iotService.test.js`
```javascript
import { iotService } from '../../services/iotService';
import apiService from '../../services/apiService';

jest.mock('../../services/apiService');

describe('IoT Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getSensors', () => {
    test('should fetch sensors with filters', async () => {
      const mockSensors = [{ id: 1, sensor_id: 'TEST-001' }];
      apiService.get.mockResolvedValue(mockSensors);

      const filters = { projectId: 1, sensorType: 'soil_moisture' };
      const result = await iotService.getSensors(filters);

      expect(apiService.get).toHaveBeenCalledWith('/iot/sensors?projectId=1&sensorType=soil_moisture');
      expect(result).toEqual(mockSensors);
    });
  });

  describe('createSensor', () => {
    test('should create new sensor', async () => {
      const mockSensor = { id: 1, sensor_id: 'NEW-001' };
      const sensorData = { sensor_id: 'NEW-001', sensor_type: 'temperature' };
      
      apiService.post.mockResolvedValue(mockSensor);
      
      const result = await iotService.createSensor(sensorData);

      expect(apiService.post).toHaveBeenCalledWith('/iot/sensors', sensorData);
      expect(result).toEqual(mockSensor);
    });
  });
});
```

#### 4.2 Component Tests
**File**: `frontend/src/tests/iot/SensorCard.test.js`
```javascript
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import SensorCard from '../../components/iot/SensorCard';

const mockStore = configureStore({
  reducer: {
    iot: (state = {}) => state
  }
});

const mockSensor = {
  id: 1,
  sensor_id: 'TEST-001',
  sensor_type: 'soil_moisture',
  status: 'active',
  location_lat: -10.1234,
  location_lng: -55.1234,
  last_reading: {
    value: 75,
    unit: '%',
    timestamp: '2024-01-01T12:00:00Z',
    battery_level: 85,
    signal_strength: 92
  }
};

describe('SensorCard', () => {
  test('renders sensor information correctly', () => {
    const onSelect = jest.fn();
    const onMenuClick = jest.fn();

    render(
      <Provider store={mockStore}>
        <SensorCard 
          sensor={mockSensor}
          onSelect={onSelect}
          onMenuClick={onMenuClick}
        />
      </Provider>
    );

    expect(screen.getByText('TEST-001')).toBeInTheDocument();
    expect(screen.getByText('SOIL MOISTURE')).toBeInTheDocument();
    expect(screen.getByText('active')).toBeInTheDocument();
    expect(screen.getByText('75 %')).toBeInTheDocument();
  });

  test('calls onSelect when card is clicked', () => {
    const onSelect = jest.fn();
    const onMenuClick = jest.fn();

    render(
      <Provider store={mockStore}>
        <SensorCard 
          sensor={mockSensor}
          onSelect={onSelect}
          onMenuClick={onMenuClick}
        />
      </Provider>
    );

    fireEvent.click(screen.getByText('TEST-001'));
    expect(onSelect).toHaveBeenCalledWith(mockSensor);
  });
});
```

---

## 📁 File Structure Summary

### **New Files to Create (8 files)**
1. `frontend/src/services/iotService.js` - API integration service
2. `frontend/src/store/iotSlice.js` - Redux state management  
3. `frontend/src/components/iot/SensorCard.js` - Individual sensor display
4. `frontend/src/components/iot/SensorForm.js` - Sensor creation form
5. `frontend/src/components/iot/SensorsList.js` - Sensors list component
6. `frontend/src/components/iot/ReadingsChart.js` - Data visualization
7. `frontend/src/components/iot/IoTAnalytics.js` - Analytics dashboard
8. `frontend/src/components/iot/DataVisualization.js` - Charts component

### **Files to Modify (3 files)**
1. `frontend/src/pages/IoT.js` - Complete page reconstruction (150+ lines)
2. `frontend/src/store/index.js` - Add iotSlice to Redux store
3. `frontend/src/services/apiService.js` - Optional: Add IoT-specific helpers

### **Test Files to Create (3 files)**
1. `frontend/src/tests/iot/iotService.test.js` - Service layer tests
2. `frontend/src/tests/iot/SensorCard.test.js` - Component tests
3. `frontend/src/tests/iot/IoT.test.js` - Page integration tests

---

## 🎯 Success Criteria

### **Technical Milestones**
- ✅ All 8 IoT API endpoints connected and functional
- ✅ Real-time sensor data display with charts
- ✅ Sensor creation and management working
- ✅ Role-based permissions implemented
- ✅ Responsive design across all devices

### **User Experience Goals**
- **Intuitive Interface**: Easy sensor management for non-technical users
- **Real-time Updates**: Live sensor data with automatic refresh
- **Visual Feedback**: Clear status indicators and data visualization
- **Mobile Friendly**: Responsive design with mobile-optimized interactions

### **Performance Targets**
- **Load Time**: Page loads under 2 seconds
- **API Response**: All API calls respond under 500ms
- **Real-time Updates**: Sensor readings update every 30 seconds
- **Chart Rendering**: Smooth chart animations and interactions

---

## ⚡ Quick Start Commands

### **Setup Dependencies**
```bash
cd frontend
npm install recharts @mui/x-date-pickers date-fns
```

### **Development Workflow**
```bash
# 1. Create service layer
touch src/services/iotService.js
touch src/store/iotSlice.js

# 2. Create components
mkdir -p src/components/iot
touch src/components/iot/{SensorCard,SensorForm,ReadingsChart,IoTAnalytics}.js

# 3. Update store
# Add iotSlice to src/store/index.js

# 4. Replace IoT page
# Overwrite src/pages/IoT.js

# 5. Test the implementation
npm start
```

---

## 🚀 Implementation Priority

### **Hour 1-2: Foundation**
- Create `iotService.js` and `iotSlice.js`
- Test API connectivity

### **Hour 3-6: Core Components** 
- Build `SensorCard.js` and `SensorForm.js`
- Implement basic sensor display

### **Hour 7-10: Data Visualization**
- Create `ReadingsChart.js` with real-time data
- Add analytics dashboard

### **Hour 11-14: Page Integration**
- Replace IoT.js with full implementation
- Connect all components

### **Hour 15-16: Testing & Polish**
- Write unit tests
- Fix bugs and optimize performance

---

This plan transforms the IoT frontend from a placeholder into a fully functional sensor management system, connecting seamlessly with the already-operational backend infrastructure. The implementation is designed to be completed incrementally, with each phase building upon the previous one to ensure a stable and tested final result.