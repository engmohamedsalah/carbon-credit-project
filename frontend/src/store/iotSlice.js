import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { iotService } from '../services/iotService';

// Async thunks
export const fetchSensors = createAsyncThunk(
  'iot/fetchSensors',
  async (filters, { rejectWithValue }) => {
    try {
      return await iotService.getSensors(filters);
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message);
    }
  }
);

export const createSensor = createAsyncThunk(
  'iot/createSensor',
  async (sensorData, { rejectWithValue }) => {
    try {
      // Validate sensor data before submission
      iotService.validateSensorData(sensorData);
      return await iotService.createSensor(sensorData);
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message);
    }
  }
);

export const updateSensor = createAsyncThunk(
  'iot/updateSensor',
  async ({ sensorId, updateData }, { rejectWithValue }) => {
    try {
      return await iotService.updateSensor(sensorId, updateData);
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message);
    }
  }
);

export const deleteSensor = createAsyncThunk(
  'iot/deleteSensor',
  async (sensorId, { rejectWithValue }) => {
    try {
      await iotService.deleteSensor(sensorId);
      return sensorId;
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message);
    }
  }
);

export const fetchSensorReadings = createAsyncThunk(
  'iot/fetchSensorReadings',
  async ({ sensorId, timeRange }, { rejectWithValue }) => {
    try {
      const readings = await iotService.getSensorReadings(sensorId, timeRange);
      return { sensorId, readings };
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message);
    }
  }
);

export const recordReading = createAsyncThunk(
  'iot/recordReading',
  async (readingData, { rejectWithValue }) => {
    try {
      return await iotService.recordReading(readingData);
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message);
    }
  }
);

export const fetchAnalytics = createAsyncThunk(
  'iot/fetchAnalytics',
  async (projectId, { rejectWithValue }) => {
    try {
      return await iotService.getAnalytics(projectId);
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message);
    }
  }
);

export const fetchRealtimeData = createAsyncThunk(
  'iot/fetchRealtimeData',
  async (sensorId, { rejectWithValue }) => {
    try {
      return await iotService.getRealtimeData(sensorId);
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message);
    }
  }
);

const iotSlice = createSlice({
  name: 'iot',
  initialState: {
    sensors: [],
    selectedSensor: null,
    readings: {}, // { sensorId: [readings] }
    analytics: null,
    realtimeData: {}, // { sensorId: latestReading }
    loading: {
      sensors: false,
      readings: false,
      analytics: false,
      creating: false,
      updating: false,
      deleting: false,
      realtime: false
    },
    error: null,
    filters: {
      projectId: null,
      sensorType: null,
      status: null
    },
    pagination: {
      page: 1,
      pageSize: 10,
      total: 0
    },
    lastUpdated: null
  },
  reducers: {
    setSelectedSensor: (state, action) => {
      state.selectedSensor = action.payload;
    },
    
    clearSelectedSensor: (state) => {
      state.selectedSensor = null;
    },
    
    setFilters: (state, action) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    
    clearFilters: (state) => {
      state.filters = {
        projectId: null,
        sensorType: null,
        status: 'active'
      };
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    updateSensorReading: (state, action) => {
      const { sensorId, reading } = action.payload;
      
      // Add to readings array
      if (!state.readings[sensorId]) {
        state.readings[sensorId] = [];
      }
      state.readings[sensorId].unshift(reading);
      
      // Keep only last 100 readings per sensor to prevent memory issues
      if (state.readings[sensorId].length > 100) {
        state.readings[sensorId] = state.readings[sensorId].slice(0, 100);
      }
      
      // Update realtime data
      state.realtimeData[sensorId] = reading;
      
      // Update sensor's last reading
      const sensor = state.sensors.find(s => s.id === sensorId);
      if (sensor) {
        sensor.last_reading = reading;
      }
    },
    
    updateSensorStatus: (state, action) => {
      const { sensorId, status } = action.payload;
      const sensor = state.sensors.find(s => s.id === sensorId);
      if (sensor) {
        sensor.status = status;
      }
    },
    
    setPagination: (state, action) => {
      state.pagination = { ...state.pagination, ...action.payload };
    },
    
    // Batch update multiple sensors (for real-time updates)
    batchUpdateSensors: (state, action) => {
      const updates = action.payload;
      updates.forEach(update => {
        const sensor = state.sensors.find(s => s.id === update.sensorId);
        if (sensor) {
          Object.assign(sensor, update.data);
        }
      });
      state.lastUpdated = new Date().toISOString();
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
        const payload = action.payload;
        state.sensors = Array.isArray(payload)
          ? payload
          : (Array.isArray(payload?.sensors) ? payload.sensors : []);
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchSensors.rejected, (state, action) => {
        state.loading.sensors = false;
        state.error = action.payload;
      });

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
      });

    // Update sensor
    builder
      .addCase(updateSensor.pending, (state) => {
        state.loading.updating = true;
        state.error = null;
      })
      .addCase(updateSensor.fulfilled, (state, action) => {
        state.loading.updating = false;
        const index = state.sensors.findIndex(s => s.id === action.payload.id);
        if (index !== -1) {
          state.sensors[index] = action.payload;
        }
        // Update selected sensor if it's the one being updated
        if (state.selectedSensor?.id === action.payload.id) {
          state.selectedSensor = action.payload;
        }
      })
      .addCase(updateSensor.rejected, (state, action) => {
        state.loading.updating = false;
        state.error = action.payload;
      });

    // Delete sensor
    builder
      .addCase(deleteSensor.pending, (state) => {
        state.loading.deleting = true;
        state.error = null;
      })
      .addCase(deleteSensor.fulfilled, (state, action) => {
        state.loading.deleting = false;
        const sensorId = action.payload;
        state.sensors = state.sensors.filter(s => s.id !== sensorId);
        
        // Clear related data
        delete state.readings[sensorId];
        delete state.realtimeData[sensorId];
        
        // Clear selected sensor if it was deleted
        if (state.selectedSensor?.id === sensorId) {
          state.selectedSensor = null;
        }
      })
      .addCase(deleteSensor.rejected, (state, action) => {
        state.loading.deleting = false;
        state.error = action.payload;
      });

    // Fetch sensor readings
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

    // Record reading
    builder
      .addCase(recordReading.fulfilled, (state, action) => {
        const reading = action.payload;
        // Add the new reading to the appropriate sensor
        if (state.readings[reading.sensor_id]) {
          state.readings[reading.sensor_id].unshift(reading);
        }
      });

    // Fetch analytics
    builder
      .addCase(fetchAnalytics.pending, (state) => {
        state.loading.analytics = true;
      })
      .addCase(fetchAnalytics.fulfilled, (state, action) => {
        state.loading.analytics = false;
        state.analytics = action.payload;
      })
      .addCase(fetchAnalytics.rejected, (state, action) => {
        state.loading.analytics = false;
        state.error = action.payload;
      });

    // Fetch realtime data
    builder
      .addCase(fetchRealtimeData.pending, (state) => {
        state.loading.realtime = true;
      })
      .addCase(fetchRealtimeData.fulfilled, (state, action) => {
        state.loading.realtime = false;
        const reading = action.payload;
        state.realtimeData[reading.sensorId] = reading;
      })
      .addCase(fetchRealtimeData.rejected, (state, action) => {
        state.loading.realtime = false;
        // Don't set error for realtime data as it's not critical
      });
  }
});

export const {
  setSelectedSensor,
  clearSelectedSensor,
  setFilters,
  clearFilters,
  clearError,
  updateSensorReading,
  updateSensorStatus,
  setPagination,
  batchUpdateSensors
} = iotSlice.actions;

// Selectors
export const selectAllSensors = (state) => state.iot.sensors;
export const selectSensorById = (sensorId) => (state) => 
  state.iot.sensors.find(sensor => sensor.id === sensorId);
export const selectSensorsByProject = (projectId) => (state) =>
  state.iot.sensors.filter(sensor => sensor.project_id === projectId);
export const selectSensorsByType = (sensorType) => (state) =>
  state.iot.sensors.filter(sensor => sensor.sensor_type === sensorType);
export const selectActiveSensors = (state) =>
  state.iot.sensors.filter(sensor => sensor.status === 'active');
export const selectSensorReadings = (sensorId) => (state) =>
  state.iot.readings[sensorId] || [];
export const selectRealtimeData = (sensorId) => (state) =>
  state.iot.realtimeData[sensorId];
export const selectIoTAnalytics = (state) => state.iot.analytics;
export const selectIoTError = (state) => state.iot.error;
export const selectIoTLoading = (state) => state.iot.loading;

export default iotSlice.reducer;