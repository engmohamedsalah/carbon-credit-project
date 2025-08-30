import apiService from './apiService';

export const iotService = {
  // Sensor Management
  async getSensors(filters = {}) {
    try {
      const params = new URLSearchParams();
      if (filters.projectId) params.append('project_id', filters.projectId);
      if (filters.sensorType) params.append('sensor_type', filters.sensorType);
      
      const queryString = params.toString();
      const url = queryString ? `/iot/sensors?${queryString}` : '/iot/sensors';
      
      const response = await apiService.get(url);
      return response.data; // Return only the data, not the full Axios response
    } catch (error) {
      console.error('Failed to get sensors:', error);
      throw error;
    }
  },

  async createSensor(sensorData) {
    try {
      const response = await apiService.post('/iot/sensors', sensorData);
      return response.data;
    } catch (error) {
      console.error('Failed to create sensor:', error);
      throw error;
    }
  },

  async updateSensor(sensorId, updateData) {
    try {
      const response = await apiService.put(`/iot/sensors/${sensorId}`, updateData);
      return response.data;
    } catch (error) {
      console.error('Failed to update sensor:', error);
      throw error;
    }
  },

  async deleteSensor(sensorId) {
    try {
      const response = await apiService.delete(`/iot/sensors/${sensorId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to delete sensor:', error);
      throw error;
    }
  },

  // Sensor Readings
  async getSensorReadings(sensorId, timeRange = {}) {
    try {
      const params = new URLSearchParams();
      if (timeRange.start) params.append('start', timeRange.start);
      if (timeRange.end) params.append('end', timeRange.end);
      if (timeRange.period) params.append('period', timeRange.period);
      
      const queryString = params.toString();
      const url = queryString ? `/iot/readings/${sensorId}?${queryString}` : `/iot/readings/${sensorId}`;
      
      const response = await apiService.get(url);
      return response.data;
    } catch (error) {
      console.error('Failed to get sensor readings:', error);
      throw error;
    }
  },

  async recordReading(readingData) {
    try {
      const response = await apiService.post('/iot/readings', readingData);
      return response.data;
    } catch (error) {
      console.error('Failed to record reading:', error);
      throw error;
    }
  },

  // Analytics
  async getAnalytics(projectId = null) {
    try {
      const params = projectId ? `?project_id=${projectId}` : '';
      const response = await apiService.get(`/iot/analytics${params}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get analytics:', error);
      throw error;
    }
  },

  // Real-time data simulation for demo
  async getRealtimeData(sensorId) {
    try {
      return await apiService.get(`/iot/sensors/${sensorId}/realtime`);
    } catch (error) {
      // Fallback to simulated data if endpoint doesn't exist
      return this.generateSimulatedReading(sensorId);
    }
  },

  // Helper function for simulated data
  generateSimulatedReading(sensorId) {
    const sensorTypes = {
      'soil_moisture': { min: 20, max: 80, unit: '%' },
      'co2_flux': { min: 380, max: 420, unit: 'ppm' },
      'temperature': { min: 15, max: 35, unit: '°C' },
      'tree_growth': { min: 0, max: 100, unit: 'cm' },
      'humidity': { min: 40, max: 90, unit: '%' },
      'light_intensity': { min: 0, max: 100000, unit: 'lux' }
    };

    // Simple simulation based on sensor type
    const type = 'soil_moisture'; // Default for demo
    const config = sensorTypes[type];
    const value = Math.random() * (config.max - config.min) + config.min;

    return {
      sensorId,
      value: Math.round(value * 100) / 100,
      unit: config.unit,
      timestamp: new Date().toISOString(),
      battery_level: Math.floor(Math.random() * 40) + 60, // 60-100%
      signal_strength: Math.floor(Math.random() * 30) + 70 // 70-100%
    };
  },

  // Utility functions
  validateSensorData(sensorData) {
    const required = ['sensor_id', 'sensor_type', 'location_lat', 'location_lng', 'project_id'];
    const missing = required.filter(field => !sensorData[field]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }

    // Validate coordinates
    if (Math.abs(sensorData.location_lat) > 90) {
      throw new Error('Invalid latitude: must be between -90 and 90');
    }
    if (Math.abs(sensorData.location_lng) > 180) {
      throw new Error('Invalid longitude: must be between -180 and 180');
    }

    return true;
  },

  formatReadingValue(value, sensorType) {
    const decimals = {
      'soil_moisture': 1,
      'co2_flux': 0,
      'temperature': 1,
      'tree_growth': 2,
      'humidity': 1,
      'light_intensity': 0
    };

    return Number(value).toFixed(decimals[sensorType] || 1);
  },

  getSensorTypeConfig(sensorType) {
    const configs = {
      'soil_moisture': {
        label: 'Soil Moisture Sensor',
        unit: '%',
        icon: 'water_drop',
        color: '#2196f3',
        range: { min: 0, max: 100 }
      },
      'co2_flux': {
        label: 'CO₂ Flux Meter',
        unit: 'ppm',
        icon: 'co2',
        color: '#4caf50',
        range: { min: 300, max: 500 }
      },
      'temperature': {
        label: 'Temperature Probe',
        unit: '°C',
        icon: 'thermostat',
        color: '#ff9800',
        range: { min: -20, max: 50 }
      },
      'tree_growth': {
        label: 'Tree Growth Monitor',
        unit: 'cm',
        icon: 'park',
        color: '#8bc34a',
        range: { min: 0, max: 500 }
      },
      'humidity': {
        label: 'Humidity Sensor',
        unit: '%',
        icon: 'humidity_percentage',
        color: '#03a9f4',
        range: { min: 0, max: 100 }
      },
      'light_intensity': {
        label: 'Light Intensity Meter',
        unit: 'lux',
        icon: 'light_mode',
        color: '#ffeb3b',
        range: { min: 0, max: 100000 }
      }
    };

    return configs[sensorType] || {
      label: 'Unknown Sensor',
      unit: '',
      icon: 'sensors',
      color: '#666666',
      range: { min: 0, max: 100 }
    };
  }
};