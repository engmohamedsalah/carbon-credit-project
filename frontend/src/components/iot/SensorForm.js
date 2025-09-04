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
  Divider,
  Alert,
  Grid,
  InputAdornment,
  CircularProgress
} from '@mui/material';
import {
  LocationOn,
  Sensors,
  Settings,
  Save,
  Cancel
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { createSensor, updateSensor } from '../../store/iotSlice';
import { iotService } from '../../services/iotService';

const sensorTypes = [
  { value: 'soil_moisture', label: 'Soil Moisture Sensor', unit: '%', description: 'Monitor soil water content for carbon sequestration analysis' },
  { value: 'co2_flux', label: 'CO₂ Flux Meter', unit: 'ppm', description: 'Real-time carbon dioxide emission and absorption monitoring' },
  { value: 'temperature', label: 'Temperature Probe', unit: '°C', description: 'Microclimate monitoring for forest health assessment' },
  { value: 'tree_growth', label: 'Tree Growth Monitor', unit: 'cm', description: 'Automated measurement of biomass growth rates' },
  { value: 'humidity', label: 'Humidity Sensor', unit: '%', description: 'Relative humidity monitoring for ecosystem analysis' },
  { value: 'light_intensity', label: 'Light Intensity Meter', unit: 'lux', description: 'Photosynthetic light availability measurement' }
];

const SensorForm = ({ open, onClose, projectId, editSensor = null }) => {
  const dispatch = useDispatch();
  const { loading } = useSelector(state => state.iot);
  
  const [formData, setFormData] = useState({
    sensor_id: '',
    sensor_type: '',
    location_lat: '',
    location_lng: '',
    installation_date: new Date().toISOString().split('T')[0],
    calibration_data: {},
    project_id: projectId
  });

  const [calibrationFields, setCalibrationFields] = useState({
    min_value: '',
    max_value: '',
    accuracy: '',
    resolution: '',
    calibration_date: new Date().toISOString().split('T')[0]
  });

  const [errors, setErrors] = useState({});
  const [validationMessage, setValidationMessage] = useState('');

  // Effect to populate form when editSensor changes
  React.useEffect(() => {
    if (editSensor) {
      console.log('DEBUG - editSensor:', editSensor);
      console.log('DEBUG - projectId prop:', projectId);
      console.log('DEBUG - editSensor.project_id:', editSensor.project_id);
      
      setFormData({
        sensor_id: editSensor.sensor_id || '',
        sensor_type: editSensor.sensor_type || '',
        location_lat: editSensor.location_lat || '',
        location_lng: editSensor.location_lng || '',
        installation_date: editSensor.installation_date || new Date().toISOString().split('T')[0],
        calibration_data: editSensor.calibration_data || {},
        project_id: editSensor.project_id || projectId
      });
      
      const calibData = editSensor.calibration_data || {};
      setCalibrationFields({
        min_value: calibData.min_value || '',
        max_value: calibData.max_value || '',
        accuracy: calibData.accuracy || '',
        resolution: calibData.resolution || '',
        calibration_date: calibData.calibration_date || new Date().toISOString().split('T')[0]
      });
    } else {
      resetForm();
    }
  }, [editSensor, projectId]);

  const handleChange = (field) => (event) => {
    const value = event.target.value;
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error for this field
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const handleCalibrationChange = (field) => (event) => {
    const value = event.target.value;
    setCalibrationFields(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.sensor_id.trim()) {
      newErrors.sensor_id = 'Sensor ID is required';
    }
    
    if (!formData.sensor_type) {
      newErrors.sensor_type = 'Sensor type is required';
    }
    
    if (!formData.location_lat || isNaN(formData.location_lat)) {
      newErrors.location_lat = 'Valid latitude is required';
    } else if (Math.abs(parseFloat(formData.location_lat)) > 90) {
      newErrors.location_lat = 'Latitude must be between -90 and 90';
    }
    
    if (!formData.location_lng || isNaN(formData.location_lng)) {
      newErrors.location_lng = 'Valid longitude is required';
    } else if (Math.abs(parseFloat(formData.location_lng)) > 180) {
      newErrors.location_lng = 'Longitude must be between -180 and 180';
    }

    // Validate calibration data
    if (calibrationFields.min_value && calibrationFields.max_value) {
      const min = parseFloat(calibrationFields.min_value);
      const max = parseFloat(calibrationFields.max_value);
      if (min >= max) {
        newErrors.calibration = 'Maximum value must be greater than minimum value';
      }
    }

    if (calibrationFields.accuracy && (parseFloat(calibrationFields.accuracy) < 0 || parseFloat(calibrationFields.accuracy) > 100)) {
      newErrors.accuracy = 'Accuracy must be between 0 and 100';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!validateForm()) return;
    
    try {
      const sensorData = {
        ...formData,
        project_id: formData.project_id || projectId,
        location_lat: parseFloat(formData.location_lat),
        location_lng: parseFloat(formData.location_lng),
        calibration_data: {
          ...calibrationFields,
          min_value: calibrationFields.min_value ? parseFloat(calibrationFields.min_value) : null,
          max_value: calibrationFields.max_value ? parseFloat(calibrationFields.max_value) : null,
          accuracy: calibrationFields.accuracy ? parseFloat(calibrationFields.accuracy) : null,
          resolution: calibrationFields.resolution ? parseFloat(calibrationFields.resolution) : null
        }
      };

      console.log('DEBUG - Final sensorData being sent:', sensorData);
      console.log('DEBUG - sensorData.project_id:', sensorData.project_id);

      // Validate using service function
      iotService.validateSensorData(sensorData);
      
      if (editSensor) {
        // Update existing sensor
        await dispatch(updateSensor({ sensorId: editSensor.id, updateData: sensorData })).unwrap();
        setValidationMessage('Sensor updated successfully!');
      } else {
        // Create new sensor
        await dispatch(createSensor(sensorData)).unwrap();
        setValidationMessage('Sensor created successfully!');
      }
      
      onClose();
      resetForm();
    } catch (error) {
      setValidationMessage(error.message || (editSensor ? 'Failed to update sensor' : 'Failed to create sensor'));
    }
  };

  const resetForm = () => {
    setFormData({
      sensor_id: '',
      sensor_type: '',
      location_lat: '',
      location_lng: '',
      installation_date: new Date().toISOString().split('T')[0],
      calibration_data: {},
      project_id: projectId
    });
    setCalibrationFields({
      min_value: '',
      max_value: '',
      accuracy: '',
      resolution: '',
      calibration_date: new Date().toISOString().split('T')[0]
    });
    setErrors({});
    setValidationMessage('');
  };

  const handleClose = () => {
    onClose();
    resetForm();
  };

  const selectedSensorType = sensorTypes.find(type => type.value === formData.sensor_type);

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
        <Sensors sx={{ mr: 1, color: 'primary.main' }} />
        {editSensor ? 'Edit IoT Sensor' : 'Add New IoT Sensor'}
      </DialogTitle>
      
      <form onSubmit={handleSubmit}>
        <DialogContent>
          {validationMessage && (
            <Alert 
              severity={validationMessage.includes('success') ? 'success' : 'error'} 
              sx={{ mb: 2 }}
              onClose={() => setValidationMessage('')}
            >
              {validationMessage}
            </Alert>
          )}

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, pt: 1 }}>
            {/* Basic Information */}
            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <Sensors sx={{ mr: 1 }} />
                Sensor Information
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Sensor ID"
                    value={formData.sensor_id}
                    onChange={handleChange('sensor_id')}
                    required
                    fullWidth
                    error={!!errors.sensor_id}
                    helperText={errors.sensor_id || "Unique identifier (e.g., SOIL-001, CO2-FLUX-01)"}
                    InputProps={{
                      startAdornment: <InputAdornment position="start"><Sensors /></InputAdornment>
                    }}
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth required error={!!errors.sensor_type}>
                    <InputLabel>Sensor Type</InputLabel>
                    <Select
                      value={formData.sensor_type}
                      onChange={handleChange('sensor_type')}
                      label="Sensor Type"
                    >
                      {sensorTypes.map(type => (
                        <MenuItem key={type.value} value={type.value}>
                          <Box>
                            <Typography variant="body1">{type.label}</Typography>
                            <Typography variant="caption" color="text.secondary">
                              {type.description}
                            </Typography>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                    {errors.sensor_type && (
                      <Typography variant="caption" color="error" sx={{ mt: 0.5, ml: 1 }}>
                        {errors.sensor_type}
                      </Typography>
                    )}
                  </FormControl>
                </Grid>
              </Grid>
            </Box>

            {/* Location Information */}
            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <LocationOn sx={{ mr: 1 }} />
                Location
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Latitude"
                    type="number"
                    value={formData.location_lat}
                    onChange={handleChange('location_lat')}
                    required
                    fullWidth
                    error={!!errors.location_lat}
                    helperText={errors.location_lat || "Decimal degrees (e.g., -10.1234)"}
                    inputProps={{ step: 'any', min: -90, max: 90 }}
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Longitude"
                    type="number"
                    value={formData.location_lng}
                    onChange={handleChange('location_lng')}
                    required
                    fullWidth
                    error={!!errors.location_lng}
                    helperText={errors.location_lng || "Decimal degrees (e.g., -55.1234)"}
                    inputProps={{ step: 'any', min: -180, max: 180 }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    label="Installation Date"
                    type="date"
                    value={formData.installation_date}
                    onChange={handleChange('installation_date')}
                    fullWidth
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
              </Grid>
            </Box>

            <Divider />
            
            {/* Calibration Data */}
            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <Settings sx={{ mr: 1 }} />
                Calibration Data
                {selectedSensorType && (
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                    ({selectedSensorType.unit})
                  </Typography>
                )}
              </Typography>

              {errors.calibration && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {errors.calibration}
                </Alert>
              )}

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Min Value"
                    type="number"
                    value={calibrationFields.min_value}
                    onChange={handleCalibrationChange('min_value')}
                    fullWidth
                    inputProps={{ step: 'any' }}
                    helperText="Minimum expected sensor reading"
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Max Value"
                    type="number"
                    value={calibrationFields.max_value}
                    onChange={handleCalibrationChange('max_value')}
                    fullWidth
                    inputProps={{ step: 'any' }}
                    helperText="Maximum expected sensor reading"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Accuracy (%)"
                    type="number"
                    value={calibrationFields.accuracy}
                    onChange={handleCalibrationChange('accuracy')}
                    fullWidth
                    inputProps={{ step: 'any', min: 0, max: 100 }}
                    helperText="Sensor accuracy percentage"
                    error={!!errors.accuracy}
                  />
                  {errors.accuracy && (
                    <Typography variant="caption" color="error">
                      {errors.accuracy}
                    </Typography>
                  )}
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Resolution"
                    type="number"
                    value={calibrationFields.resolution}
                    onChange={handleCalibrationChange('resolution')}
                    fullWidth
                    inputProps={{ step: 'any' }}
                    helperText="Smallest detectable change"
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    label="Calibration Date"
                    type="date"
                    value={calibrationFields.calibration_date}
                    onChange={handleCalibrationChange('calibration_date')}
                    fullWidth
                    InputLabelProps={{ shrink: true }}
                    helperText="Date when sensor was last calibrated"
                  />
                </Grid>
              </Grid>
            </Box>
          </Box>
        </DialogContent>
        
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={handleClose} startIcon={<Cancel />}>
            Cancel
          </Button>
          <Button 
            type="submit" 
            variant="contained"
            disabled={loading.creating || loading.updating}
            startIcon={(loading.creating || loading.updating) ? <CircularProgress size={16} /> : <Save />}
          >
            {loading.creating || loading.updating ? (editSensor ? 'Updating...' : 'Creating...') : (editSensor ? 'Update Sensor' : 'Add Sensor')}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default SensorForm;