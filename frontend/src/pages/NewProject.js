import React, { useState } from 'react';
import { 
  Container, 
  Typography, 
  TextField, 
  Button, 
  Paper, 
  Box, 
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { createProject } from '../store/projectSlice';
import MapComponent from '../components/MapComponent';

const NewProject = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { loading, error } = useSelector(state => state.projects);
  
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    location_name: '',
    project_type: '',
    start_date: '',
    end_date: '',
    area_hectares: '',
    estimated_carbon_credits: ''
  });
  
  const [geometry, setGeometry] = useState(null);
  const [formErrors, setFormErrors] = useState({});
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    
    // Clear error for this field
    if (formErrors[name]) {
      setFormErrors({
        ...formErrors,
        [name]: null
      });
    }
  };
  
  const handleGeometryChange = (newGeometry) => {
    setGeometry(newGeometry);
    if (formErrors.geometry) {
      setFormErrors({
        ...formErrors,
        geometry: null
      });
    }
  };
  
  const validateForm = () => {
    const errors = {};
    
    if (!formData.name.trim()) {
      errors.name = 'Project name is required';
    }
    
    if (!formData.project_type) {
      errors.project_type = 'Project type is required';
    }
    
    if (!geometry) {
      errors.geometry = 'Please draw the project area on the map';
    }
    
    if (formData.area_hectares && isNaN(parseFloat(formData.area_hectares))) {
      errors.area_hectares = 'Area must be a number';
    }
    
    if (formData.estimated_carbon_credits && isNaN(parseFloat(formData.estimated_carbon_credits))) {
      errors.estimated_carbon_credits = 'Estimated carbon credits must be a number';
    }
    
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    const projectData = {
      ...formData,
      geometry: geometry,
      area_hectares: formData.area_hectares ? parseFloat(formData.area_hectares) : null,
      estimated_carbon_credits: formData.estimated_carbon_credits ? parseFloat(formData.estimated_carbon_credits) : null
    };
    
    const resultAction = await dispatch(createProject(projectData));
    
    if (createProject.fulfilled.match(resultAction)) {
      navigate(`/projects/${resultAction.payload.id}`);
    }
  };
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Create New Project
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Paper sx={{ p: 3 }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Project Name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                error={!!formErrors.name}
                helperText={formErrors.name}
                required
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Location Name"
                name="location_name"
                value={formData.location_name}
                onChange={handleChange}
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                name="description"
                value={formData.description}
                onChange={handleChange}
                multiline
                rows={4}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth error={!!formErrors.project_type}>
                <InputLabel>Project Type</InputLabel>
                <Select
                  name="project_type"
                  value={formData.project_type}
                  onChange={handleChange}
                  label="Project Type"
                  required
                >
                  <MenuItem value="reforestation">Reforestation</MenuItem>
                  <MenuItem value="avoided_deforestation">Avoided Deforestation</MenuItem>
                  <MenuItem value="agroforestry">Agroforestry</MenuItem>
                  <MenuItem value="soil_carbon">Soil Carbon</MenuItem>
                  <MenuItem value="mangrove_restoration">Mangrove Restoration</MenuItem>
                  <MenuItem value="other">Other</MenuItem>
                </Select>
                {formErrors.project_type && (
                  <Typography variant="caption" color="error">
                    {formErrors.project_type}
                  </Typography>
                )}
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Area (hectares)"
                name="area_hectares"
                type="number"
                value={formData.area_hectares}
                onChange={handleChange}
                error={!!formErrors.area_hectares}
                helperText={formErrors.area_hectares}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Start Date"
                name="start_date"
                type="date"
                value={formData.start_date}
                onChange={handleChange}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="End Date"
                name="end_date"
                type="date"
                value={formData.end_date}
                onChange={handleChange}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Estimated Carbon Credits"
                name="estimated_carbon_credits"
                type="number"
                value={formData.estimated_carbon_credits}
                onChange={handleChange}
                error={!!formErrors.estimated_carbon_credits}
                helperText={formErrors.estimated_carbon_credits}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Project Area
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Draw the project area on the map below
              </Typography>
              
              <Box sx={{ height: 400, mb: 2 }}>
                <MapComponent onGeometryChange={handleGeometryChange} />
              </Box>
              
              {formErrors.geometry && (
                <Typography variant="caption" color="error">
                  {formErrors.geometry}
                </Typography>
              )}
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                <Button 
                  variant="outlined" 
                  onClick={() => navigate('/projects')}
                >
                  Cancel
                </Button>
                <Button 
                  type="submit" 
                  variant="contained" 
                  color="primary"
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Create Project'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>
    </Container>
  );
};

export default NewProject;
