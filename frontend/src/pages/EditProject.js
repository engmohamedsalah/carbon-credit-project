import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  Box,
  CircularProgress,
  Alert,
  MenuItem,
  FormControl,
  InputLabel,
  Select
} from '@mui/material';
import { useNavigate, useParams } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { fetchProjectById, updateProject } from '../store/projectSlice';

const EditProject = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  
  const { currentProject, loading, error } = useSelector(state => state.projects);
  
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    location_name: '',
    area_hectares: '',
    project_type: 'Reforestation',
    start_date: '',
    end_date: '',
    estimated_carbon_credits: ''
  });
  
  const [formError, setFormError] = useState('');
  const [saving, setSaving] = useState(false);

  // Project type options
  const projectTypes = [
    'Reforestation',
    'Forest_Conservation',
    'Wetland_Restoration',
    'Grassland_Management',
    'Soil_Carbon',
    'Blue_Carbon',
    'Other'
  ];

  useEffect(() => {
    dispatch(fetchProjectById(id));
  }, [dispatch, id]);

  useEffect(() => {
    if (currentProject) {
      setFormData({
        name: currentProject.name || '',
        description: currentProject.description || '',
        location_name: currentProject.location_name || '',
        area_hectares: currentProject.area_hectares || '',
        project_type: currentProject.project_type || 'Reforestation',
        start_date: currentProject.start_date || '',
        end_date: currentProject.end_date || '',
        estimated_carbon_credits: currentProject.estimated_carbon_credits || ''
      });
    }
  }, [currentProject]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (formError) setFormError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormError('');
    setSaving(true);

    // Basic validation
    if (!formData.name.trim()) {
      setFormError('Project name is required');
      setSaving(false);
      return;
    }

    if (!formData.location_name.trim()) {
      setFormError('Location is required');
      setSaving(false);
      return;
    }

    if (formData.area_hectares && isNaN(parseFloat(formData.area_hectares))) {
      setFormError('Area must be a valid number');
      setSaving(false);
      return;
    }

    try {
      await dispatch(updateProject({ 
        id: parseInt(id), 
        projectData: {
          ...formData,
          area_hectares: formData.area_hectares ? parseFloat(formData.area_hectares) : null,
          estimated_carbon_credits: formData.estimated_carbon_credits ? parseFloat(formData.estimated_carbon_credits) : null
        }
      })).unwrap();
      
      navigate(`/projects/${id}`);
    } catch (error) {
      console.error('Failed to update project:', error);
      setFormError(error.detail || 'Failed to update project. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  if (loading && !currentProject) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (error && !currentProject) {
    return (
      <Container maxWidth="md" sx={{ mt: 4 }}>
        <Alert severity="error">
          {error}
        </Alert>
        <Box sx={{ mt: 2 }}>
          <Button variant="outlined" onClick={() => navigate('/projects')}>
            Back to Projects
          </Button>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Edit Project
      </Typography>

      {formError && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {formError}
        </Alert>
      )}

      <Paper sx={{ p: 4 }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {/* Project Name */}
            <Grid item xs={12}>
              <TextField
                required
                fullWidth
                name="name"
                label="Project Name"
                value={formData.name}
                onChange={handleChange}
                helperText="A descriptive name for your carbon credit project"
              />
            </Grid>

            {/* Description */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                name="description"
                label="Description"
                value={formData.description}
                onChange={handleChange}
                helperText="Provide details about your project goals and activities"
              />
            </Grid>

            {/* Location */}
            <Grid item xs={12} md={6}>
              <TextField
                required
                fullWidth
                name="location_name"
                label="Location"
                value={formData.location_name}
                onChange={handleChange}
                helperText="City, region, or area where project is located"
              />
            </Grid>

            {/* Project Type */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Project Type</InputLabel>
                <Select
                  name="project_type"
                  value={formData.project_type}
                  label="Project Type"
                  onChange={handleChange}
                >
                  {projectTypes.map((type) => (
                    <MenuItem key={type} value={type}>
                      {type.replace('_', ' ')}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Area */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                name="area_hectares"
                label="Area (Hectares)"
                value={formData.area_hectares}
                onChange={handleChange}
                helperText="Total project area in hectares"
                inputProps={{ min: 0, step: 0.1 }}
              />
            </Grid>

            {/* Estimated Carbon Credits */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                name="estimated_carbon_credits"
                label="Estimated Carbon Credits"
                value={formData.estimated_carbon_credits}
                onChange={handleChange}
                helperText="Expected carbon credits (tCO2e)"
                inputProps={{ min: 0, step: 0.1 }}
              />
            </Grid>

            {/* Start Date */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="date"
                name="start_date"
                label="Start Date"
                value={formData.start_date}
                onChange={handleChange}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>

            {/* End Date */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="date"
                name="end_date"
                label="End Date"
                value={formData.end_date}
                onChange={handleChange}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>

            {/* Action Buttons */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                <Button 
                  variant="outlined" 
                  onClick={() => navigate(`/projects/${id}`)}
                  disabled={saving}
                >
                  Cancel
                </Button>
                <Button 
                  type="submit" 
                  variant="contained" 
                  color="primary"
                  disabled={saving}
                >
                  {saving ? <CircularProgress size={24} /> : 'Update Project'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>
    </Container>
  );
};

export default EditProject; 