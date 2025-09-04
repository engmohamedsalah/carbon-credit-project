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
  Fab,
  CircularProgress,
  TextField,
  InputAdornment,
  Menu,
  MenuItem as MenuItemComponent,
  IconButton
} from '@mui/material';
import {
  Sensors as SensorsIcon,
  Add as AddIcon,
  Dashboard as DashboardIcon,
  FilterList as FilterIcon,
  Search as SearchIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreVertIcon,
  Edit as EditIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { 
  fetchSensors, 
  setFilters, 
  setSelectedSensor,
  clearSelectedSensor,
  deleteSensor
} from '../store/iotSlice';
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
  const { sensors, selectedSensor, loading, filters, error } = useSelector(state => state.iot);
  
  const [showSensorForm, setShowSensorForm] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [selectedProject, setSelectedProject] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [anchorEl, setAnchorEl] = useState(null);
  const [menuSensor, setMenuSensor] = useState(null);
  const [editSensor, setEditSensor] = useState(null);

  useEffect(() => {
    // Load sensors with current filters
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

  const handleStatusFilter = (event) => {
    dispatch(setFilters({ status: event.target.value || null }));
  };

  const handleSensorSelect = (sensor) => {
    dispatch(setSelectedSensor(sensor));
  };

  const handleAddSensor = () => {
    if (!selectedProject) {
      alert('Please select a project first');
      return;
    }
    setEditSensor(null);
    setShowSensorForm(true);
  };

  const handleRefresh = () => {
    dispatch(fetchSensors(filters));
  };

  const handleMenuClick = (event, sensor) => {
    if (event) {
      event.stopPropagation();
      setAnchorEl(event.currentTarget);
    }
    setMenuSensor(sensor);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setMenuSensor(null);
  };

  const handleEditSensor = () => {
    setEditSensor(menuSensor);
    setShowSensorForm(true);
    handleMenuClose();
  };

  const handleDeleteSensor = async () => {
    if (window.confirm(`Are you sure you want to delete sensor ${menuSensor.sensor_id}?`)) {
      try {
        await dispatch(deleteSensor(menuSensor.id)).unwrap();
        handleMenuClose();
      } catch (error) {
        console.error('Failed to delete sensor:', error);
      }
    }
  };

  const getSensorStats = () => {
    // Ensure sensors is always an array
    const sensorsArray = Array.isArray(sensors) ? sensors : [];
    const total = sensorsArray.length;
    const active = sensorsArray.filter(s => s.status === 'active').length;
    const offline = sensorsArray.filter(s => s.status === 'offline').length;
    const maintenance = sensorsArray.filter(s => s.status === 'maintenance').length;
    
    return { total, active, offline, maintenance };
  };

  const getFilteredSensors = () => {
    // Ensure sensors is always an array
    const sensorsArray = Array.isArray(sensors) ? sensors : [];
    return sensorsArray.filter(sensor => 
      sensor.sensor_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sensor.sensor_type.toLowerCase().includes(searchTerm.toLowerCase())
    );
  };

  const stats = getSensorStats();
  const filteredSensors = getFilteredSensors();
  const canManageSensors = user?.role === 'admin' || user?.role === 'project_developer' || user?.role === 'Project Developer';

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
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={loading.sensors}
          >
            Refresh
          </Button>
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

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => dispatch({ type: 'iot/clearError' })}>
          <Typography variant="body1">
            <strong>Error:</strong> {error}
          </Typography>
        </Alert>
      )}

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

            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Status</InputLabel>
              <Select
                value={filters.status || ''}
                onChange={handleStatusFilter}
                label="Status"
              >
                <MenuItem value="">All Status</MenuItem>
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="offline">Offline</MenuItem>
                <MenuItem value="maintenance">Maintenance</MenuItem>
              </Select>
            </FormControl>

            <TextField
              size="small"
              placeholder="Search sensors..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
              sx={{ minWidth: 200 }}
            />

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
              Sensor Network ({filteredSensors.length} sensors)
            </Typography>
            
            {filteredSensors.length > 0 ? (
              <Grid container spacing={2}>
                {filteredSensors.map(sensor => (
                  <Grid item xs={12} sm={6} md={4} key={sensor.id}>
                    <SensorCard
                      sensor={sensor}
                      onSelect={handleSensorSelect}
                      onMenuClick={handleMenuClick}
                    />
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Card>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <SensorsIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    {searchTerm ? 'No sensors found matching your search' : 'No Sensors Found'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {searchTerm 
                      ? 'Try adjusting your search terms or filters.'
                      : selectedProject 
                        ? 'No sensors are deployed for the selected project and filters.'
                        : 'Select a project to view its sensors or add new sensors to get started.'
                    }
                  </Typography>
                  {canManageSensors && selectedProject && !searchTerm && (
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
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Sensor: {selectedSensor.sensor_id}
                  </Typography>
                  <Button
                    size="small"
                    onClick={() => dispatch(clearSelectedSensor())}
                  >
                    Close
                  </Button>
                </Box>
                <ReadingsChart 
                  sensorId={selectedSensor.id}
                  sensorType={selectedSensor.sensor_type}
                  sensor={selectedSensor}
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

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItemComponent onClick={handleEditSensor}>
          <EditIcon sx={{ mr: 1 }} />
          Edit Sensor
        </MenuItemComponent>
        <MenuItemComponent onClick={handleDeleteSensor} sx={{ color: 'error.main' }}>
          <DeleteIcon sx={{ mr: 1 }} />
          Delete Sensor
        </MenuItemComponent>
      </Menu>

      {/* Dialogs */}
      <SensorForm
        open={showSensorForm}
        onClose={() => {
          setShowSensorForm(false);
          setEditSensor(null);
        }}
        projectId={selectedProject}
        editSensor={editSensor}
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