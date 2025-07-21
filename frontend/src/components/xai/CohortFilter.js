import React from 'react';
import {
  Box,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Button,
  Grid,
  Divider
} from '@mui/material';
import {
  FilterList as FilterIcon,
  Clear as ClearIcon,
  Public as RegionIcon,
  Category as TypeIcon,
  Verified as StatusIcon
} from '@mui/icons-material';

const CohortFilter = ({ 
  filters, 
  onFiltersChange, 
  availableRegions = [], 
  availableProjectTypes = [],
  availableStatuses = [],
  sampleCount = 0
}) => {
  
  const handleFilterChange = (filterType, value) => {
    onFiltersChange({
      ...filters,
      [filterType]: value
    });
  };

  const handleClearFilters = () => {
    onFiltersChange({
      region: 'all',
      projectType: 'all', 
      verificationStatus: 'all',
      sliceExpression: 'ALL'
    });
  };

  const getActiveFilterCount = () => {
    return Object.values(filters || {}).filter(value => value && value !== 'all').length;
  };

  const generateSliceExpression = () => {
    const conditions = [];
    
    if (filters.region && filters.region !== 'all') {
      conditions.push(`region="${filters.region}"`);
    }
    if (filters.projectType && filters.projectType !== 'all') {
      conditions.push(`type="${filters.projectType}"`);
    }
    if (filters.verificationStatus && filters.verificationStatus !== 'all') {
      conditions.push(`status="${filters.verificationStatus}"`);
    }
    
    return conditions.length > 0 ? conditions.join(' AND ') : 'ALL';
  };

  // Default options if not provided
  const defaultRegions = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania'];
  const defaultProjectTypes = ['Reforestation', 'Conservation', 'Restoration', 'Agroforestry', 'REDD+'];
  const defaultStatuses = ['Verified', 'Under Review', 'Pending', 'Rejected'];

  const regions = availableRegions.length > 0 ? availableRegions : defaultRegions;
  const projectTypes = availableProjectTypes.length > 0 ? availableProjectTypes : defaultProjectTypes;
  const statuses = availableStatuses.length > 0 ? availableStatuses : defaultStatuses;

  return (
    <Paper sx={{ p: 3, mb: 3, bgcolor: 'grey.50' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <FilterIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Cohort Filtering
          </Typography>
          {getActiveFilterCount() > 0 && (
            <Chip 
              label={`${getActiveFilterCount()} active`}
              size="small"
              color="primary"
              sx={{ fontWeight: 500 }}
            />
          )}
        </Box>
        
        <Button
          startIcon={<ClearIcon />}
          onClick={handleClearFilters}
          disabled={getActiveFilterCount() === 0}
          size="small"
          sx={{ textTransform: 'none' }}
        >
          Clear All
        </Button>
      </Box>

      {/* Filter Controls */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel id="region-filter-label">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <RegionIcon fontSize="small" />
                Region
              </Box>
            </InputLabel>
            <Select
              labelId="region-filter-label"
              value={filters?.region || 'all'}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <RegionIcon fontSize="small" />
                  Region
                </Box>
              }
              onChange={(e) => handleFilterChange('region', e.target.value)}
            >
              <MenuItem value="all">
                <em>All Regions</em>
              </MenuItem>
              {regions.map(region => (
                <MenuItem key={region} value={region}>
                  {region}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel id="project-type-filter-label">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <TypeIcon fontSize="small" />
                Project Type
              </Box>
            </InputLabel>
            <Select
              labelId="project-type-filter-label"
              value={filters?.projectType || 'all'}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <TypeIcon fontSize="small" />
                  Project Type
                </Box>
              }
              onChange={(e) => handleFilterChange('projectType', e.target.value)}
            >
              <MenuItem value="all">
                <em>All Types</em>
              </MenuItem>
              {projectTypes.map(type => (
                <MenuItem key={type} value={type}>
                  {type}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel id="status-filter-label">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <StatusIcon fontSize="small" />
                Verification Status
              </Box>
            </InputLabel>
            <Select
              labelId="status-filter-label"
              value={filters?.verificationStatus || 'all'}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <StatusIcon fontSize="small" />
                  Verification Status
                </Box>
              }
              onChange={(e) => handleFilterChange('verificationStatus', e.target.value)}
            >
              <MenuItem value="all">
                <em>All Statuses</em>
              </MenuItem>
              {statuses.map(status => (
                <MenuItem key={status} value={status}>
                  {status}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
      </Grid>

      <Divider sx={{ my: 2 }} />

      {/* Filter Summary */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="body2" color="text.secondary">
            <strong>Sample Size:</strong> {sampleCount.toLocaleString()} projects
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>Filter Expression:</strong> <code>{generateSliceExpression()}</code>
          </Typography>
        </Box>

        {/* Active Filter Tags */}
        {getActiveFilterCount() > 0 && (
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {filters?.region && filters.region !== 'all' && (
              <Chip
                label={`Region: ${filters.region}`}
                onDelete={() => handleFilterChange('region', 'all')}
                size="small"
                color="primary"
                variant="outlined"
              />
            )}
            {filters?.projectType && filters.projectType !== 'all' && (
              <Chip
                label={`Type: ${filters.projectType}`}
                onDelete={() => handleFilterChange('projectType', 'all')}
                size="small"
                color="primary"
                variant="outlined"
              />
            )}
            {filters?.verificationStatus && filters.verificationStatus !== 'all' && (
              <Chip
                label={`Status: ${filters.verificationStatus}`}
                onDelete={() => handleFilterChange('verificationStatus', 'all')}
                size="small"
                color="primary"
                variant="outlined"
              />
            )}
          </Box>
        )}
      </Box>

      {/* Help Text */}
      <Box sx={{ mt: 2, p: 2, bgcolor: 'info.50', borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary">
          💡 <strong>Tip:</strong> Use filters to analyze specific cohorts of projects. 
          Changes will automatically update all visualizations below with filtered data.
          Filter expressions follow the format: field="value" AND field2="value2"
        </Typography>
      </Box>
    </Paper>
  );
};

export default CohortFilter; 