import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  Alert,
  Divider
} from '@mui/material';
import {
  BarChart as BarChartIcon,
  Download as DownloadIcon,
  Fullscreen as FullscreenIcon,
  Assessment as AssessmentIcon,
  ScatterPlot as ScatterPlotIcon,
  TrendingUp as TrendingUpIcon,
  CompareArrows as CompareArrowsIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import CohortFilter from './CohortFilter';

const VisualizationsTab = ({ 
  explanation, 
  initialTab = 0,
  initialPrimaryFeature = 'forest_area',
  initialInteractionFeature = 'soil_quality',
  onUrlParamsChange = () => {}
}) => {
  const [activeTab, setActiveTab] = useState(initialTab);
  const [selectedPrimaryFeature, setSelectedPrimaryFeature] = useState(initialPrimaryFeature);
  const [selectedInteractionFeature, setSelectedInteractionFeature] = useState(initialInteractionFeature);
  
  // Cohort filtering state
  const [cohortFilters, setCohortFilters] = useState({
    region: 'all',
    projectType: 'all',
    verificationStatus: 'all',
    sliceExpression: 'ALL'
  });

  // Enhanced handlers with URL updates
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    onUrlParamsChange({ vizTab: newValue });
  };

  const handlePrimaryFeatureChange = (newFeature) => {
    setSelectedPrimaryFeature(newFeature);
    onUrlParamsChange({ primary: newFeature });
  };

  const handleInteractionFeatureChange = (newFeature) => {
    setSelectedInteractionFeature(newFeature);
    onUrlParamsChange({ interaction: newFeature });
  };

  const handleDownloadChart = (chartKey, imageData) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = `${chartKey}_analysis.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleFullscreen = (chartKey, imageData) => {
    const newWindow = window.open('', '_blank');
    newWindow.document.write(`
      <html>
        <head>
          <title>${chartKey.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} - Full View</title>
          <style>
            body { margin: 0; background: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
            img { max-width: 100%; max-height: 100%; object-fit: contain; }
          </style>
        </head>
        <body>
          <img src="${imageData}" alt="${chartKey}" />
        </body>
      </html>
    `);
  };

  // Tab Panel Component
  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`viz-tabpanel-${index}`}
      aria-labelledby={`viz-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );

  // Beeswarm Plot Component
  const BeeswarmPlot = ({ globalData }) => {
    if (!globalData?.features || globalData.features.length === 0) {
      return (
        <Alert severity="info">
          Global feature importance data not available for beeswarm visualization
        </Alert>
      );
    }

    // Prepare data for beeswarm plot (showing top 10 features)
    const topFeatures = globalData.features.slice(0, 10);
    const plotData = [];

    topFeatures.forEach((feature, index) => {
      // Sample down to max 500 points for performance
      const sampleSize = Math.min(feature.values.length, 500);
      const sampledValues = feature.values.slice(0, sampleSize);
      
      plotData.push({
        y: Array(sampleSize).fill(feature.display_name),
        x: sampledValues,
        type: 'box',
        name: feature.display_name,
        boxpoints: 'all',
        pointpos: 0,
        jitter: 0.8,
        marker: {
          color: sampledValues,
          colorscale: 'RdBu',
          size: 4,
          opacity: 0.7
        },
        hovertemplate: '<b>%{y}</b><br>SHAP Value: %{x:.3f}<extra></extra>'
      });
    });

    return (
      <Box>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          🎯 Global Feature Importance (Beeswarm Plot)
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Shows the distribution of SHAP values for each feature across all {globalData.sample_count} samples. 
          Red indicates positive impact, blue indicates negative impact on the prediction.
        </Typography>
        
        <Paper sx={{ p: 2 }}>
          <Plot
            data={plotData}
            layout={{
              title: `${globalData.method?.toUpperCase()} Global Feature Importance`,
              xaxis: { title: 'SHAP Value' },
              yaxis: { title: 'Features' },
              height: 600,
              margin: { l: 150, r: 50, t: 50, b: 50 },
              showlegend: false,
              hovermode: 'closest'
            }}
            config={{ responsive: true, displayModeBar: true }}
            style={{ width: '100%' }}
          />
        </Paper>

        <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
          <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
            💡 Key Insights:
          </Typography>
          <Typography variant="body2">
            • <strong>{topFeatures[0]?.display_name}</strong> has the highest average impact on predictions<br/>
            • Features are ranked by mean absolute SHAP value<br/>
            • Wider distributions indicate more variable feature importance across samples<br/>
            • Analysis based on {globalData.sample_count} real project samples
          </Typography>
        </Box>
      </Box>
    );
  };

  // Dependence Plot Component
  const DependencePlot = ({ interactionsData }) => {
    if (!interactionsData?.dependence_plots || interactionsData.dependence_plots.length === 0) {
      return (
        <Alert severity="info">
          Feature interaction data not available for dependence plots
        </Alert>
      );
    }

    // Find the selected plot
    const selectedPlot = interactionsData.dependence_plots.find(plot => 
      plot.primary_feature === selectedPrimaryFeature && 
      plot.interaction_feature === selectedInteractionFeature
    ) || interactionsData.dependence_plots[0];

    if (!selectedPlot) {
      return <Alert severity="warning">No data available for selected feature combination</Alert>;
    }

    const plotData = [{
      x: selectedPlot.data_points.map(point => point.x),
      y: selectedPlot.data_points.map(point => point.shap),
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: selectedPlot.data_points.map(point => point.interaction),
        colorscale: 'Viridis',
        size: 6,
        opacity: 0.7,
        colorbar: {
          title: selectedPlot.interaction_display
        }
      },
      hovertemplate: `<b>${selectedPlot.primary_display}: %{x:.3f}</b><br>` +
                    `SHAP Value: %{y:.3f}<br>` +
                    `${selectedPlot.interaction_display}: %{marker.color:.3f}<extra></extra>`
    }];

    return (
      <Box>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          🔗 Feature Dependence Plot
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Shows how the SHAP value of a feature depends on its value, colored by interaction with another feature.
        </Typography>

        {/* Feature Selection Controls */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth size="small">
              <InputLabel>Primary Feature</InputLabel>
              <Select
                value={selectedPrimaryFeature}
                label="Primary Feature"
                onChange={(e) => handlePrimaryFeatureChange(e.target.value)}
              >
                {interactionsData.primary_features.map(feature => (
                  <MenuItem key={feature.name} value={feature.name}>
                    {feature.display_name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth size="small">
              <InputLabel>Interaction Feature</InputLabel>
              <Select
                value={selectedInteractionFeature}
                label="Interaction Feature"
                onChange={(e) => handleInteractionFeatureChange(e.target.value)}
              >
                {interactionsData.interaction_features?.map(feature => (
                  <MenuItem key={feature.name} value={feature.name}>
                    {feature.display_name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>

        <Paper sx={{ p: 2 }}>
          <Plot
            data={plotData}
            layout={{
              title: `${selectedPlot.primary_display} Dependence (colored by ${selectedPlot.interaction_display})`,
              xaxis: { title: selectedPlot.primary_display },
              yaxis: { title: 'SHAP Value' },
              height: 500,
              margin: { l: 80, r: 50, t: 50, b: 60 },
              hovermode: 'closest'
            }}
            config={{ responsive: true, displayModeBar: true }}
            style={{ width: '100%' }}
          />
        </Paper>

        <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
          <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
            🔍 Understanding This Plot:
          </Typography>
          <Typography variant="body2">
            • X-axis shows values of <strong>{selectedPlot.primary_display}</strong><br/>
            • Y-axis shows SHAP values (feature importance for each sample)<br/>
            • Color represents <strong>{selectedPlot.interaction_display}</strong> values<br/>
            • Patterns reveal how features interact to influence predictions
          </Typography>
        </Box>
      </Box>
    );
  };

  // Method Stability Component
  const MethodStability = ({ stabilityData }) => {
    if (!stabilityData?.correlations || Object.keys(stabilityData.correlations).length === 0) {
      return (
        <Alert severity="info">
          Method stability analysis not available
        </Alert>
      );
    }

    // Prepare correlation data
    const correlationEntries = Object.entries(stabilityData.correlations);
    const plotData = [{
      x: correlationEntries.map(([key, _]) => key.replace(/_vs_/g, ' vs ').toUpperCase()),
      y: correlationEntries.map(([_, value]) => value.tau),
      type: 'bar',
      marker: {
        color: correlationEntries.map(([_, value]) => 
          Math.abs(value.tau) > 0.7 ? '#4CAF50' : 
          Math.abs(value.tau) > 0.4 ? '#FF9800' : '#F44336'
        ),
        opacity: 0.8
      },
      hovertemplate: '<b>%{x}</b><br>Kendall τ: %{y:.3f}<br>%{text}<extra></extra>',
      text: correlationEntries.map(([_, value]) => value.interpretation)
    }];

    return (
      <Box>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          ⚖️ Method Stability Analysis
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Kendall-tau correlations between different XAI methods. Higher values indicate more consistent feature rankings.
        </Typography>

        <Paper sx={{ p: 2, mb: 3 }}>
          <Plot
            data={plotData}
            layout={{
              title: 'XAI Method Correlations (Kendall τ)',
              xaxis: { title: 'Method Pairs' },
              yaxis: { title: 'Kendall τ Correlation', range: [-1, 1] },
              height: 400,
              margin: { l: 80, r: 50, t: 50, b: 100 },
              shapes: [
                { type: 'line', x0: -0.5, x1: correlationEntries.length - 0.5, y0: 0.7, y1: 0.7, 
                  line: { color: 'green', width: 2, dash: 'dash' } },
                { type: 'line', x0: -0.5, x1: correlationEntries.length - 0.5, y0: 0.4, y1: 0.4, 
                  line: { color: 'orange', width: 2, dash: 'dash' } }
              ],
              annotations: [
                { x: correlationEntries.length - 1, y: 0.72, text: 'Strong (>0.7)', showarrow: false },
                { x: correlationEntries.length - 1, y: 0.42, text: 'Moderate (>0.4)', showarrow: false }
              ]
            }}
            config={{ responsive: true, displayModeBar: true }}
            style={{ width: '100%' }}
          />
        </Paper>

        {/* Feature Stability Table */}
        {stabilityData.feature_stability && stabilityData.feature_stability.length > 0 && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
              Most Stable Features Across Methods:
            </Typography>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 1 }}>
              {stabilityData.feature_stability.slice(0, 5).map((feature, index) => (
                <Box key={feature.feature} sx={{ p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    #{index + 1} {feature.display_name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Stability: {(feature.stability_score * 100).toFixed(1)}%
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        )}
      </Box>
    );
  };

  if (!explanation) {
    return (
      <Box sx={{ textAlign: 'center', py: 6 }}>
        <BarChartIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No analysis available
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Generate an explanation to see advanced visualizations
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 2, md: 3 } }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
        <PsychologyIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Advanced XAI Visualizations
        </Typography>
        <Chip 
          label={explanation.method?.toUpperCase() || 'AI Analysis'}
          size="small"
          color="primary"
          variant="outlined"
          sx={{ fontWeight: 500 }}
        />
      </Box>

      {/* Cohort Filtering */}
      <CohortFilter
        filters={cohortFilters}
        onFiltersChange={setCohortFilters}
        sampleCount={explanation.global_analysis?.sample_count || 0}
        availableRegions={['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania']}
        availableProjectTypes={['Reforestation', 'Conservation', 'Restoration', 'Agroforestry', 'REDD+']}
        availableStatuses={['Verified', 'Under Review', 'Pending', 'Rejected']}
      />

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          aria-label="XAI visualization tabs"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab 
            label="Global Importance" 
            icon={<BarChartIcon />} 
            id="viz-tab-0"
            aria-controls="viz-tabpanel-0"
          />
          <Tab 
            label="Feature Interactions" 
            icon={<ScatterPlotIcon />} 
            id="viz-tab-1"
            aria-controls="viz-tabpanel-1"
          />
          <Tab 
            label="Method Stability" 
            icon={<CompareArrowsIcon />} 
            id="viz-tab-2"
            aria-controls="viz-tabpanel-2"
          />
          <Tab 
            label="Legacy Charts" 
            icon={<AssessmentIcon />} 
            id="viz-tab-3"
            aria-controls="viz-tabpanel-3"
          />
        </Tabs>

        {/* Tab Panels */}
        <TabPanel value={activeTab} index={0}>
          <BeeswarmPlot globalData={explanation.global_analysis} />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <DependencePlot interactionsData={explanation.interactions} />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <MethodStability stabilityData={explanation.method_stability} />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          {/* Legacy Chart Grid */}
          {explanation.visualizations && Object.keys(explanation.visualizations).length > 0 ? (
            <Box>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                📊 Legacy Visualization Charts
              </Typography>
              <Box sx={{ 
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
                gap: 3
              }}>
                {Object.entries(explanation.visualizations).map(([chartKey, imageData]) => (
                  <Paper key={chartKey} sx={{ overflow: 'hidden' }}>
                    <Box sx={{ 
                      p: 2, 
                      bgcolor: 'grey.50', 
                      borderBottom: 1, 
                      borderColor: 'divider',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {chartKey.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Tooltip title="Download Chart">
                          <IconButton 
                            size="small"
                            onClick={() => handleDownloadChart(chartKey, imageData)}
                          >
                            <DownloadIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="View Fullscreen">
                          <IconButton 
                            size="small"
                            onClick={() => handleFullscreen(chartKey, imageData)}
                          >
                            <FullscreenIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    <Box sx={{ p: 2, textAlign: 'center' }}>
                      <img 
                        src={imageData}
                        alt={chartKey}
                        style={{ 
                          maxWidth: '100%', 
                          height: 'auto',
                          borderRadius: '4px'
                        }}
                      />
                    </Box>
                  </Paper>
                ))}
              </Box>
            </Box>
          ) : (
            <Alert severity="info">
              No legacy charts available. Advanced visualizations are shown in other tabs.
            </Alert>
          )}
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default React.memo(VisualizationsTab); 