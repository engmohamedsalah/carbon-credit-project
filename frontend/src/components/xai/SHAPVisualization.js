import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Info as InfoIcon,
  ShowChart as WaterfallIcon,
  BarChart as BarChartIcon
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend
} from 'recharts';

const SHAPVisualization = ({ data }) => {
  const [activeTab, setActiveTab] = useState(0);

  if (!data) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          No SHAP data available
        </Typography>
      </Box>
    );
  }

  const {
    waterfallData,
    featureImportance,
    baseValue,
    predictedValue,
    confidence
  } = data;

  // Colors for visualization
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  // Custom tooltip for bar chart
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Paper sx={{ p: 2 }}>
          <Typography variant="subtitle2">{label}</Typography>
          <Typography variant="body2">
            Value: {data.value?.toFixed(3)}
          </Typography>
          <Typography variant="body2" color={data.contribution > 0 ? 'success.main' : 'error.main'}>
            Contribution: {data.formattedContribution}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  // Tab panel component
  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`shap-tabpanel-${index}`}
      aria-labelledby={`shap-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TrendingUpIcon />
          SHAP Explanation
          <Tooltip title="SHAP (SHapley Additive exPlanations) provides feature importance using Shapley values from game theory">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="primary">
                {baseValue?.toFixed(3)}
              </Typography>
              <Typography variant="caption">Base Value</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="secondary">
                {predictedValue?.toFixed(3)}
              </Typography>
              <Typography variant="caption">Predicted Value</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="success.main">
                {confidence ? (confidence * 100).toFixed(1) + '%' : 'N/A'}
              </Typography>
              <Typography variant="caption">Confidence</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="warning.main">
                {waterfallData?.length || 0}
              </Typography>
              <Typography variant="caption">Features</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Visualization Tabs */}
      <Paper>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab label="Waterfall Chart" icon={<WaterfallIcon />} />
          <Tab label="Feature Importance" icon={<BarChartIcon />} />
          <Tab label="Data Table" icon={<TrendingUpIcon />} />
        </Tabs>

        <TabPanel value={activeTab} index={0}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              SHAP Waterfall Chart
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Shows how each feature contributes to the final prediction, starting from the base value.
            </Typography>
            
            {waterfallData && waterfallData.length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={waterfallData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="feature" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                  />
                  <YAxis />
                  <RechartsTooltip content={<CustomTooltip />} />
                  <Bar dataKey="contribution" name="Contribution">
                    {waterfallData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.contribution > 0 ? '#4CAF50' : '#F44336'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No waterfall data available
              </Typography>
            )}
          </Box>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Feature Importance Distribution
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Overall feature importance percentages across the model.
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                {featureImportance && featureImportance.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={featureImportance}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ feature, percentage }) => `${feature}: ${percentage}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="importance"
                      >
                        {featureImportance.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No feature importance data available
                  </Typography>
                )}
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle2" gutterBottom>
                  Feature Rankings
                </Typography>
                {featureImportance && featureImportance
                  .sort((a, b) => b.importance - a.importance)
                  .map((feature, index) => (
                    <Box key={feature.feature} sx={{ mb: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2">
                          {index + 1}. {feature.feature}
                        </Typography>
                        <Chip 
                          label={`${feature.percentage}%`} 
                          size="small" 
                          color={index === 0 ? 'primary' : 'default'}
                        />
                      </Box>
                    </Box>
                  ))
                }
              </Grid>
            </Grid>
          </Box>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              SHAP Values Data
            </Typography>
            
            {/* Waterfall Data Table */}
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
              Feature Contributions
            </Typography>
            <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Feature</strong></TableCell>
                    <TableCell align="right"><strong>Value</strong></TableCell>
                    <TableCell align="right"><strong>Contribution</strong></TableCell>
                    <TableCell align="center"><strong>Impact</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {waterfallData?.map((row, index) => (
                    <TableRow key={index}>
                      <TableCell>{row.feature}</TableCell>
                      <TableCell align="right">{row.value?.toFixed(3)}</TableCell>
                      <TableCell align="right">{row.formattedContribution}</TableCell>
                      <TableCell align="center">
                        <Chip
                          label={row.contribution > 0 ? 'Positive' : 'Negative'}
                          color={row.contribution > 0 ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Feature Importance Table */}
            <Typography variant="subtitle2" gutterBottom>
              Feature Importance Summary
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Rank</strong></TableCell>
                    <TableCell><strong>Feature</strong></TableCell>
                    <TableCell align="right"><strong>Importance</strong></TableCell>
                    <TableCell align="right"><strong>Percentage</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {featureImportance?.sort((a, b) => b.importance - a.importance)
                    .map((row, index) => (
                      <TableRow key={index}>
                        <TableCell>{index + 1}</TableCell>
                        <TableCell>{row.feature}</TableCell>
                        <TableCell align="right">{row.importance?.toFixed(3)}</TableCell>
                        <TableCell align="right">{row.percentage}%</TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default SHAPVisualization; 