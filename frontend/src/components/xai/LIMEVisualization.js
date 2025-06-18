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
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  Info as InfoIcon,
  GridOn as GridOnIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

const LIMEVisualization = ({ data }) => {
  const [activeTab, setActiveTab] = useState(0);

  if (!data) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          No LIME data available
        </Typography>
      </Box>
    );
  }

  const { segments, summary, confidence, explanation } = data;

  // Tab panel component
  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`lime-tabpanel-${index}`}
      aria-labelledby={`lime-tab-${index}`}
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
          <VisibilityIcon />
          LIME Explanation
          <Tooltip title="LIME (Local Interpretable Model-agnostic Explanations) provides local explanations for individual predictions">
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
                {summary?.totalSegments || 0}
              </Typography>
              <Typography variant="caption">Total Segments</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="success.main">
                {summary?.positiveSegments || 0}
              </Typography>
              <Typography variant="caption">Positive</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="error.main">
                {summary?.negativeSegments || 0}
              </Typography>
              <Typography variant="caption">Negative</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="secondary">
                {confidence ? (confidence * 100).toFixed(1) + '%' : 'N/A'}
              </Typography>
              <Typography variant="caption">Confidence</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Explanation Text */}
      {explanation && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: 'info.light', color: 'info.contrastText' }}>
          <Typography variant="body1">
            <strong>Local Explanation:</strong> {explanation}
          </Typography>
        </Paper>
      )}

      {/* Visualization Tabs */}
      <Paper>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab label="Segment Importance" icon={<AssessmentIcon />} />
          <Tab label="Segment Details" icon={<GridOnIcon />} />
        </Tabs>

        <TabPanel value={activeTab} index={0}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Image Segment Importance
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Shows the importance of each image segment in the model's prediction.
            </Typography>
            
            {segments && segments.length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={segments} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="id" />
                  <YAxis />
                  <RechartsTooltip 
                    formatter={(value, name) => [
                      `${value.toFixed(3)}`, 
                      'Importance'
                    ]}
                    labelFormatter={(label) => `Segment ${label}`}
                  />
                  <Bar dataKey="importance" name="Importance">
                    {segments.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.importance > 0 ? '#4CAF50' : '#F44336'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No segment data available
              </Typography>
            )}

            {/* Top Important Segments */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Most Important Segments
              </Typography>
              <Grid container spacing={2}>
                {segments && segments
                  .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
                  .slice(0, 6)
                  .map((segment, index) => (
                    <Grid item xs={12} sm={6} md={4} key={segment.id}>
                      <Card variant="outlined">
                        <CardContent sx={{ p: 2 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography variant="subtitle2">
                              Segment {segment.id}
                            </Typography>
                            <Chip 
                              label={segment.type}
                              color={segment.type === 'positive' ? 'success' : 'error'}
                              size="small"
                            />
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            Importance: {segment.formattedImportance}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Area: {segment.formattedArea}
                          </Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={Math.abs(segment.importance) * 100} 
                            color={segment.type === 'positive' ? 'success' : 'error'}
                            sx={{ mt: 1 }}
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                  ))
                }
              </Grid>
            </Box>
          </Box>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Segment Analysis Details
            </Typography>
            
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Segment ID</strong></TableCell>
                    <TableCell align="right"><strong>Importance</strong></TableCell>
                    <TableCell align="right"><strong>Area %</strong></TableCell>
                    <TableCell align="center"><strong>Type</strong></TableCell>
                    <TableCell align="center"><strong>Impact</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {segments?.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
                    .map((segment) => (
                      <TableRow key={segment.id}>
                        <TableCell>Segment {segment.id}</TableCell>
                        <TableCell align="right">{segment.formattedImportance}</TableCell>
                        <TableCell align="right">{segment.formattedArea}</TableCell>
                        <TableCell align="center">
                          <Chip
                            label={segment.type}
                            color={segment.type === 'positive' ? 'success' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="center">
                          <LinearProgress 
                            variant="determinate" 
                            value={Math.abs(segment.importance) * 100} 
                            color={segment.type === 'positive' ? 'success' : 'error'}
                            sx={{ width: 60 }}
                          />
                        </TableCell>
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

export default LIMEVisualization; 