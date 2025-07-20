import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  InputAdornment,
  IconButton,
  Menu,
  MenuItem,
  Checkbox,
  FormControl,
  InputLabel,
  Select,
  Card,
  CardContent,
  Grid,
  Divider,
  Alert,
  Tooltip,
  Badge
} from '@mui/material';
import {
  History as HistoryIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Assessment as AssessmentIcon,
  PictureAsPdf as PdfIcon,
  Delete as DeleteIcon,
  Archive as ArchiveIcon,
  DateRange as DateRangeIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Psychology as PsychologyIcon,
  TrendingUp as TrendingUpIcon
} from '@mui/icons-material';

const HistoryReportsTab = ({ 
  explanationHistory = [], 
  onRefresh,
  onDeleteExplanation,
  onArchiveExplanation,
  loading = false 
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterMethod, setFilterMethod] = useState('all');
  const [filterConfidence, setFilterConfidence] = useState('all');
  const [selectedItems, setSelectedItems] = useState([]);
  const [anchorEl, setAnchorEl] = useState(null);
  const [dateFilter, setDateFilter] = useState('all');

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceIcon = (confidence) => {
    if (confidence >= 0.8) return <CheckCircleIcon />;
    if (confidence >= 0.6) return <WarningIcon />;
    return <ErrorIcon />;
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  // Filter and search explanations
  const filteredExplanations = useMemo(() => {
    let filtered = [...explanationHistory];

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(exp => 
        exp.business_summary?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        exp.method?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Method filter
    if (filterMethod !== 'all') {
      filtered = filtered.filter(exp => exp.method === filterMethod);
    }

    // Confidence filter
    if (filterConfidence !== 'all') {
      filtered = filtered.filter(exp => {
        const label = getConfidenceLabel(exp.confidence_score);
        return label.toLowerCase() === filterConfidence;
      });
    }

    // Date filter
    if (dateFilter !== 'all') {
      const now = new Date();
      const filterDate = new Date();
      
      if (dateFilter === 'today') {
        filterDate.setHours(0, 0, 0, 0);
      } else if (dateFilter === 'week') {
        filterDate.setDate(now.getDate() - 7);
      } else if (dateFilter === 'month') {
        filterDate.setMonth(now.getMonth() - 1);
      }

      filtered = filtered.filter(exp => new Date(exp.timestamp) >= filterDate);
    }

    return filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }, [explanationHistory, searchTerm, filterMethod, filterConfidence, dateFilter]);

  const handleSelectAll = (checked) => {
    if (checked) {
      setSelectedItems(filteredExplanations.map(exp => exp.explanation_id));
    } else {
      setSelectedItems([]);
    }
  };

  const handleSelectItem = (explanationId, checked) => {
    if (checked) {
      setSelectedItems(prev => [...prev, explanationId]);
    } else {
      setSelectedItems(prev => prev.filter(id => id !== explanationId));
    }
  };

  const handleBulkExport = () => {
    if (selectedItems.length === 0) return;

    const selectedExplanations = explanationHistory.filter(exp => 
      selectedItems.includes(exp.explanation_id)
    );

    // Generate bulk report
    const bulkReport = {
      report_metadata: {
        generated_at: new Date().toLocaleString(),
        total_analyses: selectedExplanations.length,
        date_range: {
          from: new Date(Math.min(...selectedExplanations.map(exp => new Date(exp.timestamp)))).toLocaleDateString(),
          to: new Date(Math.max(...selectedExplanations.map(exp => new Date(exp.timestamp)))).toLocaleDateString()
        }
      },
      analyses: selectedExplanations,
      summary: {
        average_confidence: selectedExplanations.reduce((sum, exp) => sum + exp.confidence_score, 0) / selectedExplanations.length,
        methods_used: [...new Set(selectedExplanations.map(exp => exp.method))],
        high_confidence_count: selectedExplanations.filter(exp => exp.confidence_score >= 0.8).length
      }
    };

    const dataStr = JSON.stringify(bulkReport, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `xai_bulk_report_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleBulkDelete = () => {
    if (selectedItems.length === 0) return;
    
    selectedItems.forEach(id => {
      if (onDeleteExplanation) {
        onDeleteExplanation(id);
      }
    });
    setSelectedItems([]);
  };

  const clearFilters = () => {
    setSearchTerm('');
    setFilterMethod('all');
    setFilterConfidence('all');
    setDateFilter('all');
  };

  // Statistics
  const stats = useMemo(() => {
    const total = explanationHistory.length;
    const highConfidence = explanationHistory.filter(exp => exp.confidence_score >= 0.8).length;
    const methods = [...new Set(explanationHistory.map(exp => exp.method))];
    const avgConfidence = total > 0 
      ? explanationHistory.reduce((sum, exp) => sum + exp.confidence_score, 0) / total 
      : 0;

    return { total, highConfidence, methods: methods.length, avgConfidence };
  }, [explanationHistory]);

  if (explanationHistory.length === 0) {
    return (
      <Box sx={{ p: { xs: 2, md: 3 } }}>
        <Box sx={{ textAlign: 'center', py: 6 }}>
          <HistoryIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No Analysis History
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Your completed analyses will appear here for review and reporting
          </Typography>
          <Alert severity="info" sx={{ maxWidth: 500, mx: 'auto' }}>
            <Typography variant="body2">
              Generate your first XAI analysis to see it appear in the history. 
              You can then export reports, compare methods, and track analysis trends over time.
            </Typography>
          </Alert>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 2, md: 3 } }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <HistoryIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Analysis History & Reports
          </Typography>
          <Badge badgeContent={stats.total} color="primary">
            <Chip 
              label="Total Analyses"
              size="small"
              variant="outlined"
              sx={{ fontWeight: 500 }}
            />
          </Badge>
        </Box>
        <Typography variant="body2" color="text.secondary">
          View, filter, and export your XAI analysis history
        </Typography>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={2} sx={{ mb: 4 }}>
        <Grid item xs={6} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary" sx={{ fontWeight: 700 }}>
                {stats.total}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Total Analyses
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main" sx={{ fontWeight: 700 }}>
                {stats.highConfidence}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                High Confidence
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info.main" sx={{ fontWeight: 700 }}>
                {stats.methods}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Methods Used
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="secondary.main" sx={{ fontWeight: 700 }}>
                {(stats.avgConfidence * 100).toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Avg Confidence
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Filters and Search */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Filters & Search
        </Typography>
        
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              placeholder="Search analyses..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                )
              }}
              size="small"
            />
          </Grid>
          
          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Method</InputLabel>
              <Select
                value={filterMethod}
                label="Method"
                onChange={(e) => setFilterMethod(e.target.value)}
              >
                <MenuItem value="all">All Methods</MenuItem>
                <MenuItem value="shap">SHAP</MenuItem>
                <MenuItem value="lime">LIME</MenuItem>
                <MenuItem value="integrated_gradients">Integrated Gradients</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Confidence</InputLabel>
              <Select
                value={filterConfidence}
                label="Confidence"
                onChange={(e) => setFilterConfidence(e.target.value)}
              >
                <MenuItem value="all">All Levels</MenuItem>
                <MenuItem value="high">High (≥80%)</MenuItem>
                <MenuItem value="medium">Medium (60-79%)</MenuItem>
                <MenuItem value="low">Low (&lt;60%)</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Date Range</InputLabel>
              <Select
                value={dateFilter}
                label="Date Range"
                onChange={(e) => setDateFilter(e.target.value)}
              >
                <MenuItem value="all">All Time</MenuItem>
                <MenuItem value="today">Today</MenuItem>
                <MenuItem value="week">Last Week</MenuItem>
                <MenuItem value="month">Last Month</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={3}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button 
                variant="outlined" 
                onClick={clearFilters}
                size="small"
                sx={{ flex: 1 }}
              >
                Clear Filters
              </Button>
              <IconButton 
                onClick={onRefresh}
                disabled={loading}
                size="small"
              >
                <RefreshIcon />
              </IconButton>
            </Box>
          </Grid>
        </Grid>
        
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Showing {filteredExplanations.length} of {explanationHistory.length} analyses
        </Typography>
      </Paper>

      {/* Bulk Actions */}
      {selectedItems.length > 0 && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: 'primary.50' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              {selectedItems.length} item(s) selected
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                size="small"
                startIcon={<DownloadIcon />}
                onClick={handleBulkExport}
              >
                Export Selected
              </Button>
              <Button
                size="small"
                startIcon={<ArchiveIcon />}
                onClick={() => selectedItems.forEach(id => onArchiveExplanation?.(id))}
              >
                Archive
              </Button>
              <Button
                size="small"
                startIcon={<DeleteIcon />}
                color="error"
                onClick={handleBulkDelete}
              >
                Delete
              </Button>
            </Box>
          </Box>
        </Paper>
      )}

      {/* History Table */}
      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedItems.length === filteredExplanations.length && filteredExplanations.length > 0}
                    indeterminate={selectedItems.length > 0 && selectedItems.length < filteredExplanations.length}
                    onChange={(e) => handleSelectAll(e.target.checked)}
                  />
                </TableCell>
                <TableCell>Method</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell>Confidence</TableCell>
                <TableCell>Summary</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredExplanations.map((explanation) => (
                <TableRow 
                  key={explanation.explanation_id} 
                  hover
                  selected={selectedItems.includes(explanation.explanation_id)}
                >
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedItems.includes(explanation.explanation_id)}
                      onChange={(e) => handleSelectItem(explanation.explanation_id, e.target.checked)}
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <PsychologyIcon fontSize="small" color="primary" />
                      <Typography variant="body2" fontWeight={500}>
                        {explanation.method?.toUpperCase() || 'Unknown'}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {new Date(explanation.timestamp).toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip 
                      icon={getConfidenceIcon(explanation.confidence_score)}
                      label={`${(explanation.confidence_score * 100).toFixed(1)}% (${getConfidenceLabel(explanation.confidence_score)})`}
                      color={getConfidenceColor(explanation.confidence_score)}
                      size="small"
                      sx={{ fontWeight: 500 }}
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" noWrap sx={{ maxWidth: 300 }}>
                      {explanation.business_summary?.substring(0, 150) || 'No summary available'}
                      {explanation.business_summary?.length > 150 && '...'}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Tooltip title="Export Report">
                        <IconButton size="small">
                          <PdfIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="View Details">
                        <IconButton size="small">
                          <AssessmentIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton 
                          size="small" 
                          color="error"
                          onClick={() => onDeleteExplanation?.(explanation.explanation_id)}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        {filteredExplanations.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              No analyses match your current filters
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Export Options */}
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Export Options
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Generate comprehensive reports from your analysis history
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={4}>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<PdfIcon />}
              onClick={() => {
                // Generate summary report
                console.log('Generate summary report');
              }}
            >
              Summary Report
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<TrendingUpIcon />}
              onClick={() => {
                // Generate trend analysis
                console.log('Generate trend analysis');
              }}
            >
              Trend Analysis
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<AssessmentIcon />}
              onClick={() => {
                // Generate detailed report
                console.log('Generate detailed report');
              }}
            >
              Detailed Report
            </Button>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default React.memo(HistoryReportsTab); 