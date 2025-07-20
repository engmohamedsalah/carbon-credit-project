import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  BarChart as BarChartIcon,
  Download as DownloadIcon,
  Fullscreen as FullscreenIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';

const VisualizationsTab = ({ explanation }) => {
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

  if (!explanation?.visualizations || Object.keys(explanation.visualizations).length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 6 }}>
        <BarChartIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No visualizations available
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Charts and graphs will appear here when analysis is complete
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 2, md: 3 } }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <BarChartIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Visualizations & Charts
          </Typography>
          <Chip 
            label={`${Object.keys(explanation.visualizations).length} Charts`}
            size="small"
            color="primary"
            variant="outlined"
            sx={{ fontWeight: 500 }}
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          Interactive data visualizations and analysis charts
        </Typography>
      </Box>

      {/* Responsive Chart Grid */}
      <Box sx={{
        display: 'grid',
        gridTemplateColumns: {
          xs: '1fr',
          sm: 'repeat(auto-fit, minmax(320px, 1fr))',
          lg: 'repeat(auto-fit, minmax(400px, 1fr))',
          xl: 'repeat(auto-fit, minmax(450px, 1fr))'
        },
        gap: { xs: 3, md: 4 },
        width: '100%'
      }}>
        {Object.entries(explanation.visualizations).map(([key, imageData]) => (
          <Paper 
            key={key}
            elevation={2}
            sx={{ 
              overflow: 'hidden',
              minHeight: { xs: 300, md: 350 },
              display: 'flex',
              flexDirection: 'column',
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                elevation: 4,
                transform: 'translateY(-2px)'
              }
            }}
          >
            {/* Chart Header */}
            <Box sx={{ 
              p: 2, 
              bgcolor: 'grey.100',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.1rem' }}>
                {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 0.5 }}>
                <Tooltip title="Download Chart">
                  <IconButton 
                    size="small"
                    onClick={() => handleDownloadChart(key, imageData)}
                    sx={{ 
                      color: 'primary.main',
                      '&:hover': { bgcolor: 'primary.50' }
                    }}
                  >
                    <DownloadIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title="View Fullscreen">
                  <IconButton 
                    size="small"
                    onClick={() => handleFullscreen(key, imageData)}
                    sx={{ 
                      color: 'primary.main',
                      '&:hover': { bgcolor: 'primary.50' }
                    }}
                  >
                    <FullscreenIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>

            {/* Chart Content */}
            <Box sx={{ 
              flex: 1, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              p: 2,
              bgcolor: 'white'
            }}>
              <img 
                src={imageData} 
                alt={key}
                style={{ 
                  width: '100%',
                  height: 'auto',
                  minHeight: '250px',
                  maxHeight: '400px',
                  objectFit: 'contain',
                  borderRadius: '4px'
                }}
                loading="lazy"
              />
            </Box>

            {/* Chart Footer */}
            <Box sx={{ 
              p: 1.5, 
              bgcolor: 'grey.50',
              borderTop: '1px solid',
              borderColor: 'grey.200'
            }}>
              <Typography variant="caption" color="text.secondary">
                Generated: {new Date(explanation.timestamp).toLocaleString()}
              </Typography>
            </Box>
          </Paper>
        ))}
      </Box>

      {/* Charts Summary */}
      <Paper sx={{ mt: 4, p: 3, bgcolor: 'primary.50' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <AssessmentIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Analysis Summary
          </Typography>
        </Box>
        
        <Typography variant="body1" sx={{ mb: 2, lineHeight: 1.6 }}>
          📊 <strong>{Object.keys(explanation.visualizations).length}</strong> real-time visualization{Object.keys(explanation.visualizations).length !== 1 ? 's' : ''} generated from production ML models
        </Typography>
        
        <Typography variant="body2" color="text.secondary">
          These charts represent live analysis results from our AI verification system. 
          Each visualization provides insights into different aspects of the carbon credit verification process, 
          including feature importance, model confidence, and risk assessment factors.
        </Typography>
      </Paper>

      {/* Chart Types Information */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Chart Information
        </Typography>
        <Box sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            md: 'repeat(2, 1fr)',
            lg: 'repeat(3, 1fr)'
          },
          gap: 2
        }}>
          {Object.keys(explanation.visualizations).map((key) => (
            <Paper key={key} sx={{ p: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {key.includes('feature') ? 'Shows the importance of different features in the AI decision-making process' :
                 key.includes('confidence') ? 'Displays confidence levels across different analysis dimensions' :
                 key.includes('risk') ? 'Visualizes risk factors and assessment criteria' :
                 key.includes('time') ? 'Temporal analysis showing changes over time' :
                 'AI-generated visualization providing insights into the verification process'}
              </Typography>
            </Paper>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

export default React.memo(VisualizationsTab); 