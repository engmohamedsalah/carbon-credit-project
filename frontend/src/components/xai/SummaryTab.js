import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';

const SummaryTab = ({ explanation }) => {
  if (!explanation?.business_summary) {
    return (
      <Box sx={{ textAlign: 'center', py: 6 }}>
        <AssessmentIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No summary available
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 2, md: 3 } }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <TrendingUpIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Executive Summary
          </Typography>
          <Chip 
            label="AI Analysis"
            size="small"
            color="primary"
            variant="outlined"
            sx={{ fontWeight: 500 }}
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          Comprehensive analysis of carbon credit verification results
        </Typography>
      </Box>

      {/* Enhanced Summary Content */}
      <Paper sx={{ p: { xs: 3, md: 4 }, bgcolor: 'grey.50' }}>
        <Box sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            lg: 'repeat(2, 1fr)'
          },
          gap: { xs: 3, lg: 5 },
          '& h3': {
            fontSize: '1.2rem',
            fontWeight: 600,
            color: 'primary.main',
            mt: { xs: 2, lg: 3 },
            mb: 1.5,
            '&:first-of-type': { mt: 0 },
            display: 'flex',
            alignItems: 'center',
            gap: 1
          },
          '& p': {
            lineHeight: 1.7,
            mb: 2,
            color: 'text.primary',
            fontSize: '1rem'
          },
          '& ul': {
            pl: 0,
            listStyle: 'none',
            '& li': {
              mb: 1,
              lineHeight: 1.6,
              position: 'relative',
              pl: 3,
              '&:before': {
                content: '"•"',
                color: 'primary.main',
                fontWeight: 'bold',
                position: 'absolute',
                left: 0,
                fontSize: '1.2rem'
              }
            }
          },
          '& strong': {
            fontWeight: 600,
            color: 'text.primary'
          }
        }}>
          <Box sx={{ gridColumn: { xs: '1', lg: 'span 2' } }}>
            <Typography 
              variant="body1" 
              component="div"
              sx={{ 
                fontSize: '1rem',
                lineHeight: 1.7,
                color: 'text.primary'
              }}
              dangerouslySetInnerHTML={{
                __html: explanation.business_summary
                  // Convert **text** to headings with icons
                  .replace(/\*\*(.*?)\*\*/g, '<h3><span style="color: #1976d2;">📊</span> $1</h3>')
                  // Convert * items to list items
                  .replace(/\* (.*?)(?=\n|$)/g, '<li>$1</li>')
                  // Wrap consecutive list items in ul
                  .replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>')
                  // Convert double newlines to paragraph breaks
                  .replace(/\n\n/g, '</p><p>')
                  // Wrap in paragraphs
                  .replace(/^/, '<p>')
                  .replace(/$/, '</p>')
                  // Clean up empty paragraphs
                  .replace(/<p><\/p>/g, '')
                  // Make important numbers stand out
                  .replace(/(\d+\.?\d*%)/g, '<strong>$1</strong>')
                  .replace(/(\$[\d,]+)/g, '<strong>$1</strong>')
              }}
            />
          </Box>
        </Box>
      </Paper>

      {/* Key Metrics Summary */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Key Insights
        </Typography>
        <Box sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            sm: 'repeat(2, 1fr)',
            md: 'repeat(3, 1fr)'
          },
          gap: 2
        }}>
          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.50' }}>
            <Typography variant="h4" color="success.main" sx={{ fontWeight: 700 }}>
              {(explanation.confidence_score * 100).toFixed(1)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              AI Confidence Score
            </Typography>
          </Paper>
          
          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.50' }}>
            <Typography variant="h4" color="info.main" sx={{ fontWeight: 700 }}>
              {explanation.method?.toUpperCase() || 'AI'}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Analysis Method
            </Typography>
          </Paper>
          
          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.50' }}>
            <Typography variant="h4" color="primary.main" sx={{ fontWeight: 700 }}>
              {new Date(explanation.timestamp).toLocaleDateString()}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Analysis Date
            </Typography>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default React.memo(SummaryTab); 