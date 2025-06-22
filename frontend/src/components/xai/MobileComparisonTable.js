import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  Stack,
  IconButton,
  Collapse,
  useMediaQuery,
  useTheme
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckIcon,
  Compare as CompareIcon
} from '@mui/icons-material';

const MobileComparisonTable = ({ 
  explanations = [], 
  selectedExplanations = [], 
  onExplanationSelect,
  onCompare
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [expandedCards, setExpandedCards] = useState(new Set());

  const toggleCardExpansion = (explanationId) => {
    const newExpanded = new Set(expandedCards);
    if (newExpanded.has(explanationId)) {
      newExpanded.delete(explanationId);
    } else {
      newExpanded.add(explanationId);
    }
    setExpandedCards(newExpanded);
  };

  const isSelected = (explanationId) => selectedExplanations.includes(explanationId);

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  if (!isMobile) {
    // Return traditional table for desktop
    return null; // Let parent component handle desktop table
  }

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">
          Select Explanations ({selectedExplanations.length})
        </Typography>
        <Button
          variant="contained"
          size="small"
          disabled={selectedExplanations.length < 2}
          onClick={onCompare}
          startIcon={<CompareIcon />}
        >
          Compare
        </Button>
      </Box>

      <Stack spacing={2}>
        {explanations.map((explanation) => {
          const expanded = expandedCards.has(explanation.explanation_id);
          const selected = isSelected(explanation.explanation_id);

          return (
            <Card 
              key={explanation.explanation_id}
              variant={selected ? "elevation" : "outlined"}
              sx={{
                border: selected ? 2 : 1,
                borderColor: selected ? 'primary.main' : 'divider',
                position: 'relative'
              }}
            >
              {/* Selection indicator */}
              {selected && (
                <Box sx={{
                  position: 'absolute',
                  top: 8,
                  right: 8,
                  zIndex: 1
                }}>
                  <CheckIcon color="primary" />
                </Box>
              )}

              <CardContent 
                sx={{ 
                  pb: expanded ? 2 : '16px !important',
                  cursor: 'pointer'
                }}
                onClick={() => onExplanationSelect(explanation.explanation_id, !selected)}
              >
                {/* Main info */}
                <Box sx={{ pr: 4 }}> {/* Leave space for check icon */}
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Chip 
                      label={explanation.method?.toUpperCase()} 
                      size="small" 
                      color="primary"
                      variant={selected ? "filled" : "outlined"}
                    />
                    <Chip 
                      label={`${(explanation.confidence_score * 100).toFixed(1)}%`}
                      color={getConfidenceColor(explanation.confidence_score)}
                      size="small"
                    />
                  </Box>

                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {new Date(explanation.timestamp).toLocaleString()}
                  </Typography>

                  <Typography variant="body2" sx={{ 
                    display: '-webkit-box',
                    WebkitLineClamp: expanded ? 'none' : 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}>
                    {explanation.business_summary || 'No summary available'}
                  </Typography>
                </Box>

                {/* Expandable details */}
                <Collapse in={expanded}>
                  <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                    <Stack spacing={2}>
                      {explanation.risk_assessment && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Risk Level
                          </Typography>
                          <Chip 
                            label={explanation.risk_assessment.level}
                            color={explanation.risk_assessment.level === 'Low' ? 'success' : 
                                   explanation.risk_assessment.level === 'Medium' ? 'warning' : 'error'}
                            size="small"
                          />
                        </Box>
                      )}

                      {explanation.regulatory_notes && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Compliance Status
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {explanation.regulatory_notes.eu_ai_act_compliance ? 
                              'EU AI Act Compliant' : 'Compliance Pending'}
                          </Typography>
                        </Box>
                      )}

                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Project ID
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {explanation.project_id}
                        </Typography>
                      </Box>
                    </Stack>
                  </Box>
                </Collapse>

                {/* Expand button */}
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'center', 
                  mt: 1,
                  pt: 1,
                  borderTop: expanded ? 0 : 1,
                  borderColor: 'divider'
                }}>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleCardExpansion(explanation.explanation_id);
                    }}
                    sx={{
                      transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                      transition: 'transform 0.3s ease'
                    }}
                  >
                    <ExpandMoreIcon />
                  </IconButton>
                </Box>
              </CardContent>
            </Card>
          );
        })}
      </Stack>

      {explanations.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body2" color="text.secondary">
            No explanations available for comparison
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default MobileComparisonTable; 