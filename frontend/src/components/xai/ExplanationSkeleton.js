import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Skeleton,
  Grid,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import { ExpandMore as ExpandMoreIcon } from '@mui/icons-material';

const ExplanationSkeleton = ({ variant = 'full' }) => {
  if (variant === 'compact') {
    return (
      <Card>
        <CardContent>
          <Stack spacing={2}>
            <Skeleton variant="rectangular" height={40} />
            <Skeleton variant="rectangular" height={100} />
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Skeleton variant="rounded" width={80} height={24} />
              <Skeleton variant="rounded" width={60} height={24} />
            </Box>
          </Stack>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Header skeleton */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <Skeleton variant="circular" width={32} height={32} />
            <Skeleton variant="text" width={200} height={32} />
            <Box sx={{ ml: 'auto' }}>
              <Skeleton variant="rounded" width={100} height={24} />
            </Box>
          </Box>
          <Skeleton variant="text" width="60%" height={20} />
        </CardContent>
      </Card>

      {/* Executive Summary skeleton */}
      <Accordion defaultExpanded disabled>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Skeleton variant="text" width={150} height={24} />
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={1}>
            <Skeleton variant="text" width="100%" />
            <Skeleton variant="text" width="90%" />
            <Skeleton variant="text" width="75%" />
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* Risk Assessment skeleton */}
      <Accordion disabled>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Skeleton variant="circular" width={20} height={20} />
            <Skeleton variant="text" width={120} height={24} />
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={2}>
            <Skeleton variant="text" width="100%" />
            <Skeleton variant="text" width="85%" />
            <Box>
              <Skeleton variant="text" width={120} height={20} />
              <Stack spacing={0.5} sx={{ ml: 2, mt: 1 }}>
                <Skeleton variant="text" width="95%" />
                <Skeleton variant="text" width="88%" />
                <Skeleton variant="text" width="92%" />
              </Stack>
            </Box>
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* Visualizations skeleton */}
      <Accordion defaultExpanded disabled>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Skeleton variant="text" width={180} height={24} />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Skeleton variant="text" width={120} height={20} sx={{ mx: 'auto', mb: 2 }} />
                  <Skeleton variant="rectangular" height={200} />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Skeleton variant="text" width={100} height={20} sx={{ mx: 'auto', mb: 2 }} />
                  <Skeleton variant="rectangular" height={200} />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
            <Skeleton variant="text" width="60%" height={16} />
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Loading indicator */}
      <Box sx={{ 
        position: 'fixed', 
        bottom: 24, 
        right: 24, 
        zIndex: 1000,
        bgcolor: 'primary.main',
        color: 'white',
        px: 3,
        py: 1.5,
        borderRadius: 3,
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
      }}>
        <Box sx={{ 
          width: 16, 
          height: 16, 
          border: '2px solid currentColor',
          borderTopColor: 'transparent',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          '@keyframes spin': {
            '0%': { transform: 'rotate(0deg)' },
            '100%': { transform: 'rotate(360deg)' }
          }
        }} />
        <Box sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
          Generating AI explanation...
        </Box>
      </Box>
    </Box>
  );
};

export default ExplanationSkeleton; 