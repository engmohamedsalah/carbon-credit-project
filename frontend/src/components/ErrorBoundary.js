import React from 'react';
import { Box, Typography, Button, Paper, Alert } from '@mui/material';
import { Refresh as RefreshIcon, Home as HomeIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

/**
 * Professional Error Boundary Component
 * Catches JavaScript errors in child components and displays fallback UI
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null 
    };
  }

  static getDerivedStateFromError(error) {
    // Update state to trigger fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details for debugging
    console.error('Error Boundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo
    });

    // Report to error monitoring service (e.g., Sentry)
    if (window.reportError) {
      window.reportError(error, errorInfo);
    }
  }

  handleRetry = () => {
    // Reset error state to retry rendering
    this.setState({ 
      hasError: false, 
      error: null, 
      errorInfo: null 
    });
  };

  render() {
    if (this.state.hasError) {
      return <ErrorFallback 
        error={this.state.error}
        errorInfo={this.state.errorInfo}
        onRetry={this.handleRetry}
        componentName={this.props.fallbackProps?.componentName}
      />;
    }

    return this.props.children;
  }
}

/**
 * Error Fallback Component
 * Professional error UI with retry and navigation options
 */
const ErrorFallback = ({ error, errorInfo, onRetry, componentName }) => {
  const navigate = useNavigate();

  const isDevelopment = process.env.NODE_ENV === 'development';

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '400px',
        p: 3,
        textAlign: 'center'
      }}
    >
      <Paper 
        elevation={1}
        sx={{ 
          p: 4, 
          maxWidth: 600, 
          width: '100%',
          borderRadius: 2
        }}
      >
        <Typography variant="h5" color="error" gutterBottom>
          Something went wrong
        </Typography>
        
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          {componentName 
            ? `There was an error in the ${componentName} component.`
            : 'An unexpected error occurred while loading this page.'
          }
        </Typography>

        {isDevelopment && error && (
          <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
            <Typography variant="subtitle2" gutterBottom>
              Error Details (Development Mode):
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
              {error.toString()}
            </Typography>
          </Alert>
        )}

        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Button 
            variant="contained" 
            startIcon={<RefreshIcon />}
            onClick={onRetry}
            sx={{ minWidth: 120 }}
          >
            Try Again
          </Button>
          
          <Button 
            variant="outlined" 
            startIcon={<HomeIcon />}
            onClick={() => navigate('/dashboard')}
            sx={{ minWidth: 120 }}
          >
            Go Home
          </Button>
        </Box>

        <Typography variant="caption" color="text.secondary" sx={{ mt: 3, display: 'block' }}>
          If this problem persists, please contact support or refresh the page.
        </Typography>
      </Paper>
    </Box>
  );
};

export default ErrorBoundary; 