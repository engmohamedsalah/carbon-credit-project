import React, { useState, Suspense, lazy } from 'react';
import {
  Typography,
  TextField,
  Button,
  Paper,
  Box,
  Grid,
  Alert,
  CircularProgress,
  Link,
  useMediaQuery,
  InputAdornment,
  IconButton
} from '@mui/material';
import { 
  Visibility, 
  VisibilityOff,
  Email,
  Lock
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { login, clearError } from '../store/authSlice';

// Lazy-load the heavy 3D globe to avoid blocking first paint
const CarbonGlobe = lazy(() => import('../components/visual/CarbonGlobe'));

const Login = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { loading, error } = useSelector(state => state.auth);
  const reduceMotion = useMediaQuery('(prefers-reduced-motion: reduce)');

  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  const [formErrors, setFormErrors] = useState({});
  const [showPassword, setShowPassword] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;

    // Clear error when user starts typing
    if (error) {
      dispatch(clearError());
    }

    // Clear field-specific error
    if (formErrors[name]) {
      setFormErrors({
        ...formErrors,
        [name]: null
      });
    }

    setFormData({
      ...formData,
      [name]: value
    });
  };

  const validateForm = () => {
    const errors = {};

    if (!formData.email.trim()) {
      errors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      errors.email = 'Email is invalid';
    }

    if (!formData.password) {
      errors.password = 'Password is required';
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (loading) return;
    if (!validateForm()) return;

    const resultAction = await dispatch(login(formData));
    if (login.fulfilled.match(resultAction)) {
      navigate('/dashboard');
    }
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', overflow: 'hidden', width: '100%', bgcolor: '#000' }}>
      {/* Left side - Globe */}
      <Box sx={{ 
        position: 'relative',
        flex: { xs: 0, md: 1 },
        minWidth: 0,
        display: { xs: 'none', md: 'flex' },
        alignItems: 'center',
        justifyContent: 'center',
        background: 'radial-gradient(ellipse at center, #1a3a5c 0%, #0d1e3a 40%, #030a1a 100%)',
        overflow: 'hidden'
      }}>
        <Box sx={{ 
          width: '100%', 
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Suspense fallback={
            <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <CircularProgress size={60} sx={{ color: '#00ff88' }} />
            </Box>
          }>
            <CarbonGlobe />
          </Suspense>
        </Box>
      </Box>

      {/* Right side - Login form */}
      <Box sx={{ 
        flex: { xs: '1 1 100%', md: '0 0 500px' },
        minWidth: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        p: { xs: 3, sm: 4, md: 5 },
        background: {
          xs: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
          md: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'
        }
      }}>
        {/* Mobile background globe */}
        {!reduceMotion && (
          <Box sx={{ 
            position: 'absolute', 
            inset: 0,
            opacity: 0.3,
            display: { xs: 'block', md: 'none' }
          }}>
            <Suspense fallback={null}>
              <CarbonGlobe />
            </Suspense>
          </Box>
        )}

        <Box sx={{ width: '100%', maxWidth: 400, position: 'relative', zIndex: 1 }}>
          <Typography component="h1" variant="h3" sx={{ mb: 2, fontWeight: 800, color: '#eafff5', textShadow: '0 2px 12px rgba(0,255,136,0.25)' }}>
            Carbon Credit Portal
          </Typography>
          <Typography variant="body1" sx={{ mb: 3, color: 'rgba(255,255,255,0.85)' }}>
            Verify impact. Mint trust. Grow forests.
          </Typography>

          <Paper
            elevation={0}
            sx={{
              p: 4,
              borderRadius: 3,
              background: 'rgba(255,255,255,0.08)',
              border: '1px solid rgba(255,255,255,0.15)',
              backdropFilter: 'blur(16px) saturate(180%)',
              color: '#fff',
              boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.2)',
              transition: 'all 0.3s ease'
            }}
          >
            <Typography component="h2" variant="h5" sx={{ mb: 2, textAlign: 'center', fontWeight: 700 }}>
              Sign In
            </Typography>

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <form onSubmit={handleSubmit} noValidate>
              <TextField
                margin="normal"
                required
                fullWidth
                id="email"
                label="Email Address"
                name="email"
                autoComplete="email"
                autoFocus
                value={formData.email}
                onChange={handleChange}
                error={!!formErrors.email}
                helperText={formErrors.email}
                variant="outlined"
                InputProps={{ 
                  sx: { 
                    backgroundColor: 'rgba(255,255,255,0.06)',
                    color: 'white',
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(255,255,255,0.3)'
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0,255,136,0.6)'
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                      borderColor: '#00ff88',
                      borderWidth: '2px'
                    },
                    '& input': {
                      color: 'white',
                      '&::placeholder': {
                        color: 'rgba(255,255,255,0.7)',
                        opacity: 1
                      }
                    }
                  },
                  startAdornment: (
                    <InputAdornment position="start">
                      <Email sx={{ color: 'rgba(255,255,255,0.7)' }} />
                    </InputAdornment>
                  ),
                }}
                InputLabelProps={{ 
                  sx: { 
                    color: 'rgba(255,255,255,0.7)',
                    '&.Mui-focused': {
                      color: '#00ff88',
                      fontWeight: 500
                    },
                    '&.MuiFormLabel-filled': {
                      color: 'rgba(255,255,255,0.6)'
                    }
                  } 
                }}
                FormHelperTextProps={{
                  sx: { color: '#ff6b6b' }
                }}
              />

              <TextField
                margin="normal"
                required
                fullWidth
                name="password"
                label="Password"
                type={showPassword ? 'text' : 'password'}
                id="password"
                autoComplete="current-password"
                value={formData.password}
                onChange={handleChange}
                error={!!formErrors.password}
                helperText={formErrors.password}
                variant="outlined"
                InputProps={{ 
                  sx: { 
                    backgroundColor: 'rgba(255,255,255,0.06)',
                    color: 'white',
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(255,255,255,0.3)'
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0,255,136,0.6)'
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                      borderColor: '#00ff88',
                      borderWidth: '2px'
                    },
                    '& input': {
                      color: 'white',
                      '&::placeholder': {
                        color: 'rgba(255,255,255,0.7)',
                        opacity: 1
                      }
                    }
                  },
                  startAdornment: (
                    <InputAdornment position="start">
                      <Lock sx={{ color: 'rgba(255,255,255,0.7)' }} />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        aria-label="toggle password visibility"
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                        sx={{ color: 'rgba(255,255,255,0.7)' }}
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                InputLabelProps={{ 
                  sx: { 
                    color: 'rgba(255,255,255,0.9)',
                    '&.Mui-focused': {
                      color: '#00ff88'
                    }
                  } 
                }}
                FormHelperTextProps={{
                  sx: { color: '#ff6b6b' }
                }}
              />

              <Button
                type="submit"
                fullWidth
                size="large"
                variant="contained"
                sx={{
                  mt: 3,
                  mb: 2,
                  py: 1.5,
                  fontWeight: 600,
                  fontSize: '1.1rem',
                  background: '#00ff88',
                  color: '#000',
                  border: 'none',
                  backdropFilter: 'blur(10px)',
                  textTransform: 'none',
                  letterSpacing: '0.5px',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  '&:hover': {
                    background: '#00ff88',
                    boxShadow: '0 8px 40px rgba(0, 255, 136, 0.4)',
                    transform: 'translateY(-1px)',
                    border: '1px solid rgba(0, 255, 136, 0.5)'
                  },
                  '&:active': {
                    transform: 'translateY(0)'
                  },
                  '&:disabled': {
                    background: 'rgba(255,255,255,0.08)',
                    color: 'rgba(255,255,255,0.3)',
                    border: '1px solid rgba(255,255,255,0.1)'
                  }
                }}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} sx={{ color: 'rgba(0, 0, 0, 0.9)' }} /> : 'Sign In'}
              </Button>

              <Grid container justifyContent="flex-end">
                <Grid item>
                  <Link
                    component="button"
                    variant="body2"
                    onClick={(e) => {
                      e.preventDefault();
                      navigate('/register');
                    }}
                    sx={{ color: 'rgba(255,255,255,0.9)' }}
                  >
                    {"Don't have an account? Sign Up"}
                  </Link>
                </Grid>
              </Grid>
            </form>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default Login;
