import React, { useState } from 'react';
import {
  Typography, TextField, Button, Paper, Box, Alert, CircularProgress, Link, InputAdornment, IconButton,
} from '@mui/material';
import { Lock, Visibility, VisibilityOff } from '@mui/icons-material';
import { useNavigate, useSearchParams } from 'react-router-dom';
import apiService from '../services/apiService';

const cardSx = {
  p: 4, borderRadius: 3, width: '100%', maxWidth: 420,
  background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
  backdropFilter: 'blur(16px) saturate(180%)', color: '#fff',
  boxShadow: '0 8px 32px 0 rgba(0,0,0,0.2)',
};
const fieldSx = {
  '& .MuiOutlinedInput-root': { backgroundColor: 'rgba(255,255,255,0.06)', color: '#fff' },
  '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255,255,255,0.3)' },
  '& .MuiInputLabel-root': { color: 'rgba(255,255,255,0.7)' },
  '& .MuiInputLabel-root.Mui-focused': { color: '#00ff88' },
};

const ResetPassword = () => {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const token = params.get('token') || '';

  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [show, setShow] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;
    if (password.length < 8) { setError('Password must be at least 8 characters'); return; }
    if (password !== confirm) { setError('Passwords do not match'); return; }
    setError(''); setLoading(true);
    try {
      await apiService.auth.resetPassword(token, password);
      setDone(true);
      setTimeout(() => navigate('/login'), 2500);
    } catch (err) {
      setError(err.response?.data?.detail || 'Invalid or expired reset link. Request a new one.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3,
      background: 'linear-gradient(135deg, #0d1e3a 0%, #1e293b 50%, #334155 100%)' }}>
      <Paper elevation={0} sx={cardSx}>
        <Typography component="h1" variant="h5" sx={{ mb: 3, fontWeight: 700, textAlign: 'center' }}>
          Set a new password
        </Typography>

        {!token ? (
          <Alert severity="error" sx={{ mb: 2 }}>
            Missing reset token. Please use the link from your reset email.
          </Alert>
        ) : done ? (
          <Alert severity="success" sx={{ mb: 2 }}>
            Password reset. Redirecting you to sign in…
          </Alert>
        ) : (
          <form onSubmit={handleSubmit} noValidate>
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            <TextField
              fullWidth required autoFocus label="New password" type={show ? 'text' : 'password'}
              value={password} onChange={(e) => setPassword(e.target.value)} sx={fieldSx} margin="normal"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start"><Lock sx={{ color: 'rgba(255,255,255,0.7)' }} /></InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton onClick={() => setShow(!show)} edge="end" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                      {show ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            <TextField
              fullWidth required label="Confirm new password" type={show ? 'text' : 'password'}
              value={confirm} onChange={(e) => setConfirm(e.target.value)} sx={fieldSx} margin="normal"
              InputProps={{ startAdornment: (
                <InputAdornment position="start"><Lock sx={{ color: 'rgba(255,255,255,0.7)' }} /></InputAdornment>
              ) }}
            />
            <Button type="submit" fullWidth size="large" variant="contained" disabled={loading}
              sx={{ mt: 3, mb: 2, py: 1.5, fontWeight: 600, textTransform: 'none', background: '#00ff88', color: '#000',
                '&:hover': { background: '#00ff88', boxShadow: '0 8px 40px rgba(0,255,136,0.4)' } }}>
              {loading ? <CircularProgress size={24} sx={{ color: '#000' }} /> : 'Reset password'}
            </Button>
          </form>
        )}

        <Box sx={{ textAlign: 'center', mt: 1 }}>
          <Link component="button" variant="body2" onClick={() => navigate('/login')}
            sx={{ color: 'rgba(255,255,255,0.9)' }}>
            Back to Sign In
          </Link>
        </Box>
      </Paper>
    </Box>
  );
};

export default ResetPassword;
