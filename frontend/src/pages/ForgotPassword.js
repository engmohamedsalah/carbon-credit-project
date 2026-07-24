import React, { useState } from 'react';
import {
  Typography, TextField, Button, Paper, Box, Alert, CircularProgress, Link, InputAdornment,
} from '@mui/material';
import { Email } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
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

const ForgotPassword = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [demoLink, setDemoLink] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;
    if (!/\S+@\S+\.\S+/.test(email)) { setError('Enter a valid email address'); return; }
    setError(''); setLoading(true);
    try {
      const res = await apiService.auth.forgotPassword(email.trim().toLowerCase());
      setMessage(res.data?.message || 'If an account with that email exists, a reset link has been sent.');
      // Demo convenience: backend returns reset_link only when explicitly enabled (no email service).
      if (res.data?.reset_link) setDemoLink(res.data.reset_link);
    } catch (err) {
      setError('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3,
      background: 'linear-gradient(135deg, #0d1e3a 0%, #1e293b 50%, #334155 100%)' }}>
      <Paper elevation={0} sx={cardSx}>
        <Typography component="h1" variant="h5" sx={{ mb: 1, fontWeight: 700, textAlign: 'center' }}>
          Reset your password
        </Typography>
        <Typography variant="body2" sx={{ mb: 3, textAlign: 'center', color: 'rgba(255,255,255,0.75)' }}>
          Enter your email and we'll send you a link to reset it.
        </Typography>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        {message ? (
          <>
            <Alert severity="success" sx={{ mb: 2 }}>{message}</Alert>
            {demoLink && (
              <Alert severity="info" sx={{ mb: 2, wordBreak: 'break-all' }}>
                Email delivery isn't configured, so here's your reset link:{' '}
                <Link href={demoLink} sx={{ color: '#00ff88' }}>{demoLink}</Link>
              </Alert>
            )}
          </>
        ) : (
          <form onSubmit={handleSubmit} noValidate>
            <TextField
              fullWidth required autoFocus label="Email Address" type="email" value={email}
              onChange={(e) => setEmail(e.target.value)} sx={fieldSx} margin="normal"
              InputProps={{ startAdornment: (
                <InputAdornment position="start"><Email sx={{ color: 'rgba(255,255,255,0.7)' }} /></InputAdornment>
              ) }}
            />
            <Button type="submit" fullWidth size="large" variant="contained" disabled={loading}
              sx={{ mt: 3, mb: 2, py: 1.5, fontWeight: 600, textTransform: 'none', background: '#00ff88', color: '#000',
                '&:hover': { background: '#00ff88', boxShadow: '0 8px 40px rgba(0,255,136,0.4)' } }}>
              {loading ? <CircularProgress size={24} sx={{ color: '#000' }} /> : 'Send reset link'}
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

export default ForgotPassword;
