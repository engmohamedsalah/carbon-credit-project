import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import {
  Container,
  Box,
  Typography,
  Paper,
  Grid,
  Chip,
  Button,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Tooltip,
  IconButton,
} from '@mui/material';
import { ArrowBack, Verified, Visibility, ContentCopy, OpenInNew } from '@mui/icons-material';
import { fetchVerificationById, certifyVerification } from '../store/verificationSlice';

const VerificationDetail = () => {
  const { id } = useParams();
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const { currentVerification, loading } = useSelector(state => state.verifications);
  const { projects } = useSelector(state => state.projects);
  const { user } = useSelector(state => state.auth);

  const [recipientDialog, setRecipientDialog] = useState({ open: false, address: '' });
  const [certifying, setCertifying] = useState(false);

  useEffect(() => {
    if (id) {
      dispatch(fetchVerificationById(parseInt(id)));
    }
  }, [dispatch, id]);

  const projectName = projects?.find(p => p.id === currentVerification?.project_id)?.name;

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      try {
        const el = document.createElement('textarea');
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
      } catch {}
    }
  };

  const getExplorerTxUrl = (hash) => `https://mumbai.polygonscan.com/tx/${hash}`;

  const openRecipientDialog = () => setRecipientDialog({ open: true, address: '' });
  const handleConfirmCertify = async () => {
    try {
      setCertifying(true);
      await dispatch(
        certifyVerification({
          projectId: currentVerification.project_id,
          carbonAmount: Math.round(currentVerification.carbon_impact || 0),
          recipientAddress: recipientDialog.address || undefined,
        })
      );
      setRecipientDialog({ open: false, address: '' });
      // Refresh
      dispatch(fetchVerificationById(parseInt(id)));
    } catch (e) {
      console.error('Certification failed:', e);
    } finally {
      setCertifying(false);
    }
  };

  if (loading || !currentVerification) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, mb: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography variant="body2" sx={{ mt: 2 }}>Loading verification…</Typography>
      </Container>
    );
  }

  const v = currentVerification;
  const canCertify = user?.role?.toLowerCase() === 'admin' &&
    String(v.status || '').toLowerCase() === 'approved' && !v.blockchain_certified;

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Button startIcon={<ArrowBack />} onClick={() => navigate(-1)}>Back</Button>
      </Box>

      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h5">Verification #{v.id}</Typography>
          <Chip
            label={(v.status || 'created').toUpperCase()}
            color={String(v.status || '').toLowerCase() === 'approved' ? 'success' :
                   String(v.status || '').toLowerCase() === 'rejected' ? 'error' : 'warning'}
          />
        </Box>

        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">Project</Typography>
            <Typography variant="body1">{projectName || `Project ${v.project_id}`}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">Carbon Impact</Typography>
            <Typography variant="body1">{v.carbon_impact ? `${v.carbon_impact} tonnes CO₂/year` : 'N/A'}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">AI Confidence</Typography>
            <Typography variant="body1">{v.ai_confidence ? `${(v.ai_confidence * 100).toFixed(1)}%` : 'N/A'}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">Human Verified</Typography>
            <Typography variant="body1">{v.human_verified ? 'Yes' : 'No'}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">Blockchain</Typography>
            <Typography variant="body1" color={v.blockchain_certified ? 'success.main' : 'text.secondary'}>
              {v.blockchain_certified ? 'Certified' : 'Not certified'}
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">Tx Hash</Typography>
            {v.certificate_id ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                  {String(v.certificate_id).slice(0, 16)}…
                </Typography>
                <Tooltip title="Copy tx hash">
                  <IconButton size="small" onClick={() => copyToClipboard(v.certificate_id)}>
                    <ContentCopy fontSize="inherit" />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Open in explorer">
                  <IconButton
                    size="small"
                    component="a"
                    href={getExplorerTxUrl(v.certificate_id)}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <OpenInNew fontSize="inherit" />
                  </IconButton>
                </Tooltip>
              </Box>
            ) : (
              <Typography variant="body1">—</Typography>
            )}
          </Grid>
        </Grid>

        <Box sx={{ mt: 3, display: 'flex', gap: 1 }}>
          <Button variant="outlined" startIcon={<Visibility />} onClick={() => navigate(`/projects/${v.project_id}`)}>
            View Project
          </Button>
          {canCertify && (
            <Button
              variant="contained"
              startIcon={<Verified />}
              onClick={openRecipientDialog}
              disabled={certifying || !v.carbon_impact}
            >
              Certify on Blockchain
            </Button>
          )}
        </Box>
      </Paper>

      <Dialog open={recipientDialog.open} onClose={() => setRecipientDialog({ open: false, address: '' })}>
        <DialogTitle>Certify on Blockchain</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Optionally specify a recipient wallet address. If left empty, the default demo address will be used.
          </Typography>
          <TextField
            fullWidth
            label="Recipient Address (0x...)"
            placeholder="0x..."
            value={recipientDialog.address}
            onChange={(e) => setRecipientDialog(prev => ({ ...prev, address: e.target.value }))}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRecipientDialog({ open: false, address: '' })}>Cancel</Button>
          <Button variant="contained" onClick={handleConfirmCertify} disabled={certifying}>
            {certifying ? 'Certifying...' : 'Certify'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default VerificationDetail;

