import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Button,
  TextField,
  CircularProgress,
  Alert,
  Chip,
  Divider,
  Card,
  CardContent,
  CardActions,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useNavigate } from 'react-router-dom';
import { fetchVerificationById } from '../store/verificationSlice';
import { submitHumanReview, certifyVerification } from '../store/verificationSlice';
import MapComponent from '../components/MapComponent';

const Verification = () => {
  const { id } = useParams();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  
  const { currentVerification, loading, error, certificationResult } = useSelector(state => state.verifications);
  const { user } = useSelector(state => state.auth);
  
  const [reviewNotes, setReviewNotes] = useState('');
  const [reviewDecision, setReviewDecision] = useState(null);
  
  useEffect(() => {
    dispatch(fetchVerificationById(id));
  }, [dispatch, id]);
  
  const handleSubmitReview = async () => {
    await dispatch(submitHumanReview({
      id,
      approved: reviewDecision === 'approve',
      notes: reviewNotes
    }));
  };
  
  const handleCertify = async () => {
    await dispatch(certifyVerification(id));
  };
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
        return 'default';
      case 'in_progress':
        return 'primary';
      case 'human_review':
        return 'warning';
      case 'approved':
        return 'success';
      case 'rejected':
        return 'error';
      default:
        return 'default';
    }
  };
  
  const formatDate = (dateString) => {
    if (!dateString) return 'Not set';
    return new Date(dateString).toLocaleDateString();
  };
  
  const canReview = () => {
    if (!currentVerification || !user) return false;
    
    // Only verifiers and admins can review
    if (user.role !== 'verifier' && user.role !== 'admin') return false;
    
    // Can only review if status is human_review or if the user is an admin
    return currentVerification.status === 'human_review' || user.role === 'admin';
  };
  
  const canCertify = () => {
    if (!currentVerification || !user) return false;
    
    // Only verifiers and admins can certify
    if (user.role !== 'verifier' && user.role !== 'admin') return false;
    
    // Can only certify if status is approved and not already certified
    return currentVerification.status === 'approved' && !currentVerification.blockchain_transaction_hash;
  };
  
  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, textAlign: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }
  
  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }
  
  if (!currentVerification) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="info">Verification not found</Alert>
      </Container>
    );
  }
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" gutterBottom>
          Verification #{id}
        </Typography>
        
        <Chip 
          label={currentVerification.status.replace('_', ' ').toUpperCase()} 
          color={getStatusColor(currentVerification.status)}
          sx={{ textTransform: 'capitalize' }}
        />
      </Box>
      
      {certificationResult && (
        <Alert severity="success" sx={{ mb: 3 }}>
          <Typography variant="subtitle1">
            Successfully certified on blockchain!
          </Typography>
          <Typography variant="body2">
            Transaction Hash: {certificationResult.transaction_hash}
          </Typography>
          <Typography variant="body2">
            Token ID: {certificationResult.token_id}
          </Typography>
        </Alert>
      )}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Verification Details
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Project
                  </Typography>
                  <Typography variant="body1">
                    {currentVerification.project?.name || 'Unknown Project'}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Verification Date
                  </Typography>
                  <Typography variant="body1">
                    {formatDate(currentVerification.verification_date || currentVerification.created_at)}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Verified Carbon Credits
                  </Typography>
                  <Typography variant="body1">
                    {currentVerification.verified_carbon_credits ? `${currentVerification.verified_carbon_credits} tonnes CO₂e` : 'Not specified'}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Confidence Score
                  </Typography>
                  <Typography variant="body1">
                    {currentVerification.confidence_score ? `${(currentVerification.confidence_score * 100).toFixed(1)}%` : 'Not specified'}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Verification Notes
                  </Typography>
                  <Typography variant="body1">
                    {currentVerification.verification_notes || 'No notes provided.'}
                  </Typography>
                </Box>
              </Grid>
              
              {currentVerification.human_reviewed && (
                <Grid item xs={12}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Human Review Notes
                    </Typography>
                    <Typography variant="body1">
                      {currentVerification.human_review_notes || 'No review notes provided.'}
                    </Typography>
                  </Box>
                </Grid>
              )}
              
              {currentVerification.blockchain_transaction_hash && (
                <Grid item xs={12}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Blockchain Transaction
                    </Typography>
                    <Typography variant="body1">
                      {currentVerification.blockchain_transaction_hash}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Timestamp
                    </Typography>
                    <Typography variant="body1">
                      {formatDate(currentVerification.blockchain_timestamp)}
                    </Typography>
                  </Box>
                </Grid>
              )}
            </Grid>
          </Paper>
          
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Satellite Analysis
            </Typography>
            
            {currentVerification.satellite_analyses && currentVerification.satellite_analyses.length > 0 ? (
              currentVerification.satellite_analyses.map((analysis) => (
                <Accordion key={analysis.id}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>{analysis.analysis_type.replace('_', ' ')}</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          Analysis Date
                        </Typography>
                        <Typography variant="body1">
                          {formatDate(analysis.analysis_date)}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          Confidence Score
                        </Typography>
                        <Typography variant="body1">
                          {analysis.confidence_score ? `${(analysis.confidence_score * 100).toFixed(1)}%` : 'Not specified'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="body2" color="text.secondary">
                          Results
                        </Typography>
                        <pre>
                          {JSON.stringify(analysis.result_data, null, 2)}
                        </pre>
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              ))
            ) : (
              <Alert severity="info">No satellite analyses available for this verification.</Alert>
            )}
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          {canReview() && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Human Review
                </Typography>
                
                <TextField
                  fullWidth
                  label="Review Notes"
                  multiline
                  rows={4}
                  value={reviewNotes}
                  onChange={(e) => setReviewNotes(e.target.value)}
                  sx={{ mb: 2 }}
                />
                
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button 
                    variant="contained" 
                    color="success"
                    onClick={() => setReviewDecision('approve')}
                    sx={{ flex: 1 }}
                  >
                    Approve
                  </Button>
                  
                  <Button 
                    variant="contained" 
                    color="error"
                    onClick={() => setReviewDecision('reject')}
                    sx={{ flex: 1 }}
                  >
                    Reject
                  </Button>
                </Box>
              </CardContent>
              
              {reviewDecision && (
                <CardActions>
                  <Button 
                    fullWidth
                    variant="contained"
                    onClick={handleSubmitReview}
                    disabled={loading}
                  >
                    {loading ? <CircularProgress size={24} /> : `Submit ${reviewDecision === 'approve' ? 'Approval' : 'Rejection'}`}
                  </Button>
                </CardActions>
              )}
            </Card>
          )}
          
          {canCertify() && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Blockchain Certification
                </Typography>
                
                <Typography variant="body2" paragraph>
                  This verification has been approved and is ready to be certified on the blockchain.
                </Typography>
              </CardContent>
              
              <CardActions>
                <Button 
                  fullWidth
                  variant="contained"
                  color="primary"
                  onClick={handleCertify}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Certify on Blockchain'}
                </Button>
              </CardActions>
            </Card>
          )}
          
          {currentVerification.explanation_data && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  AI Explanation
                </Typography>
                
                <Typography variant="body2" paragraph>
                  The AI model has provided the following explanation for its verification decision:
                </Typography>
                
                <pre>
                  {JSON.stringify(currentVerification.explanation_data, null, 2)}
                </pre>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default Verification;
