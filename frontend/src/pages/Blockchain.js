import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Card,
  CardContent,
  Alert,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  Chip,
  CircularProgress
} from '@mui/material';
import BlockIcon from '@mui/icons-material/ViewInAr';
import SearchIcon from '@mui/icons-material/Search';
import VerifiedIcon from '@mui/icons-material/Verified';
import { COMMON_STYLES } from '../theme/constants';
import apiService from '../services/apiService';
import { API_ENDPOINTS } from '../config/api';
import { useLocation } from 'react-router-dom';

const Blockchain = () => {
  const [tokenId, setTokenId] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [networkInfo, setNetworkInfo] = useState(null);
  const [blockchainEnabled, setBlockchainEnabled] = useState(null); // null = unknown until fetched
  const location = useLocation();

  useEffect(() => {
    // Load network info on component mount
    loadNetworkInfo();
    apiService.get(API_ENDPOINTS.blockchain.enabled)
      .then(res => setBlockchainEnabled(Boolean((res.data ?? res).enabled)))
      .catch(() => setBlockchainEnabled(false));
  }, []);

  // Auto-verify if token_id_or_hash is provided in URL
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const prefill = params.get('token_id_or_hash');
    if (prefill && prefill.trim()) {
      setTokenId(prefill);
      handleVerifyToken(prefill);
    }
  }, [location.search]);

  const loadNetworkInfo = async () => {
    try {
      const response = await apiService.get(API_ENDPOINTS.blockchain.status);
      setNetworkInfo(response);
    } catch (error) {
      console.error('Failed to load network info:', error);
      setNetworkInfo({ error: 'Failed to connect to blockchain' });
    }
  };

  const handleVerifyToken = async (valueOverride) => {
    const value = typeof valueOverride === 'string' ? valueOverride : tokenId;
    if (!value.trim()) return;
    
    setLoading(true);
    try {
      const response = await apiService.get(`${API_ENDPOINTS.blockchain.verify}?token_id_or_hash=${encodeURIComponent(value)}`);
      const data = response.data || response; // Handle both axios response object and direct data
      
      setSearchResult({
        tokenId: data.token_id || value,
        isValid: true,
        projectName: data.project_name,
        carbonCredits: data.carbon_amount,
        verificationDate: new Date(data.verification_date * 1000).toLocaleDateString(),
        transactionHash: value.startsWith('0x') ? value : 'N/A',
        status: data.is_retired ? 'retired' : 'active',
        location: data.location,
        verificationHash: data.verification_hash
      });
    } catch (error) {
      console.error('Verification failed:', error);
      setSearchResult({
        tokenId: value,
        isValid: false,
        error: error.response?.data?.detail || 'Verification failed'
      });
    } finally {
      setLoading(false);
    }
  };

  const blockchainFeatures = [
    {
      title: 'Polygon Network',
      description: 'Energy-efficient Layer 2 blockchain for minimal environmental impact'
    },
    {
      title: 'NFT Certificates',
      description: 'ERC-721 tokens representing verified carbon credits'
    },
    {
      title: 'Immutable Records',
      description: 'Tamper-proof verification history and audit trails'
    }
  ];

  return (
    <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <BlockIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Typography variant="h4" gutterBottom>
          Blockchain Explorer
        </Typography>
      </Box>

      {blockchainEnabled === false ? (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body1">
            On-chain certification is not configured in this environment.
          </Typography>
        </Alert>
      ) : (
        <Alert severity={networkInfo?.connected ? "success" : "warning"} sx={{ mb: 3 }}>
          <Typography variant="body1">
            <strong>Blockchain Status:</strong> {
              networkInfo?.connected
                ? `Connected to ${networkInfo.network}. Real-time certificate verification is active.`
                : 'Blockchain service is currently unavailable.'
            }
          </Typography>
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Verify Carbon Credit Certificate
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <TextField
                label="Token ID or Transaction Hash"
                value={tokenId}
                onChange={(e) => setTokenId(e.target.value)}
                placeholder="Enter token ID to verify certificate"
                fullWidth
                disabled={blockchainEnabled === false}
              />
              <Button
                variant="contained"
                onClick={handleVerifyToken}
                startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <SearchIcon />}
                disabled={loading || !tokenId.trim() || blockchainEnabled === false}
                sx={{ minWidth: 120 }}
              >
                {loading ? 'Verifying...' : 'Verify'}
              </Button>
            </Box>

            {searchResult && (
              <Card sx={{ mt: 2 }}>
                <CardContent>
                  {searchResult.isValid ? (
                    <>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <VerifiedIcon color="success" sx={{ mr: 1 }} />
                        <Typography variant="h6">Certificate Verified</Typography>
                        <Chip 
                          label={searchResult.status} 
                          color={searchResult.status === 'retired' ? 'default' : 'success'}
                          size="small" 
                          sx={{ ml: 2, textTransform: 'capitalize' }}
                        />
                      </Box>
                      <List dense>
                        <ListItem>
                          <ListItemText 
                            primary="Project Name" 
                            secondary={searchResult.projectName} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="Carbon Credits" 
                            secondary={`${searchResult.carbonCredits} tCO₂e`} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="Location" 
                            secondary={searchResult.location} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="Verification Date" 
                            secondary={searchResult.verificationDate} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="Transaction Hash" 
                            secondary={searchResult.transactionHash} 
                          />
                        </ListItem>
                        {searchResult.verificationHash && (
                          <ListItem>
                            <ListItemText 
                              primary="Verification Hash" 
                              secondary={searchResult.verificationHash.substring(0, 16) + '...'} 
                            />
                          </ListItem>
                        )}
                      </List>
                    </>
                  ) : (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      <Typography variant="body1">
                        <strong>Verification Failed:</strong> {searchResult.error}
                      </Typography>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Blockchain Features
              </Typography>
              <List dense>
                {blockchainFeatures.map((feature, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={feature.title}
                      secondary={feature.description}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>

          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Network Information
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="Network" 
                    secondary={networkInfo ? networkInfo.network || 'Connecting...' : 'Loading...'} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Connection Status" 
                    secondary={networkInfo ? (networkInfo.connected ? 'Connected' : 'Disconnected') : 'Checking...'} 
                  />
                </ListItem>
                {networkInfo?.latest_block && (
                  <ListItem>
                    <ListItemText 
                      primary="Latest Block" 
                      secondary={networkInfo.latest_block.toLocaleString()} 
                    />
                  </ListItem>
                )}
                <ListItem>
                  <ListItemText primary="Block Time" secondary="~2 seconds" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Transaction Cost" secondary="~$0.01" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          How Blockchain Certification Works
        </Typography>
        <Typography variant="body1">
          When a carbon credit verification is completed, a unique NFT certificate is minted on the Polygon blockchain. 
          This creates an immutable record that prevents double-counting and ensures transparency in carbon credit trading.
        </Typography>
      </Paper>
    </Container>
  );
};

export default Blockchain; 