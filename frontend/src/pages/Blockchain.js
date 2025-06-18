import React, { useState } from 'react';
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
  Chip
} from '@mui/material';
import BlockIcon from '@mui/icons-material/ViewInAr';
import SearchIcon from '@mui/icons-material/Search';
import VerifiedIcon from '@mui/icons-material/Verified';
import { COMMON_STYLES } from '../theme/constants';

const Blockchain = () => {
  const [tokenId, setTokenId] = useState('');
  const [searchResult, setSearchResult] = useState(null);

  const handleVerifyToken = () => {
    // Mock verification for demonstration
    if (tokenId) {
      setSearchResult({
        tokenId: tokenId,
        isValid: true,
        projectName: 'Amazon Reforestation Project',
        carbonCredits: 1250,
        verificationDate: '2024-01-15',
        transactionHash: '0x1234...abcd',
        status: 'verified'
      });
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

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Blockchain Integration:</strong> Certificate verification is implemented in the backend. 
          The interactive blockchain explorer interface is under development.
        </Typography>
      </Alert>

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
              />
              <Button 
                variant="contained" 
                onClick={handleVerifyToken}
                startIcon={<SearchIcon />}
                sx={{ minWidth: 120 }}
              >
                Verify
              </Button>
            </Box>

            {searchResult && (
              <Card sx={{ mt: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <VerifiedIcon color="success" sx={{ mr: 1 }} />
                    <Typography variant="h6">Certificate Verified</Typography>
                    <Chip 
                      label={searchResult.status} 
                      color="success" 
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
                  </List>
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
                  <ListItemText primary="Network" secondary="Polygon Mainnet" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Block Time" secondary="~2 seconds" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Transaction Cost" secondary="~$0.01" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Energy Efficiency" secondary="99.95% less than Ethereum" />
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