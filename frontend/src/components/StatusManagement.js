import React, { useState } from 'react';
import {
  Box,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Typography,
  CircularProgress,
  Tooltip,
  Alert
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  Edit as EditIcon
} from '@mui/icons-material';
import { 
  PROJECT_STATUS, 
  getStatusColor, 
  getStatusIcon, 
  getNextPossibleStatuses,
  STATUS_DESCRIPTIONS 
} from '../utils/statusUtils';

const StatusManagement = ({ 
  currentStatus, 
  projectId, 
  onStatusUpdate, 
  canUpdate = false,
  loading = false,
  compact = false 
}) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedStatus, setSelectedStatus] = useState('');
  const [reason, setReason] = useState('');
  const [notes, setNotes] = useState('');
  const [updating, setUpdating] = useState(false);

  const menuOpen = Boolean(anchorEl);
  const nextStatuses = getNextPossibleStatuses(currentStatus);

  const handleMenuClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleStatusSelect = (status) => {
    setSelectedStatus(status);
    setReason('');
    setNotes('');
    setDialogOpen(true);
    handleMenuClose();
  };

  const handleDialogClose = () => {
    setDialogOpen(false);
    setSelectedStatus('');
    setReason('');
    setNotes('');
  };

  const handleStatusUpdate = async () => {
    if (selectedStatus === PROJECT_STATUS.REJECTED && !reason.trim()) {
      return; // Validation handled in UI
    }

    setUpdating(true);
    try {
      await onStatusUpdate({
        status: selectedStatus,
        reason: reason.trim(),
        notes: notes.trim()
      });
      handleDialogClose();
    } catch (error) {
      console.error('Status update failed:', error);
      // Error handling is managed by parent component
    } finally {
      setUpdating(false);
    }
  };

  const getStatusMessage = (status) => {
    switch (status) {
      case PROJECT_STATUS.VERIFIED:
        return 'Mark this project as successfully verified';
      case PROJECT_STATUS.REJECTED:
        return 'Reject this project due to issues';
      case PROJECT_STATUS.PENDING:
        return 'Move this project back to pending review';
      default:
        return 'Update project status';
    }
  };

  if (compact) {
    // Compact version for lists/tables
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Chip
          icon={getStatusIcon(currentStatus)}
          label={currentStatus || PROJECT_STATUS.PENDING}
          color={getStatusColor(currentStatus)}
          size="small"
          variant="outlined"
        />
        {canUpdate && (
          <Tooltip title="Change Status">
            <IconButton 
              size="small" 
              onClick={handleMenuClick}
              disabled={loading}
            >
              <EditIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </Box>
    );
  }

  // Full version for detail pages
  return (
    <>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Chip
          icon={getStatusIcon(currentStatus)}
          label={currentStatus || PROJECT_STATUS.PENDING}
          color={getStatusColor(currentStatus)}
          size="medium"
          variant="filled"
          sx={{ 
            fontWeight: 'bold',
            minWidth: 120,
            '& .MuiChip-icon': {
              fontSize: '1.1rem'
            }
          }}
        />
        
        {canUpdate && (
          <Tooltip title="Change Project Status">
            <IconButton 
              onClick={handleMenuClick}
              disabled={loading}
              sx={{ 
                bgcolor: 'action.hover',
                '&:hover': { bgcolor: 'action.selected' }
              }}
            >
              {loading ? <CircularProgress size={20} /> : <MoreVertIcon />}
            </IconButton>
          </Tooltip>
        )}
      </Box>

      {/* Status Change Menu */}
      <Menu
        anchorEl={anchorEl}
        open={menuOpen}
        onClose={handleMenuClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        {nextStatuses.map((status) => (
          <MenuItem 
            key={status} 
            onClick={() => handleStatusSelect(status)}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              minWidth: 200
            }}
          >
            {getStatusIcon(status)}
            <Box>
              <Typography variant="body2" fontWeight="medium">
                {status}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {STATUS_DESCRIPTIONS[status]}
              </Typography>
            </Box>
          </MenuItem>
        ))}
      </Menu>

      {/* Status Change Dialog */}
      <Dialog 
        open={dialogOpen} 
        onClose={handleDialogClose}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {getStatusIcon(selectedStatus)}
            Update Project Status
          </Box>
        </DialogTitle>
        
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <Alert severity="info" sx={{ mb: 2 }}>
              {getStatusMessage(selectedStatus)}
            </Alert>
            
            <Typography variant="body1" gutterBottom>
              Change status from <strong>{currentStatus}</strong> to <strong>{selectedStatus}</strong>
            </Typography>
            
            {selectedStatus === PROJECT_STATUS.REJECTED && (
              <TextField
                fullWidth
                required
                label="Reason for Rejection"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                multiline
                rows={3}
                sx={{ mt: 2 }}
                helperText="Please provide a clear reason for rejecting this project"
                error={selectedStatus === PROJECT_STATUS.REJECTED && !reason.trim()}
              />
            )}
            
            <TextField
              fullWidth
              label={
                selectedStatus === PROJECT_STATUS.VERIFIED ? 'Verification Notes' :
                selectedStatus === PROJECT_STATUS.REJECTED ? 'Additional Notes' :
                'Notes'
              }
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              multiline
              rows={2}
              sx={{ mt: 2 }}
              helperText={`Optional additional information about this ${selectedStatus.toLowerCase()} decision`}
            />
          </Box>
        </DialogContent>
        
        <DialogActions>
          <Button 
            onClick={handleDialogClose}
            disabled={updating}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleStatusUpdate}
            variant="contained"
            disabled={updating || (selectedStatus === PROJECT_STATUS.REJECTED && !reason.trim())}
            color={selectedStatus === PROJECT_STATUS.VERIFIED ? 'success' : 
                   selectedStatus === PROJECT_STATUS.REJECTED ? 'error' : 'primary'}
          >
            {updating ? <CircularProgress size={20} /> : `Mark as ${selectedStatus}`}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default StatusManagement; 