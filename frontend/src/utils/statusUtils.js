/**
 * Status utilities for consistent status handling across the app
 * Eliminates DRY violation from multiple getStatusColor implementations
 */

import {
  CheckCircle as CheckCircleIcon,
  Pending as PendingIcon,
  Autorenew as AutorenewIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';

export const PROJECT_STATUS = {
  DRAFT: 'Draft',
  PENDING: 'Pending',
  VERIFIED: 'Verified',
  REJECTED: 'Rejected'
};

/**
 * Get Material-UI color for project status
 * @param {string} status - Project status
 * @returns {string} Material-UI color variant
 */
export const getStatusColor = (status) => {
  switch (status) {
    case PROJECT_STATUS.VERIFIED:
      return 'success';
    case PROJECT_STATUS.PENDING:
      return 'warning';
    case PROJECT_STATUS.REJECTED:
      return 'error';
    case PROJECT_STATUS.DRAFT:
      return 'default';
    default:
      return 'default';
  }
};

/**
 * Get appropriate icon for project status
 * @param {string} status - Project status
 * @returns {JSX.Element} Material-UI icon component
 */
export const getStatusIcon = (status) => {
  switch (status) {
    case PROJECT_STATUS.VERIFIED:
      return <CheckCircleIcon />;
    case PROJECT_STATUS.PENDING:
      return <PendingIcon />;
    case PROJECT_STATUS.REJECTED:
      return <ErrorIcon />;
    case PROJECT_STATUS.DRAFT:
      return <InfoIcon />;
    default:
      return <InfoIcon />;
  }
};

/**
 * Get human-readable status text
 * @param {string} status - Project status
 * @returns {string} Formatted status text
 */
export const getStatusText = (status) => {
  if (!status) return 'Unknown';
  return status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
};

/**
 * Check if status indicates completion
 * @param {string} status - Project status
 * @returns {boolean} True if status indicates completion
 */
export const isCompletedStatus = (status) => {
  return status?.toLowerCase() === PROJECT_STATUS.VERIFIED;
};

/**
 * Check if status indicates pending state
 * @param {string} status - Project status
 * @returns {boolean} True if status indicates pending
 */
export const isPendingStatus = (status) => {
  const pendingStatuses = [PROJECT_STATUS.PENDING, PROJECT_STATUS.DRAFT];
  return pendingStatuses.includes(status);
}; 