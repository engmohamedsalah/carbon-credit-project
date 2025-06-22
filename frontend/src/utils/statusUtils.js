/**
 * Status utilities for consistent status handling across the app
 * Simplified to match database reality - three-status workflow
 */

import {
  CheckCircle as CheckCircleIcon,
  Pending as PendingIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon
} from '@mui/icons-material';

// Simplified status constants matching database reality
export const PROJECT_STATUS = {
  PENDING: 'Pending',
  VERIFIED: 'Verified', 
  REJECTED: 'Rejected'
};

// Status workflow descriptions
export const STATUS_DESCRIPTIONS = {
  [PROJECT_STATUS.PENDING]: 'Awaiting review and verification',
  [PROJECT_STATUS.VERIFIED]: 'Successfully verified and approved',
  [PROJECT_STATUS.REJECTED]: 'Verification failed or rejected'
};

// Status progression flow (for UI guidance)
export const STATUS_FLOW = [
  PROJECT_STATUS.PENDING,
  PROJECT_STATUS.VERIFIED // or REJECTED
];

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
      return <ScheduleIcon />;
    case PROJECT_STATUS.REJECTED:
      return <ErrorIcon />;
    default:
      return <PendingIcon />;
  }
};

/**
 * Get human-readable status text
 * @param {string} status - Project status
 * @returns {string} Formatted status text
 */
export const getStatusText = (status) => {
  if (!status) return PROJECT_STATUS.PENDING;
  return status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
};

/**
 * Check if status indicates completion (success or failure)
 * @param {string} status - Project status
 * @returns {boolean} True if status indicates completion
 */
export const isCompletedStatus = (status) => {
  return status === PROJECT_STATUS.VERIFIED || status === PROJECT_STATUS.REJECTED;
};

/**
 * Check if status indicates pending state
 * @param {string} status - Project status
 * @returns {boolean} True if status indicates pending
 */
export const isPendingStatus = (status) => {
  return status === PROJECT_STATUS.PENDING || !status;
};

/**
 * Get next possible statuses for progression
 * @param {string} currentStatus - Current project status
 * @returns {string[]} Array of possible next statuses
 */
export const getNextPossibleStatuses = (currentStatus) => {
  switch (currentStatus) {
    case PROJECT_STATUS.PENDING:
      return [PROJECT_STATUS.VERIFIED, PROJECT_STATUS.REJECTED];
    case PROJECT_STATUS.VERIFIED:
      return [PROJECT_STATUS.REJECTED]; // Can be reverted if issues found
    case PROJECT_STATUS.REJECTED:
      return [PROJECT_STATUS.PENDING]; // Can be resubmitted
    default:
      return [PROJECT_STATUS.PENDING];
  }
};

/**
 * Get status priority for sorting (lower number = higher priority)
 * @param {string} status - Project status
 * @returns {number} Priority number
 */
export const getStatusPriority = (status) => {
  switch (status) {
    case PROJECT_STATUS.PENDING:
      return 1; // Highest priority - needs action
    case PROJECT_STATUS.REJECTED:
      return 2; // Medium priority - needs attention
    case PROJECT_STATUS.VERIFIED:
      return 3; // Lowest priority - completed
    default:
      return 1;
  }
}; 