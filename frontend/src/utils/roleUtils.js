// Role-based access control utilities
// Centralized role management for the Carbon Credit Verification System

// Define all available roles as constants
export const ROLES = {
  ADMIN: 'Admin',
  LEGACY_ADMIN: 'admin', // Legacy lowercase admin role
  PROJECT_DEVELOPER: 'Project Developer',
  VERIFIER: 'Verifier',
  SCIENTIST: 'Scientist',
  RESEARCHER: 'Researcher',
  INVESTOR: 'Investor',
  BROKER: 'Broker',
  REGULATOR: 'Regulator',
  MONITOR: 'Monitor',
  AUDITOR: 'Auditor'
};

// Define role hierarchies and permissions
export const ROLE_PERMISSIONS = {
  // Administrative roles - highest permissions
  ADMIN_ROLES: [ROLES.ADMIN, ROLES.LEGACY_ADMIN],
  
  // Verification and quality assurance
  VERIFICATION_ROLES: [ROLES.VERIFIER, ROLES.AUDITOR],
  
  // Scientific and research roles
  SCIENTIFIC_ROLES: [ROLES.SCIENTIST, ROLES.RESEARCHER],
  
  // Business and finance roles
  BUSINESS_ROLES: [ROLES.INVESTOR, ROLES.BROKER],
  
  // Regulatory and compliance roles
  REGULATORY_ROLES: [ROLES.REGULATOR, ROLES.MONITOR],
  
  // Development roles
  DEVELOPMENT_ROLES: [ROLES.PROJECT_DEVELOPER],
  
  // All roles that can access basic features
  ALL_AUTHENTICATED_ROLES: Object.values(ROLES)
};

// Define feature access groups
export const FEATURE_ACCESS = {
  // Core features accessible to most users
  CORE_FEATURES: [
    ...ROLE_PERMISSIONS.ADMIN_ROLES,
    ...ROLE_PERMISSIONS.VERIFICATION_ROLES,
    ...ROLE_PERMISSIONS.SCIENTIFIC_ROLES,
    ...ROLE_PERMISSIONS.BUSINESS_ROLES,
    ...ROLE_PERMISSIONS.REGULATORY_ROLES,
    ...ROLE_PERMISSIONS.DEVELOPMENT_ROLES
  ],
  
  // ML and AI features
  ML_FEATURES: [
    ...ROLE_PERMISSIONS.ADMIN_ROLES,
    ...ROLE_PERMISSIONS.VERIFICATION_ROLES,
    ...ROLE_PERMISSIONS.SCIENTIFIC_ROLES,
    ...ROLE_PERMISSIONS.DEVELOPMENT_ROLES
  ],
  
  // XAI (Explainable AI) features - requires higher permissions
  XAI_FEATURES: [
    ...ROLE_PERMISSIONS.ADMIN_ROLES,
    ...ROLE_PERMISSIONS.VERIFICATION_ROLES,
    ...ROLE_PERMISSIONS.SCIENTIFIC_ROLES,
    ...ROLE_PERMISSIONS.REGULATORY_ROLES
  ],
  
  // Verification workflow
  VERIFICATION_WORKFLOW: [
    ...ROLE_PERMISSIONS.ADMIN_ROLES,
    ...ROLE_PERMISSIONS.VERIFICATION_ROLES,
    ...ROLE_PERMISSIONS.SCIENTIFIC_ROLES,
    ...ROLE_PERMISSIONS.DEVELOPMENT_ROLES
  ],
  
  // IoT and monitoring
  IOT_MONITORING: [
    ...ROLE_PERMISSIONS.ADMIN_ROLES,
    ...ROLE_PERMISSIONS.VERIFICATION_ROLES,
    ...ROLE_PERMISSIONS.SCIENTIFIC_ROLES,
    ...ROLE_PERMISSIONS.REGULATORY_ROLES,
    ...ROLE_PERMISSIONS.DEVELOPMENT_ROLES
  ],
  
  // Blockchain and certificates
  BLOCKCHAIN_FEATURES: [
    ...ROLE_PERMISSIONS.ADMIN_ROLES,
    ...ROLE_PERMISSIONS.VERIFICATION_ROLES,
    ...ROLE_PERMISSIONS.BUSINESS_ROLES,
    ...ROLE_PERMISSIONS.REGULATORY_ROLES,
    ...ROLE_PERMISSIONS.DEVELOPMENT_ROLES
  ],
  
  // Reporting and analytics
  REPORTING: [
    ...ROLE_PERMISSIONS.ADMIN_ROLES,
    ...ROLE_PERMISSIONS.VERIFICATION_ROLES,
    ...ROLE_PERMISSIONS.REGULATORY_ROLES
  ],
  
  // System administration
  ADMIN_FEATURES: [
    ...ROLE_PERMISSIONS.ADMIN_ROLES
  ]
};

// Helper functions for role checking
export const hasRole = (userRole, allowedRoles) => {
  if (!userRole || !allowedRoles) return false;
  return allowedRoles.includes(userRole);
};

export const hasAnyRole = (userRole, roleGroups) => {
  if (!userRole || !roleGroups) return false;
  return roleGroups.some(group => group.includes(userRole));
};

export const isAdmin = (userRole) => {
  return hasRole(userRole, ROLE_PERMISSIONS.ADMIN_ROLES);
};

export const canAccessFeature = (userRole, featureKey) => {
  const allowedRoles = FEATURE_ACCESS[featureKey];
  return hasRole(userRole, allowedRoles);
};

export const getUserRoleDisplayName = (role) => {
  // Map internal role names to user-friendly display names
  const displayNames = {
    [ROLES.ADMIN]: 'Administrator',
    [ROLES.LEGACY_ADMIN]: 'Administrator',
    [ROLES.PROJECT_DEVELOPER]: 'Project Developer',
    [ROLES.VERIFIER]: 'Carbon Verifier',
    [ROLES.SCIENTIST]: 'Environmental Scientist',
    [ROLES.RESEARCHER]: 'Climate Researcher',
    [ROLES.INVESTOR]: 'Carbon Credit Investor',
    [ROLES.BROKER]: 'Carbon Credit Broker',
    [ROLES.REGULATOR]: 'Environmental Regulator',
    [ROLES.MONITOR]: 'Environmental Monitor',
    [ROLES.AUDITOR]: 'Third-Party Auditor'
  };
  
  return displayNames[role] || role;
};

export const getRoleDescription = (role) => {
  const descriptions = {
    [ROLES.ADMIN]: 'Full system administration and oversight',
    [ROLES.LEGACY_ADMIN]: 'Full system administration and oversight',
    [ROLES.PROJECT_DEVELOPER]: 'Create and manage carbon credit projects',
    [ROLES.VERIFIER]: 'Verify and approve carbon credit claims',
    [ROLES.SCIENTIST]: 'Environmental analysis and research',
    [ROLES.RESEARCHER]: 'Climate research and data analysis',
    [ROLES.INVESTOR]: 'Investment analysis and portfolio management',
    [ROLES.BROKER]: 'Carbon credit trading and marketplace',
    [ROLES.REGULATOR]: 'Regulatory compliance and oversight',
    [ROLES.MONITOR]: 'Environmental monitoring and tracking',
    [ROLES.AUDITOR]: 'Independent verification and auditing'
  };
  
  return descriptions[role] || 'System user';
};

// Menu item configuration using feature access
export const getMenuItemsForRole = (userRole) => {
  const menuConfig = [
    {
      text: 'Dashboard',
      path: '/dashboard',
      feature: 'CORE_FEATURES',
      description: 'Overview and quick actions'
    },
    {
      text: 'Projects',
      path: '/projects',
      feature: 'CORE_FEATURES',
      description: 'Manage carbon credit projects'
    },
    {
      text: 'AI Verification',
      path: '/verification',
      feature: 'VERIFICATION_WORKFLOW',
      description: 'ML-powered satellite analysis'
    },
    {
      text: 'Explainable AI',
      path: '/xai',
      feature: 'XAI_FEATURES',
      description: 'Model explanations and transparency'
    },
    {
      text: 'IoT Sensors',
      path: '/iot',
      feature: 'IOT_MONITORING',
      description: 'Ground-based sensor data'
    },
    {
      text: 'Analytics',
      path: '/analytics',
      feature: 'CORE_FEATURES',
      description: 'Performance insights and trends'
    },
    {
      text: 'Blockchain',
      path: '/blockchain',
      feature: 'BLOCKCHAIN_FEATURES',
      description: 'Certificate verification and explorer'
    },
    {
      text: 'Reports',
      path: '/reports',
      feature: 'REPORTING',
      description: 'Verification certificates and audits'
    },
    {
      text: 'Settings',
      path: '/settings',
      feature: 'ADMIN_FEATURES',
      description: 'Account and system preferences'
    }
  ];
  
  // Filter menu items based on user role and feature access
  return menuConfig.filter(item => 
    canAccessFeature(userRole, item.feature)
  );
}; 