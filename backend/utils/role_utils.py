"""
Role-based access control utilities for the Carbon Credit Verification System
Centralized role management to maintain DRY principles and consistency
"""

from typing import List, Optional
from enum import Enum

class Roles(str, Enum):
    """Enumeration of all available roles in the system"""
    # Admin roles (legacy support)
    ADMINISTRATOR = "Administrator"  # Standard admin role
    ADMIN = "admin"  # Legacy lowercase admin role
    
    # User roles
    PROJECT_DEVELOPER = "Project Developer"
    VERIFIER = "Verifier"
    SCIENTIST = "Scientist"
    RESEARCHER = "Researcher"
    INVESTOR = "Investor"
    BROKER = "Broker"
    REGULATOR = "Regulator"
    MONITOR = "Monitor"
    AUDITOR = "Auditor"

class RolePermissions:
    """Define role hierarchies and permissions"""
    # Administrative roles - highest permissions
    ADMIN_ROLES = [Roles.ADMINISTRATOR, Roles.ADMIN]
    
    # Verification and quality assurance
    VERIFICATION_ROLES = [Roles.VERIFIER, Roles.AUDITOR]
    
    # Scientific and research roles
    SCIENTIFIC_ROLES = [Roles.SCIENTIST, Roles.RESEARCHER]
    
    # Business and finance roles
    BUSINESS_ROLES = [Roles.INVESTOR, Roles.BROKER]
    
    # Regulatory and compliance roles
    REGULATORY_ROLES = [Roles.REGULATOR, Roles.MONITOR]
    
    # Development roles
    DEVELOPMENT_ROLES = [Roles.PROJECT_DEVELOPER]
    
    # All roles that can access basic features
    ALL_AUTHENTICATED_ROLES = list(Roles)

def has_role(user_role: Optional[str], allowed_roles: List[str]) -> bool:
    """
    Check if user has one of the allowed roles
    
    Args:
        user_role: The user's role
        allowed_roles: List of allowed roles
        
    Returns:
        bool: True if user has permission, False otherwise
    """
    if not user_role or not allowed_roles:
        return False
    return user_role in allowed_roles

def is_admin(user_role: Optional[str]) -> bool:
    """
    Check if user has admin privileges
    
    Args:
        user_role: The user's role
        
    Returns:
        bool: True if user is admin, False otherwise
    """
    return has_role(user_role, RolePermissions.ADMIN_ROLES)

def can_manage_projects(user_role: Optional[str]) -> bool:
    """
    Check if user can manage projects (admin or project developer)
    
    Args:
        user_role: The user's role
        
    Returns:
        bool: True if user can manage projects, False otherwise
    """
    allowed_roles = RolePermissions.ADMIN_ROLES + RolePermissions.DEVELOPMENT_ROLES
    return has_role(user_role, allowed_roles)

def can_verify_projects(user_role: Optional[str]) -> bool:
    """
    Check if user can verify projects
    
    Args:
        user_role: The user's role
        
    Returns:
        bool: True if user can verify projects, False otherwise
    """
    allowed_roles = RolePermissions.ADMIN_ROLES + RolePermissions.VERIFICATION_ROLES
    return has_role(user_role, allowed_roles)

def can_access_analytics(user_role: Optional[str]) -> bool:
    """
    Check if user can access analytics features
    
    Args:
        user_role: The user's role
        
    Returns:
        bool: True if user can access analytics, False otherwise
    """
    allowed_roles = (RolePermissions.ADMIN_ROLES + 
                    RolePermissions.VERIFICATION_ROLES + 
                    RolePermissions.SCIENTIFIC_ROLES)
    return has_role(user_role, allowed_roles)

def can_manage_users(user_role: Optional[str]) -> bool:
    """
    Check if user can manage other users
    
    Args:
        user_role: The user's role
        
    Returns:
        bool: True if user can manage users, False otherwise
    """
    return has_role(user_role, RolePermissions.ADMIN_ROLES)

def can_access_blockchain(user_role: Optional[str]) -> bool:
    """
    Check if user can access blockchain features
    
    Args:
        user_role: The user's role
        
    Returns:
        bool: True if user can access blockchain, False otherwise
    """
    allowed_roles = (RolePermissions.ADMIN_ROLES + 
                    RolePermissions.VERIFICATION_ROLES)
    return has_role(user_role, allowed_roles)

def normalize_role(role: Optional[str]) -> Optional[str]:
    """
    Normalize role string to handle case variations
    
    Args:
        role: The role string to normalize
        
    Returns:
        str: Normalized role string or None if invalid
    """
    if not role:
        return None
    
    # Handle common variations
    role_mappings = {
        'admin': Roles.ADMIN,
        'administrator': Roles.ADMINISTRATOR,
        'project developer': Roles.PROJECT_DEVELOPER,
        'verifier': Roles.VERIFIER,
        'scientist': Roles.SCIENTIST,
        'researcher': Roles.RESEARCHER,
        'investor': Roles.INVESTOR,
        'broker': Roles.BROKER,
        'regulator': Roles.REGULATOR,
        'monitor': Roles.MONITOR,
        'auditor': Roles.AUDITOR,
    }
    
    # First try exact match
    if role in [r.value for r in Roles]:
        return role
    
    # Try case-insensitive mapping
    normalized = role_mappings.get(role.lower())
    return normalized if normalized else role

# Decorator for admin-only endpoints
def require_admin(func):
    """Decorator to require admin role for endpoint access"""
    def wrapper(*args, **kwargs):
        # This would be used with dependency injection in FastAPI
        # Implementation depends on how current_user is passed
        return func(*args, **kwargs)
    return wrapper