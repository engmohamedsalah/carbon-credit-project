import { 
  ROLES, 
  ROLE_PERMISSIONS, 
  hasRole, 
  hasAnyRole, 
  isAdmin,
  canAccessFeature,
  getUserRoleDisplayName,
  getRoleDescription
} from './roleUtils';

describe('Role Utils', () => {
  test('ROLES object is defined', () => {
    expect(ROLES).toBeDefined();
    expect(typeof ROLES).toBe('object');
  });

  test('ROLE_PERMISSIONS object is defined', () => {
    expect(ROLE_PERMISSIONS).toBeDefined();
    expect(typeof ROLE_PERMISSIONS).toBe('object');
  });

  test('hasRole function exists', () => {
    expect(typeof hasRole).toBe('function');
  });

  test('hasAnyRole function exists', () => {
    expect(typeof hasAnyRole).toBe('function');
  });

  test('isAdmin function exists', () => {
    expect(typeof isAdmin).toBe('function');
  });

  test('canAccessFeature function exists', () => {
    expect(typeof canAccessFeature).toBe('function');
  });

  test('getUserRoleDisplayName function exists', () => {
    expect(typeof getUserRoleDisplayName).toBe('function');
  });

  test('getRoleDescription function exists', () => {
    expect(typeof getRoleDescription).toBe('function');
  });

  test('basic role checks work', () => {
    // These should be functions that return boolean values
    expect(typeof isAdmin('admin')).toBe('boolean');
    expect(typeof hasRole('Project Developer', [ROLES.PROJECT_DEVELOPER])).toBe('boolean');
    expect(typeof hasRole('Verifier', [ROLES.VERIFIER])).toBe('boolean');
  });
}); 