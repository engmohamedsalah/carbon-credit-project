## User Roles and Permissions System for Carbon Credit Verification SaaS

Implementing a Role-Based Access Control (RBAC) system is crucial for managing user access and ensuring data security within the Carbon Credit Verification SaaS application. This section outlines the design and implementation steps for such a system.

### 1. Defining Roles and Permissions

We will define the following roles:

1.  **Project Owner**: Users who create and manage carbon credit projects. They can upload data, initiate verification, and view results for their own projects.
2.  **Verifier**: Users (potentially internal experts or third-party auditors) responsible for reviewing verification results (human-in-the-loop). They can access assigned verification tasks and submit reviews.
3.  **Administrator**: Users with full access to the system. They can manage users, roles, system settings, oversee all projects and verifications, and potentially perform administrative actions on the blockchain integration.

**Permissions Matrix (Example)**:

| Action                      | Project Owner | Verifier | Administrator |
| :-------------------------- | :-----------: | :------: | :-----------: |
| **User Management**         |               |          |       ✅       |
| - Create/Edit/Delete User |               |          |       ✅       |
| - Assign Roles              |               |          |       ✅       |
| **Project Management**      |               |          |       ✅       |
| - Create Project            |       ✅       |          |       ✅       |
| - View Own Projects         |       ✅       |          |       ✅       |
| - View All Projects         |               |          |       ✅       |
| - Edit Own Project          |       ✅       |          |       ✅       |
| - Delete Own Project        |       ✅       |          |       ✅       |
| - Upload Data (Own Project) |       ✅       |          |       ✅       |
| **Verification**            |               |          |       ✅       |
| - Initiate Verification     |       ✅       |          |       ✅       |
| - View Own Verification     |       ✅       |          |       ✅       |
| - View Assigned Verification|               |    ✅     |       ✅       |
| - Review Verification       |               |    ✅     |       ✅       |
| - View All Verifications    |               |          |       ✅       |
| **Blockchain**              |               |          |       ✅       |
| - Issue Certificate         |               |          |       ✅       |
| - View Certificate Details  |       ✅       |    ✅     |       ✅       |
| **System Settings**         |               |          |       ✅       |

### 2. Database Schema Design

We need to modify the database schema to support roles and permissions.

```python
# backend/app/models/user.py (Updates)
from sqlalchemy import Column, Integer, String, Boolean, Table, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

# Association table for User and Role
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
)

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String)
    users = relationship("User", secondary=user_roles, back_populates="roles")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    projects = relationship("Project", back_populates="owner")

    @property
    def is_admin(self):
        return any(role.name == "Administrator" for role in self.roles)

# Add relationships in other models if needed (e.g., Project)
# backend/app/models/project.py
class Project(Base):
    # ... existing columns ...
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="projects")
    verifications = relationship("Verification", back_populates="project")
```

**Note**: A more granular permission system could involve a `Permissions` table and a `RolePermissions` association table, but for these defined roles, checking the role name might suffice initially.

### 3. Backend Implementation (FastAPI)

#### 3.1 Update Schemas

```python
# backend/app/schemas/user.py (Updates)
from pydantic import BaseModel, EmailStr
from typing import List, Optional

class RoleBase(BaseModel):
    name: str
    description: Optional[str] = None

class RoleCreate(RoleBase):
    pass

class Role(RoleBase):
    id: int
    class Config:
        orm_mode = True

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    roles: List[Role] = [] # Include roles in the User schema
    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None
    role_ids: Optional[List[int]] = None # For updating roles
```

#### 3.2 Authentication and Dependency Functions

Modify the authentication dependency to load user roles and create dependencies to check roles.

```python
# backend/app/api/deps.py (New or Updated)
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import SessionLocal
from app.models import user as user_model
from app.schemas import user as user_schema
from app.services import auth_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> user_model.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = auth_service.get_user_by_email_with_roles(db, email=email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: user_model.User = Depends(get_current_user)
) -> user_model.User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Role Checking Dependencies ---

def require_role(required_role: str):
    """Dependency factory to require a specific role."""
    async def role_checker(
        current_user: user_model.User = Depends(get_current_active_user)
    ):
        if not any(role.name == required_role for role in current_user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have the required '{required_role}' role"
            )
        return current_user
    return role_checker

def require_admin(current_user: user_model.User = Depends(require_role("Administrator"))):
    """Dependency to require Administrator role."""
    return current_user

def require_verifier(current_user: user_model.User = Depends(require_role("Verifier"))):
    """Dependency to require Verifier role."""
    return current_user

def require_project_owner(current_user: user_model.User = Depends(require_role("Project Owner"))):
    """Dependency to require Project Owner role."""
    return current_user
```

Update `auth_service` to include `get_user_by_email_with_roles` which eagerly loads roles.

```python
# backend/app/services/auth_service.py (Update)
from sqlalchemy.orm import Session, joinedload
from app.models import user as user_model

def get_user_by_email_with_roles(db: Session, email: str):
    return db.query(user_model.User).options(joinedload(user_model.User.roles)).filter(user_model.User.email == email).first()
```

#### 3.3 Protect API Endpoints

Apply the role-checking dependencies to API endpoints.

```python
# backend/app/api/projects.py (Example)
from app.api import deps

@router.post("/", response_model=project_schema.Project)
def create_project(
    *, 
    db: Session = Depends(deps.get_db),
    project_in: project_schema.ProjectCreate,
    # Require Project Owner or Administrator
    current_user: user_model.User = Depends(deps.get_current_active_user) 
):
    if not any(role.name in ["Project Owner", "Administrator"] for role in current_user.roles):
         raise HTTPException(status_code=403, detail="Not authorized to create projects")
    # ... creation logic, assign current_user.id as owner_id ...
    return project_service.create_project(db=db, project=project_in, owner_id=current_user.id)

@router.get("/", response_model=List[project_schema.Project])
def read_projects(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: user_model.User = Depends(deps.get_current_active_user)
):
    if current_user.is_admin:
        projects = project_service.get_projects(db, skip=skip, limit=limit)
    else: # Assume Project Owner
        projects = project_service.get_projects_by_owner(db, owner_id=current_user.id, skip=skip, limit=limit)
    return projects

@router.get("/{project_id}", response_model=project_schema.Project)
def read_project(
    project_id: int,
    db: Session = Depends(deps.get_db),
    current_user: user_model.User = Depends(deps.get_current_active_user)
):
    project = project_service.get_project(db, project_id=project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    # Check ownership or admin role
    if not current_user.is_admin and project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this project")
    return project

# Apply similar logic to PUT, DELETE, and other endpoints
# Use deps.require_admin for admin-only endpoints like user management
```

#### 3.4 User/Role Management API (Admin Only)

Create new API endpoints (e.g., `/api/admin/users`, `/api/admin/roles`) protected by `deps.require_admin` to manage users and assign roles.

### 4. Frontend Implementation (React)

#### 4.1 Update State Management (Redux)

-   Modify the authentication slice (`authSlice.js`) to store user roles upon login.
-   Create selectors to easily access the current user's roles.

```javascript
// src/store/authSlice.js (Updates)
import { createSlice } from '@reduxjs/toolkit';
import { api } from '../services/api';

const initialState = {
  user: null,
  token: localStorage.getItem('token') || null,
  roles: [], // Add roles
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    logout: (state) => {
      state.user = null;
      state.token = null;
      state.roles = [];
      localStorage.removeItem('token');
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      api.endpoints.login.matchFulfilled,
      (state, { payload }) => {
        state.token = payload.access_token;
        state.user = payload.user; // Assuming API returns user object with roles
        state.roles = payload.user.roles.map(role => role.name); // Store role names
        localStorage.setItem('token', payload.access_token);
      }
    );
    // Add matcher for fetching user profile if login doesn't return full user data
  },
});

export const { logout } = authSlice.actions;

// Selectors
export const selectCurrentUser = (state) => state.auth.user;
export const selectIsLoggedIn = (state) => !!state.auth.token;
export const selectUserRoles = (state) => state.auth.roles;
export const selectIsAdmin = (state) => state.auth.roles.includes('Administrator');
export const selectIsVerifier = (state) => state.auth.roles.includes('Verifier');
export const selectIsProjectOwner = (state) => state.auth.roles.includes('Project Owner');

export default authSlice.reducer;
```

#### 4.2 Conditional Rendering

Use role selectors to conditionally render UI elements.

```javascript
// src/components/Layout.js (Example)
import React from 'react';
import { useSelector } from 'react-redux';
import { Link } from 'react-router-dom';
import { selectIsLoggedIn, selectIsAdmin } from '../store/authSlice';

const Layout = ({ children }) => {
  const isLoggedIn = useSelector(selectIsLoggedIn);
  const isAdmin = useSelector(selectIsAdmin);

  return (
    <div>
      <nav>
        <Link to="/">Home</Link>
        {isLoggedIn && <Link to="/dashboard">Dashboard</Link>}
        {isAdmin && <Link to="/admin">Admin Panel</Link>} {/* Admin only link */}
        {/* Other links */}
      </nav>
      <main>{children}</main>
    </div>
  );
};
```

#### 4.3 Protected Routes

Create a higher-order component or use route properties to protect routes based on roles.

```javascript
// src/components/ProtectedRoute.js (Updated)
import React from 'react';
import { useSelector } from 'react-redux';
import { Navigate, useLocation } from 'react-router-dom';
import { selectIsLoggedIn, selectUserRoles } from '../store/authSlice';

const ProtectedRoute = ({ children, allowedRoles }) => {
  const isLoggedIn = useSelector(selectIsLoggedIn);
  const userRoles = useSelector(selectUserRoles);
  const location = useLocation();

  if (!isLoggedIn) {
    // Redirect to login if not logged in
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Check if user has at least one of the allowed roles
  const hasRequiredRole = allowedRoles 
    ? allowedRoles.some(role => userRoles.includes(role))
    : true; // If no roles specified, just require login

  if (!hasRequiredRole) {
    // Redirect to an unauthorized page or dashboard
    return <Navigate to="/unauthorized" replace />;
  }

  return children;
};

export default ProtectedRoute;

// Usage in App.js
import ProtectedRoute from './components/ProtectedRoute';
import AdminPage from './pages/AdminPage';

<Routes>
  {/* Public routes */}
  <Route path="/login" element={<Login />} />
  
  {/* Protected routes */}
  <Route 
    path="/dashboard" 
    element={
      <ProtectedRoute>
        <Dashboard />
      </ProtectedRoute>
    }
  />
  <Route 
    path="/admin" 
    element={
      <ProtectedRoute allowedRoles={["Administrator"]}> {/* Admin only route */}
        <AdminPage />
      </ProtectedRoute>
    }
  />
  {/* Other routes */}
</Routes>
```

### 5. Initial Setup

-   **Database Seeding**: Create initial roles (`Administrator`, `Verifier`, `Project Owner`) in the database when the application starts or via a migration script.
-   **First Admin User**: Implement a command-line script or initial setup process to create the first administrator user.

This RBAC system provides a solid foundation for managing user access. It can be further extended with more granular permissions if needed.
