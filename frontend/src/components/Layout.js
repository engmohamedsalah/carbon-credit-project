import React, { useEffect, useMemo } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Box, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Divider, 
  IconButton,
  Avatar,
  Menu,
  MenuItem,
  Chip
} from '@mui/material';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { logout, getCurrentUser } from '../store/authSlice';
import {
  getMenuItemsForRole,
  getUserRoleDisplayName,
  isAdmin,
  ROLES
} from '../utils/roleUtils';

// Icons
import DashboardIcon from '@mui/icons-material/Dashboard';
import ForestIcon from '@mui/icons-material/Forest';
import VerifiedIcon from '@mui/icons-material/Verified';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import LogoutIcon from '@mui/icons-material/Logout';
import BlockIcon from '@mui/icons-material/ViewInAr';
import SensorsIcon from '@mui/icons-material/Sensors';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import AssessmentIcon from '@mui/icons-material/Assessment';
import SettingsIcon from '@mui/icons-material/Settings';
import PsychologyIcon from '@mui/icons-material/Psychology';
import AdminPanelSettingsIcon from '@mui/icons-material/AdminPanelSettings';

const drawerWidth = 240;

// Icon mapping for menu items
const iconMap = {
  'Dashboard': <DashboardIcon />,
  'Projects': <ForestIcon />,
  'AI Verification': <VerifiedIcon />,
  'Explainable AI': <PsychologyIcon />,
  'IoT Sensors': <SensorsIcon />,
  'Analytics': <AnalyticsIcon />,
  'Blockchain': <BlockIcon />,
  'Reports': <AssessmentIcon />,
  'Settings': <SettingsIcon />
};

const Layout = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const location = useLocation();
  
  const user = useSelector((state) => state.auth.user);
  const isAuthenticated = useSelector((state) => state.auth.isAuthenticated);

  // Get user role with fallback
  const userRole = user?.role || ROLES.PROJECT_DEVELOPER;
  
  // Get menu items based on user role using our professional role system
  const menuItems = useMemo(() => {
    return getMenuItemsForRole(userRole);
  }, [userRole]);

  // Show user-friendly role display name
  const roleDisplayName = getUserRoleDisplayName(userRole);
  const showAdminBadge = isAdmin(userRole);

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login');
    }
  }, [isAuthenticated, navigate]);
  
  // Fetch user data when authenticated but user data is not loaded
  useEffect(() => {
    if (isAuthenticated && !user) {
      dispatch(getCurrentUser());
    }
  }, [dispatch, isAuthenticated, user]);
  
  // Debug: log user role and filtered menu items
  useEffect(() => {
    console.log('🔍 Debug - User Role:', userRole);
    console.log('👤 User Object:', user);
    console.log('📋 Available Menu Items:', menuItems.length);
    console.log('📝 Items:', menuItems.map(item => item.text));
  }, [userRole, menuItems, user]);
  
  const [anchorEl, setAnchorEl] = React.useState(null);
  const open = Boolean(anchorEl);
  
  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleLogout = () => {
    dispatch(logout());
    navigate('/login');
  };
  
  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Carbon Credit Verification
          </Typography>
          
          {/* User role indicator */}
          {showAdminBadge && (
            <Chip 
              icon={<AdminPanelSettingsIcon />}
              label="Admin"
              color="error"
              size="small"
              sx={{ mr: 2 }}
            />
          )}
          
          <Box sx={{ mr: 2, textAlign: 'right' }}>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              {user?.full_name || 'User'}
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.7 }}>
              {roleDisplayName}
            </Typography>
          </Box>
          
          <IconButton
            onClick={handleProfileMenuOpen}
            size="large"
            edge="end"
            color="inherit"
          >
            <Avatar sx={{ bgcolor: 'secondary.main' }}>
              {user?.full_name?.charAt(0) || 'U'}
            </Avatar>
          </IconButton>
          
          <Menu
            anchorEl={anchorEl}
            open={open}
            onClose={handleMenuClose}
            MenuListProps={{
              'aria-labelledby': 'basic-button',
            }}
          >
            <MenuItem onClick={() => navigate('/profile')}>
              <ListItemIcon>
                <AccountCircleIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Profile</ListItemText>
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <ListItemIcon>
                <LogoutIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Logout</ListItemText>
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>
      
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {menuItems.map((item) => (
              <ListItem 
                button 
                key={item.text} 
                onClick={() => navigate(item.path)}
                selected={location.pathname === item.path}
                sx={{
                  '&.Mui-selected': {
                    bgcolor: 'primary.light',
                    '&:hover': {
                      bgcolor: 'primary.light',
                    },
                  },
                }}
              >
                <ListItemIcon>
                  {iconMap[item.text] || <DashboardIcon />}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text}
                  secondary={item.description}
                  secondaryTypographyProps={{
                    variant: 'caption',
                    sx: { fontSize: '0.65rem' }
                  }}
                />
              </ListItem>
            ))}
          </List>
          <Divider />
          
          {/* Role information at bottom of sidebar */}
          <Box sx={{ p: 2, mt: 'auto' }}>
            <Typography variant="caption" color="text.secondary">
              Role: {roleDisplayName}
            </Typography>
            <Typography variant="caption" display="block" color="text.secondary">
              {menuItems.length} features available
            </Typography>
          </Box>
        </Box>
      </Drawer>
      
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  );
};

export default Layout;
