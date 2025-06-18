import React, { useEffect } from 'react';
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
  MenuItem
} from '@mui/material';
import { Outlet, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { logout, getCurrentUser } from '../store/authSlice';

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

const drawerWidth = 240;

const Layout = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { user, isAuthenticated } = useSelector(state => state.auth);
  
  // Fetch user data when authenticated but user data is not loaded
  useEffect(() => {
    if (isAuthenticated && !user) {
      dispatch(getCurrentUser());
    }
  }, [dispatch, isAuthenticated, user]);
  
  // Debug: log user role and filtered menu items
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const userRole = user?.role || 'viewer';
    const visibleItems = menuItems.filter(item => item.roles.includes(userRole));
    console.log('🔍 Debug - User Role:', userRole);
    console.log('👤 User Object:', user);
    console.log('📋 Visible Menu Items:', visibleItems.length, 'of', menuItems.length);
    console.log('📝 Items:', visibleItems.map(item => item.text));
  }, [user]);
  
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
  
  const menuItems = [
    {
      text: 'Dashboard',
      icon: <DashboardIcon />,
      path: '/dashboard',
      roles: ['viewer', 'project_developer', 'verifier', 'admin'],
      description: 'Overview and quick actions'
    },
    {
      text: 'Projects',
      icon: <ForestIcon />,
      path: '/projects',
      roles: ['viewer', 'project_developer', 'verifier', 'admin'],
      description: 'Manage carbon credit projects'
    },
    {
      text: 'AI Verification',
      icon: <VerifiedIcon />,
      path: '/verification',
      roles: ['project_developer', 'verifier', 'admin'],
      description: 'ML-powered satellite analysis'
    },
    {
      text: 'Explainable AI',
      icon: <PsychologyIcon />,
      path: '/xai',
      roles: ['verifier', 'admin'],
      description: 'Model explanations and transparency'
    },
    {
      text: 'IoT Sensors',
      icon: <SensorsIcon />,
      path: '/iot',
      roles: ['project_developer', 'verifier', 'admin'],
      description: 'Ground-based sensor data'
    },
    {
      text: 'Analytics',
      icon: <AnalyticsIcon />,
      path: '/analytics',
      roles: ['viewer', 'project_developer', 'verifier', 'admin'],
      description: 'Performance insights and trends'
    },
    {
      text: 'Blockchain',
      icon: <BlockIcon />,
      path: '/blockchain',
      roles: ['viewer', 'project_developer', 'verifier', 'admin'],
      description: 'Certificate verification and explorer'
    },
    {
      text: 'Reports',
      icon: <AssessmentIcon />,
      path: '/reports',
      roles: ['verifier', 'admin'],
      description: 'Verification certificates and audits'
    },
    {
      text: 'Settings',
      icon: <SettingsIcon />,
      path: '/settings',
      roles: ['project_developer', 'admin'],
      description: 'Account and system preferences'
    }
  ];
  
  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Carbon Credit Verification
          </Typography>
          
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
              item.roles.includes(user?.role || 'viewer') && (
                <ListItem 
                  button 
                  key={item.text} 
                  onClick={() => navigate(item.path)}
                >
                  <ListItemIcon>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItem>
              )
            ))}
          </List>
          <Divider />
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
