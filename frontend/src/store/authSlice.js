import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import apiService from '../services/apiService';
import { formatApiError } from '../utils/errorUtils';

// Async thunks
export const login = createAsyncThunk(
  'auth/login',
  async ({ email, password }, { rejectWithValue }) => {
    try {
      // Use URLSearchParams to send form data instead of JSON
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);
      
      const response = await apiService.auth.login(formData);
      localStorage.setItem('token', response.data.access_token);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Login failed');
    }
  }
);

export const register = createAsyncThunk(
  'auth/register',
  async ({ email, password, fullName, role }, { rejectWithValue }) => {
    try {
      const response = await apiService.auth.register({ 
        email, 
        password, 
        full_name: fullName,
        role 
      });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Registration failed');
    }
  }
);

export const getCurrentUser = createAsyncThunk(
  'auth/getCurrentUser',
  async (_, { rejectWithValue }) => {
    try {
      const response = await apiService.auth.getCurrentUser();
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Failed to get user data');
    }
  }
);

// Initial state
const initialState = {
  user: null,
  isAuthenticated: false,
  loading: false,
  error: null,
};

// Slice
const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    logout: (state) => {
      localStorage.removeItem('token');
      state.user = null;
      state.isAuthenticated = false;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Login
      .addCase(login.pending, (state) => {
        state.loading = true;
        state.error = null; // Clear previous errors when starting new request
      })
      .addCase(login.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.error = null;
      })
      .addCase(login.rejected, (state, action) => {
        state.loading = false;
        state.error = formatApiError(action, 'Login failed');
      })
      // Register
      .addCase(register.pending, (state) => {
        state.loading = true;
        state.error = null; // Clear previous errors when starting new request
      })
      .addCase(register.fulfilled, (state) => {
        state.loading = false;
        state.error = null;
      })
      .addCase(register.rejected, (state, action) => {
        state.loading = false;
        state.error = formatApiError(action, 'Registration failed');
      })
      // Get current user
      .addCase(getCurrentUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(getCurrentUser.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
        state.isAuthenticated = true;
      })
      .addCase(getCurrentUser.rejected, (state, action) => {
        state.loading = false;
        state.user = null;
        state.isAuthenticated = false;
        state.error = formatApiError(action, 'Failed to get user data');
      });
  },
});

export const { logout, clearError } = authSlice.actions;

export default authSlice.reducer;
