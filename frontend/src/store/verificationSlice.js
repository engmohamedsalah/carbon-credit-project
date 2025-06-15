import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import apiService from '../services/apiService';

// Async thunks
export const fetchVerifications = createAsyncThunk(
  'verifications/fetchVerifications',
  async ({ projectId, status }, { rejectWithValue }) => {
    try {
      const params = new URLSearchParams();
      
      if (projectId) params.append('project_id', projectId);
      if (status) params.append('status', status);
      
      const response = await apiService.verification.list(Object.fromEntries(params));
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);

export const fetchVerificationById = createAsyncThunk(
  'verifications/fetchVerificationById',
  async (id, { rejectWithValue }) => {
    try {
      const response = await apiService.verification.getById(id);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);

export const createVerification = createAsyncThunk(
  'verifications/createVerification',
  async (verificationData, { rejectWithValue }) => {
    try {
      const response = await apiService.verification.create(verificationData);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);

export const submitHumanReview = createAsyncThunk(
  'verifications/submitHumanReview',
  async ({ id, approved, notes }, { rejectWithValue }) => {
    try {
      const response = await apiService.verification.submitHumanReview(id, {
        approved,
        notes
      });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);

export const certifyVerification = createAsyncThunk(
  'verifications/certifyVerification',
  async (verificationId, { rejectWithValue }) => {
    try {
      const response = await apiService.verification.certify(verificationId);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);

// Initial state
const initialState = {
  verifications: [],
  currentVerification: null,
  loading: false,
  error: null,
  certificationResult: null,
};

// Slice
const verificationSlice = createSlice({
  name: 'verifications',
  initialState,
  reducers: {
    clearCurrentVerification: (state) => {
      state.currentVerification = null;
    },
    clearError: (state) => {
      state.error = null;
    },
    clearCertificationResult: (state) => {
      state.certificationResult = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch verifications
      .addCase(fetchVerifications.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchVerifications.fulfilled, (state, action) => {
        state.loading = false;
        state.verifications = action.payload;
      })
      .addCase(fetchVerifications.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to fetch verifications';
      })
      // Fetch verification by ID
      .addCase(fetchVerificationById.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchVerificationById.fulfilled, (state, action) => {
        state.loading = false;
        state.currentVerification = action.payload;
      })
      .addCase(fetchVerificationById.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to fetch verification';
      })
      // Create verification
      .addCase(createVerification.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createVerification.fulfilled, (state, action) => {
        state.loading = false;
        state.verifications.push(action.payload);
      })
      .addCase(createVerification.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to create verification';
      })
      // Submit human review
      .addCase(submitHumanReview.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(submitHumanReview.fulfilled, (state, action) => {
        state.loading = false;
        const index = state.verifications.findIndex(v => v.id === action.payload.id);
        if (index !== -1) {
          state.verifications[index] = action.payload;
        }
        if (state.currentVerification?.id === action.payload.id) {
          state.currentVerification = action.payload;
        }
      })
      .addCase(submitHumanReview.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to submit human review';
      })
      // Certify verification
      .addCase(certifyVerification.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(certifyVerification.fulfilled, (state, action) => {
        state.loading = false;
        state.certificationResult = action.payload;
      })
      .addCase(certifyVerification.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to certify verification';
      });
  },
});

export const { clearCurrentVerification, clearError, clearCertificationResult } = verificationSlice.actions;

export default verificationSlice.reducer;
