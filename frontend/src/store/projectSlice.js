import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import apiService from '../services/apiService';

// Async thunks
export const fetchProjects = createAsyncThunk(
  'projects/fetchProjects',
  async (_, { rejectWithValue }) => {
    try {
      const response = await apiService.projects.list();
      // Handle the API response format: {projects: [], total: 0, page: 1, page_size: 20}
      return response.data.projects || response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Failed to fetch projects');
    }
  }
);

export const fetchProjectById = createAsyncThunk(
  'projects/fetchProjectById',
  async (id, { rejectWithValue }) => {
    try {
      const response = await apiService.projects.getById(id);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Failed to fetch project');
    }
  }
);

export const createProject = createAsyncThunk(
  'projects/createProject',
  async (projectData, { rejectWithValue }) => {
    try {
      const response = await apiService.projects.create({
        name: projectData.name,
        location_name: projectData.location_name || 'Unknown Location',
        area_size: parseFloat(projectData.area_hectares) || 0,
        description: projectData.description || '',
        project_type: projectData.project_type || 'Reforestation'
      });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || { detail: 'Network error' });
    }
  }
);

export const updateProject = createAsyncThunk(
  'projects/updateProject',
  async ({ id, projectData }, { rejectWithValue }) => {
    try {
      const response = await apiService.projects.update(id, projectData);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Failed to update project');
    }
  }
);

export const deleteProject = createAsyncThunk(
  'projects/deleteProject',
  async (projectId, { rejectWithValue }) => {
    try {
      const response = await apiService.projects.delete(projectId);
      return { projectId, message: response.data.message };
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Failed to delete project');
    }
  }
);

export const updateProjectStatus = createAsyncThunk(
  'projects/updateProjectStatus',
  async ({ projectId, status }, { rejectWithValue }) => {
    try {
      const response = await apiService.projects.updateStatus(projectId, status);
      return { projectId, status, ...response.data };
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Failed to update project status');
    }
  }
);

export const uploadSatelliteImage = createAsyncThunk(
  'projects/uploadSatelliteImage',
  async ({ projectId, file }, { rejectWithValue }) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('project_id', projectId);
      
      const response = await apiService.satellite.upload(formData);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message || 'Failed to upload satellite image');
    }
  }
);

export const verifyProject = createAsyncThunk(
  'projects/verifyProject',
  async (projectId, { rejectWithValue }) => {
    try {
      const response = await apiService.verification.list({ project_id: projectId });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || { detail: 'Network error' });
    }
  }
);

// Initial state
const initialState = {
  projects: [],
  currentProject: null,
  satelliteImages: [],
  verificationResults: null,
  loading: false,
  error: null,
};

// Slice
const projectSlice = createSlice({
  name: 'projects',
  initialState,
  reducers: {
    clearCurrentProject: (state) => {
      state.currentProject = null;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch projects
      .addCase(fetchProjects.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchProjects.fulfilled, (state, action) => {
        state.loading = false;
        state.projects = action.payload;
      })
      .addCase(fetchProjects.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to fetch projects';
      })
      // Fetch project by ID
      .addCase(fetchProjectById.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchProjectById.fulfilled, (state, action) => {
        state.loading = false;
        state.currentProject = action.payload;
      })
      .addCase(fetchProjectById.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to fetch project';
      })
      // Create project
      .addCase(createProject.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createProject.fulfilled, (state, action) => {
        state.loading = false;
        state.projects.push(action.payload);
      })
      .addCase(createProject.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to create project';
      })
      // Update project
      .addCase(updateProject.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateProject.fulfilled, (state, action) => {
        state.loading = false;
        const index = state.projects.findIndex(p => p.id === action.payload.id);
        if (index !== -1) {
          state.projects[index] = action.payload;
        }
        if (state.currentProject?.id === action.payload.id) {
          state.currentProject = action.payload;
        }
      })
      .addCase(updateProject.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to update project';
      })
      // Delete project
      .addCase(deleteProject.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(deleteProject.fulfilled, (state, action) => {
        state.loading = false;
        state.projects = state.projects.filter(p => p.id !== action.payload.projectId);
        if (state.currentProject?.id === action.payload.projectId) {
          state.currentProject = null;
        }
      })
      .addCase(deleteProject.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to delete project';
      })
      // Update project status
      .addCase(updateProjectStatus.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateProjectStatus.fulfilled, (state, action) => {
        state.loading = false;
        const index = state.projects.findIndex(p => p.id === action.payload.projectId);
        if (index !== -1) {
          state.projects[index] = action.payload;
        }
        if (state.currentProject?.id === action.payload.projectId) {
          state.currentProject = action.payload;
        }
      })
      .addCase(updateProjectStatus.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to update project status';
      })
      // Upload satellite image
      .addCase(uploadSatelliteImage.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(uploadSatelliteImage.fulfilled, (state, action) => {
        state.loading = false;
        if (state.currentProject) {
          state.satelliteImages.push(action.payload);
        }
      })
      .addCase(uploadSatelliteImage.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to upload satellite image';
      })
      // Verify project
      .addCase(verifyProject.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(verifyProject.fulfilled, (state, action) => {
        state.loading = false;
        state.verificationResults = action.payload;
        if (state.currentProject && action.payload.project_id === state.currentProject.id) {
          state.currentProject.status = action.payload.status;
        }
      })
      .addCase(verifyProject.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.detail || 'Failed to verify project';
      });
  },
});

export const { clearCurrentProject, clearError } = projectSlice.actions;

// Selectors
export const selectProjects = (state) => state.projects.projects;
export const selectCurrentProject = (state) => state.projects.currentProject;
export const selectProjectsLoading = (state) => state.projects.loading;
export const selectProjectsError = (state) => state.projects.error;
export const selectSatelliteImages = (state) => state.projects.satelliteImages;
export const selectVerificationResults = (state) => state.projects.verificationResults;

export default projectSlice.reducer;
