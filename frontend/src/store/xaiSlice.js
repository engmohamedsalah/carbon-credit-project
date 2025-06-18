import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import xaiService from '../services/xaiService';

// Async thunks for XAI operations
export const generateExplanation = createAsyncThunk(
  'xai/generateExplanation',
  async (request, { rejectWithValue }) => {
    try {
      const response = await xaiService.generateExplanation(request);
      return response;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const getExplanation = createAsyncThunk(
  'xai/getExplanation',
  async (explanationId, { rejectWithValue }) => {
    try {
      const response = await xaiService.getExplanation(explanationId);
      return response;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const compareExplanations = createAsyncThunk(
  'xai/compareExplanations',
  async (request, { rejectWithValue }) => {
    try {
      const response = await xaiService.compareExplanations(request);
      return response;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const getAvailableMethods = createAsyncThunk(
  'xai/getAvailableMethods',
  async (_, { rejectWithValue }) => {
    try {
      const response = await xaiService.getAvailableMethods();
      return response;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const exportExplanation = createAsyncThunk(
  'xai/exportExplanation',
  async ({ explanationId, format }, { rejectWithValue }) => {
    try {
      const blob = await xaiService.exportExplanation(explanationId, format);
      
      // Create download URL
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `explanation_${explanationId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      return { explanationId, format, success: true };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// Initial state
const initialState = {
  // Current explanation being viewed
  currentExplanation: null,
  
  // All explanations history
  explanations: [],
  
  // Available XAI methods
  availableMethods: [],
  
  // Comparison results
  comparisons: [],
  currentComparison: null,
  
  // UI state
  selectedMethod: 'shap',
  selectedProjectId: null,
  
  // Generation settings
  settings: {
    visualizationType: 'interactive',
    exportFormat: 'png',
    comparisonType: 'side_by_side',
    autoRefresh: false
  },
  
  // Loading states
  loading: {
    generating: false,
    fetching: false,
    comparing: false,
    exporting: false,
    methods: false
  },
  
  // Error handling
  error: null,
  
  // Service status
  serviceStatus: 'unknown' // 'operational', 'unavailable', 'unknown'
};

// XAI slice
const xaiSlice = createSlice({
  name: 'xai',
  initialState,
  reducers: {
    // UI actions
    setSelectedMethod: (state, action) => {
      state.selectedMethod = action.payload;
    },
    
    setSelectedProjectId: (state, action) => {
      state.selectedProjectId = action.payload;
    },
    
    setCurrentExplanation: (state, action) => {
      state.currentExplanation = action.payload;
    },
    
    setCurrentComparison: (state, action) => {
      state.currentComparison = action.payload;
    },
    
    // Settings actions
    updateSettings: (state, action) => {
      state.settings = { ...state.settings, ...action.payload };
    },
    
    // Clear actions
    clearCurrentExplanation: (state) => {
      state.currentExplanation = null;
    },
    
    clearCurrentComparison: (state) => {
      state.currentComparison = null;
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    clearExplanations: (state) => {
      state.explanations = [];
      state.currentExplanation = null;
    },
    
    // Add explanation to history
    addExplanationToHistory: (state, action) => {
      const explanation = action.payload;
      // Avoid duplicates
      const existingIndex = state.explanations.findIndex(exp => exp.explanation_id === explanation.explanation_id);
      if (existingIndex >= 0) {
        state.explanations[existingIndex] = explanation;
      } else {
        state.explanations.unshift(explanation); // Add to beginning
      }
      
      // Keep only last 50 explanations
      if (state.explanations.length > 50) {
        state.explanations = state.explanations.slice(0, 50);
      }
    },
    
    // Remove explanation from history
    removeExplanationFromHistory: (state, action) => {
      const explanationId = action.payload;
      state.explanations = state.explanations.filter(exp => exp.explanation_id !== explanationId);
      
      // Clear current if it matches
      if (state.currentExplanation?.explanation_id === explanationId) {
        state.currentExplanation = null;
      }
    }
  },
  
  extraReducers: (builder) => {
    builder
      // Generate explanation
      .addCase(generateExplanation.pending, (state) => {
        state.loading.generating = true;
        state.error = null;
      })
      .addCase(generateExplanation.fulfilled, (state, action) => {
        state.loading.generating = false;
        state.currentExplanation = action.payload;
        
        // Add to history
        xaiSlice.caseReducers.addExplanationToHistory(state, action);
      })
      .addCase(generateExplanation.rejected, (state, action) => {
        state.loading.generating = false;
        state.error = action.payload;
      })
      
      // Get explanation
      .addCase(getExplanation.pending, (state) => {
        state.loading.fetching = true;
        state.error = null;
      })
      .addCase(getExplanation.fulfilled, (state, action) => {
        state.loading.fetching = false;
        state.currentExplanation = action.payload;
        
        // Add to history
        xaiSlice.caseReducers.addExplanationToHistory(state, action);
      })
      .addCase(getExplanation.rejected, (state, action) => {
        state.loading.fetching = false;
        state.error = action.payload;
      })
      
      // Compare explanations
      .addCase(compareExplanations.pending, (state) => {
        state.loading.comparing = true;
        state.error = null;
      })
      .addCase(compareExplanations.fulfilled, (state, action) => {
        state.loading.comparing = false;
        state.currentComparison = action.payload;
        
        // Add to comparisons history
        state.comparisons.unshift(action.payload);
        if (state.comparisons.length > 20) {
          state.comparisons = state.comparisons.slice(0, 20);
        }
      })
      .addCase(compareExplanations.rejected, (state, action) => {
        state.loading.comparing = false;
        state.error = action.payload;
      })
      
      // Get available methods
      .addCase(getAvailableMethods.pending, (state) => {
        state.loading.methods = true;
        state.error = null;
      })
      .addCase(getAvailableMethods.fulfilled, (state, action) => {
        state.loading.methods = false;
        state.availableMethods = action.payload.methods || [];
        state.serviceStatus = action.payload.service_status || 'unknown';
      })
      .addCase(getAvailableMethods.rejected, (state, action) => {
        state.loading.methods = false;
        state.error = action.payload;
        state.serviceStatus = 'unavailable';
      })
      
      // Export explanation
      .addCase(exportExplanation.pending, (state) => {
        state.loading.exporting = true;
        state.error = null;
      })
      .addCase(exportExplanation.fulfilled, (state) => {
        state.loading.exporting = false;
        // Export success is handled in the thunk (download)
      })
      .addCase(exportExplanation.rejected, (state, action) => {
        state.loading.exporting = false;
        state.error = action.payload;
      });
  }
});

// Action creators
export const {
  setSelectedMethod,
  setSelectedProjectId,
  setCurrentExplanation,
  setCurrentComparison,
  updateSettings,
  clearCurrentExplanation,
  clearCurrentComparison,
  clearError,
  clearExplanations,
  addExplanationToHistory,
  removeExplanationFromHistory
} = xaiSlice.actions;

// Selectors
export const selectXAI = (state) => state.xai;
export const selectCurrentExplanation = (state) => state.xai.currentExplanation;
export const selectExplanations = (state) => state.xai.explanations;
export const selectAvailableMethods = (state) => state.xai.availableMethods;
export const selectCurrentComparison = (state) => state.xai.currentComparison;
export const selectXAILoading = (state) => state.xai.loading;
export const selectXAIError = (state) => state.xai.error;
export const selectXAISettings = (state) => state.xai.settings;
export const selectSelectedMethod = (state) => state.xai.selectedMethod;
export const selectSelectedProjectId = (state) => state.xai.selectedProjectId;
export const selectServiceStatus = (state) => state.xai.serviceStatus;

// Complex selectors
export const selectExplanationsByProject = (state, projectId) => 
  state.xai.explanations.filter(exp => exp.project_id === projectId);

export const selectExplanationsByMethod = (state, method) => 
  state.xai.explanations.filter(exp => 
    exp.method === method || exp.methods_used?.includes(method)
  );

export const selectIsMethodAvailable = (state, method) => 
  state.xai.availableMethods.some(m => m.name === method);

export const selectFormattedCurrentExplanation = (state) => {
  const explanation = state.xai.currentExplanation;
  if (!explanation || !explanation.analysis) return null;
  
  const formatted = {};
  
  // Format SHAP data if available
  if (explanation.analysis.shap) {
    formatted.shap = xaiService.formatSHAPData(explanation.analysis.shap);
  }
  
  // Format LIME data if available
  if (explanation.analysis.lime) {
    formatted.lime = xaiService.formatLIMEData(explanation.analysis.lime);
  }
  
  // Format Integrated Gradients data if available
  if (explanation.analysis.integrated_gradients) {
    formatted.integrated_gradients = xaiService.formatIntegratedGradientsData(explanation.analysis.integrated_gradients);
  }
  
  return {
    ...explanation,
    formattedAnalysis: formatted
  };
};

export default xaiSlice.reducer; 