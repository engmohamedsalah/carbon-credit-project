import apiClient from '../config/api';

/**
 * XAI Service - Handles all Explainable AI API calls
 * Provides methods for generating, retrieving, and comparing AI explanations
 */
class XAIService {
  /**
   * Generate an AI explanation for a model prediction
   * @param {Object} request - Explanation request parameters
   * @param {number} request.project_id - Project ID
   * @param {string} request.explanation_method - Method to use ("shap", "lime", "integrated_gradients", "all")
   * @param {string} [request.prediction_id] - ID of previous prediction to explain
   * @param {string} [request.image_path] - Path to image for explanation
   * @returns {Promise<Object>} Explanation response
   */
  async generateExplanation(request) {
    try {
      const response = await apiClient.post('/api/v1/xai/generate-explanation', request);
      return response.data;
    } catch (error) {
      console.error('Failed to generate explanation:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Retrieve a generated explanation by ID
   * @param {string} explanationId - Explanation ID
   * @returns {Promise<Object>} Explanation data
   */
  async getExplanation(explanationId) {
    try {
      const response = await apiClient.get(`/api/v1/xai/explanation/${explanationId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to retrieve explanation:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Compare multiple explanations
   * @param {Object} request - Comparison request
   * @param {string[]} request.explanation_ids - Array of explanation IDs
   * @param {string} request.comparison_type - Type of comparison ("side_by_side", "overlay", "difference")
   * @returns {Promise<Object>} Comparison results
   */
  async compareExplanations(request) {
    try {
      const response = await apiClient.post('/api/v1/xai/compare-explanations', request);
      return response.data;
    } catch (error) {
      console.error('Failed to compare explanations:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Get available XAI explanation methods
   * @returns {Promise<Object>} Available methods and their configurations
   */
  async getAvailableMethods() {
    try {
      const response = await apiClient.get('/api/v1/xai/methods');
      return response.data;
    } catch (error) {
      console.error('Failed to get XAI methods:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Generate explanation for existing ML analysis
   * @param {number} projectId - Project ID
   * @param {string} method - XAI method to use
   * @param {string} [predictionId] - Existing prediction ID
   * @returns {Promise<Object>} Explanation response
   */
  async explainExistingPrediction(projectId, method, predictionId = null) {
    return this.generateExplanation({
      project_id: projectId,
      explanation_method: method,
      prediction_id: predictionId
    });
  }

  /**
   * Generate explanation for uploaded image
   * @param {number} projectId - Project ID
   * @param {string} method - XAI method to use
   * @param {string} imagePath - Path to uploaded image
   * @returns {Promise<Object>} Explanation response
   */
  async explainImagePrediction(projectId, method, imagePath) {
    return this.generateExplanation({
      project_id: projectId,
      explanation_method: method,
      image_path: imagePath
    });
  }

  /**
   * Get explanation history for a project
   * @param {number} projectId - Project ID
   * @returns {Promise<Object[]>} Array of explanations for the project
   */
  async getProjectExplanations(projectId) {
    // Note: This would require a backend endpoint to list explanations by project
    // For now, we'll return empty array
    try {
      // Future endpoint: GET /api/v1/xai/explanations?project_id=${projectId}
      console.warn('Project explanations endpoint not implemented yet');
      return [];
    } catch (error) {
      console.error('Failed to get project explanations:', error);
      return [];
    }
  }

  /**
   * Export explanation results
   * @param {string} explanationId - Explanation ID
   * @param {string} format - Export format ("json", "pdf", "png")
   * @returns {Promise<Blob>} Exported file
   */
  async exportExplanation(explanationId, format = 'json') {
    try {
      // Future endpoint implementation
      console.warn('Export explanation endpoint not implemented yet');
      
      // For now, return mock download
      const explanation = await this.getExplanation(explanationId);
      const blob = new Blob([JSON.stringify(explanation, null, 2)], {
        type: 'application/json'
      });
      return blob;
    } catch (error) {
      console.error('Failed to export explanation:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Handle API errors with user-friendly messages
   * @param {Error} error - Original error
   * @returns {Error} Formatted error
   */
  handleError(error) {
    if (error.response) {
      const status = error.response.status;
      const message = error.response.data?.detail || error.response.data?.message || 'Unknown error';
      
      switch (status) {
        case 404:
          return new Error('Explanation not found');
        case 403:
          return new Error('Not authorized to access this explanation');
        case 503:
          return new Error('XAI service is currently unavailable');
        case 500:
          return new Error('Server error while processing explanation');
        default:
          return new Error(`Explanation service error: ${message}`);
      }
    } else if (error.request) {
      return new Error('Unable to connect to XAI service');
    } else {
      return new Error('Failed to process explanation request');
    }
  }

  /**
   * Validate explanation method
   * @param {string} method - Method to validate
   * @returns {boolean} Whether method is valid
   */
  isValidMethod(method) {
    const validMethods = ['shap', 'lime', 'integrated_gradients', 'all'];
    return validMethods.includes(method);
  }

  /**
   * Get method display information
   * @param {string} method - Method name
   * @returns {Object} Method information
   */
  getMethodInfo(method) {
    const methodInfo = {
      shap: {
        name: 'SHAP',
        fullName: 'SHapley Additive exPlanations',
        description: 'Feature importance using Shapley values from game theory',
        visualizations: ['Waterfall Plot', 'Force Plot', 'Summary Plot'],
        bestFor: 'Global feature importance and individual prediction explanation'
      },
      lime: {
        name: 'LIME',
        fullName: 'Local Interpretable Model-agnostic Explanations',
        description: 'Local explanations for individual predictions',
        visualizations: ['Image Segments', 'Feature Importance'],
        bestFor: 'Understanding individual predictions in detail'
      },
      integrated_gradients: {
        name: 'Integrated Gradients',
        fullName: 'Integrated Gradients Attribution',
        description: 'Attribution method for deep learning models',
        visualizations: ['Attribution Map', 'Sensitivity Analysis'],
        bestFor: 'Deep learning model interpretation and gradient analysis'
      },
      all: {
        name: 'All Methods',
        fullName: 'Comprehensive XAI Analysis',
        description: 'Generate explanations using all available methods',
        visualizations: ['Combined Analysis', 'Method Comparison'],
        bestFor: 'Complete understanding and method comparison'
      }
    };

    return methodInfo[method] || null;
  }

  /**
   * Parse SHAP values for visualization
   * @param {Object} shapData - Raw SHAP data from API
   * @returns {Object} Formatted data for charts
   */
  formatSHAPData(shapData) {
    if (!shapData.waterfall_data) return null;

    return {
      waterfallData: shapData.waterfall_data.map(item => ({
        feature: item.feature,
        value: item.value,
        contribution: item.contribution,
        formattedContribution: item.contribution > 0 ? `+${item.contribution.toFixed(3)}` : item.contribution.toFixed(3)
      })),
      featureImportance: Object.entries(shapData.feature_importance || {}).map(([feature, importance]) => ({
        feature,
        importance: importance,
        percentage: (importance * 100).toFixed(1)
      })),
      baseValue: shapData.base_value,
      predictedValue: shapData.predicted_value,
      confidence: shapData.confidence
    };
  }

  /**
   * Parse LIME values for visualization
   * @param {Object} limeData - Raw LIME data from API
   * @returns {Object} Formatted data for charts
   */
  formatLIMEData(limeData) {
    if (!limeData.segment_importance) return null;

    return {
      segments: limeData.segment_importance.map(segment => ({
        id: segment.segment_id,
        importance: segment.importance,
        area: segment.area_percentage,
        type: segment.importance > 0 ? 'positive' : 'negative',
        formattedImportance: segment.importance.toFixed(3),
        formattedArea: `${segment.area_percentage.toFixed(1)}%`
      })),
      summary: {
        totalSegments: limeData.image_segments?.total_segments || 0,
        importantSegments: limeData.image_segments?.important_segments || 0,
        positiveSegments: limeData.image_segments?.positive_segments || 0,
        negativeSegments: limeData.image_segments?.negative_segments || 0
      },
      confidence: limeData.prediction_confidence,
      explanation: limeData.local_explanation
    };
  }

  /**
   * Parse Integrated Gradients data for visualization
   * @param {Object} igData - Raw IG data from API
   * @returns {Object} Formatted data for charts
   */
  formatIntegratedGradientsData(igData) {
    if (!igData.attribution_map) return null;

    return {
      attributionStats: {
        min: igData.attribution_map.min_attribution,
        max: igData.attribution_map.max_attribution,
        mean: igData.attribution_map.mean_attribution,
        std: igData.attribution_map.std_attribution
      },
      pathIntegration: {
        steps: igData.path_integration?.steps || 0,
        baseline: igData.path_integration?.baseline || 'unknown',
        convergence: igData.path_integration?.convergence || 0
      },
      sensitivity: {
        inputSensitivity: igData.sensitivity_analysis?.input_sensitivity || 0,
        noiseRobustness: igData.sensitivity_analysis?.noise_robustness || 0,
        spatialCoherence: igData.sensitivity_analysis?.spatial_coherence || 0
      }
    };
  }
}

// Create and export a singleton instance
const xaiService = new XAIService();
export default xaiService; 