/**
 * Enhanced XAI Service for Carbon Credit Verification
 * Provides business-focused AI explanations with regulatory compliance features
 */

import apiClient from './apiService';

class XAIService {
  constructor() {
    this.baseURL = '/xai';
  }

  /**
   * Generate enhanced AI explanation with business context
   */
  async generateExplanation(explanationData) {
    try {
      const response = await apiClient.post(`${this.baseURL}/generate-explanation`, {
        model_id: explanationData.modelId || 'forest_cover_ensemble',
        instance_data: explanationData.instanceData,
        explanation_method: explanationData.method || 'shap',
        business_friendly: explanationData.businessFriendly !== false,
        include_uncertainty: explanationData.includeUncertainty !== false
      });
      
      return response.data;
    } catch (error) {
      console.error('Failed to generate explanation:', error);
      throw new Error(error.response?.data?.detail || 'Failed to generate explanation');
    }
  }

  /**
   * Retrieve explanation by ID
   */
  async getExplanation(explanationId) {
    try {
      const response = await apiClient.get(`${this.baseURL}/explanation/${explanationId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to retrieve explanation:', error);
      throw new Error(error.response?.data?.detail || 'Failed to retrieve explanation');
    }
  }

  /**
   * Compare multiple explanations with business analysis
   */
  async compareExplanations(explanationIds, comparisonType = 'side_by_side') {
    try {
      const response = await apiClient.post(`${this.baseURL}/compare-explanations`, {
        explanation_ids: explanationIds,
        comparison_type: comparisonType
      });
      
      return response.data;
    } catch (error) {
      console.error('Failed to compare explanations:', error);
      throw new Error(error.response?.data?.detail || 'Failed to compare explanations');
    }
  }

  /**
   * Generate professional report from explanation
   */
  async generateReport(explanationId, format = 'pdf', includeBusinessSummary = true) {
    try {
      const response = await apiClient.post(`${this.baseURL}/generate-report`, {
        explanation_id: explanationId,
        format: format,
        include_business_summary: includeBusinessSummary
      });
      
      return response.data;
    } catch (error) {
      console.error('Failed to generate report:', error);
      throw new Error(error.response?.data?.detail || 'Failed to generate report');
    }
  }

  /**
   * Get explanation history for a project
   */
  async getExplanationHistory(projectId) {
    try {
      const response = await apiClient.get(`${this.baseURL}/explanation-history/${projectId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to retrieve explanation history:', error);
      throw new Error(error.response?.data?.detail || 'Failed to retrieve explanation history');
    }
  }

  /**
   * Get available XAI methods and capabilities
   */
  async getAvailableMethods() {
    try {
      const response = await apiClient.get(`${this.baseURL}/methods`);
      return response.data;
    } catch (error) {
      console.error('Failed to retrieve XAI methods:', error);
      throw new Error(error.response?.data?.detail || 'Failed to retrieve XAI methods');
    }
  }

  /**
   * Download report as file
   */
  downloadReport(reportData, filename) {
    try {
      // Extract base64 data from data URL
      const base64Data = reportData.data.split(',')[1];
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { 
        type: reportData.format === 'pdf' ? 'application/pdf' : 'application/json' 
      });
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || reportData.filename || `xai_report.${reportData.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      return true;
    } catch (error) {
      console.error('Failed to download report:', error);
      throw new Error('Failed to download report');
    }
  }

  /**
   * Format explanation for display
   */
  formatExplanationForDisplay(explanation) {
    if (!explanation) return null;

    return {
      id: explanation.explanation_id,
      timestamp: new Date(explanation.timestamp).toLocaleString(),
      method: this.getMethodDisplayName(explanation.method),
      confidence: explanation.confidence_score,
      businessSummary: explanation.business_summary,
      riskLevel: explanation.risk_assessment?.level || 'Unknown',
      hasVisualizations: explanation.visualizations && Object.keys(explanation.visualizations).length > 0,
      complianceStatus: explanation.regulatory_notes?.eu_ai_act_compliance ? 'Compliant' : 'Pending'
    };
  }

  /**
   * Get display name for XAI method
   */
  getMethodDisplayName(method) {
    const methodMap = {
      'shap': 'SHAP',
      'lime': 'LIME',
      'integrated_gradients': 'Integrated Gradients'
    };
    return methodMap[method] || method.toUpperCase();
  }

  /**
   * Get confidence level description
   */
  getConfidenceDescription(confidence) {
    if (confidence >= 0.9) return 'Very High';
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.7) return 'Medium';
    if (confidence >= 0.6) return 'Moderate';
    return 'Low';
  }

  /**
   * Get risk level color
   */
  getRiskLevelColor(riskLevel) {
    const colorMap = {
      'Low': 'success',
      'Medium': 'warning',
      'High': 'error'
    };
    return colorMap[riskLevel] || 'default';
  }

  /**
   * Validate explanation data before generation
   */
  validateExplanationData(data) {
    const errors = [];

    if (!data.instanceData) {
      errors.push('Instance data is required');
    }

    if (data.method && !['shap', 'lime', 'integrated_gradients'].includes(data.method)) {
      errors.push('Invalid explanation method');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Create sample explanation data for testing
   */
  createSampleExplanationData(projectId, imageData = null) {
    return {
      modelId: 'forest_cover_ensemble',
      instanceData: {
        project_id: projectId,
        image_data: imageData,
        features: {
          ndvi: 0.7,
          temperature: 25.5,
          precipitation: 1200,
          elevation: 500,
          slope: 15,
          aspect: 180
        }
      },
      method: 'shap',
      businessFriendly: true,
      includeUncertainty: true
    };
  }
}

// Create and export singleton instance
const xaiService = new XAIService();
export default xaiService; 