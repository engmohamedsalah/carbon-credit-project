/**
 * ML Service for Carbon Credit Verification
 * Handles all ML-related API calls
 */

import apiService from './apiService';

class MLService {
  /**
   * Get ML service status
   */
  async getMLStatus() {
    try {
      const response = await apiService.get('/ml/status');
      return response.data;
    } catch (error) {
      console.error('ML Status check failed:', error);
      throw error;
    }
  }

  /**
   * Analyze forest cover from uploaded satellite image
   */
  async analyzeForestCover(projectId, imageFile) {
    try {
      const formData = new FormData();
      formData.append('project_id', projectId);
      formData.append('file', imageFile);

      const response = await apiService.post('/ml/forest-cover', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Forest cover analysis failed:', error);
      throw error;
    }
  }

  /**
   * Detect changes between two satellite images
   */
  async detectChanges(projectId, beforeImage, afterImage) {
    try {
      const formData = new FormData();
      formData.append('project_id', projectId);
      formData.append('before_image', beforeImage);
      formData.append('after_image', afterImage);

      const response = await apiService.post('/ml/change-detection', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Change detection failed:', error);
      throw error;
    }
  }

  /**
   * Run comprehensive ML analysis for a project
   */
  async runComprehensiveAnalysis(projectId, analysisData) {
    try {
      const results = {};

      // Forest cover analysis if image provided
      if (analysisData.forestCoverImage) {
        results.forestCoverAnalysis = await this.analyzeForestCover(
          projectId,
          analysisData.forestCoverImage
        );
      }

      // Change detection if before/after images provided
      if (analysisData.beforeImage && analysisData.afterImage) {
        results.changeDetection = await this.detectChanges(
          projectId,
          analysisData.beforeImage,
          analysisData.afterImage
        );
      }

      return {
        projectId,
        status: 'completed',
        timestamp: new Date().toISOString(),
        results
      };

    } catch (error) {
      console.error('Comprehensive analysis failed:', error);
      throw error;
    }
  }

  /**
   * Format analysis results for display
   */
  formatAnalysisResults(results) {
    if (!results) return null;

    const formatted = {
      status: results.status,
      timestamp: results.timestamp,
      projectId: results.project_id,
      summary: {}
    };

    // Populate summary from the real forest-cover analysis response.
    // Endpoint envelope: { results: { forest_prediction, carbon_impact, ... } }
    const forest = results.results?.forestCoverAnalysis?.results;
    if (forest) {
      const carbon = forest.carbon_impact || {};
      formatted.summary.forestCoverage = carbon.forest_coverage_percent;
      formatted.summary.forestArea = carbon.forest_area_hectares;
      formatted.summary.carbonEstimate = carbon.total_carbon_tons;
      formatted.summary.confidenceScore = forest.forest_prediction?.mean_probability;
    }

    // Fall back to change-detection "after" state when no forest-cover image was run.
    // Envelope: { results: { after_analysis: { forest_prediction, carbon_impact }, ... } }
    const change = results.results?.changeDetection?.results;
    if (change && !forest) {
      const carbon = change.after_analysis?.carbon_impact || {};
      formatted.summary.forestCoverage = carbon.forest_coverage_percent;
      formatted.summary.forestArea = carbon.forest_area_hectares;
      formatted.summary.carbonEstimate = carbon.total_carbon_tons;
      formatted.summary.confidenceScore = change.after_analysis?.forest_prediction;
    }

    // Format individual analysis results
    if (results.results) {
      formatted.details = results.results;
    }

    return formatted;
  }

  /**
   * Calculate carbon credit eligibility
   */
  calculateEligibility(analysisResults) {
    if (!analysisResults) return null;

    const { summary } = analysisResults;
    
    let eligibilityScore = 0;
    let factors = [];

    // Forest coverage factor (0-30 points)
    if (summary.forestCoverage >= 60) {
      eligibilityScore += 30;
      factors.push({ factor: 'High Forest Coverage', score: 30, status: 'excellent' });
    } else if (summary.forestCoverage >= 40) {
      eligibilityScore += 20;
      factors.push({ factor: 'Moderate Forest Coverage', score: 20, status: 'good' });
    } else if (summary.forestCoverage >= 20) {
      eligibilityScore += 10;
      factors.push({ factor: 'Low Forest Coverage', score: 10, status: 'fair' });
    } else {
      factors.push({ factor: 'Very Low Forest Coverage', score: 0, status: 'poor' });
    }

    // Confidence score factor (0-25 points)
    if (summary.confidenceScore >= 0.8) {
      eligibilityScore += 25;
      factors.push({ factor: 'High Model Confidence', score: 25, status: 'excellent' });
    } else if (summary.confidenceScore >= 0.6) {
      eligibilityScore += 15;
      factors.push({ factor: 'Good Model Confidence', score: 15, status: 'good' });
    } else {
      eligibilityScore += 5;
      factors.push({ factor: 'Low Model Confidence', score: 5, status: 'fair' });
    }

    // Carbon potential factor (0-25 points)
    if (summary.carbonEstimate >= 2000) {
      eligibilityScore += 25;
      factors.push({ factor: 'High Carbon Potential', score: 25, status: 'excellent' });
    } else if (summary.carbonEstimate >= 1000) {
      eligibilityScore += 15;
      factors.push({ factor: 'Moderate Carbon Potential', score: 15, status: 'good' });
    } else if (summary.carbonEstimate >= 500) {
      eligibilityScore += 10;
      factors.push({ factor: 'Low Carbon Potential', score: 10, status: 'fair' });
    } else {
      factors.push({ factor: 'Very Low Carbon Potential', score: 0, status: 'poor' });
    }

    // Area size factor (0-20 points)
    if (summary.forestArea >= 1000) {
      eligibilityScore += 20;
      factors.push({ factor: 'Large Project Area', score: 20, status: 'excellent' });
    } else if (summary.forestArea >= 500) {
      eligibilityScore += 15;
      factors.push({ factor: 'Medium Project Area', score: 15, status: 'good' });
    } else if (summary.forestArea >= 100) {
      eligibilityScore += 10;
      factors.push({ factor: 'Small Project Area', score: 10, status: 'fair' });
    } else {
      factors.push({ factor: 'Very Small Project Area', score: 0, status: 'poor' });
    }

    let recommendation;
    let status;

    if (eligibilityScore >= 80) {
      recommendation = 'Highly Recommended for Carbon Credit Certification';
      status = 'excellent';
    } else if (eligibilityScore >= 60) {
      recommendation = 'Recommended for Carbon Credit Certification';
      status = 'good';
    } else if (eligibilityScore >= 40) {
      recommendation = 'Conditionally Eligible - Improvements Needed';
      status = 'fair';
    } else {
      recommendation = 'Not Eligible for Carbon Credit Certification';
      status = 'poor';
    }

    return {
      eligibilityScore,
      maxScore: 100,
      percentage: Math.round((eligibilityScore / 100) * 100),
      recommendation,
      status,
      factors,
      nextSteps: this.getNextSteps(eligibilityScore)
    };
  }

  /**
   * Get recommended next steps based on eligibility score
   */
  getNextSteps(score) {
    if (score >= 80) {
      return [
        'Proceed with detailed field verification',
        'Prepare comprehensive documentation',
        'Submit for official certification',
        'Establish monitoring protocols'
      ];
    } else if (score >= 60) {
      return [
        'Conduct additional field surveys',
        'Improve forest management practices',
        'Enhance monitoring systems',
        'Resubmit for analysis in 6 months'
      ];
    } else if (score >= 40) {
      return [
        'Implement significant forest restoration',
        'Address identified deficiencies',
        'Develop comprehensive management plan',
        'Consider alternative carbon projects'
      ];
    } else {
      return [
        'Reassess project viability',
        'Consider site relocation',
        'Implement major forest restoration',
        'Explore alternative environmental projects'
      ];
    }
  }
}

// Export singleton instance
const mlService = new MLService();
export default mlService; 