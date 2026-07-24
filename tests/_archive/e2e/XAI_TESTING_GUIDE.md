# XAI (Explainable AI) Testing Guide

## Overview

This guide explains how to test the XAI (Explainable AI) functionality in the Carbon Credit Verification SaaS application. The XAI system provides business-friendly AI explanations with regulatory compliance features.

> **Current status:** The XAI service is **DISABLED** in the application build (`backend/services/xai_service.py` returns an explicit "Explainability is not available in this build" result). The endpoints and flows described below are the intended design. Until the service is re-enabled and wired to real per-prediction inputs, these calls return an "unavailable" response, so this guide documents expected behavior rather than a live feature.

## XAI Testing Flow

### 1. **System Architecture Flow**
```
User Request → Authentication → XAI API → ML Models → Explanation Generation → Business Intelligence → Response
```

### 2. **Testing Levels**

#### **API Level Testing**
- Authentication with JWT tokens
- XAI methods endpoint validation
- Explanation generation with different methods (SHAP, LIME, Integrated Gradients)
- Report generation (PDF/JSON)
- Explanation comparison and history
- Error handling and validation

#### **UI Level Testing** 
- Page accessibility and navigation
- Tab functionality (Generate, Compare, History, Business Intelligence)
- Form interactions and validation
- Real-time explanation generation
- Report download functionality

#### **Integration Testing**
- End-to-end workflow testing
- Business intelligence features
- Regulatory compliance validation
- Performance and response time testing

## Available XAI Methods

### 1. **SHAP (SHapley Additive exPlanations)**
- **Purpose**: Feature importance using Shapley values with business context
- **Supported Models**: forest_cover, change_detection, ensemble
- **Visualizations**: waterfall, force_plot, summary_plot, business_dashboard
- **Business Features**: plain_language_explanations, financial_impact, risk_assessment

### 2. **LIME (Local Interpretable Model-agnostic Explanations)**
- **Purpose**: Local explanations with regulatory compliance features
- **Supported Models**: forest_cover, change_detection
- **Visualizations**: image_segments, feature_importance, business_metrics
- **Business Features**: stakeholder_reports, uncertainty_quantification, compliance_notes

### 3. **Integrated Gradients**
- **Purpose**: Attribution method with professional reporting capabilities
- **Supported Models**: forest_cover, change_detection, time_series
- **Visualizations**: attribution_map, sensitivity_analysis, confidence_intervals
- **Business Features**: executive_summaries, audit_trails, regulatory_compliance

## XAI API Endpoints

### Authentication
```bash
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded
Body: username=$ADMIN_EMAIL&password=$ADMIN_PASSWORD
```

### Core XAI Endpoints

#### 1. Get Available Methods
```bash
GET /api/v1/xai/methods
Authorization: Bearer <token>
```

#### 2. Generate Explanation
```bash
POST /api/v1/xai/generate-explanation
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_id": "forest_cover_ensemble",
  "instance_data": {
    "project_id": 1,
    "location": "Test Forest Area",
    "area_hectares": 150.0,
    "forest_type": "Mixed Deciduous",
    "satellite_date": "2024-01-15"
  },
  "explanation_method": "shap",
  "business_friendly": true,
  "include_uncertainty": true
}
```

#### 3. Generate Report
```bash
POST /api/v1/xai/generate-report
Authorization: Bearer <token>
Content-Type: application/json

{
  "explanation_id": "<explanation-id>",
  "format": "pdf",
  "include_business_summary": true
}
```

#### 4. Compare Explanations
```bash
POST /api/v1/xai/compare-explanations
Authorization: Bearer <token>
Content-Type: application/json

{
  "explanation_ids": ["<id1>", "<id2>"],
  "comparison_type": "side_by_side"
}
```

#### 5. Get Explanation History
```bash
GET /api/v1/xai/explanation-history/<project_id>
Authorization: Bearer <token>
```

## Running XAI Tests

### Prerequisites
1. **Backend Running**: `cd backend && python main.py`
2. **Frontend Running**: `cd frontend && npm start`
3. **Dependencies Installed**: `pip install playwright requests`

### API Tests Only
```bash
python tests/e2e/test_xai_api_only.py
```

### Full Playwright Tests
```bash
python tests/e2e/test_xai_functionality.py
```

### Quick Test Runner
```bash
python tests/e2e/run_xai_tests.py
```

## Test Results Analysis

### Expected API Test Results

#### ✅ **Successful Tests**
- **Authentication**: JWT token generation
- **Explanation Generation**: SHAP/LIME/IG explanations with confidence scores
- **Report Generation**: PDF reports with business summaries
- **Explanation Comparison**: Side-by-side method comparisons

#### ⚠️ **Common Issues**
- **History Retrieval**: May fail if project doesn't exist
- **Error Handling**: Some invalid data might be processed (expected behavior for demo)
- **UI Tests**: May require specific test data or user interactions

### Performance Metrics

No performance figures have been measured: the XAI service is disabled in the
current build (see Conclusion), so generation latency, response time, and
confidence-score ranges are not yet available.

## Business Intelligence Features

### 1. **Executive Summaries**
- Plain language explanations
- Business impact assessments
- Financial implications
- Risk level categorization

### 2. **Regulatory Compliance**
- EU AI Act compliance indicators
- Carbon Standards (VCS/Gold Standard) readiness
- Audit trail documentation
- Transparency requirements fulfillment

### 3. **Uncertainty Quantification**
- Confidence intervals
- Model uncertainty assessment
- Data quality indicators
- Risk mitigation recommendations

## UI Testing Flow

### 1. **Page Navigation**
```
Login → Dashboard → XAI Page → Tab Selection → Form Interaction → Results Display
```

### 2. **Tab Functionality**
- **Generate Tab**: Create new explanations
- **Compare Tab**: Side-by-side comparisons
- **History Tab**: View past explanations
- **Business Intelligence Tab**: Executive dashboards

### 3. **User Interactions**
- Project selection from dropdown
- XAI method selection (SHAP/LIME/IG)
- Business-friendly options toggle
- Uncertainty quantification toggle
- Report generation and download

## Troubleshooting

### Common Issues

#### 1. **Authentication Failures**
- **Issue**: Login endpoint expects form data, not JSON
- **Solution**: Use `requests.post(url, data={...})` instead of `json={...}`

#### 2. **Project Not Found Errors**
- **Issue**: History endpoint requires valid project IDs
- **Solution**: Create test projects or use existing project IDs

#### 3. **Playwright Browser Issues**
- **Issue**: Browser crashes or timeouts
- **Solution**: Install browser with `playwright install chromium`

#### 4. **Server Connection Issues**
- **Issue**: Backend/Frontend not accessible
- **Solution**: Ensure both servers are running on correct ports

### Performance Optimization
- Use parallel test execution where possible
- Cache authentication tokens
- Implement proper timeout handling
- Use headless browsers for CI/CD

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run XAI Tests
  run: |
    python tests/e2e/test_xai_api_only.py
    python tests/e2e/run_xai_tests.py
```

### Test Coverage Goals
- **API Coverage**: 100% of XAI endpoints
- **UI Coverage**: All major user workflows
- **Error Handling**: All validation scenarios
- **Performance**: Response time benchmarks

## Future Enhancements

### Planned Testing Improvements
1. **Visual Regression Testing**: Screenshot comparisons
2. **Load Testing**: Multiple concurrent users
3. **Security Testing**: Authentication and authorization
4. **Mobile Testing**: Responsive design validation
5. **Accessibility Testing**: WCAG compliance

### Advanced XAI Testing
1. **Model Accuracy Testing**: Explanation quality validation
2. **Business Logic Testing**: Financial calculations
3. **Compliance Testing**: Regulatory requirement verification
4. **Integration Testing**: Third-party service connections

## Conclusion

The XAI testing suite documents comprehensive coverage of the explainable AI functionality, so that business users can trust and understand AI decisions in carbon credit verification once the feature is live. Note that the XAI service is currently disabled in the application build, so these tests describe the intended behavior rather than a live capability; they will report "unavailable" until the service is re-enabled and wired to real per-prediction inputs. 