import { test, expect } from '@playwright/test';

/**
 * Real XAI Integration Tests
 * Tests the complete workflow from login to generating real XAI explanations
 */

test.describe('Real XAI Integration Tests', () => {
  
  // Test data
  const adminUser = {
    email: 'testadmin@example.com',
    password: 'password123'
  };
  
  const testProject = {
    name: 'XAI Test Forest Project',
    description: 'Test project for real XAI explanation generation',
    location: 'Amazon Basin Test Area',
    area: '250',
    projectType: 'Reforestation'
  };

  test.beforeEach(async ({ page }) => {
    // Login as admin user before each test
    await page.goto('/login');
    await page.fill('input[type="email"]', adminUser.email);
    await page.fill('input[type="password"]', adminUser.password);
    await page.click('button[type="submit"]');
    
    // Wait for successful login and redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Wait for page to load - look for any of these indicators
    await Promise.race([
      page.waitForSelector('text=Welcome', { timeout: 5000 }).catch(() => null),
      page.waitForSelector('h1', { timeout: 5000 }).catch(() => null),
      page.waitForSelector('[data-testid="dashboard"]', { timeout: 5000 }).catch(() => null),
      page.waitForTimeout(2000) // Fallback timeout
    ]);
  });

  test('should access XAI page with admin permissions', async ({ page }) => {
    // Navigate to XAI page
    await page.click('text=XAI');
    await expect(page).toHaveURL('/xai');
    
    // Verify XAI page loads without permission error
    await expect(page.locator('text=Explainable AI Analysis')).toBeVisible();
    await expect(page.locator('text=You don\'t have permission')).not.toBeVisible();
    
    // Verify XAI method selector is present
    await expect(page.locator('[data-testid="method-selector"]')).toBeVisible();
  });

  test('should generate real SHAP explanation', async ({ page }) => {
    // Navigate to XAI page
    await page.goto('/xai');
    
    // Create a test project first
    await page.goto('/projects/new');
    await page.fill('input[name="name"]', testProject.name);
    await page.fill('textarea[name="description"]', testProject.description);
    await page.fill('input[name="location_name"]', testProject.location);
    await page.fill('input[name="area_hectares"]', testProject.area);
    await page.selectOption('select[name="project_type"]', testProject.projectType);
    
    // Submit project creation
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL(/\/projects\/\d+/);
    
    // Get project ID from URL
    const projectId = page.url().match(/\/projects\/(\d+)/)[1];
    
    // Navigate back to XAI page
    await page.goto('/xai');
    
    // Select SHAP method
    await page.selectOption('[data-testid="method-selector"]', 'shap');
    
    // Select the test project
    await page.selectOption('[data-testid="project-selector"]', projectId);
    
    // Generate explanation
    await page.click('[data-testid="generate-explanation-btn"]');
    
    // Wait for explanation to be generated (real XAI takes time)
    await expect(page.locator('[data-testid="explanation-loading"]')).toBeVisible();
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Verify SHAP explanation is displayed
    await expect(page.locator('[data-testid="shap-visualization"]')).toBeVisible();
    await expect(page.locator('text=SHAP Explanation')).toBeVisible();
    
    // Verify real computation indicators
    await expect(page.locator('text=Real XAI')).toBeVisible();
    await expect(page.locator('text=confidence')).toBeVisible();
    
    // Verify waterfall chart is present
    await expect(page.locator('[data-testid="waterfall-chart"]')).toBeVisible();
    
    // Verify feature importance data
    await expect(page.locator('[data-testid="feature-importance"]')).toBeVisible();
    
    // Check for business summary
    await expect(page.locator('[data-testid="business-summary"]')).toBeVisible();
    await expect(page.locator('text=SHAP analysis')).toBeVisible();
  });

  test('should generate real LIME explanation', async ({ page }) => {
    // Navigate to XAI page
    await page.goto('/xai');
    
    // Select LIME method
    await page.selectOption('[data-testid="method-selector"]', 'lime');
    
    // Use existing project or create one
    const projects = await page.locator('[data-testid="project-selector"] option').count();
    if (projects > 1) {
      await page.selectOption('[data-testid="project-selector"]', { index: 1 });
    }
    
    // Generate LIME explanation
    await page.click('[data-testid="generate-explanation-btn"]');
    
    // Wait for explanation
    await expect(page.locator('[data-testid="explanation-loading"]')).toBeVisible();
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Verify LIME explanation is displayed
    await expect(page.locator('[data-testid="lime-visualization"]')).toBeVisible();
    await expect(page.locator('text=LIME Explanation')).toBeVisible();
    
    // Verify segment analysis
    await expect(page.locator('text=Segment')).toBeVisible();
    await expect(page.locator('[data-testid="segment-importance"]')).toBeVisible();
    
    // Verify real computation
    await expect(page.locator('text=Real')).toBeVisible();
  });

  test('should generate real Integrated Gradients explanation', async ({ page }) => {
    // Navigate to XAI page
    await page.goto('/xai');
    
    // Select Integrated Gradients method
    await page.selectOption('[data-testid="method-selector"]', 'integrated_gradients');
    
    // Use existing project
    const projects = await page.locator('[data-testid="project-selector"] option').count();
    if (projects > 1) {
      await page.selectOption('[data-testid="project-selector"]', { index: 1 });
    }
    
    // Generate IG explanation
    await page.click('[data-testid="generate-explanation-btn"]');
    
    // Wait for explanation
    await expect(page.locator('[data-testid="explanation-loading"]')).toBeVisible();
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Verify IG explanation is displayed
    await expect(page.locator('[data-testid="ig-visualization"]')).toBeVisible();
    await expect(page.locator('text=Integrated Gradients')).toBeVisible();
    
    // Verify attribution analysis
    await expect(page.locator('text=Attribution')).toBeVisible();
    await expect(page.locator('[data-testid="attribution-stats"]')).toBeVisible();
    
    // Verify convergence information
    await expect(page.locator('text=Convergence')).toBeVisible();
  });

  test('should compare multiple XAI methods', async ({ page }) => {
    // Navigate to XAI page
    await page.goto('/xai');
    
    // Generate SHAP explanation first
    await page.selectOption('[data-testid="method-selector"]', 'shap');
    const projects = await page.locator('[data-testid="project-selector"] option').count();
    if (projects > 1) {
      await page.selectOption('[data-testid="project-selector"]', { index: 1 });
    }
    await page.click('[data-testid="generate-explanation-btn"]');
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Store first explanation
    const shapExplanationId = await page.locator('[data-testid="explanation-id"]').textContent();
    
    // Generate LIME explanation
    await page.selectOption('[data-testid="method-selector"]', 'lime');
    await page.click('[data-testid="generate-explanation-btn"]');
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Access comparison feature
    await page.click('[data-testid="comparison-tab"]');
    
    // Verify comparison interface
    await expect(page.locator('[data-testid="explanation-comparison"]')).toBeVisible();
    await expect(page.locator('text=Method Comparison')).toBeVisible();
    
    // Verify both explanations are listed
    await expect(page.locator('text=SHAP')).toBeVisible();
    await expect(page.locator('text=LIME')).toBeVisible();
  });

  test('should display real XAI performance metrics', async ({ page }) => {
    // Navigate to XAI page
    await page.goto('/xai');
    
    // Generate any explanation to get metrics
    await page.selectOption('[data-testid="method-selector"]', 'shap');
    const projects = await page.locator('[data-testid="project-selector"] option').count();
    if (projects > 1) {
      await page.selectOption('[data-testid="project-selector"]', { index: 1 });
    }
    await page.click('[data-testid="generate-explanation-btn"]');
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Check for performance metrics
    await expect(page.locator('[data-testid="processing-time"]')).toBeVisible();
    await expect(page.locator('[data-testid="confidence-score"]')).toBeVisible();
    await expect(page.locator('[data-testid="model-info"]')).toBeVisible();
    
    // Verify real computation indicators
    await expect(page.locator('text=Real Models')).toBeVisible();
    await expect(page.locator('text=XAI Libraries')).toBeVisible();
    
    // Check for library version information
    await page.click('[data-testid="technical-details-tab"]');
    await expect(page.locator('text=SHAP')).toBeVisible();
    await expect(page.locator('text=LIME')).toBeVisible();
    await expect(page.locator('text=Captum')).toBeVisible();
  });

  test('should handle XAI explanation errors gracefully', async ({ page }) => {
    // Navigate to XAI page
    await page.goto('/xai');
    
    // Try to generate explanation without selecting project
    await page.selectOption('[data-testid="method-selector"]', 'shap');
    await page.click('[data-testid="generate-explanation-btn"]');
    
    // Verify error handling
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('text=Please select a project')).toBeVisible();
    
    // Verify system remains stable
    await expect(page.locator('[data-testid="method-selector"]')).toBeVisible();
    await expect(page.locator('[data-testid="generate-explanation-btn"]')).toBeVisible();
  });

  test('should export XAI explanation report', async ({ page }) => {
    // Navigate to XAI page and generate explanation
    await page.goto('/xai');
    await page.selectOption('[data-testid="method-selector"]', 'shap');
    const projects = await page.locator('[data-testid="project-selector"] option').count();
    if (projects > 1) {
      await page.selectOption('[data-testid="project-selector"]', { index: 1 });
    }
    await page.click('[data-testid="generate-explanation-btn"]');
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Test PDF export
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="export-pdf-btn"]');
    const download = await downloadPromise;
    
    // Verify download
    expect(download.suggestedFilename()).toContain('xai_explanation');
    expect(download.suggestedFilename()).toContain('.pdf');
    
    // Test JSON export
    await page.click('[data-testid="export-json-btn"]');
    await expect(page.locator('[data-testid="json-export-modal"]')).toBeVisible();
    await expect(page.locator('text=explanation_id')).toBeVisible();
    await expect(page.locator('text=method')).toBeVisible();
    await expect(page.locator('text=confidence_score')).toBeVisible();
  });

  test('should validate real XAI backend integration', async ({ page }) => {
    // Test direct API access through browser network
    await page.goto('/xai');
    
    // Monitor network requests
    const requests = [];
    page.on('request', request => {
      if (request.url().includes('/api/v1/xai/')) {
        requests.push(request);
      }
    });
    
    // Generate explanation to trigger API call
    await page.selectOption('[data-testid="method-selector"]', 'shap');
    const projects = await page.locator('[data-testid="project-selector"] option').count();
    if (projects > 1) {
      await page.selectOption('[data-testid="project-selector"]', { index: 1 });
    }
    await page.click('[data-testid="generate-explanation-btn"]');
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Verify API call was made
    expect(requests.length).toBeGreaterThan(0);
    expect(requests[0].url()).toContain('/api/v1/xai/explain');
    expect(requests[0].method()).toBe('POST');
    
    // Verify response contains real XAI data
    await expect(page.locator('[data-testid="real-computation-badge"]')).toBeVisible();
    await expect(page.locator('text=Production Models')).toBeVisible();
  });

  test('should display regulatory compliance information', async ({ page }) => {
    // Navigate to XAI page and generate explanation
    await page.goto('/xai');
    await page.selectOption('[data-testid="method-selector"]', 'shap');
    const projects = await page.locator('[data-testid="project-selector"] option').count();
    if (projects > 1) {
      await page.selectOption('[data-testid="project-selector"]', { index: 1 });
    }
    await page.click('[data-testid="generate-explanation-btn"]');
    await expect(page.locator('[data-testid="explanation-loading"]')).not.toBeVisible({ timeout: 30000 });
    
    // Check regulatory compliance section
    await page.click('[data-testid="compliance-tab"]');
    await expect(page.locator('[data-testid="regulatory-notes"]')).toBeVisible();
    await expect(page.locator('text=EU AI Act')).toBeVisible();
    await expect(page.locator('text=Compliant')).toBeVisible();
    await expect(page.locator('text=Carbon Standards')).toBeVisible();
    
    // Verify audit trail information
    await expect(page.locator('[data-testid="audit-trail"]')).toBeVisible();
    await expect(page.locator('text=Explanation ID')).toBeVisible();
    await expect(page.locator('text=Timestamp')).toBeVisible();
    await expect(page.locator('text=Model Version')).toBeVisible();
  });

}); 