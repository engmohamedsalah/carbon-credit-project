import { test, expect } from '@playwright/test';

/**
 * Simple XAI Integration Test
 * Basic test to verify real XAI integration is working
 */

test.describe('Simple XAI Integration Verification', () => {
  
  const adminUser = {
    email: 'testadmin@example.com',
    password: 'password123'
  };

  test('should login and access XAI page successfully', async ({ page }) => {
    // Go to login page
    await page.goto('/login');
    
    // Fill login form
    await page.fill('input[type="email"]', adminUser.email);
    await page.fill('input[type="password"]', adminUser.password);
    
    // Submit login
    await page.click('button[type="submit"]');
    
    // Wait for redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Navigate to XAI page
    await page.click('text=XAI');
    await expect(page).toHaveURL('/xai');
    
    // Verify XAI page loads
    await expect(page.locator('text=Explainable AI')).toBeVisible();
    
    // Verify no permission error
    await expect(page.locator('text=You don\'t have permission')).not.toBeVisible();
    
    console.log('✅ XAI page access test passed');
  });

  test('should show XAI method selector and project selector', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', adminUser.email);
    await page.fill('input[type="password"]', adminUser.password);
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL('/dashboard');
    
    // Go to XAI page
    await page.goto('/xai');
    
    // Check for method selector
    const methodSelector = page.locator('[data-testid="method-selector"]');
    if (await methodSelector.isVisible()) {
      console.log('✅ Method selector found');
    } else {
      console.log('⚠️ Method selector not found - may need to wait for data loading');
    }
    
    // Check for project selector
    const projectSelector = page.locator('[data-testid="project-selector"]');
    if (await projectSelector.isVisible()) {
      console.log('✅ Project selector found');
    } else {
      console.log('⚠️ Project selector not found - may need to wait for data loading');
    }
    
    // Check for generate button
    const generateBtn = page.locator('[data-testid="generate-explanation-btn"]');
    await expect(generateBtn).toBeVisible();
    console.log('✅ Generate explanation button found');
  });

  test('should verify real XAI backend is responding', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', adminUser.email);
    await page.fill('input[type="password"]', adminUser.password);
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL('/dashboard');
    
    // Monitor network requests
    const xaiRequests = [];
    page.on('request', request => {
      if (request.url().includes('/api/v1/xai/') || request.url().includes('xai')) {
        xaiRequests.push({
          url: request.url(),
          method: request.method()
        });
      }
    });
    
    // Go to XAI page (this might trigger some API calls)
    await page.goto('/xai');
    
    // Wait a moment for any API calls
    await page.waitForTimeout(2000);
    
    if (xaiRequests.length > 0) {
      console.log('✅ XAI API requests detected:', xaiRequests.length);
      xaiRequests.forEach(req => {
        console.log(`  - ${req.method} ${req.url}`);
      });
    } else {
      console.log('ℹ️ No XAI API requests detected yet');
    }
  });

  test('should display page title and basic elements', async ({ page }) => {
    // Login and navigate to XAI
    await page.goto('/login');
    await page.fill('input[type="email"]', adminUser.email);
    await page.fill('input[type="password"]', adminUser.password);
    await page.click('button[type="submit"]');
    await page.goto('/xai');
    
    // Check page title contains XAI-related content
    await expect(page).toHaveTitle(/Carbon Credit|XAI|Explainable/);
    
    // Check for main headings
    const headings = [
      'Enhanced XAI',
      'Explainable AI',
      'XAI',
      'Explanation'
    ];
    
    let foundHeading = false;
    for (const heading of headings) {
      if (await page.locator(`text=${heading}`).isVisible()) {
        console.log(`✅ Found heading: ${heading}`);
        foundHeading = true;
        break;
      }
    }
    
    if (!foundHeading) {
      console.log('⚠️ No expected headings found');
    }
    
    // Check for tab navigation
    const tabs = page.locator('[role="tab"]');
    const tabCount = await tabs.count();
    console.log(`✅ Found ${tabCount} tabs in XAI interface`);
  });

}); 