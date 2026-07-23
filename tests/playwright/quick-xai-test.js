const { chromium } = require('playwright');

/**
 * Quick XAI Test - Direct browser automation
 * This will show the XAI system in action
 */

async function runXaiTest() {
  console.log('🧪 Starting Quick XAI Test...');
  
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  try {
    // Step 1: Navigate to login
    console.log('Step 1: Navigating to login page...');
    await page.goto('http://localhost:3000/login');
    await page.waitForTimeout(2000);
    console.log('✅ Login page loaded');
    
    // Step 2: Login
    console.log('Step 2: Logging in as admin...');
    await page.fill('input[type="email"]', 'testadmin@example.com');
    await page.fill('input[type="password"]', 'changeme');
    await page.click('button[type="submit"]');
    
    // Wait for redirect
    await page.waitForTimeout(3000);
    console.log('✅ Login completed');
    
    // Step 3: Navigate to XAI
    console.log('Step 3: Navigating to XAI page...');
    await page.goto('http://localhost:3000/xai');
    await page.waitForTimeout(5000); // Give time for page and data to load
    
    const currentUrl = page.url();
    console.log(`✅ Current URL: ${currentUrl}`);
    
    // Step 4: Check page content
    console.log('Step 4: Checking page content...');
    const pageTitle = await page.title();
    console.log(`📄 Page title: ${pageTitle}`);
    
    // Look for XAI-related content
    const bodyText = await page.textContent('body');
    const hasXaiContent = bodyText.includes('XAI') || bodyText.includes('Explainable') || bodyText.includes('SHAP');
    console.log(`📝 Has XAI content: ${hasXaiContent}`);
    
    if (hasXaiContent) {
      console.log('✅ XAI page loaded successfully!');
    }
    
    // Step 5: Monitor network requests
    console.log('Step 5: Monitoring XAI API requests...');
    const xaiRequests = [];
    
    page.on('request', request => {
      if (request.url().includes('/api/v1/xai/')) {
        xaiRequests.push({
          method: request.method(),
          url: request.url(),
          timestamp: new Date().toISOString()
        });
        console.log(`🌐 XAI API Request: ${request.method()} ${request.url()}`);
      }
    });
    
    // Step 6: Try to interact with XAI controls
    console.log('Step 6: Looking for XAI controls...');
    
    // Wait for controls to load
    await page.waitForTimeout(3000);
    
    // Look for method selector
    const methodSelector = page.locator('select').first();
    const isMethodVisible = await methodSelector.isVisible().catch(() => false);
    console.log(`🎛️ Method selector visible: ${isMethodVisible}`);
    
    // Look for any buttons
    const buttons = await page.locator('button').allTextContents();
    console.log(`🔘 Found buttons: ${buttons.join(', ')}`);
    
    // Look for any selects
    const selects = await page.locator('select').count();
    console.log(`📋 Found ${selects} select elements`);
    
    // Step 7: Try to trigger XAI generation
    if (selects > 0) {
      console.log('Step 7: Attempting to trigger XAI generation...');
      
      // Try to find and click a generate button
      const generateButtons = await page.locator('button:has-text("Generate"), button:has-text("Explain")').count();
      console.log(`🚀 Found ${generateButtons} potential generate buttons`);
      
      if (generateButtons > 0) {
        try {
          await page.locator('button:has-text("Generate"), button:has-text("Explain")').first().click();
          console.log('✅ Clicked generate button');
          
          // Wait for API request
          await page.waitForTimeout(5000);
          
          if (xaiRequests.length > 0) {
            console.log('🎉 SUCCESS! XAI API requests detected:');
            xaiRequests.forEach((req, index) => {
              console.log(`  ${index + 1}. ${req.method} ${req.url}`);
            });
          } else {
            console.log('⚠️ No XAI API requests detected yet');
          }
          
        } catch (error) {
          console.log('⚠️ Could not click generate button:', error.message);
        }
      }
    }
    
    // Step 8: Take screenshot
    console.log('Step 8: Taking screenshot...');
    await page.screenshot({ path: 'xai-test-screenshot.png', fullPage: true });
    console.log('📸 Screenshot saved: xai-test-screenshot.png');
    
    // Step 9: Final summary
    console.log('Step 9: Test summary...');
    console.log(`📊 Total XAI API requests: ${xaiRequests.length}`);
    console.log(`📍 Final URL: ${page.url()}`);
    console.log(`✅ XAI page accessible: ${page.url().includes('/xai')}`);
    
    if (xaiRequests.length > 0) {
      console.log('🎉 SUCCESS: Real XAI system is working!');
    } else {
      console.log('ℹ️ XAI system accessible but no API calls triggered');
    }
    
  } catch (error) {
    console.error('❌ Test error:', error.message);
  } finally {
    await browser.close();
  }
  
  console.log('🎯 Quick XAI Test Complete!');
}

// Run the test
runXaiTest().catch(console.error); 