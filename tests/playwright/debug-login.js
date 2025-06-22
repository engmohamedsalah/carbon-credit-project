const { chromium } = require('playwright');

/**
 * Debug Login Page - Check what selectors are available
 */

async function debugLogin() {
  console.log('🔍 Debugging login page...');
  
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  try {
    // Go to login page
    console.log('📄 Navigating to login page...');
    await page.goto('http://localhost:3000/login');
    await page.waitForTimeout(3000);
    
    // Take screenshot
    await page.screenshot({ path: 'debug-login-page.png' });
    console.log('📸 Screenshot saved: debug-login-page.png');
    
    // Check page title
    const title = await page.title();
    console.log(`📋 Page title: ${title}`);
    
    // Check for all input elements
    console.log('🔍 Looking for input elements...');
    const inputs = await page.locator('input').evaluateAll(elements => 
      elements.map(el => ({
        type: el.type,
        name: el.name,
        id: el.id,
        placeholder: el.placeholder,
        className: el.className
      }))
    );
    
    console.log('📝 Found inputs:', inputs);
    
    // Check for buttons
    console.log('🔍 Looking for buttons...');
    const buttons = await page.locator('button').evaluateAll(elements => 
      elements.map(el => ({
        text: el.textContent,
        type: el.type,
        className: el.className
      }))
    );
    
    console.log('🔘 Found buttons:', buttons);
    
    // Check for forms
    const forms = await page.locator('form').count();
    console.log(`📋 Found ${forms} forms`);
    
    // Get page HTML for inspection
    const bodyHTML = await page.locator('body').innerHTML();
    console.log('📄 Body HTML (first 500 chars):', bodyHTML.substring(0, 500));
    
    // Wait so you can see the page
    console.log('⏸️ Pausing for 10 seconds so you can see the page...');
    await page.waitForTimeout(10000);
    
  } catch (error) {
    console.error('❌ Debug error:', error.message);
  } finally {
    await browser.close();
  }
  
  console.log('🎯 Debug complete!');
}

// Run the debug
debugLogin().catch(console.error); 