const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

/**
 * Debug Login Page - Check what selectors are available
 * Now also records a video of the session.
 */

async function debugLogin() {
  console.log('🔍 Debugging login page...');

  // Ensure video output directory exists
  const videoDir = path.resolve(process.cwd(), 'tests/playwright/test-results/videos');
  fs.mkdirSync(videoDir, { recursive: true });

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({
    recordVideo: { dir: videoDir, size: { width: 1280, height: 720 } }
  });
  const page = await context.newPage();

  // Keep a handle to the video artifact
  let videoHandle = null;
  try {
    // Go to login page
    console.log('📄 Navigating to login page...');
    await page.goto('http://localhost:3000/login');
    await page.waitForTimeout(3000);

    // Take screenshot
    const screenshotPath = path.resolve(process.cwd(), 'debug-login-page.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log('📸 Screenshot saved:', screenshotPath);

    // Capture video handle for later
    videoHandle = page.video ? page.video() : null;

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
    console.log('⏸️ Pausing for 5 seconds so you can see the login page...');
    await page.waitForTimeout(5000);

    // Try full login flow by first registering a test user
    console.log('🧪 Registering a test user...');
    const unique = Date.now();
    const testEmail = `playwright_${unique}@example.com`;
    const testPassword = 'testpass123';

    await page.goto('http://localhost:3000/register');
    await page.fill('input[name="fullName"]', 'PW User');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.fill('input[name="confirmPassword"]', testPassword);
    await page.click('button[type="submit"]');

    // Wait for redirect back to login
    await page.waitForURL('**/login', { timeout: 20000 });
    console.log('✅ Registration complete, now logging in...');

    // Login with the created user
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');

    // Wait for dashboard
    await page.waitForURL('**/dashboard', { timeout: 20000 });
    console.log('🎉 Login successful, dashboard loaded');

    const postLoginShot = path.resolve(process.cwd(), 'debug-post-login.png');
    await page.screenshot({ path: postLoginShot, fullPage: true });
    console.log('📸 Post-login screenshot saved:', postLoginShot);
  } catch (error) {
    console.error('❌ Debug error:', error.message);
  } finally {
    // Close page/context to finalize video
    try { await page.close(); } catch {}
    let videoPath = null;
    try { await context.close(); } catch {}
    try {
      if (videoHandle) {
        videoPath = await videoHandle.path();
      }
    } catch {}
    try { await browser.close(); } catch {}

    if (videoPath) {
      console.log('🎥 Video saved at:', videoPath);
    } else {
      console.log('🎥 Video directory:', videoDir, '(check for latest .webm)');
    }
  }

  console.log('🎯 Debug complete!');
}

// Run the debug
debugLogin().catch(console.error);
