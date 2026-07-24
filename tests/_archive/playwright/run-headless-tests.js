#!/usr/bin/env node

/**
 * Headless Test Runner for Carbon Credit Verification
 * Runs all tests in background without opening browser windows
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Starting Carbon Credit Verification Tests (HEADLESS MODE)');
console.log('📍 No browser windows will open during testing');
console.log('⏰ Test started at:', new Date().toLocaleString());
console.log('============================================================\n');

// Ensure we're in the right directory
process.chdir(path.join(__dirname));

// Run Playwright tests with explicit headless flags
const testProcess = spawn('npx', [
  'playwright', 
  'test',
  '--project=chromium',  // Run only Chrome for speed
  '--reporter=list',  // Simple console output
  '--max-failures=5'  // Stop after 5 failures
], {
  stdio: 'inherit',
  env: {
    ...process.env,
    HEADLESS: 'true',
    CI: 'true'  // Force CI mode for more stable testing
  }
});

testProcess.on('close', (code) => {
  console.log('\n============================================================');
  if (code === 0) {
    console.log('✅ All tests completed successfully!');
  } else {
    console.log(`❌ Tests failed with exit code: ${code}`);
  }
  console.log('⏰ Test completed at:', new Date().toLocaleString());
  process.exit(code);
});

testProcess.on('error', (error) => {
  console.error('❌ Failed to start test process:', error);
  process.exit(1);
}); 