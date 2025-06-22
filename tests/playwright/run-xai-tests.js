#!/usr/bin/env node

/**
 * XAI Integration Test Runner
 * Runs Playwright tests for the real XAI integration
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🧪 Starting Real XAI Integration Tests...\n');

// Check if backend and frontend are running
async function checkServices() {
  console.log('🔍 Checking if services are running...');
  
  try {
    const http = require('http');
    
    // Check backend
    await new Promise((resolve, reject) => {
      const req = http.get('http://localhost:8000/docs', (res) => {
        console.log('✅ Backend is running on port 8000');
        resolve();
      });
      req.on('error', () => {
        console.log('❌ Backend not running. Please start with: cd backend && python main.py');
        reject();
      });
      req.setTimeout(2000, () => {
        req.destroy();
        reject();
      });
    });
    
    // Check frontend
    await new Promise((resolve, reject) => {
      const req = http.get('http://localhost:3000', (res) => {
        console.log('✅ Frontend is running on port 3000');
        resolve();
      });
      req.on('error', () => {
        console.log('❌ Frontend not running. Please start with: cd frontend && npm start');
        reject();
      });
      req.setTimeout(2000, () => {
        req.destroy();
        reject();
      });
    });
    
  } catch (error) {
    console.log('\n❌ Services not running. Please start both backend and frontend first.');
    process.exit(1);
  }
}

// Run the tests
async function runTests() {
  try {
    await checkServices();
    
    console.log('\n🚀 Running XAI Integration Tests...\n');
    
    const testProcess = spawn('npx', [
      'playwright', 
      'test', 
      '--config=tests/playwright/playwright.config.js',
      'tests/playwright/tests/real-xai-integration.spec.js',
      '--reporter=html',
      '--headed'  // Run in headed mode to see the tests
    ], {
      stdio: 'inherit',
      cwd: process.cwd()
    });
    
    testProcess.on('close', (code) => {
      if (code === 0) {
        console.log('\n✅ All XAI integration tests passed!');
        console.log('📊 Test report: playwright-report/index.html');
      } else {
        console.log('\n❌ Some tests failed. Check the report for details.');
        console.log('📊 Test report: playwright-report/index.html');
      }
      process.exit(code);
    });
    
  } catch (error) {
    console.error('❌ Test execution failed:', error);
    process.exit(1);
  }
}

// Handle command line arguments
const args = process.argv.slice(2);

if (args.includes('--help') || args.includes('-h')) {
  console.log(`
🧪 XAI Integration Test Runner

Usage: node run-xai-tests.js [options]

Options:
  --help, -h     Show this help message
  --headless     Run tests in headless mode (default: headed)
  --debug        Run with debug output
  --single       Run a single test by name

Examples:
  node run-xai-tests.js                    # Run all tests (headed)
  node run-xai-tests.js --headless         # Run all tests (headless)
  node run-xai-tests.js --debug            # Run with debug output

Prerequisites:
  - Backend running on http://localhost:8000
  - Frontend running on http://localhost:3000
  - Admin user: testadmin@example.com / password123
`);
  process.exit(0);
}

// Start the test execution
runTests(); 