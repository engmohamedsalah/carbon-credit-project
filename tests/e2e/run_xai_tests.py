#!/usr/bin/env python3
"""
XAI Test Runner
Simple script to run XAI tests with proper setup
"""

import os
import sys
import subprocess
import time
import requests
import asyncio
from playwright.async_api import async_playwright

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        return response.status_code == 200
    except:
        return False

def check_frontend_health():
    """Check if frontend is running"""
    try:
        response = requests.get("http://localhost:3000")
        return response.status_code == 200
    except:
        return False

async def run_xai_api_tests():
    """Run XAI API tests"""
    print("🧪 Running XAI API Tests...")
    
    # Test authentication
    try:
        auth_response = requests.post("http://localhost:8000/api/v1/auth/login", json={
            "email": "testadmin@example.com",
            "password": "password123"
        })
        
        if auth_response.status_code != 200:
            print("❌ Authentication failed")
            return False
            
        token = auth_response.json()["access_token"]
        print("✅ Authentication successful")
        
        # Test XAI methods endpoint
        methods_response = requests.get(
            "http://localhost:8000/api/v1/xai/methods",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if methods_response.status_code != 200:
            print("❌ XAI methods endpoint failed")
            return False
            
        methods_data = methods_response.json()
        print(f"✅ XAI methods available: {list(methods_data.get('methods', {}).keys())}")
        
        # Test explanation generation
        explanation_data = {
            "model_id": "forest_cover_ensemble",
            "instance_data": {
                "project_id": 1,
                "location": "Test Forest",
                "area_hectares": 100.0
            },
            "explanation_method": "shap",
            "business_friendly": True,
            "include_uncertainty": True
        }
        
        explanation_response = requests.post(
            "http://localhost:8000/api/v1/xai/generate-explanation",
            json=explanation_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if explanation_response.status_code != 200:
            print(f"❌ Explanation generation failed: {explanation_response.text}")
            return False
            
        explanation = explanation_response.json()
        print(f"✅ Explanation generated with ID: {explanation.get('explanation_id')}")
        print(f"   Confidence: {explanation.get('confidence_score', 0):.2f}")
        print(f"   Risk Level: {explanation.get('risk_assessment', {}).get('level', 'Unknown')}")
        
        # Test report generation
        report_data = {
            "explanation_id": explanation["explanation_id"],
            "format": "pdf",
            "include_business_summary": True
        }
        
        report_response = requests.post(
            "http://localhost:8000/api/v1/xai/generate-report",
            json=report_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if report_response.status_code != 200:
            print(f"❌ Report generation failed: {report_response.text}")
            return False
            
        report = report_response.json()
        print(f"✅ Report generated: {report.get('filename')}")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

async def run_xai_ui_tests():
    """Run XAI UI tests with Playwright"""
    print("🎭 Running XAI UI Tests...")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, slow_mo=500)
            context = await browser.new_context(viewport={"width": 1920, "height": 1080})
            page = await context.new_page()
            
            # Enable console logging
            page.on("console", lambda msg: print(f"🖥️  Console: {msg.text}"))
            page.on("pageerror", lambda error: print(f"🚨 Page Error: {error}"))
            
            # Login
            print("🔐 Logging in...")
            await page.goto("http://localhost:3000")
            await page.wait_for_selector('input[type="email"]', timeout=10000)
            
            await page.fill('input[type="email"]', "testadmin@example.com")
            await page.fill('input[type="password"]', "password123")
            await page.click('button[type="submit"]')
            
            # Wait for login
            await page.wait_for_url("http://localhost:3000/dashboard", timeout=10000)
            print("✅ Login successful")
            
            # Navigate to XAI page
            print("🧠 Navigating to XAI page...")
            await page.goto("http://localhost:3000/xai")
            await page.wait_for_load_state("networkidle")
            
            # Check page elements
            title = await page.title()
            print(f"📄 Page title: {title}")
            
            # Check for tabs
            tabs = await page.query_selector_all('[role="tab"]')
            print(f"📑 Found {len(tabs)} tabs")
            
            if len(tabs) > 0:
                # Click first tab
                await tabs[0].click()
                await page.wait_for_timeout(1000)
                
                # Look for form elements
                selects = await page.query_selector_all('select, [role="combobox"]')
                inputs = await page.query_selector_all('input')
                buttons = await page.query_selector_all('button')
                
                print(f"🎛️  Form elements: {len(selects)} selects, {len(inputs)} inputs, {len(buttons)} buttons")
                
                # Try to find generate button
                for button in buttons:
                    text = await button.inner_text()
                    if "generate" in text.lower():
                        print(f"🎯 Found generate button: '{text}'")
                        break
            
            print("✅ UI navigation successful")
            
            await context.close()
            await browser.close()
            
            return True
            
    except Exception as e:
        print(f"❌ UI test failed: {str(e)}")
        return False

def main():
    """Main test runner"""
    print("🚀 Starting XAI Test Suite")
    print("=" * 50)
    
    # Check if servers are running
    print("🔍 Checking server status...")
    
    if not check_backend_health():
        print("❌ Backend not running on http://localhost:8000")
        print("   Please start the backend with: cd backend && python main.py")
        return False
        
    if not check_frontend_health():
        print("❌ Frontend not running on http://localhost:3000")
        print("   Please start the frontend with: cd frontend && npm start")
        return False
        
    print("✅ Both servers are running")
    print()
    
    # Run API tests
    api_success = asyncio.run(run_xai_api_tests())
    print()
    
    # Run UI tests
    ui_success = asyncio.run(run_xai_ui_tests())
    print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    print(f"   API Tests: {'✅ PASSED' if api_success else '❌ FAILED'}")
    print(f"   UI Tests:  {'✅ PASSED' if ui_success else '❌ FAILED'}")
    
    if api_success and ui_success:
        print("🎉 All XAI tests passed!")
        return True
    else:
        print("💥 Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 