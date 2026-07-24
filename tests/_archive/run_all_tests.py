#!/usr/bin/env python3
"""
Comprehensive Test Runner
Executes all test suites and generates detailed reports
"""
import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

def run_test_suite(test_file, suite_name):
    """Run a specific test suite and return results"""
    print(f"\n{'='*60}")
    print(f"Running {suite_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse pytest output
        output_lines = result.stdout.split('\n')
        test_results = {
            'suite_name': suite_name,
            'file': test_file,
            'duration': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0
        }
        
        # Count test results from output
        for line in output_lines:
            if 'PASSED' in line:
                test_results['passed'] += 1
            elif 'FAILED' in line:
                test_results['failed'] += 1
            elif 'ERROR' in line:
                test_results['errors'] += 1
            elif 'SKIPPED' in line:
                test_results['skipped'] += 1
        
        # Print summary
        print(f"✅ {suite_name} completed in {duration:.2f}s")
        print(f"   Passed: {test_results['passed']}")
        print(f"   Failed: {test_results['failed']}")
        print(f"   Errors: {test_results['errors']}")
        print(f"   Skipped: {test_results['skipped']}")
        
        if result.returncode != 0:
            print(f"❌ {suite_name} had failures")
            if result.stderr:
                print("Errors:")
                print(result.stderr)
        else:
            print(f"✅ {suite_name} passed successfully")
        
        return test_results
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {suite_name} timed out after 5 minutes")
        return {
            'suite_name': suite_name,
            'file': test_file,
            'duration': 300,
            'return_code': -1,
            'stdout': '',
            'stderr': 'Test suite timed out',
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'skipped': 0
        }
    except Exception as e:
        print(f"💥 {suite_name} failed to run: {e}")
        return {
            'suite_name': suite_name,
            'file': test_file,
            'duration': 0,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'skipped': 0
        }

def generate_test_report(all_results):
    """Generate a comprehensive test report"""
    total_passed = sum(r['passed'] for r in all_results)
    total_failed = sum(r['failed'] for r in all_results)
    total_errors = sum(r['errors'] for r in all_results)
    total_skipped = sum(r['skipped'] for r in all_results)
    total_tests = total_passed + total_failed + total_errors + total_skipped
    total_duration = sum(r['duration'] for r in all_results)
    
    # Calculate success rate
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'errors': total_errors,
            'skipped': total_skipped,
            'success_rate': success_rate,
            'total_duration': total_duration
        },
        'suites': all_results,
        'failed_suites': [r for r in all_results if r['return_code'] != 0],
        'passed_suites': [r for r in all_results if r['return_code'] == 0]
    }
    
    return report

def print_summary_report(report):
    """Print a summary of the test results"""
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Total Duration: {report['summary']['total_duration']:.2f}s")
    print(f"")
    print(f"📈 SUMMARY:")
    print(f"   Total Tests: {report['summary']['total_tests']}")
    print(f"   ✅ Passed: {report['summary']['passed']}")
    print(f"   ❌ Failed: {report['summary']['failed']}")
    print(f"   💥 Errors: {report['summary']['errors']}")
    print(f"   ⏭️  Skipped: {report['summary']['skipped']}")
    print(f"   📊 Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"")
    
    print(f"📋 SUITE RESULTS:")
    for suite in report['suites']:
        status = "✅ PASS" if suite['return_code'] == 0 else "❌ FAIL"
        print(f"   {status} {suite['suite_name']} ({suite['duration']:.2f}s)")
        print(f"      Passed: {suite['passed']}, Failed: {suite['failed']}, Errors: {suite['errors']}")
    
    print(f"")
    if report['failed_suites']:
        print(f"❌ FAILED SUITES:")
        for suite in report['failed_suites']:
            print(f"   - {suite['suite_name']}")
            if suite['stderr']:
                print(f"     Error: {suite['stderr'][:100]}...")
    else:
        print(f"🎉 ALL SUITES PASSED!")
    
    print(f"{'='*80}")

def save_report_to_file(report, filename="test_report.json"):
    """Save the test report to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Test report saved to {filename}")

def main():
    """Main test runner function"""
    print("🚀 Starting Comprehensive Test Suite")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Define test suites
    test_suites = [
        ("tests/test_api_comprehensive.py", "Comprehensive API Tests"),
        ("tests/test_iot_integration.py", "IoT Integration Tests"),
        ("tests/test_performance.py", "Performance Tests"),
    ]
    
    # Check if backend is running
    print("\n🔍 Checking if backend is running...")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("⚠️  Backend responded but not healthy")
    except Exception as e:
        print(f"❌ Backend is not running: {e}")
        print("Please start the backend with: cd backend && python main.py")
        return 1
    
    # Run all test suites
    all_results = []
    
    for test_file, suite_name in test_suites:
        if os.path.exists(test_file):
            result = run_test_suite(test_file, suite_name)
            all_results.append(result)
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    # Generate and display report
    report = generate_test_report(all_results)
    print_summary_report(report)
    
    # Save report
    save_report_to_file(report)
    
    # Return appropriate exit code
    if report['summary']['failed'] > 0 or report['summary']['errors'] > 0:
        print(f"\n❌ Tests completed with failures")
        return 1
    else:
        print(f"\n🎉 All tests passed successfully!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 