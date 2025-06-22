#!/usr/bin/env python3

"""
Headless E2E Test Runner for Carbon Credit Verification
Runs all E2E tests in background without opening browser windows
"""

import subprocess
import sys
import os
from datetime import datetime

def main():
    print('🚀 Starting Carbon Credit E2E Tests (HEADLESS MODE)')
    print('📍 No browser windows will open during testing')
    print(f'⏰ Test started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('============================================================\n')
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Set environment variables for headless mode
    env = os.environ.copy()
    env.update({
        'HEADLESS': 'true',
        'BROWSER': 'none',
        'DISPLAY': ':99' if sys.platform.startswith('linux') else env.get('DISPLAY', ''),
        'PYTEST_CURRENT_TEST': 'headless_mode'
    })
    
    # Run pytest with headless configuration
    cmd = [
        'python', '-m', 'pytest',
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--maxfail=5',  # Stop after 5 failures
        '--disable-warnings',  # Reduce noise
        'test_authentication.py',  # Start with auth tests
        'test_user_workflow.py',   # Then user workflow
        '--asyncio-mode=auto'  # Handle async tests properly
    ]
    
    try:
        result = subprocess.run(cmd, env=env, check=False)
        
        print('\n============================================================')
        if result.returncode == 0:
            print('✅ All E2E tests completed successfully!')
        else:
            print(f'❌ E2E tests failed with exit code: {result.returncode}')
        print(f'⏰ Test completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        return result.returncode
        
    except KeyboardInterrupt:
        print('\n⚠️  Tests interrupted by user')
        return 1
    except Exception as e:
        print(f'❌ Failed to run tests: {e}')
        return 1

if __name__ == '__main__':
    sys.exit(main()) 