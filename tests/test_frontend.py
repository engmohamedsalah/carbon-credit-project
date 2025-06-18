"""
Frontend Component Tests
Tests for React components and frontend functionality
"""

import pytest
import json
import subprocess
import time
import requests
from pathlib import Path


class TestFrontendComponents:
    """Test suite for frontend React components"""
    
    def test_package_json_dependencies(self):
        """Test that all required dependencies are present in package.json"""
        frontend_dir = Path(__file__).parent.parent / "frontend"
        package_json_path = frontend_dir / "package.json"
        
        assert package_json_path.exists(), "package.json not found"
        
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
        
        # Check required dependencies
        required_deps = [
            "@mui/material",
            "@mui/icons-material", 
            "@reduxjs/toolkit",
            "react-redux",
            "react-router-dom",
            "axios"
        ]
        
        dependencies = package_data.get("dependencies", {})
        for dep in required_deps:
            assert dep in dependencies, f"Missing required dependency: {dep}"
    
    def test_frontend_build_process(self):
        """Test that frontend builds successfully"""
        frontend_dir = Path(__file__).parent.parent / "frontend"
        
        # Test npm install
        result = subprocess.run(
            ["npm", "install"], 
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"npm install failed: {result.stderr}"
        
        # Test build process
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"npm build failed: {result.stderr}"
        
        # Check that build directory exists
        build_dir = frontend_dir / "build"
        assert build_dir.exists(), "Build directory not created"
        assert (build_dir / "index.html").exists(), "index.html not found in build"
    
    def test_component_imports(self):
        """Test that all custom components exist and are properly structured"""
        frontend_dir = Path(__file__).parent.parent / "frontend" / "src"
        
        # Check required component files exist
        required_components = [
            "components/Layout.js",
            "components/MapComponent.js", 
            "components/MLAnalysis.js",
            "components/ProtectedRoute.js",
            "pages/Dashboard.js",
            "pages/Login.js",
            "pages/NewProject.js",
            "pages/ProjectDetail.js",
            "pages/Register.js",
            "pages/Verification.js",
            "services/apiService.js",
            "services/mlService.js"
        ]
        
        for component_path in required_components:
            full_path = frontend_dir / component_path
            assert full_path.exists(), f"Component file missing: {component_path}"
            
            # Check file is not empty
            assert full_path.stat().st_size > 0, f"Component file is empty: {component_path}"
    
    def test_ml_analysis_component_structure(self):
        """Test MLAnalysis component has proper structure"""
        ml_component_path = Path(__file__).parent.parent / "frontend" / "src" / "components" / "MLAnalysis.js"
        
        with open(ml_component_path, 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            "import React",
            "from '@mui/material'",
            "from '@mui/icons-material'"
        ]
        
        for import_statement in required_imports:
            assert import_statement in content, f"Missing import: {import_statement}"
        
        # Check for key component features
        required_features = [
            "useState",
            "useEffect", 
            "upload",
            "CircularProgress",
            "handleFile",
            "MLAnalysis"
        ]
        
        for feature in required_features:
            assert feature in content, f"Missing feature in MLAnalysis: {feature}"
    
    def test_verification_page_structure(self):
        """Test Verification page component structure"""
        verification_path = Path(__file__).parent.parent / "frontend" / "src" / "pages" / "Verification.js"
        
        with open(verification_path, 'r') as f:
            content = f.read()
        
        # Check for required features
        required_features = [
            "import React",
            "MLAnalysis",
            "useSelector",
            "useLocation",
            "project_id"
        ]
        
        for feature in required_features:
            assert feature in content, f"Missing feature in Verification page: {feature}"
    
    def test_ml_service_structure(self):
        """Test ML service has proper API integration"""
        ml_service_path = Path(__file__).parent.parent / "frontend" / "src" / "services" / "mlService.js"
        
        with open(ml_service_path, 'r') as f:
            content = f.read()
        
        # Check for required API methods
        required_methods = [
            "analyzeLocation",
            "analyzeForestCover", 
            "detectChanges",
            "calculateEligibility",
            "apiService"
        ]
        
        for method in required_methods:
            assert method in content, f"Missing method in ML service: {method}"
    
    def test_redux_store_setup(self):
        """Test Redux store is properly configured"""
        store_path = Path(__file__).parent.parent / "frontend" / "src" / "store" / "index.js"
        
        with open(store_path, 'r') as f:
            content = f.read()
        
        # Check for required Redux setup
        required_redux = [
            "configureStore",
            "authSlice",
            "projectSlice",
            "verificationSlice"
        ]
        
        for redux_item in required_redux:
            assert redux_item in content, f"Missing Redux item: {redux_item}"


class TestFrontendIntegration:
    """Integration tests for frontend with backend"""
    
    @pytest.mark.integration
    def test_frontend_backend_communication(self):
        """Test that frontend can communicate with backend (requires running backend)"""
        # Check if backend is running
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
            backend_running = response.status_code == 200
        except requests.exceptions.RequestException:
            backend_running = False
        
        if not backend_running:
            pytest.skip("Backend not running - skipping integration test")
        
        # Test API endpoints are accessible
        endpoints_to_test = [
            "/api/v1/health",
            "/api/v1/ml/status"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
                # Some endpoints may require auth, so we check they at least respond
                assert response.status_code in [200, 401, 403], f"Endpoint {endpoint} not responding properly"
            except requests.exceptions.RequestException as e:
                pytest.fail(f"Failed to connect to endpoint {endpoint}: {e}")


class TestUIComponentQuality:
    """Test UI component quality and best practices"""
    
    def test_no_console_errors_in_components(self):
        """Test that components don't have obvious console.log statements"""
        frontend_src = Path(__file__).parent.parent / "frontend" / "src"
        
        # Find all JavaScript files
        js_files = list(frontend_src.rglob("*.js"))
        
        for js_file in js_files:
            with open(js_file, 'r') as f:
                content = f.read()
            
            # Check for potential issues
            issues = []
            
            if "console.log(" in content:
                issues.append("Contains console.log statements")
            
            if "debugger;" in content:
                issues.append("Contains debugger statements")
            
            # Allow console.error and console.warn for legitimate error handling
            if issues:
                print(f"Warning in {js_file.relative_to(frontend_src)}: {', '.join(issues)}")
    
    def test_proper_error_handling_in_components(self):
        """Test that components have proper error handling"""
        ml_analysis_path = Path(__file__).parent.parent / "frontend" / "src" / "components" / "MLAnalysis.js"
        
        with open(ml_analysis_path, 'r') as f:
            content = f.read()
        
        # Check for error handling patterns
        error_handling_patterns = [
            "try {",
            "catch",
            "error",
            "setError"
        ]
        
        found_patterns = sum(1 for pattern in error_handling_patterns if pattern in content)
        assert found_patterns >= 3, "MLAnalysis component should have proper error handling"
    
    def test_accessibility_considerations(self):
        """Test that components consider accessibility"""
        component_files = [
            "components/MLAnalysis.js",
            "pages/Verification.js",
            "pages/Dashboard.js"
        ]
        
        frontend_src = Path(__file__).parent.parent / "frontend" / "src"
        
        for component_file in component_files:
            component_path = frontend_src / component_file
            if component_path.exists():
                with open(component_path, 'r') as f:
                    content = f.read()
                
                # Check for accessibility features
                accessibility_features = [
                    "aria-",
                    "alt=",
                    "title=",
                    "role="
                ]
                
                has_accessibility = any(feature in content for feature in accessibility_features)
                if not has_accessibility:
                    print(f"Warning: {component_file} may lack accessibility features")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
