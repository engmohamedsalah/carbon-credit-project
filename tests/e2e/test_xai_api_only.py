#!/usr/bin/env python3
"""
XAI API Testing Suite
Simple API-only tests for XAI functionality
"""

import requests
import json
import time
from datetime import datetime

class XAIAPITester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.test_user = {
            "email": "testadmin@example.com",
            "password": "password123"
        }
        self.token = None
        
    def authenticate(self):
        """Authenticate and get token"""
        print("🔐 Authenticating...")
        
        response = requests.post(f"{self.backend_url}/api/v1/auth/login", data={
            "username": self.test_user["email"],
            "password": self.test_user["password"]
        })
        
        if response.status_code != 200:
            print(f"❌ Authentication failed: {response.text}")
            return False
            
        self.token = response.json()["access_token"]
        print("✅ Authentication successful")
        return True
    
    def get_headers(self):
        """Get authorization headers"""
        return {"Authorization": f"Bearer {self.token}"}
    
    def test_xai_methods(self):
        """Test XAI methods endpoint"""
        print("\n🧪 Testing XAI Methods Endpoint...")
        
        response = requests.get(
            f"{self.backend_url}/api/v1/xai/methods",
            headers=self.get_headers()
        )
        
        if response.status_code != 200:
            print(f"❌ XAI methods failed: {response.text}")
            return False
            
        data = response.json()
        print(f"✅ XAI methods retrieved successfully")
        methods = data.get('methods', {})
        if isinstance(methods, dict):
            print(f"   Available methods: {list(methods.keys())}")
        else:
            print(f"   Available methods: {methods}")
        print(f"   Business features: {data.get('business_features', [])}")
        print(f"   Compliance status: {data.get('compliance_status', 'Unknown')}")
        
        return True
    
    def test_explanation_generation(self):
        """Test explanation generation"""
        print("\n🧠 Testing Explanation Generation...")
        
        explanation_data = {
            "model_id": "forest_cover_ensemble",
            "instance_data": {
                "project_id": 1,
                "location": "Test Forest Area",
                "area_hectares": 150.0,
                "forest_type": "Mixed Deciduous",
                "satellite_date": "2024-01-15"
            },
            "explanation_method": "shap",
            "business_friendly": True,
            "include_uncertainty": True
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.backend_url}/api/v1/xai/generate-explanation",
            json=explanation_data,
            headers=self.get_headers()
        )
        end_time = time.time()
        
        if response.status_code != 200:
            print(f"❌ Explanation generation failed: {response.text}")
            return False, None
            
        explanation = response.json()
        
        print(f"✅ Explanation generated successfully")
        print(f"   Explanation ID: {explanation.get('explanation_id')}")
        print(f"   Method: {explanation.get('method', 'Unknown')}")
        print(f"   Confidence: {explanation.get('confidence_score', 0):.3f}")
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        
        # Check business features
        if "business_summary" in explanation:
            print(f"   Business summary: {explanation['business_summary'][:100]}...")
            
        if "risk_assessment" in explanation:
            risk = explanation["risk_assessment"]
            print(f"   Risk level: {risk.get('level', 'Unknown')}")
            
        if "regulatory_notes" in explanation:
            reg = explanation["regulatory_notes"]
            print(f"   EU AI Act compliance: {reg.get('eu_ai_act_compliance', 'Unknown')}")
            
        return True, explanation
    
    def test_report_generation(self, explanation_id):
        """Test report generation"""
        print("\n📄 Testing Report Generation...")
        
        report_data = {
            "explanation_id": explanation_id,
            "format": "pdf",
            "include_business_summary": True
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.backend_url}/api/v1/xai/generate-report",
            json=report_data,
            headers=self.get_headers()
        )
        end_time = time.time()
        
        if response.status_code != 200:
            print(f"❌ Report generation failed: {response.text}")
            return False
            
        report = response.json()
        
        print(f"✅ Report generated successfully")
        print(f"   Report ID: {report.get('report_id')}")
        print(f"   Filename: {report.get('filename')}")
        print(f"   Format: {report.get('format')}")
        print(f"   Generation time: {end_time - start_time:.2f} seconds")
        
        # Check if PDF data is present
        if "data" in report and report["data"]:
            print(f"   PDF data size: {len(report['data'])} characters")
            
        return True
    
    def test_explanation_comparison(self):
        """Test explanation comparison"""
        print("\n🔍 Testing Explanation Comparison...")
        
        # Generate two explanations with different methods
        explanation_ids = []
        
        for method in ["shap", "lime"]:
            explanation_data = {
                "model_id": "forest_cover_ensemble",
                "instance_data": {
                    "project_id": 1,
                    "method_test": method,
                    "comparison_test": True
                },
                "explanation_method": method,
                "business_friendly": True,
                "include_uncertainty": True
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/xai/generate-explanation",
                json=explanation_data,
                headers=self.get_headers()
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to generate {method} explanation")
                continue
                
            explanation_ids.append(response.json()["explanation_id"])
            print(f"   Generated {method.upper()} explanation")
        
        if len(explanation_ids) < 2:
            print("❌ Could not generate enough explanations for comparison")
            return False
        
        # Compare explanations
        comparison_data = {
            "explanation_ids": explanation_ids,
            "comparison_type": "side_by_side"
        }
        
        response = requests.post(
            f"{self.backend_url}/api/v1/xai/compare-explanations",
            json=comparison_data,
            headers=self.get_headers()
        )
        
        if response.status_code != 200:
            print(f"❌ Comparison failed: {response.text}")
            return False
            
        comparison = response.json()
        
        print(f"✅ Comparison completed successfully")
        print(f"   Comparison ID: {comparison.get('comparison_id')}")
        print(f"   Explanations compared: {len(comparison.get('explanations', []))}")
        print(f"   Analysis provided: {'Yes' if 'analysis' in comparison else 'No'}")
        print(f"   Recommendations: {len(comparison.get('recommendations', []))}")
        
        return True
    
    def test_explanation_history(self):
        """Test explanation history"""
        print("\n📚 Testing Explanation History...")
        
        response = requests.get(
            f"{self.backend_url}/api/v1/xai/explanation-history/1",
            headers=self.get_headers()
        )
        
        if response.status_code != 200:
            print(f"❌ History retrieval failed: {response.text}")
            return False
            
        history = response.json()
        
        print(f"✅ History retrieved successfully")
        print(f"   Project ID: {history.get('project_id')}")
        print(f"   Total explanations: {history.get('total_count', 0)}")
        print(f"   Explanations in response: {len(history.get('explanations', []))}")
        
        return True
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n🚨 Testing Error Handling...")
        
        # Test invalid data
        invalid_data = {
            "model_id": "",
            "instance_data": {},
            "explanation_method": "invalid_method"
        }
        
        response = requests.post(
            f"{self.backend_url}/api/v1/xai/generate-explanation",
            json=invalid_data,
            headers=self.get_headers()
        )
        
        if response.status_code == 200:
            print("❌ Error handling failed - should have rejected invalid data")
            return False
            
        print(f"✅ Error handling working - rejected invalid data (status: {response.status_code})")
        
        # Test invalid explanation ID
        response = requests.get(
            f"{self.backend_url}/api/v1/xai/explanation/invalid-id-12345",
            headers=self.get_headers()
        )
        
        if response.status_code == 200:
            print("❌ Error handling failed - should have rejected invalid ID")
            return False
            
        print(f"✅ Error handling working - rejected invalid ID (status: {response.status_code})")
        
        return True
    
    def run_all_tests(self):
        """Run all XAI tests"""
        print("🚀 Starting XAI API Test Suite")
        print("=" * 60)
        
        # Authenticate
        if not self.authenticate():
            return False
        
        # Run tests
        tests = [
            ("XAI Methods", self.test_xai_methods),
            ("Explanation Generation", self.test_explanation_generation),
            ("Explanation Comparison", self.test_explanation_comparison),
            ("Explanation History", self.test_explanation_history),
            ("Error Handling", self.test_error_handling),
        ]
        
        results = {}
        explanation_id = None
        
        for test_name, test_func in tests:
            try:
                if test_name == "Explanation Generation":
                    success, explanation = test_func()
                    results[test_name] = success
                    if success and explanation:
                        explanation_id = explanation.get("explanation_id")
                else:
                    results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {str(e)}")
                results[test_name] = False
        
        # Test report generation if we have an explanation
        if explanation_id:
            try:
                results["Report Generation"] = self.test_report_generation(explanation_id)
            except Exception as e:
                print(f"❌ Report Generation failed with exception: {str(e)}")
                results["Report Generation"] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        print("-" * 40)
        
        passed = 0
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {test_name:<25} {status}")
            if success:
                passed += 1
        
        print("-" * 40)
        print(f"   Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All XAI API tests passed!")
            return True
        else:
            print("💥 Some tests failed")
            return False

def main():
    """Main function"""
    tester = XAIAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 