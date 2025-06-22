"""
Real XAI Service for Carbon Credit Verification
Processes REAL data ONLY - NO DEMO OR MOCK DATA
"""

import os
import json
import uuid
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io

# Import ONLY real XAI service - NO FALLBACKS
from .real_xai_service import real_data_xai_service

class RealOnlyXAIService:
    """Real XAI Service - PROCESSES REAL DATA ONLY"""
    
    def __init__(self):
        if real_data_xai_service is None:
            raise Exception("❌ CRITICAL: Real XAI service is required but not available")
        
        self.real_xai_service = real_data_xai_service
        # Expose the explanation cache from the real service
        self.explanation_cache = self.real_xai_service.explanation_cache
        print("🚀 Real-Only XAI Service initialized - NO FALLBACKS, NO DEMOS")
        
    async def generate_explanation(
        self,
        model_id: str,
        instance_data: Dict[str, Any],
        explanation_method: str = "shap",
        business_friendly: bool = True,
        include_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """Generate REAL AI explanation - NO DEMO DATA"""
        
        print(f"🔍 REAL XAI Processing: {explanation_method} for project {instance_data.get('project_id')}")
        
        # Direct call to real service - NO FALLBACKS
        result = await self.real_xai_service.generate_explanation(
            model_id=model_id,
            instance_data=instance_data,
            explanation_method=explanation_method,
            business_friendly=business_friendly,
            include_uncertainty=include_uncertainty
        )
        
        if "error" in result:
            print(f"❌ Real XAI processing failed: {result['error']}")
        else:
            print(f"✅ Real XAI explanation generated: {result.get('explanation_id')}")
        
        return result
    
    async def get_explanation(self, explanation_id: str) -> Optional[Dict[str, Any]]:
        """Get real explanation - NO DEMO DATA"""
        return await self.real_xai_service.get_explanation(explanation_id)
    
    async def compare_explanations(self, explanation_ids: List[str]) -> Dict[str, Any]:
        """Compare real explanations - NO DEMO DATA"""
        return await self.real_xai_service.compare_explanations(explanation_ids)
    
    async def generate_report(
        self, 
        explanation_id: str, 
        format: str = "json",
        include_business_summary: bool = True
    ) -> Dict[str, Any]:
        """Generate real report - NO DEMO DATA"""
        
        explanation = await self.get_explanation(explanation_id)
        if not explanation:
            return {"error": "Explanation not found"}
        
        if format == "json":
            return {
                "report_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "explanation_id": explanation_id,
                "format": "json",
                "data": explanation,
                "summary": explanation.get("business_summary", ""),
                "processing_type": "real_data_analysis"
            }
        else:
            return {"error": f"Format {format} not supported"}

# Create singleton instance - REAL SERVICE ONLY
try:
    xai_service = RealOnlyXAIService()
    print("✅ Real-Only XAI Service instantiated successfully")
except Exception as e:
    print(f"❌ CRITICAL: Failed to instantiate Real-Only XAI Service: {e}")
    raise Exception(f"System requires real XAI service: {e}") 