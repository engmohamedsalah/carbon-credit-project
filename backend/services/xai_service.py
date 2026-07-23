"""
Explainability (XAI) service — DISABLED in this build.

Honest state: genuine model explanations require running an attribution method
(e.g. Integrated Gradients) on the real input imagery of a specific prediction.
The app does not yet persist a per-prediction input tensor to explain, so no real
explanation can be produced. Rather than fabricate one, every method here returns
an explicit "disabled" result and the API surfaces it as unavailable.

The genuine attribution code is preserved (dormant) in ``real_xai_service.py`` for a
future pass that wires it to real per-prediction inputs.
"""
from typing import Any, Dict, List, Optional

XAI_AVAILABLE = False
_DISABLED = {
    "error": "Explainability is not available in this build",
    "disabled": True,
}


class DisabledXAIService:
    """No-op XAI service that never returns fabricated explanations."""

    async def generate_explanation(self, *args, **kwargs) -> Dict[str, Any]:
        return dict(_DISABLED)

    async def get_explanation(self, explanation_id: str) -> Optional[Dict[str, Any]]:
        return None

    async def compare_explanations(self, explanation_ids: List[str]) -> Dict[str, Any]:
        return dict(_DISABLED)

    async def generate_report(self, *args, **kwargs) -> Dict[str, Any]:
        return dict(_DISABLED)


xai_service = DisabledXAIService()
