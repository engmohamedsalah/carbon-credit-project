"""Lightweight implementation validation for CI: key source files are present.

A cheap structural gate — catches accidental removal of the pieces the app depends on.
Runs as `python tests/validate_implementation.py` and under pytest.
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED = [
    "backend/main.py",
    "backend/config.py",
    "backend/dbdriver.py",
    "backend/services/report_service.py",
    "ml/inference/production_inference.py",
    "ml/inference/ensemble_model.py",
    "ml/acquisition/sentinel2.py",
    "ml/analyze.py",
    "frontend/src/App.js",
    "blockchain/contracts/CarbonCreditNFT.sol",
]


def test_required_files_exist():
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(_ROOT, f))]
    assert not missing, f"missing required files: {missing}"


if __name__ == "__main__":
    test_required_files_exist()
    print("implementation validation passed")
