"""Satellite-imagery acquisition for a project boundary (model-ready stacks)."""
from ml.acquisition.sentinel2 import fetch_sentinel2_stack, TRAINING_BAND_ORDER

__all__ = ["fetch_sentinel2_stack", "TRAINING_BAND_ORDER"]
