from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.models.satellite import ImageType, AnalysisType

# Satellite Image schemas
class SatelliteImageBase(BaseModel):
    image_type: ImageType
    acquisition_date: datetime
    cloud_cover_percentage: Optional[float] = None
    image_url: str
    metadata: Optional[Dict[str, Any]] = None

class SatelliteImageCreate(SatelliteImageBase):
    project_id: int

class SatelliteImageUpdate(SatelliteImageBase):
    pass

class SatelliteImageInDBBase(SatelliteImageBase):
    id: int
    created_at: datetime
    project_id: int

    class Config:
        orm_mode = True

class SatelliteImage(SatelliteImageInDBBase):
    pass

# Satellite Analysis schemas
class SatelliteAnalysisBase(BaseModel):
    analysis_type: AnalysisType
    result_data: Dict[str, Any]
    confidence_score: Optional[float] = None
    explanation_data: Optional[Dict[str, Any]] = None

class SatelliteAnalysisCreate(SatelliteAnalysisBase):
    satellite_image_id: int
    verification_id: Optional[int] = None

class SatelliteAnalysisUpdate(SatelliteAnalysisBase):
    pass

class SatelliteAnalysisInDBBase(SatelliteAnalysisBase):
    id: int
    analysis_date: datetime
    created_at: datetime
    satellite_image_id: int
    verification_id: Optional[int] = None

    class Config:
        orm_mode = True

class SatelliteAnalysis(SatelliteAnalysisInDBBase):
    pass

# Carbon Estimate schemas
class CarbonEstimateBase(BaseModel):
    carbon_tonnes: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    methodology: str

class CarbonEstimateCreate(CarbonEstimateBase):
    project_id: int

class CarbonEstimateUpdate(CarbonEstimateBase):
    pass

class CarbonEstimateInDBBase(CarbonEstimateBase):
    id: int
    estimate_date: datetime
    created_at: datetime
    project_id: int

    class Config:
        orm_mode = True

class CarbonEstimate(CarbonEstimateInDBBase):
    pass
