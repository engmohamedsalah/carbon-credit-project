from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from geojson_pydantic import Polygon
from app.models.project import ProjectStatus

# Project schemas
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    location_name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    area_hectares: Optional[float] = None
    estimated_carbon_credits: Optional[float] = None
    project_type: Optional[str] = None

class ProjectCreate(ProjectBase):
    geometry: Polygon

class ProjectUpdate(ProjectBase):
    geometry: Optional[Polygon] = None
    status: Optional[ProjectStatus] = None

class ProjectInDBBase(ProjectBase):
    id: int
    geometry: Dict[str, Any]  # GeoJSON representation
    status: ProjectStatus
    blockchain_token_id: Optional[str] = None
    blockchain_transaction_hash: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    owner_id: int

    class Config:
        orm_mode = True

class Project(ProjectInDBBase):
    pass

class ProjectWithDetails(Project):
    owner: Dict[str, Any]
    verifications: List[Dict[str, Any]] = []
    carbon_estimates: List[Dict[str, Any]] = []
