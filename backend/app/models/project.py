"""
Project models and schemas
"""
from typing import Optional
from pydantic import BaseModel, validator
from datetime import datetime
from enum import Enum


class ProjectType(str, Enum):
    """Project type enumeration"""
    REFORESTATION = "Reforestation"
    AFFORESTATION = "Afforestation"
    FOREST_CONSERVATION = "Forest Conservation"
    AGROFORESTRY = "Agroforestry"
    RENEWABLE_ENERGY = "Renewable Energy"
    ENERGY_EFFICIENCY = "Energy Efficiency"


class ProjectStatus(str, Enum):
    """Project status enumeration"""
    PENDING = "Pending"
    UNDER_REVIEW = "Under Review"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    ACTIVE = "Active"
    COMPLETED = "Completed"
    SUSPENDED = "Suspended"


class ProjectBase(BaseModel):
    """Base project model"""
    name: str
    description: Optional[str] = None
    location_name: str
    area_size: float
    project_type: ProjectType = ProjectType.REFORESTATION
    
    @validator('name')
    def validate_name(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Project name must be at least 3 characters long')
        if len(v.strip()) > 200:
            raise ValueError('Project name must be less than 200 characters')
        return v.strip()
    
    @validator('location_name')
    def validate_location(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Location name must be at least 2 characters long')
        if len(v.strip()) > 100:
            raise ValueError('Location name must be less than 100 characters')
        return v.strip()
    
    @validator('area_size')
    def validate_area_size(cls, v):
        if v <= 0:
            raise ValueError('Area size must be greater than 0')
        if v > 1000000:  # 1 million hectares max
            raise ValueError('Area size cannot exceed 1,000,000 hectares')
        return v
    
    @validator('description')
    def validate_description(cls, v):
        if v and len(v.strip()) > 1000:
            raise ValueError('Description must be less than 1000 characters')
        return v.strip() if v else None


class ProjectCreate(ProjectBase):
    """Project creation model"""
    pass


class ProjectUpdate(BaseModel):
    """Project update model"""
    name: Optional[str] = None
    description: Optional[str] = None
    location_name: Optional[str] = None
    area_size: Optional[float] = None
    project_type: Optional[ProjectType] = None
    status: Optional[ProjectStatus] = None
    
    @validator('name')
    def validate_name(cls, v):
        if v is not None:
            if len(v.strip()) < 3:
                raise ValueError('Project name must be at least 3 characters long')
            if len(v.strip()) > 200:
                raise ValueError('Project name must be less than 200 characters')
            return v.strip()
        return v
    
    @validator('location_name')
    def validate_location(cls, v):
        if v is not None:
            if len(v.strip()) < 2:
                raise ValueError('Location name must be at least 2 characters long')
            if len(v.strip()) > 100:
                raise ValueError('Location name must be less than 100 characters')
            return v.strip()
        return v
    
    @validator('area_size')
    def validate_area_size(cls, v):
        if v is not None:
            if v <= 0:
                raise ValueError('Area size must be greater than 0')
            if v > 1000000:
                raise ValueError('Area size cannot exceed 1,000,000 hectares')
        return v


class ProjectResponse(ProjectBase):
    """Project response model"""
    id: int
    status: ProjectStatus = ProjectStatus.PENDING
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    """Project list response with pagination"""
    projects: list[ProjectResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
