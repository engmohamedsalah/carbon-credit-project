from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Float, Text, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
import enum

from app.core.database import Base

class ProjectStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    UNDER_VERIFICATION = "under_verification"
    VERIFIED = "verified"
    REJECTED = "rejected"

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    location_name = Column(String)
    # GeoJSON geometry for the project area
    geometry = Column(Geometry('POLYGON', srid=4326))
    # Project metadata
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    area_hectares = Column(Float)
    estimated_carbon_credits = Column(Float)
    project_type = Column(String)  # e.g., reforestation, avoided deforestation
    status = Column(Enum(ProjectStatus), default=ProjectStatus.DRAFT)
    
    # Blockchain related fields
    blockchain_token_id = Column(String, nullable=True)
    blockchain_transaction_hash = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    verifications = relationship("Verification", back_populates="project")
    satellite_images = relationship("SatelliteImage", back_populates="project")
    carbon_estimates = relationship("CarbonEstimate", back_populates="project")
