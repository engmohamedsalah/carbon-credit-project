from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Float, Text, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
import enum

from app.core.database import Base

class ImageType(str, enum.Enum):
    SENTINEL_2 = "sentinel_2"
    LANDSAT = "landsat"
    CUSTOM = "custom"

class SatelliteImage(Base):
    __tablename__ = "satellite_images"

    id = Column(Integer, primary_key=True, index=True)
    image_type = Column(Enum(ImageType), default=ImageType.SENTINEL_2)
    acquisition_date = Column(DateTime(timezone=True))
    cloud_cover_percentage = Column(Float)
    image_url = Column(String)  # URL to the stored image
    metadata = Column(JSON)  # Additional metadata
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"))
    
    # Relationships
    project = relationship("Project", back_populates="satellite_images")
    analyses = relationship("SatelliteAnalysis", back_populates="satellite_image")

class AnalysisType(str, enum.Enum):
    LAND_COVER = "land_cover"
    DEFORESTATION = "deforestation"
    CARBON_STOCK = "carbon_stock"
    BIOMASS = "biomass"

class SatelliteAnalysis(Base):
    __tablename__ = "satellite_analyses"

    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(Enum(AnalysisType))
    analysis_date = Column(DateTime(timezone=True))
    result_data = Column(JSON)  # Analysis results
    confidence_score = Column(Float)
    
    # XAI related fields
    explanation_data = Column(JSON)  # Store SHAP/LIME values
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign keys
    satellite_image_id = Column(Integer, ForeignKey("satellite_images.id"))
    verification_id = Column(Integer, ForeignKey("verifications.id"), nullable=True)
    
    # Relationships
    satellite_image = relationship("SatelliteImage", back_populates="analyses")
    verification = relationship("Verification", back_populates="satellite_analyses")

class CarbonEstimate(Base):
    __tablename__ = "carbon_estimates"

    id = Column(Integer, primary_key=True, index=True)
    estimate_date = Column(DateTime(timezone=True))
    carbon_tonnes = Column(Float)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    methodology = Column(String)  # Description of estimation methodology
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"))
    
    # Relationships
    project = relationship("Project", back_populates="carbon_estimates")
