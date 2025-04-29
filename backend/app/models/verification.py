from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Float, Text, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base

class VerificationStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    HUMAN_REVIEW = "human_review"
    APPROVED = "approved"
    REJECTED = "rejected"

class Verification(Base):
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING)
    verification_date = Column(DateTime(timezone=True))
    verified_carbon_credits = Column(Float)
    confidence_score = Column(Float)  # AI model confidence
    verification_notes = Column(Text)
    
    # XAI related fields
    explanation_data = Column(JSON)  # Store SHAP/LIME values
    
    # Blockchain related fields
    blockchain_transaction_hash = Column(String, nullable=True)
    blockchain_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Human-in-the-loop fields
    human_reviewed = Column(Boolean, default=False)
    human_review_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"))
    verifier_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    project = relationship("Project", back_populates="verifications")
    verifier = relationship("User", back_populates="verifications")
    satellite_analyses = relationship("SatelliteAnalysis", back_populates="verification")
