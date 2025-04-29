from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.models.verification import VerificationStatus

# Verification schemas
class VerificationBase(BaseModel):
    verification_notes: Optional[str] = None
    verified_carbon_credits: Optional[float] = None
    confidence_score: Optional[float] = None

class VerificationCreate(VerificationBase):
    project_id: int

class VerificationUpdate(VerificationBase):
    status: Optional[VerificationStatus] = None
    human_reviewed: Optional[bool] = None
    human_review_notes: Optional[str] = None

class VerificationInDBBase(VerificationBase):
    id: int
    status: VerificationStatus
    verification_date: Optional[datetime] = None
    explanation_data: Optional[Dict[str, Any]] = None
    blockchain_transaction_hash: Optional[str] = None
    blockchain_timestamp: Optional[datetime] = None
    human_reviewed: bool
    human_review_notes: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    project_id: int
    verifier_id: Optional[int] = None

    class Config:
        orm_mode = True

class Verification(VerificationInDBBase):
    pass

class VerificationWithDetails(Verification):
    project: Dict[str, Any]
    verifier: Optional[Dict[str, Any]] = None
    satellite_analyses: List[Dict[str, Any]] = []
