from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import datetime

from app.models.verification import Verification, VerificationStatus
from app.schemas.verification import VerificationCreate, VerificationUpdate
from app.services.base import CRUDBase
from app.models.user import User
from app.models.project import Project, ProjectStatus
from app.services import project_service

class VerificationService:
    def create_verification(
        self, db: Session, *, obj_in: VerificationCreate, verifier_id: int
    ) -> Verification:
        """Create a new verification for a project"""
        # Check if project exists
        project = db.query(Project).filter(Project.id == obj_in.project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Update project status
        project_service.update_status(db, db_obj=project, status=ProjectStatus.UNDER_VERIFICATION)
        
        # Create verification
        db_obj = Verification(
            status=VerificationStatus.PENDING,
            verification_notes=obj_in.verification_notes,
            verified_carbon_credits=obj_in.verified_carbon_credits,
            confidence_score=obj_in.confidence_score,
            project_id=obj_in.project_id,
            verifier_id=verifier_id,
            human_reviewed=False
        )
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get(self, db: Session, id: int) -> Optional[Verification]:
        """Get verification by ID"""
        return db.query(Verification).filter(Verification.id == id).first()
    
    def get_multi(
        self, db: Session, skip: int = 0, limit: int = 100, 
        project_id: Optional[int] = None, status: Optional[str] = None
    ) -> List[Verification]:
        """Get multiple verifications with optional filters"""
        query = db.query(Verification)
        
        if project_id:
            query = query.filter(Verification.project_id == project_id)
        
        if status:
            query = query.filter(Verification.status == status)
        
        return query.offset(skip).limit(limit).all()
    
    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100,
        project_id: Optional[int] = None, status: Optional[str] = None
    ) -> List[Verification]:
        """Get verifications for projects owned by a specific user"""
        query = db.query(Verification).join(Project).filter(Project.owner_id == owner_id)
        
        if project_id:
            query = query.filter(Verification.project_id == project_id)
        
        if status:
            query = query.filter(Verification.status == status)
        
        return query.offset(skip).limit(limit).all()
    
    def update(
        self, db: Session, *, db_obj: Verification, obj_in: VerificationUpdate
    ) -> Verification:
        """Update a verification"""
        update_data = obj_in.dict(exclude_unset=True)
        
        for field in update_data:
            setattr(db_obj, field, update_data[field])
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def submit_human_review(
        self, db: Session, *, verification: Verification, 
        approved: bool, notes: Optional[str], reviewer_id: int
    ) -> Verification:
        """Submit human review for a verification"""
        # Update verification
        verification.human_reviewed = True
        verification.human_review_notes = notes
        
        if approved:
            verification.status = VerificationStatus.APPROVED
            # Update project status
            project = db.query(Project).filter(Project.id == verification.project_id).first()
            project_service.update_status(db, db_obj=project, status=ProjectStatus.VERIFIED)
        else:
            verification.status = VerificationStatus.REJECTED
            # Update project status
            project = db.query(Project).filter(Project.id == verification.project_id).first()
            project_service.update_status(db, db_obj=project, status=ProjectStatus.REJECTED)
        
        db.add(verification)
        db.commit()
        db.refresh(verification)
        return verification
    
    def can_access_verification(self, db: Session, verification: Verification, user: User) -> bool:
        """Check if user can access a verification"""
        # Admins and verifiers can access all verifications
        if user.role in ["admin", "verifier"]:
            return True
        
        # Project owners can access verifications for their projects
        project = db.query(Project).filter(Project.id == verification.project_id).first()
        return project and project.owner_id == user.id
    
    def can_modify_verification(self, db: Session, verification: Verification, user: User) -> bool:
        """Check if user can modify a verification"""
        # Only admins, the assigned verifier, or verifiers for pending verifications can modify
        if user.role == "admin":
            return True
        
        if user.role == "verifier":
            # Assigned verifier can always modify
            if verification.verifier_id == user.id:
                return True
            
            # Other verifiers can only modify pending verifications
            return verification.status == VerificationStatus.PENDING
        
        return False

verification_service = VerificationService()
