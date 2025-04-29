from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any, List, Optional

from app.core.database import get_db
from app.schemas.verification import Verification, VerificationCreate, VerificationUpdate, VerificationWithDetails
from app.services import verification_service, auth_service

router = APIRouter()

@router.post("/", response_model=Verification)
def create_verification(
    *,
    db: Session = Depends(get_db),
    verification_in: VerificationCreate,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Create a new verification for a project.
    """
    # Check if user is a verifier or admin
    if not (auth_service.is_verifier(current_user) or auth_service.is_admin(current_user)):
        raise HTTPException(
            status_code=403, 
            detail="Only verifiers and admins can create verifications"
        )
    
    return verification_service.create_verification(
        db=db, 
        obj_in=verification_in, 
        verifier_id=current_user.id
    )

@router.get("/", response_model=List[Verification])
def get_verifications(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    project_id: Optional[int] = None,
    status: Optional[str] = None,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Retrieve verifications.
    """
    if auth_service.is_admin(current_user) or auth_service.is_verifier(current_user):
        return verification_service.get_multi(
            db, skip=skip, limit=limit, project_id=project_id, status=status
        )
    else:
        # Regular users can only see verifications for their own projects
        return verification_service.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit, 
            project_id=project_id, status=status
        )

@router.get("/{id}", response_model=VerificationWithDetails)
def get_verification(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Get verification by ID.
    """
    verification = verification_service.get(db=db, id=id)
    if not verification:
        raise HTTPException(status_code=404, detail="Verification not found")
    
    # Check permissions
    if not verification_service.can_access_verification(db, verification, current_user):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return verification

@router.put("/{id}", response_model=Verification)
def update_verification(
    *,
    db: Session = Depends(get_db),
    id: int,
    verification_in: VerificationUpdate,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Update a verification.
    """
    verification = verification_service.get(db=db, id=id)
    if not verification:
        raise HTTPException(status_code=404, detail="Verification not found")
    
    # Check permissions
    if not verification_service.can_modify_verification(db, verification, current_user):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return verification_service.update(db=db, db_obj=verification, obj_in=verification_in)

@router.post("/{id}/human-review", response_model=Verification)
def submit_human_review(
    *,
    db: Session = Depends(get_db),
    id: int,
    approved: bool,
    notes: Optional[str] = None,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Submit human review for a verification.
    """
    # Check if user is a verifier or admin
    if not (auth_service.is_verifier(current_user) or auth_service.is_admin(current_user)):
        raise HTTPException(
            status_code=403, 
            detail="Only verifiers and admins can submit reviews"
        )
    
    verification = verification_service.get(db=db, id=id)
    if not verification:
        raise HTTPException(status_code=404, detail="Verification not found")
    
    return verification_service.submit_human_review(
        db=db, 
        verification=verification, 
        approved=approved, 
        notes=notes,
        reviewer_id=current_user.id
    )
