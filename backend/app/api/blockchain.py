from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional
from web3 import Web3

from app.core.database import get_db
from app.core.config import settings
from app.services import blockchain_service, auth_service

router = APIRouter()

@router.post("/certify/{verification_id}")
def certify_verification(
    *,
    db: Session = Depends(get_db),
    verification_id: int,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Dict[str, Any]:
    """
    Certify a verification on the blockchain.
    """
    # Check if user is a verifier or admin
    if not (auth_service.is_verifier(current_user) or auth_service.is_admin(current_user)):
        raise HTTPException(
            status_code=403, 
            detail="Only verifiers and admins can certify verifications"
        )
    
    # Get verification and check if it's in the right state
    verification = blockchain_service.get_verification_for_certification(db, verification_id)
    if not verification:
        raise HTTPException(status_code=404, detail="Verification not found or not ready for certification")
    
    # Certify on blockchain
    try:
        tx_hash, token_id = blockchain_service.certify_verification(db, verification)
        return {
            "success": True,
            "transaction_hash": tx_hash,
            "token_id": token_id,
            "message": "Verification successfully certified on blockchain"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Blockchain certification failed: {str(e)}"
        )

@router.get("/transaction/{tx_hash}")
def get_transaction_status(
    *,
    tx_hash: str,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Dict[str, Any]:
    """
    Get the status of a blockchain transaction.
    """
    try:
        status = blockchain_service.get_transaction_status(tx_hash)
        return status
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get transaction status: {str(e)}"
        )

@router.get("/verify/{token_id}")
def verify_token(
    *,
    token_id: str,
) -> Dict[str, Any]:
    """
    Verify a token on the blockchain (public endpoint).
    """
    try:
        token_data = blockchain_service.verify_token(token_id)
        return token_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify token: {str(e)}"
        )
