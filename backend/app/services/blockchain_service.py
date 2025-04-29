from typing import Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from web3 import Web3
from datetime import datetime
import json
import os

from app.core.config import settings
from app.models.verification import Verification, VerificationStatus
from app.models.project import Project

# Smart contract ABI (simplified for demonstration)
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "projectId", "type": "uint256"},
            {"internalType": "uint256", "name": "verificationId", "type": "uint256"},
            {"internalType": "uint256", "name": "carbonCredits", "type": "uint256"},
            {"internalType": "string", "name": "metadataURI", "type": "string"}
        ],
        "name": "mintCarbonCredit",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "name": "getTokenData",
        "outputs": [
            {"internalType": "uint256", "name": "projectId", "type": "uint256"},
            {"internalType": "uint256", "name": "verificationId", "type": "uint256"},
            {"internalType": "uint256", "name": "carbonCredits", "type": "uint256"},
            {"internalType": "string", "name": "metadataURI", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

class BlockchainService:
    def __init__(self):
        """Initialize blockchain connection"""
        self.w3 = Web3(Web3.HTTPProvider(settings.BLOCKCHAIN_PROVIDER_URL))
        
        # Check if we have a contract address
        if settings.CONTRACT_ADDRESS:
            self.contract = self.w3.eth.contract(
                address=settings.CONTRACT_ADDRESS,
                abi=CONTRACT_ABI
            )
        else:
            self.contract = None
    
    def get_verification_for_certification(self, db: Session, verification_id: int) -> Optional[Verification]:
        """Get a verification that is ready for blockchain certification"""
        verification = db.query(Verification).filter(
            Verification.id == verification_id,
            Verification.status == VerificationStatus.APPROVED,
            Verification.human_reviewed == True,
            Verification.blockchain_transaction_hash.is_(None)
        ).first()
        
        return verification
    
    def certify_verification(self, db: Session, verification: Verification) -> Tuple[str, str]:
        """Certify a verification on the blockchain"""
        if not self.contract:
            raise ValueError("Smart contract not configured")
        
        # Get project
        project = db.query(Project).filter(Project.id == verification.project_id).first()
        if not project:
            raise ValueError("Project not found")
        
        # Create metadata
        metadata = {
            "project_name": project.name,
            "project_location": project.location_name,
            "verification_date": verification.verification_date.isoformat() if verification.verification_date else datetime.now().isoformat(),
            "carbon_credits": verification.verified_carbon_credits,
            "confidence_score": verification.confidence_score,
            "verification_notes": verification.verification_notes,
            "human_review_notes": verification.human_review_notes
        }
        
        # In a real implementation, we would store this metadata on IPFS or similar
        # For demo purposes, we'll just create a local file
        os.makedirs("blockchain_metadata", exist_ok=True)
        metadata_filename = f"blockchain_metadata/verification_{verification.id}.json"
        
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f)
        
        # In a real implementation, this would be the IPFS URI
        metadata_uri = f"file://{metadata_filename}"
        
        # Prepare transaction
        # In a real implementation, we would sign and send this transaction
        # For demo purposes, we'll just simulate it
        tx_hash = f"0x{os.urandom(32).hex()}"
        token_id = f"{project.id}{verification.id}"
        
        # Update verification with blockchain info
        verification.blockchain_transaction_hash = tx_hash
        verification.blockchain_timestamp = datetime.now()
        
        # Update project with blockchain info
        project.blockchain_token_id = token_id
        project.blockchain_transaction_hash = tx_hash
        
        db.add(verification)
        db.add(project)
        db.commit()
        db.refresh(verification)
        db.refresh(project)
        
        return tx_hash, token_id
    
    def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get the status of a blockchain transaction"""
        # In a real implementation, we would check the actual transaction status
        # For demo purposes, we'll just return a simulated status
        return {
            "transaction_hash": tx_hash,
            "status": "confirmed",
            "block_number": 12345678,
            "timestamp": datetime.now().isoformat()
        }
    
    def verify_token(self, token_id: str) -> Dict[str, Any]:
        """Verify a token on the blockchain"""
        # In a real implementation, we would call the smart contract
        # For demo purposes, we'll just return simulated data
        
        # Try to parse project_id and verification_id from token_id
        try:
            project_id = int(token_id[:-1])
            verification_id = int(token_id[-1])
        except:
            project_id = 0
            verification_id = 0
        
        return {
            "token_id": token_id,
            "project_id": project_id,
            "verification_id": verification_id,
            "carbon_credits": 100,
            "metadata_uri": f"file://blockchain_metadata/verification_{verification_id}.json",
            "is_valid": True
        }

blockchain_service = BlockchainService()
