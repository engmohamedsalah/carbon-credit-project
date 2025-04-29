from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
import os
import uuid
import aiofiles
from datetime import datetime

from app.models.satellite import SatelliteImage, SatelliteAnalysis, ImageType, AnalysisType
from app.schemas.satellite import SatelliteImageCreate, SatelliteAnalysisCreate
from app.services.base import CRUDBase
from app.models.user import User
from app.models.project import Project
from app.services import auth_service

class SatelliteService:
    def check_project_permissions(self, db: Session, project_id: int, user: User) -> Project:
        """Check if user has permissions to access the project"""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not (auth_service.is_admin(user) or project.owner_id == user.id):
            raise HTTPException(status_code=403, detail="Not enough permissions")
        
        return project
    
    def get_satellite_image(self, db: Session, image_id: int) -> Optional[SatelliteImage]:
        """Get satellite image by ID"""
        return db.query(SatelliteImage).filter(SatelliteImage.id == image_id).first()
    
    async def create_satellite_image(self, db: Session, project_id: int, file: UploadFile) -> SatelliteImage:
        """Process and save a satellite image"""
        # Create directory for satellite images if it doesn't exist
        os.makedirs("satellite_images", exist_ok=True)
        
        # Generate unique filename
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = f"satellite_images/{unique_filename}"
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Create database record
        db_obj = SatelliteImage(
            image_type=ImageType.SENTINEL_2,  # Default, could be determined from metadata
            acquisition_date=datetime.now(),  # Should be extracted from image metadata
            cloud_cover_percentage=0.0,  # Should be extracted from image metadata
            image_url=file_path,
            metadata={},  # Should be extracted from image
            project_id=project_id
        )
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get_project_images(self, db: Session, project_id: int) -> List[SatelliteImage]:
        """Get all satellite images for a project"""
        return db.query(SatelliteImage).filter(SatelliteImage.project_id == project_id).all()
    
    def create_analysis(self, db: Session, obj_in: SatelliteAnalysisCreate) -> SatelliteAnalysis:
        """Create a new satellite image analysis"""
        db_obj = SatelliteAnalysis(
            analysis_type=obj_in.analysis_type,
            analysis_date=datetime.now(),
            result_data=obj_in.result_data,
            confidence_score=obj_in.confidence_score,
            explanation_data=obj_in.explanation_data,
            satellite_image_id=obj_in.satellite_image_id,
            verification_id=obj_in.verification_id
        )
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get_image_analyses(self, db: Session, image_id: int) -> List[SatelliteAnalysis]:
        """Get all analyses for a satellite image"""
        return db.query(SatelliteAnalysis).filter(SatelliteAnalysis.satellite_image_id == image_id).all()

satellite_service = SatelliteService()
