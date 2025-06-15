"""
Project management endpoints
"""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.models.project import ProjectCreate, ProjectResponse, ProjectUpdate, ProjectListResponse
from app.models.user import UserResponse
from app.services.auth_service import auth_service
from app.services.project_service import project_service
from app.api.v1.endpoints.auth import oauth2_scheme

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """Dependency to get current authenticated user"""
    return auth_service.get_current_user(token)


@router.get("", response_model=ProjectListResponse)
async def get_projects(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get user's projects with pagination
    
    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (max 100)
    """
    try:
        return project_service.get_user_projects(
            user_id=current_user.id,
            page=page,
            page_size=page_size
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get projects endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve projects"
        )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get a specific project by ID
    
    - **project_id**: Project ID
    """
    try:
        project = project_service.get_project_by_id(project_id)
        
        # Check if user owns the project or is admin
        if project.user_id != current_user.id and current_user.role != "Admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this project"
            )
        
        return project
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get project endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve project"
        )


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Create a new project
    
    - **name**: Project name (3-200 characters)
    - **description**: Project description (optional, max 1000 characters)
    - **location_name**: Location name (2-100 characters)
    - **area_size**: Area size in hectares (> 0, max 1,000,000)
    - **project_type**: Type of project (Reforestation, Afforestation, etc.)
    """
    try:
        return project_service.create_project(
            project_data=project_data,
            user_id=current_user.id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create project endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project"
        )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Update an existing project
    
    - **project_id**: Project ID
    - All fields are optional for updates
    """
    try:
        # Check if project exists and user owns it
        existing_project = project_service.get_project_by_id(project_id)
        
        if existing_project.user_id != current_user.id and current_user.role != "Admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this project"
            )
        
        return project_service.update_project(
            project_id=project_id,
            project_data=project_data
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update project endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update project"
        )


@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Delete a project
    
    - **project_id**: Project ID
    """
    try:
        # Check if project exists and user owns it
        existing_project = project_service.get_project_by_id(project_id)
        
        if existing_project.user_id != current_user.id and current_user.role != "Admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this project"
            )
        
        success = project_service.delete_project(project_id)
        
        if success:
            return {"message": "Project deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete project"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete project endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete project"
        ) 