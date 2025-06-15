"""
Project service layer
"""
import logging
import math
from typing import Optional
from datetime import datetime

from fastapi import HTTPException, status
from app.db.database import db_manager
from app.models.project import (
    ProjectCreate, 
    ProjectResponse, 
    ProjectUpdate, 
    ProjectListResponse,
    ProjectStatus
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class ProjectService:
    """Professional project service"""
    
    @staticmethod
    def create_project(project_data: ProjectCreate, user_id: int) -> ProjectResponse:
        """Create a new project"""
        try:
            # Insert project into database
            project_id = db_manager.execute_insert(
                """
                INSERT INTO projects (name, description, location_name, area_size, project_type, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_data.name,
                    project_data.description,
                    project_data.location_name,
                    project_data.area_size,
                    project_data.project_type.value,
                    user_id
                )
            )
            
            # Get the created project
            project = db_manager.execute_query(
                """
                SELECT id, name, description, location_name, area_size, project_type, 
                       status, user_id, created_at, updated_at
                FROM projects 
                WHERE id = ?
                """,
                (project_id,)
            )
            
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create project"
                )
            
            logger.info(f"Project created successfully: {project_id}")
            
            return ProjectResponse(
                id=project["id"],
                name=project["name"],
                description=project["description"],
                location_name=project["location_name"],
                area_size=project["area_size"],
                project_type=project["project_type"],
                status=project["status"] or ProjectStatus.PENDING,
                user_id=project["user_id"],
                created_at=datetime.fromisoformat(project["created_at"]),
                updated_at=datetime.fromisoformat(project["updated_at"]) if project["updated_at"] else None
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create project"
            )
    
    @staticmethod
    def get_project_by_id(project_id: int) -> ProjectResponse:
        """Get project by ID"""
        try:
            project = db_manager.execute_query(
                """
                SELECT id, name, description, location_name, area_size, project_type, 
                       status, user_id, created_at, updated_at
                FROM projects 
                WHERE id = ?
                """,
                (project_id,)
            )
            
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            return ProjectResponse(
                id=project["id"],
                name=project["name"],
                description=project["description"],
                location_name=project["location_name"],
                area_size=project["area_size"],
                project_type=project["project_type"],
                status=project["status"] or ProjectStatus.PENDING,
                user_id=project["user_id"],
                created_at=datetime.fromisoformat(project["created_at"]),
                updated_at=datetime.fromisoformat(project["updated_at"]) if project["updated_at"] else None
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get project failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve project"
            )
    
    @staticmethod
    def get_user_projects(user_id: int, page: int = 1, page_size: int = 20) -> ProjectListResponse:
        """Get user's projects with pagination"""
        try:
            # Validate pagination parameters
            page_size = min(page_size, settings.MAX_PAGE_SIZE)
            offset = (page - 1) * page_size
            
            # Get total count
            total_result = db_manager.execute_query(
                "SELECT COUNT(*) as total FROM projects WHERE user_id = ?",
                (user_id,)
            )
            total = total_result["total"] if total_result else 0
            
            # Get projects
            projects_data = db_manager.execute_query_all(
                """
                SELECT id, name, description, location_name, area_size, project_type, 
                       status, user_id, created_at, updated_at
                FROM projects 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, page_size, offset)
            )
            
            projects = []
            for project in projects_data:
                projects.append(ProjectResponse(
                    id=project["id"],
                    name=project["name"],
                    description=project["description"],
                    location_name=project["location_name"],
                    area_size=project["area_size"],
                    project_type=project["project_type"],
                    status=project["status"] or ProjectStatus.PENDING,
                    user_id=project["user_id"],
                    created_at=datetime.fromisoformat(project["created_at"]),
                    updated_at=datetime.fromisoformat(project["updated_at"]) if project["updated_at"] else None
                ))
            
            total_pages = math.ceil(total / page_size) if total > 0 else 1
            
            return ProjectListResponse(
                projects=projects,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages
            )
            
        except Exception as e:
            logger.error(f"Get user projects failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve projects"
            )
    
    @staticmethod
    def update_project(project_id: int, project_data: ProjectUpdate) -> ProjectResponse:
        """Update an existing project"""
        try:
            # Build update query dynamically
            update_fields = []
            update_values = []
            
            if project_data.name is not None:
                update_fields.append("name = ?")
                update_values.append(project_data.name)
            
            if project_data.description is not None:
                update_fields.append("description = ?")
                update_values.append(project_data.description)
            
            if project_data.location_name is not None:
                update_fields.append("location_name = ?")
                update_values.append(project_data.location_name)
            
            if project_data.area_size is not None:
                update_fields.append("area_size = ?")
                update_values.append(project_data.area_size)
            
            if project_data.project_type is not None:
                update_fields.append("project_type = ?")
                update_values.append(project_data.project_type.value)
            
            if project_data.status is not None:
                update_fields.append("status = ?")
                update_values.append(project_data.status.value)
            
            if not update_fields:
                # No fields to update, return current project
                return ProjectService.get_project_by_id(project_id)
            
            # Add updated_at
            update_fields.append("updated_at = ?")
            update_values.append(datetime.utcnow().isoformat())
            
            # Add project_id for WHERE clause
            update_values.append(project_id)
            
            # Execute update
            affected_rows = db_manager.execute_update(
                f"""
                UPDATE projects 
                SET {', '.join(update_fields)}
                WHERE id = ?
                """,
                tuple(update_values)
            )
            
            if affected_rows == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            logger.info(f"Project updated successfully: {project_id}")
            
            # Return updated project
            return ProjectService.get_project_by_id(project_id)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Project update failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update project"
            )
    
    @staticmethod
    def delete_project(project_id: int) -> bool:
        """Delete a project"""
        try:
            affected_rows = db_manager.execute_update(
                "DELETE FROM projects WHERE id = ?",
                (project_id,)
            )
            
            if affected_rows == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            logger.info(f"Project deleted successfully: {project_id}")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Project deletion failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete project"
            )


# Global project service instance
project_service = ProjectService()
