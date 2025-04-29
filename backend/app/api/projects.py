from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Any, List, Optional

from app.core.database import get_db
from app.schemas.project import Project, ProjectCreate, ProjectUpdate, ProjectWithDetails
from app.services import project_service, auth_service

router = APIRouter()

@router.get("/", response_model=List[Project])
def get_projects(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Retrieve projects.
    """
    if auth_service.is_admin(current_user):
        projects = project_service.get_multi(db, skip=skip, limit=limit)
    else:
        projects = project_service.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit
        )
    return projects

@router.post("/", response_model=Project)
def create_project(
    *,
    db: Session = Depends(get_db),
    project_in: ProjectCreate,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Create new project.
    """
    project = project_service.create_with_owner(
        db=db, obj_in=project_in, owner_id=current_user.id
    )
    return project

@router.get("/{id}", response_model=ProjectWithDetails)
def get_project(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Get project by ID.
    """
    project = project_service.get(db=db, id=id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not auth_service.is_admin(current_user) and (project.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return project

@router.put("/{id}", response_model=Project)
def update_project(
    *,
    db: Session = Depends(get_db),
    id: int,
    project_in: ProjectUpdate,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Update a project.
    """
    project = project_service.get(db=db, id=id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not auth_service.is_admin(current_user) and (project.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    project = project_service.update(db=db, db_obj=project, obj_in=project_in)
    return project

@router.delete("/{id}", response_model=Project)
def delete_project(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: auth_service.User = Depends(auth_service.get_current_user),
) -> Any:
    """
    Delete a project.
    """
    project = project_service.get(db=db, id=id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not auth_service.is_admin(current_user) and (project.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    project = project_service.remove(db=db, id=id)
    return project
