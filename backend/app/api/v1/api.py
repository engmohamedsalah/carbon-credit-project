"""
Main API router combining all endpoint routers
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, projects, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
api_router.include_router(health.router, prefix="", tags=["Health"]) 