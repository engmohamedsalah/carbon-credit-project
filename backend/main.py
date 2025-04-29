from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.api import auth, projects, satellite, verification, blockchain
from app.core.config import settings
from app.core.database import get_db, init_db

app = FastAPI(
    title="Carbon Credit Verification API",
    description="API for verifying carbon credits using satellite imagery, AI, and blockchain",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(satellite.router, prefix="/api/satellite", tags=["Satellite Imagery"])
app.include_router(verification.router, prefix="/api/verification", tags=["Verification"])
app.include_router(blockchain.router, prefix="/api/blockchain", tags=["Blockchain"])

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
