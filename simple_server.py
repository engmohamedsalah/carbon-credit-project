"""
Simple FastAPI server for testing.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Carbon Credit Verification API",
    description="API for verifying carbon credits using satellite imagery, AI, and blockchain",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "Simple test server is running"}

@app.get("/api/test")
def test_endpoint():
    return {
        "message": "API is working",
        "phase": "Phase 1 completed",
        "project": "Carbon Credit Verification SaaS"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simple_server:app", host="0.0.0.0", port=8000, reload=True) 