"""
Test FastAPI server.
"""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from the test server!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run directly with:
# python test_server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888) 