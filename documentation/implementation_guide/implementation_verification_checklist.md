# Implementation Verification Checklist

This checklist helps verify that the Carbon Credit Verification SaaS application has been implemented correctly, from data setup through to a functional application.

## 1. Environment & Setup

- [ ] Required system dependencies installed (Python 3.10+, Docker, Docker Compose)?
- [ ] Project code cloned/present in `/home/ubuntu/carbon_credit_project`?
- [ ] Python virtual environment created and activated (if used)?
- [ ] Backend Python dependencies installed (`pip install -r backend/requirements.txt`)?
- [ ] ML Python dependencies installed (e.g., `torch`, `torchvision`, `rasterio`, `geopandas`, `sentinelsat`, `segmentation-models-pytorch`, `matplotlib`, `tqdm`, `scikit-learn`)?
- [ ] Frontend Node.js dependencies installed (`cd frontend && npm install`)?
- [ ] Necessary credentials configured (e.g., Copernicus Hub in `sample_data_preparation.py` or environment variables)?
- [ ] Database connection details confirmed (using defaults in `docker-compose.yml` or updated)?

## 2. Data Preparation

- [ ] Area(s) of Interest (AOI) defined (e.g., `sample_area.geojson`)?
- [ ] Sentinel-2 satellite imagery acquired for AOI(s) and relevant time periods?
- [ ] Hansen Global Forest Change data acquired for AOI(s)?
- [ ] Data organized in the expected directory structure (e.g., `ml/data/sentinel/sceneX`, `ml/data/hansen/sceneX_forest_change.tif`)?
- [ ] Preprocessing steps completed if necessary (e.g., masking, resampling, CRS alignment)?
- [ ] `ForestChangeDataset` class in `ml/training/train_forest_change.py` reviewed/adapted to correctly load data from your specific structure and file formats?

## 3. Model Training

- [ ] Navigated to `ml/training` directory?
- [ ] `python train_forest_change.py` executed successfully without critical errors?
- [ ] Training logs (console output) reviewed for reasonable loss reduction and no major errors?
- [ ] Trained model file `forest_change_unet.pth` generated and saved in `ml/models/`?

## 4. Docker Deployment (Local)

- [ ] Navigated to `docker` directory?
- [ ] Backend container configured to access the trained model (`forest_change_unet.pth`)? (Check `docker/backend.Dockerfile` or `docker/docker-compose.yml` for COPY command or volume mount).
- [ ] `docker-compose up --build -d` executed successfully?
- [ ] `docker-compose ps` shows `db`, `backend`, and `frontend` services in the `Up` state?
- [ ] `docker-compose logs db` shows database ready and accepting connections?
- [ ] `docker-compose logs backend` shows FastAPI server started on port 8000 and no critical errors (especially DB connection or model loading)?
- [ ] `docker-compose logs frontend` shows React development server started on port 3000?

## 5. Backend API Verification

- [ ] Backend API documentation accessible via browser at `http://<host_ip>:8000/docs`?
- [ ] `/health` endpoint (if implemented) returns a success status?
- [ ] User registration endpoint (`/api/auth/register`) works via Swagger UI or frontend?
- [ ] User login endpoint (`/api/auth/token`) works and returns a token?
- [ ] Protected endpoints require authentication?

## 6. Frontend Application Verification

- [ ] Frontend application accessible via browser at `http://<host_ip>:3000`?
- [ ] Login page displayed correctly?
- [ ] Browser developer console checked for critical JavaScript errors?
- [ ] Browser network tab checked for successful API calls to the backend (no CORS errors, 404s, 500s)?

## 7. Core Workflow End-to-End Test

- [ ] **User Auth**: Can register a new user via the frontend?
- [ ] **User Auth**: Can log in with the registered user?
- [ ] **Project Creation**: Can create a new project, define its name, and specify its boundaries (e.g., by drawing on a map or uploading GeoJSON)?
- [ ] **Data Input**: Can associate/select satellite imagery (baseline/monitoring periods) for the project?
- [ ] **Verification Trigger**: Can initiate the verification process for the project?
- [ ] **ML Inference**: Does the backend log indicate that the ML inference task was triggered and completed successfully (loading the model, processing data, generating results)?
- [ ] **Results Display**: Are the verification results (e.g., change map, statistics) displayed correctly on the project detail page in the frontend?
- [ ] **Map Interaction**: Does the map component display the AOI and results overlay correctly?
- [ ] **Human Review**: Is the interface for human review of ML results functional?
- [ ] **Blockchain Certification**: Can the process to certify the verification results on the blockchain be initiated?
- [ ] **Blockchain Interaction**: Does the backend log indicate successful interaction with the smart contract (transaction sent/confirmed)?
- [ ] **Certificate Display**: Is the resulting certificate information (e.g., transaction hash, token ID) displayed on the frontend?

## 8. General Checks

- [ ] Application behaves as expected during normal use?
- [ ] No unexpected crashes or freezes?
- [ ] Data entered is persisted correctly in the database?

This checklist provides a structured way to confirm the core functionality after completing the implementation steps.
