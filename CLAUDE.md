# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Frontend (React)
- **Start dev server**: `cd frontend && npm start` (runs on http://localhost:3000)
- **Build for production**: `cd frontend && npm run build`
- **Run tests**: `cd frontend && npm test`
- **Install dependencies**: `cd frontend && npm install`

### Backend (FastAPI)
- **Start dev server**: `cd backend && python main.py` (runs on http://localhost:8000)
- **API documentation**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **Install dependencies**: `pip install -r backend/requirements.txt`

### Unified Development
- **Start both frontend and backend**: `./run_app.sh` (requires virtual environment setup)
- **Local setup**: `./scripts/local_dev_setup.sh`
- **Stop all services**: Press Ctrl+C when using run_app.sh

### Testing
- **Run all tests**: `./scripts/run_tests.sh`
- **E2E tests**: `cd tests/e2e && ./run_tests.sh`
- **E2E with options**: `cd tests/e2e && ./run_tests.sh --headed --browser firefox`
- **Playwright tests**: `cd tests/playwright && npm test`
- **Backend tests**: `python tests/test_backend.py`

### Docker Deployment
- **Start with Docker**: `cd docker && docker-compose up`
- **Build and start**: `cd docker && docker-compose up --build`
- **Stop containers**: `cd docker && docker-compose down`

## Architecture Overview

This is a Carbon Credit Verification SaaS application with three main tiers:

### Backend (FastAPI + SQLite)
- **Main entry**: `backend/main.py`
- **Database**: SQLite at `database/carbon_credits.db`
- **Services**: ML processing, XAI (Explainable AI), reporting
- **Authentication**: OAuth2 with JWT tokens
- **Rate limiting**: Built-in with slowapi

### Frontend (React + Redux)
- **Main structure**: Component-based with pages, components, services
- **State management**: Redux Toolkit with slices (auth, projects, verification, XAI)
- **UI Library**: Material-UI (MUI) with custom theming
- **Mapping**: Leaflet.js for interactive maps
- **Charts**: Recharts, Chart.js, and D3.js for data visualization

### Machine Learning Pipeline
- **Location**: `ml/` directory
- **Models**: 
  - Forest Cover U-Net (`ml/models/forest_cover_unet_focal_alpha_0.75_threshold_0.53.pth`)
  - Change Detection Siamese U-Net (`ml/models/change_detection_siamese_unet.pth`)
  - ConvLSTM for temporal analysis (`ml/models/convlstm_fast_final.pth`)
  - Production ensemble model (`ml/inference/ensemble_model.py`)
- **Data Sources**: Sentinel-1/2 satellite imagery, Hansen Global Forest Change data
- **Purpose**: Forest change detection and carbon sequestration estimation

## Key Components and Patterns

### Authentication & Authorization
- Role-based access control (RBAC) with user roles
- Protected routes using `components/ProtectedRoute.js`
- Auth state managed in `store/authSlice.js`
- Backend auth in `main.py` with password hashing and JWT

### Data Flow
1. **Satellite imagery** → ML models → **Change detection**
2. **Change detection** → Carbon estimation → **Verification workflow**
3. **Human verification** → Blockchain certification → **Carbon credits**

### API Structure
- RESTful API with `/api/v1/` prefix
- Endpoints: `/auth/`, `/projects/`, `/verification/`, `/ml/`, `/xai/`
- File uploads handled in `backend/uploads/`
- ML service integration in `backend/services/ml_service.py`

### XAI (Explainable AI) System
- **Frontend**: Comprehensive XAI dashboard in `pages/XAI.js`
- **Components**: SHAP, LIME, Integrated Gradients visualizations
- **Backend service**: `backend/services/real_xai_service.py`
- **Features**: Method comparison, explanation history, compliance reporting

### Testing Architecture
- **E2E tests**: Playwright-based in `tests/e2e/`
- **Unit tests**: Frontend (Jest/React Testing Library), Backend (pytest)
- **Test categories**: Authentication, dashboard, XAI functionality, user workflows
- **CI/CD**: GitHub Actions integration

## Development Guidelines

### Database
- Uses SQLite for simplicity (development) and PostgreSQL for production (Docker)
- Database initialization via `backend/init_db.py`
- Schema managed through the FastAPI application

### State Management
- Redux store in `frontend/src/store/`
- Async actions with Redux Toolkit Query for API calls
- Local state for UI components, global state for data

### File Structure Conventions
- **Pages**: Route-level components in `frontend/src/pages/`
- **Components**: Reusable UI components in `frontend/src/components/`
- **Services**: API interaction logic in `frontend/src/services/`
- **Utils**: Helper functions in `frontend/src/utils/`

### ML Integration
- Production models are pre-trained and stored in `ml/models/`
- Inference happens through `ml/inference/production_inference.py`
- Real-time XAI processing via `ml/utils/real_xai_service.py`
- Carbon calculations integrated into the verification workflow

### Error Handling
- Frontend: Error boundaries and utility functions in `utils/errorUtils.js`
- Backend: Structured logging and HTTP exception handling
- Global error state management through Redux

## Important Notes

- The application is designed for carbon credit verification using satellite imagery
- ML models are production-ready and pre-trained (no retraining needed)
- Virtual environment setup is required for local development
- Docker setup provides PostgreSQL with PostGIS for spatial data
- All sensitive configuration should use environment variables
- The system includes comprehensive documentation in the `documentation/` folder

## Common Workflows

1. **Adding new features**: Start with backend API endpoint, then frontend integration
2. **ML model updates**: Replace model files in `ml/models/`, update inference scripts
3. **UI changes**: Follow Material-UI patterns, maintain responsive design
4. **Testing**: Write E2E tests for user workflows, unit tests for utilities
5. **Deployment**: Use Docker for production, local scripts for development