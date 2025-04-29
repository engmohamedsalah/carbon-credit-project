# Carbon Credit Verification SaaS Application
## Technical Architecture Overview

This document provides a comprehensive technical overview of the Carbon Credit Verification SaaS application, including architecture, components, data flow, and implementation details.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Backend Components](#backend-components)
4. [Frontend Components](#frontend-components)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [Blockchain Integration](#blockchain-integration)
7. [Data Storage](#data-storage)
8. [API Documentation](#api-documentation)
9. [Security Considerations](#security-considerations)
10. [Scalability and Performance](#scalability-and-performance)

## System Architecture

The Carbon Credit Verification SaaS application follows a modern, microservices-oriented architecture designed for scalability, maintainability, and performance. The system is composed of several key components:

![System Architecture](../images/ml_pipeline.png)

### High-Level Architecture

The application is built using a hybrid architecture:

- **Backend**: FastAPI (Python) for core services and ML model integration
- **Frontend**: React with TypeScript for the user interface
- **Database**: PostgreSQL with PostGIS extension for geospatial data
- **ML Pipeline**: PyTorch-based models for satellite imagery analysis
- **Blockchain**: Polygon (Ethereum L2) for verification certification
- **Containerization**: Docker for consistent deployment

### Component Interaction

The components interact through well-defined APIs:

1. **User Interface** → **Backend API**: Frontend communicates with backend services via RESTful APIs
2. **Backend API** → **ML Services**: Backend requests predictions and analysis from ML services
3. **ML Services** → **Data Sources**: ML services access satellite imagery and reference data
4. **Backend API** → **Blockchain**: Verification results are certified on the blockchain
5. **Backend API** → **Database**: Persistent storage of project data and verification results

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.10+
- **Database**: PostgreSQL 14+ with PostGIS extension
- **ORM**: SQLAlchemy
- **Authentication**: JWT with OAuth2
- **API Documentation**: OpenAPI (Swagger)

### Frontend
- **Framework**: React 18+
- **Language**: TypeScript
- **State Management**: Redux Toolkit
- **UI Components**: Material-UI
- **Mapping**: Leaflet.js
- **Data Visualization**: D3.js, Recharts

### Machine Learning
- **Framework**: PyTorch
- **Data Processing**: NumPy, Pandas, GDAL, Rasterio
- **Model Architecture**: U-Net for semantic segmentation
- **Explainability**: Captum (Integrated Gradients, SHAP, Occlusion)
- **Visualization**: Matplotlib, OpenCV

### Blockchain
- **Platform**: Polygon (Ethereum L2)
- **Smart Contracts**: Solidity
- **Client Library**: ethers.js
- **Standards**: ERC-721 for NFT certification

### DevOps
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)

## Backend Components

The backend is structured as a modular FastAPI application with the following components:

### Core Modules
- **app/core/**: Core functionality and configuration
  - `config.py`: Application configuration
  - `database.py`: Database connection and session management
  - `security.py`: Authentication and authorization

### API Endpoints
- **app/api/**: API route definitions
  - `auth.py`: Authentication endpoints
  - `projects.py`: Project management endpoints
  - `satellite.py`: Satellite imagery analysis endpoints
  - `verification.py`: Verification workflow endpoints
  - `blockchain.py`: Blockchain certification endpoints

### Data Models
- **app/models/**: Database models
  - `user.py`: User and role models
  - `project.py`: Project and location models
  - `verification.py`: Verification and certification models
  - `satellite.py`: Satellite imagery and analysis models

### Service Layer
- **app/services/**: Business logic
  - `auth_service.py`: Authentication and user management
  - `project_service.py`: Project operations
  - `satellite_service.py`: Satellite imagery processing
  - `verification_service.py`: Verification workflow
  - `blockchain_service.py`: Blockchain integration

### Schemas
- **app/schemas/**: Pydantic schemas for request/response validation
  - `user.py`: User-related schemas
  - `project.py`: Project-related schemas
  - `verification.py`: Verification-related schemas
  - `satellite.py`: Satellite imagery-related schemas

## Frontend Components

The frontend is a React application with TypeScript, organized as follows:

### Core Components
- **src/App.js**: Main application component
- **src/index.js**: Application entry point
- **src/components/**: Reusable UI components
  - `Layout.js`: Page layout component
  - `MapComponent.js`: Interactive map component
  - `ProtectedRoute.js`: Route protection for authentication

### Pages
- **src/pages/**: Page components
  - `Login.js`: User login page
  - `Register.js`: User registration page
  - `Dashboard.js`: Main dashboard
  - `NewProject.js`: Project creation page
  - `ProjectDetail.js`: Project details page
  - `Verification.js`: Verification workflow page

### State Management
- **src/store/**: Redux store configuration
  - `index.js`: Store configuration
  - `authSlice.js`: Authentication state
  - `projectSlice.js`: Project state
  - `verificationSlice.js`: Verification state

### Services
- **src/services/**: API service integration
  - `api.js`: API client configuration

## Machine Learning Pipeline

The ML pipeline is a critical component of the system, responsible for analyzing satellite imagery and detecting forest cover changes:

### Data Acquisition
- **ml/utils/data_preparation.py**: Downloads and prepares satellite imagery and reference data
  - Connects to Sentinel-2 API (Copernicus Open Access Hub)
  - Downloads Hansen Global Forest Change data
  - Prepares training datasets

### Model Training
- **ml/training/train_forest_change.py**: Trains the forest change detection model
  - Implements a U-Net architecture for semantic segmentation
  - Processes Sentinel-2 imagery (bands B02, B03, B04, B08)
  - Calculates NDVI (Normalized Difference Vegetation Index)
  - Trains on forest change data from Hansen dataset

### Inference
- **ml/inference/predict_forest_change.py**: Performs inference on new satellite imagery
  - Preprocesses imagery
  - Runs prediction using trained model
  - Generates explanations using XAI techniques
  - Outputs prediction maps and confidence scores

### Carbon Estimation
- **ml/inference/estimate_carbon_sequestration.py**: Estimates carbon impact
  - Compares imagery from different time points
  - Calculates forest cover changes
  - Estimates carbon sequestration or emissions
  - Generates visualizations and reports

### Visualization
- **ml/utils/visualization.py**: Creates visualizations of satellite imagery and results
- **ml/utils/xai_visualization.py**: Creates explainable AI visualizations

## Blockchain Integration

The blockchain component provides immutable certification of verification results:

### Smart Contracts
- **blockchain/contracts/VerificationCertificate.sol**: Smart contract for verification certificates
  - Implements ERC-721 standard for NFT certificates
  - Stores verification metadata and results hash
  - Includes ownership and transfer functionality

### Integration
- **app/services/blockchain_service.py**: Backend integration with blockchain
  - Connects to Polygon network
  - Deploys and interacts with smart contracts
  - Creates and verifies certificates

### Certificate Structure
Each certificate contains:
- Project identifier
- Verification timestamp
- Results hash (IPFS CID of detailed results)
- Verifier information
- Methodology reference
- Carbon impact estimation

## Data Storage

The application uses several data storage mechanisms:

### Relational Database
- PostgreSQL with PostGIS extension for geospatial data
- Stores user accounts, projects, verification records
- Maintains relationships between entities
- Supports spatial queries for location-based analysis

### File Storage
- Satellite imagery stored in efficient formats (GeoTIFF)
- ML models saved as PyTorch .pth files
- Verification results stored as JSON and GeoJSON
- Visualizations saved as PNG files

### Blockchain Storage
- Verification certificates stored on Polygon blockchain
- Detailed results stored off-chain with hash reference
- Smart contract events for audit trail

## API Documentation

The application provides comprehensive API documentation:

### Authentication Endpoints
- `POST /api/auth/login`: User login
- `POST /api/auth/register`: User registration
- `POST /api/auth/refresh`: Refresh access token

### Project Endpoints
- `GET /api/projects`: List projects
- `POST /api/projects`: Create new project
- `GET /api/projects/{id}`: Get project details
- `PUT /api/projects/{id}`: Update project
- `DELETE /api/projects/{id}`: Delete project

### Satellite Analysis Endpoints
- `POST /api/satellite/upload-satellite-bands`: Upload satellite imagery
- `GET /api/satellite/forest-change/{task_id}`: Get forest change results
- `POST /api/satellite/estimate-carbon-sequestration`: Estimate carbon sequestration
- `GET /api/satellite/carbon-sequestration/{task_id}`: Get carbon estimation results
- `GET /api/satellite/explanation/{task_id}`: Get explanation for prediction

### Verification Endpoints
- `POST /api/verification/start`: Start verification process
- `GET /api/verification/{id}`: Get verification status
- `PUT /api/verification/{id}/review`: Submit human review
- `POST /api/verification/{id}/certify`: Create blockchain certificate
- `GET /api/verification/{id}/certificate`: Get certificate details

## Security Considerations

The application implements several security measures:

### Authentication and Authorization
- JWT-based authentication
- Role-based access control
- Password hashing with bcrypt
- Token refresh mechanism

### Data Protection
- HTTPS for all communications
- Database encryption for sensitive data
- Input validation with Pydantic schemas
- CORS configuration

### API Security
- Rate limiting
- Request validation
- Error handling without sensitive information
- Audit logging

### ML Model Security
- Model versioning
- Input sanitization
- Explainability for transparency
- Human review for critical decisions

## Scalability and Performance

The application is designed for scalability and performance:

### Horizontal Scaling
- Stateless API design
- Containerization for easy deployment
- Database connection pooling
- Caching for frequent queries

### Performance Optimization
- Asynchronous processing for long-running tasks
- Efficient satellite imagery processing
- Optimized ML model inference
- Background task processing

### Monitoring and Metrics
- Performance monitoring
- Error tracking
- Usage metrics
- Resource utilization

---

This technical overview provides a comprehensive understanding of the Carbon Credit Verification SaaS application architecture and components. For detailed implementation instructions, please refer to the installation guides and user documentation.
