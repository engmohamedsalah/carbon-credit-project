# Carbon Credit Verification SaaS Implementation Guide

This comprehensive guide provides step-by-step instructions for implementing the Carbon Credit Verification SaaS application, from training the machine learning model to deploying the complete system. This document is structured to work well with Cursor IDE for efficient implementation.

## Table of Contents

1. [Machine Learning Model Training](#1-machine-learning-model-training)
2. [Backend Implementation](#2-backend-implementation)
3. [Frontend Implementation](#3-frontend-implementation)
4. [System Integration and Deployment](#4-system-integration-and-deployment)
5. [Final Steps and Considerations](#5-final-steps-and-considerations)
6. [Implementation Timeline](#6-implementation-timeline)
7. [Key Considerations and Next Steps](#7-key-considerations-and-next-steps)

## 1. Machine Learning Model Training

### 1.1 Environment Setup

```bash
# Install required dependencies
pip install rasterio sentinelsat geopandas torch torchvision captum matplotlib
pip install scikit-learn pandas numpy tqdm
```

### 1.2 Data Preparation

```python
# Run data preparation script
python ml/utils/data_preparation.py --aoi project_area.geojson --start_date 2020-01-01 --end_date 2023-01-01 --output_dir ml/data
```

#### Data Preparation Best Practices

- **Sentinel-2 Data Selection**:
  - Use Level-2A products (atmospherically corrected)
  - Filter for cloud coverage < 20%
  - Select images from similar seasons to minimize seasonal variation
  - Recommended bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)

- **Hansen Data Processing**:
  - Use the latest available Hansen Global Forest Change dataset
  - Create binary masks for forest/non-forest using 30% tree cover threshold
  - Generate change detection labels by comparing baseline with current state

- **Data Augmentation Implementation**:

```python
# Add to train_forest_change.py
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
])
```

### 1.3 Model Training

```python
# Run model training script
python ml/training/train_forest_change.py
```

#### Model Training Optimization

- **U-Net Architecture Enhancements**:

```python
# Modify UNet class in train_forest_change.py
def _encoder_block(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.2),  # Add dropout for regularization
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
```

- **Training Parameters**:

```python
# Update in train_forest_change.py
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Add early stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# Update training loop
for epoch in range(NUM_EPOCHS):
    # Training code...
    
    # Step the scheduler
    scheduler.step()
    
    # Check early stopping
    if early_stopping(val_loss):
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break
```

- **Performance Monitoring**:

```python
# Add to train_forest_change.py
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/forest_change_detection')

# In training loop
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('IoU/val', iou_score, epoch)
```

### 1.4 Model Evaluation

```python
# Run model evaluation script
python ml/inference/predict_forest_change.py --image_path ml/data/test_image --output_dir ml/results
```

#### Evaluation Metrics Implementation

```python
# Add to predict_forest_change.py
def calculate_metrics(prediction, ground_truth):
    """Calculate evaluation metrics for forest change detection."""
    # Convert to binary masks
    pred_mask = prediction > 0
    gt_mask = ground_truth > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)
    
    # Calculate precision and recall
    true_positives = intersection
    false_positives = pred_mask.sum() - true_positives
    false_negatives = gt_mask.sum() - true_positives
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
```

## 2. Backend Implementation

### 2.1 Database Setup

```bash
# Start PostgreSQL with PostGIS
docker-compose -f docker/docker-compose.yml up -d postgres

# Initialize the database
cd backend
cp .env.example .env
# Edit .env with your database credentials
python -c "from app.core.database import Base, engine; Base.metadata.create_all(bind=engine)"
```

#### PostGIS Setup

```sql
-- Run these commands in PostgreSQL
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_raster;

-- Create spatial indexes
CREATE INDEX idx_projects_geometry ON projects USING GIST (geometry);
```

### 2.2 FastAPI Implementation

#### Core API Structure

```python
# Update backend/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, projects, satellite, verification, blockchain
from app.core.config import settings

app = FastAPI(
    title="Carbon Credit Verification API",
    description="API for carbon credit verification using satellite imagery and ML",
    version="1.0.0"
)

# Add CORS middleware
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
app.include_router(satellite.router, prefix="/api/satellite", tags=["Satellite"])
app.include_router(verification.router, prefix="/api/verification", tags=["Verification"])
app.include_router(blockchain.router, prefix="/api/blockchain", tags=["Blockchain"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Carbon Credit Verification API"}
```

#### Async Request Handling

```python
# Update backend/app/api/verification.py
@router.post("/projects/{project_id}/verify", response_model=schemas.VerificationResponse)
async def create_verification(
    project_id: int,
    verification: schemas.VerificationCreate,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Create verification record
    verification_id = verification_service.create_verification(
        db, project_id, verification, current_user.id
    )
    
    # Run verification process in background
    background_tasks.add_task(
        verification_service.run_verification_process,
        db, project_id, verification_id, current_user.id
    )
    
    return {"status": "Verification process started", "id": verification_id}
```

#### API Rate Limiting and Caching

```python
# Create backend/app/core/middleware.py
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import asyncio
from app.core.cache import cache

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        
        # Clean old requests
        current_time = time.time()
        self.clients[client_ip] = [t for t in self.clients[client_ip] 
                                  if current_time - t < self.period]
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            return Response(
                content="Rate limit exceeded",
                status_code=429
            )
        
        # Add current request timestamp
        self.clients[client_ip].append(current_time)
        
        # Process request
        return await call_next(request)

# Add to main.py
from app.core.middleware import RateLimitMiddleware

app.add_middleware(
    RateLimitMiddleware,
    calls=100,
    period=60
)
```

### 2.3 ML Model Integration

```python
# Create backend/app/services/ml_service.py
import os
import sys
import logging
from fastapi import BackgroundTasks
from app.core.config import settings

# Add ML directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ml'))

# Import ML model
from inference.predict_forest_change import ForestChangePredictor

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(settings.ML_MODEL_DIR, 'forest_change_unet.pth')
    
    def load_model(self):
        """Load ML model if not already loaded."""
        if self.model is None:
            try:
                logger.info(f"Loading model from {self.model_path}")
                self.model = ForestChangePredictor(model_path=self.model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    def predict_forest_change(self, image_path, output_dir):
        """Run forest change prediction on satellite imagery."""
        self.load_model()
        return self.model.process_satellite_image(image_path, output_dir)

# Create singleton instance
ml_service = MLService()

# Update verification_service.py to use ML service
from app.services.ml_service import ml_service

def run_verification_process(db, project_id, verification_id, user_id):
    """Run the verification process for a project."""
    try:
        # Get project and verification details
        project = project_service.get_project(db, project_id)
        verification = get_verification(db, verification_id)
        
        # Get satellite imagery path
        image_path = os.path.join(settings.SATELLITE_DATA_DIR, f"project_{project_id}")
        
        # Create output directory
        output_dir = os.path.join(settings.VERIFICATION_RESULTS_DIR, f"verification_{verification_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run ML prediction
        results = ml_service.predict_forest_change(image_path, output_dir)
        
        # Update verification record with results
        update_verification(db, verification_id, {
            "status": "COMPLETED",
            "results": results,
            "forest_loss_area": results["forest_loss_area_ha"],
            "carbon_impact": results["carbon_impact"]["carbon_loss_tons"],
            "co2_emissions": results["carbon_impact"]["co2_emissions_tons"],
            "confidence": results["average_confidence"]
        })
        
        # Trigger human review if confidence below threshold
        if results["average_confidence"] < settings.CONFIDENCE_THRESHOLD:
            update_verification(db, verification_id, {"status": "NEEDS_REVIEW"})
    
    except Exception as e:
        logger.error(f"Error in verification process: {e}")
        update_verification(db, verification_id, {"status": "FAILED", "error": str(e)})
```

### 2.4 Blockchain Integration

```python
# Update backend/app/services/blockchain_service.py
from web3 import Web3
import json
import os
from app.core.config import settings

class BlockchainService:
    def __init__(self):
        # Connect to Polygon network
        self.w3 = Web3(Web3.HTTPProvider(settings.BLOCKCHAIN_RPC_URL))
        
        # Load contract ABI
        with open(settings.CONTRACT_ABI_PATH) as f:
            self.contract_abi = json.load(f)
        
        # Create contract instance
        self.contract = self.w3.eth.contract(
            address=settings.CONTRACT_ADDRESS, 
            abi=self.contract_abi
        )
        
        # Set wallet address and private key
        self.wallet_address = settings.WALLET_ADDRESS
        self.private_key = settings.PRIVATE_KEY
    
    def issue_certificate(self, verification_id, carbon_impact, metadata_uri):
        """Issue a carbon credit certificate on the blockchain."""
        # Check if connected to network
        if not self.w3.is_connected():
            raise Exception("Not connected to blockchain network")
        
        # Get nonce
        nonce = self.w3.eth.get_transaction_count(self.wallet_address)
        
        # Build transaction
        txn = self.contract.functions.issueCertificate(
            verification_id,
            self.w3.to_wei(carbon_impact, 'ether'),
            metadata_uri
        ).build_transaction({
            'chainId': settings.CHAIN_ID,
            'gas': 2000000,
            'gasPrice': self.w3.to_wei('50', 'gwei'),
            'nonce': nonce,
        })
        
        # Sign transaction
        signed_txn = self.w3.eth.account.sign_transaction(txn, self.private_key)
        
        # Send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'tx_hash': self.w3.to_hex(tx_hash),
            'block_number': tx_receipt['blockNumber'],
            'status': tx_receipt['status']
        }
    
    def verify_certificate(self, certificate_id):
        """Verify a carbon credit certificate on the blockchain."""
        certificate = self.contract.functions.getCertificate(certificate_id).call()
        
        return {
            'id': certificate_id,
            'issuer': certificate[0],
            'carbon_impact': self.w3.from_wei(certificate[1], 'ether'),
            'timestamp': certificate[2],
            'metadata_uri': certificate[3],
            'is_valid': certificate[4]
        }

# Create singleton instance
blockchain_service = BlockchainService()
```

## 3. Frontend Implementation

### 3.1 Setup and Configuration

```bash
# Install dependencies
cd frontend
npm install

# Add required packages
npm install @reduxjs/toolkit react-redux axios leaflet react-leaflet deck.gl @deck.gl/react @deck.gl/geo-layers @deck.gl/layers react-window
```

### 3.2 Redux Store Configuration

```javascript
// Update frontend/src/store/index.js
import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query/react';
import authReducer from './authSlice';
import projectReducer from './projectSlice';
import verificationReducer from './verificationSlice';
import { api } from '../services/api';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    projects: projectReducer,
    verifications: verificationReducer,
    [api.reducerPath]: api.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }).concat(api.middleware),
});

setupListeners(store.dispatch);
```

### 3.3 API Integration

```javascript
// Update frontend/src/services/api.js
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000/api',
    prepareHeaders: (headers, { getState }) => {
      const token = getState().auth.token;
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      return headers;
    },
  }),
  tagTypes: ['Projects', 'Verifications', 'User'],
  endpoints: (builder) => ({
    // Auth endpoints
    login: builder.mutation({
      query: (credentials) => ({
        url: '/auth/login',
        method: 'POST',
        body: credentials,
      }),
    }),
    register: builder.mutation({
      query: (userData) => ({
        url: '/auth/register',
        method: 'POST',
        body: userData,
      }),
    }),
    
    // Project endpoints
    getProjects: builder.query({
      query: () => '/projects',
      providesTags: ['Projects'],
    }),
    getProject: builder.query({
      query: (id) => `/projects/${id}`,
      providesTags: (result, error, id) => [{ type: 'Projects', id }],
    }),
    createProject: builder.mutation({
      query: (project) => ({
        url: '/projects',
        method: 'POST',
        body: project,
      }),
      invalidatesTags: ['Projects'],
    }),
    updateProject: builder.mutation({
      query: ({ id, ...project }) => ({
        url: `/projects/${id}`,
        method: 'PUT',
        body: project,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'Projects', id }],
    }),
    
    // Satellite endpoints
    uploadSatelliteImage: builder.mutation({
      query: ({ projectId, formData }) => ({
        url: `/satellite/projects/${projectId}/upload`,
        method: 'POST',
        body: formData,
        formData: true,
      }),
      invalidatesTags: (result, error, { projectId }) => [{ type: 'Projects', id: projectId }],
    }),
    acquireSatelliteImage: builder.mutation({
      query: ({ projectId, params }) => ({
        url: `/satellite/projects/${projectId}/acquire`,
        method: 'POST',
        body: params,
      }),
      invalidatesTags: (result, error, { projectId }) => [{ type: 'Projects', id: projectId }],
    }),
    
    // Verification endpoints
    getVerifications: builder.query({
      query: (projectId) => `/verification/projects/${projectId}`,
      providesTags: ['Verifications'],
    }),
    getVerification: builder.query({
      query: (id) => `/verification/${id}`,
      providesTags: (result, error, id) => [{ type: 'Verifications', id }],
    }),
    createVerification: builder.mutation({
      query: ({ projectId, ...verification }) => ({
        url: `/verification/projects/${projectId}/verify`,
        method: 'POST',
        body: verification,
      }),
      invalidatesTags: ['Verifications'],
    }),
    reviewVerification: builder.mutation({
      query: ({ id, review }) => ({
        url: `/verification/${id}/review`,
        method: 'POST',
        body: review,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'Verifications', id }],
    }),
    
    // Blockchain endpoints
    getCertificate: builder.query({
      query: (id) => `/blockchain/certificates/${id}`,
    }),
    verifyCertificate: builder.query({
      query: (id) => `/blockchain/certificates/${id}/verify`,
    }),
  }),
});

export const {
  useLoginMutation,
  useRegisterMutation,
  useGetProjectsQuery,
  useGetProjectQuery,
  useCreateProjectMutation,
  useUpdateProjectMutation,
  useUploadSatelliteImageMutation,
  useAcquireSatelliteImageMutation,
  useGetVerificationsQuery,
  useGetVerificationQuery,
  useCreateVerificationMutation,
  useReviewVerificationMutation,
  useGetCertificateQuery,
  useVerifyCertificateQuery,
} = api;
```

### 3.4 Map Component Implementation

```javascript
// Update frontend/src/components/MapComponent.js
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, LayersControl, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const MapComponent = ({ 
  projectBoundary, 
  forestChangeData, 
  satelliteImagery, 
  center = [0, 0], 
  zoom = 2,
  onBoundaryChange,
  editable = false
}) => {
  const [map, setMap] = useState(null);
  const [drawControl, setDrawControl] = useState(null);
  
  // Initialize draw control for editable maps
  useEffect(() => {
    if (!map || !editable) return;
    
    // Import Leaflet Draw
    import('leaflet-draw').then(() => {
      // Remove existing draw control if any
      if (drawControl) {
        map.removeControl(drawControl);
      }
      
      // Create feature group for drawn items
      const drawnItems = new L.FeatureGroup();
      map.addLayer(drawnItems);
      
      // Add existing boundary if available
      if (projectBoundary) {
        L.geoJSON(projectBoundary).eachLayer(layer => {
          drawnItems.addLayer(layer);
        });
      }
      
      // Initialize draw control
      const control = new L.Control.Draw({
        edit: {
          featureGroup: drawnItems,
        },
        draw: {
          polygon: true,
          rectangle: true,
          circle: false,
          circlemarker: false,
          marker: false,
          polyline: false,
        },
      });
      
      map.addControl(control);
      setDrawControl(control);
      
      // Handle draw events
      map.on(L.Draw.Event.CREATED, (e) => {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        
        // Convert to GeoJSON and notify parent
        const geojson = drawnItems.toGeoJSON();
        if (onBoundaryChange) {
          onBoundaryChange(geojson);
        }
      });
      
      map.on(L.Draw.Event.EDITED, (e) => {
        // Convert to GeoJSON and notify parent
        const geojson = drawnItems.toGeoJSON();
        if (onBoundaryChange) {
          onBoundaryChange(geojson);
        }
      });
      
      map.on(L.Draw.Event.DELETED, () => {
        if (onBoundaryChange) {
          onBoundaryChange(null);
        }
      });
    });
    
    return () => {
      if (map && drawControl) {
        map.removeControl(drawControl);
      }
    };
  }, [map, editable, projectBoundary, onBoundaryChange, drawControl]);
  
  // Style function for forest change data
  const forestChangeStyle = (feature) => {
    return {
      fillColor: feature.properties.change_type === 'loss' ? '#ff3300' : '#33cc33',
      weight: 1,
      opacity: 1,
      color: 'white',
      fillOpacity: 0.7
    };
  };
  
  // Style function for project boundary
  const boundaryStyle = {
    color: '#3388ff',
    weight: 2,
    opacity: 1,
    fillOpacity: 0.1
  };
  
  return (
    <MapContainer 
      center={center} 
      zoom={zoom} 
      style={{ height: '500px', width: '100%' }}
      whenCreated={setMap}
    >
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="OpenStreetMap">
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; OpenStreetMap contributors'
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Satellite">
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution='&copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community'
          />
        </LayersControl.BaseLayer>
        
        {projectBoundary && !editable && (
          <LayersControl.Overlay checked name="Project Boundary">
            <GeoJSON data={projectBoundary} style={boundaryStyle} />
          </LayersControl.Overlay>
        )}
        
        {forestChangeData && (
          <LayersControl.Overlay checked name="Forest Change">
            <GeoJSON 
              data={forestChangeData} 
              style={forestChangeStyle}
              onEachFeature={(feature, layer) => {
                if (feature.properties) {
                  layer.bindPopup(`
                    <strong>Change Type:</strong> ${feature.properties.change_type}<br>
                    <strong>Area:</strong> ${feature.properties.area_ha.toFixed(2)} ha<br>
                    <strong>Confidence:</strong> ${(feature.properties.confidence * 100).toFixed(1)}%
                  `);
                }
              }}
            />
          </LayersControl.Overlay>
        )}
        
        {satelliteImagery && (
          <LayersControl.Overlay name="Satellite Imagery">
            <TileLayer
              url={satelliteImagery.url}
              attribution={satelliteImagery.attribution}
              tms={satelliteImagery.tms}
            />
          </LayersControl.Overlay>
        )}
      </LayersControl>
    </MapContainer>
  );
};

export default MapComponent;
```

### 3.5 Verification Workflow Implementation

```javascript
// Create frontend/src/components/VerificationReview.js
import React, { useState, useEffect } from 'react';
import { useReviewVerificationMutation } from '../services/api';
import MapComponent from './MapComponent';

const VerificationReview = ({ verification, onComplete }) => {
  const [selectedArea, setSelectedArea] = useState(null);
  const [userDecisions, setUserDecisions] = useState({});
  const [reviewVerification, { isLoading, isSuccess }] = useReviewVerificationMutation();
  
  // Extract prediction areas from verification results
  const predictionAreas = verification?.results?.prediction_areas || [];
  
  // Calculate review progress
  const totalAreas = predictionAreas.length;
  const reviewedAreas = Object.keys(userDecisions).length;
  const progress = totalAreas > 0 ? (reviewedAreas / totalAreas) * 100 : 0;
  
  // Handle area selection
  const handleAreaSelect = (areaId) => {
    setSelectedArea(areaId);
  };
  
  // Handle user decision
  const handleDecision = (decision) => {
    setUserDecisions({
      ...userDecisions,
      [selectedArea]: decision
    });
  };
  
  // Handle review submission
  const handleSubmitReview = async () => {
    try {
      await reviewVerification({
        id: verification.id,
        review: {
          decisions: userDecisions,
          reviewer_notes: document.getElementById('reviewer-notes').value
        }
      }).unwrap();
      
      if (onComplete) {
        onComplete();
      }
    } catch (error) {
      console.error('Failed to submit review:', error);
    }
  };
  
  // Prepare GeoJSON for the map
  const forestChangeData = {
    type: 'FeatureCollection',
    features: predictionAreas.map(area => ({
      type: 'Feature',
      properties: {
        id: area.id,
        change_type: area.classification,
        area_ha: area.area_ha,
        confidence: area.confidence,
        selected: area.id === selectedArea
      },
      geometry: area.geometry
    }))
  };
  
  return (
    <div className="verification-review">
      <h2>Human Review: Verification #{verification.id}</h2>
      
      <div className="review-progress">
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <p>Reviewed: {reviewedAreas} of {totalAreas} areas</p>
      </div>
      
      <div className="review-container">
        <div className="map-container">
          <MapComponent 
            projectBoundary={verification.project.boundary}
            forestChangeData={forestChangeData}
            center={verification.project.center}
            zoom={12}
          />
        </div>
        
        <div className="review-panel">
          <h3>Review Predictions</h3>
          
          {selectedArea ? (
            <>
              <div className="prediction-details">
                <h4>Area #{selectedArea}</h4>
                {predictionAreas.find(a => a.id === selectedArea) && (
                  <>
                    <p><strong>AI Classification:</strong> {predictionAreas.find(a => a.id === selectedArea).classification}</p>
                    <p><strong>Area:</strong> {predictionAreas.find(a => a.id === selectedArea).area_ha.toFixed(2)} ha</p>
                    <p><strong>Confidence:</strong> {(predictionAreas.find(a => a.id === selectedArea).confidence * 100).toFixed(1)}%</p>
                    <p><strong>Your Decision:</strong> {userDecisions[selectedArea] || 'Not reviewed'}</p>
                  </>
                )}
              </div>
              
              <div className="decision-buttons">
                <button 
                  className={userDecisions[selectedArea] === 'approve' ? 'active' : ''}
                  onClick={() => handleDecision('approve')}
                >
                  Approve
                </button>
                <button 
                  className={userDecisions[selectedArea] === 'reject' ? 'active' : ''}
                  onClick={() => handleDecision('reject')}
                >
                  Reject
                </button>
                <button 
                  className={userDecisions[selectedArea] === 'modify' ? 'active' : ''}
                  onClick={() => handleDecision('modify')}
                >
                  Modify
                </button>
              </div>
            </>
          ) : (
            <p>Select an area on the map to review</p>
          )}
          
          <div className="area-list">
            <h4>Areas to Review</h4>
            <ul>
              {predictionAreas.map(area => (
                <li 
                  key={area.id}
                  className={`
                    ${area.id === selectedArea ? 'selected' : ''}
                    ${userDecisions[area.id] ? 'reviewed' : ''}
                  `}
                  onClick={() => handleAreaSelect(area.id)}
                >
                  Area #{area.id} - {area.classification}
                  {userDecisions[area.id] && (
                    <span className="decision-indicator">
                      {userDecisions[area.id]}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
          
          <div className="reviewer-notes">
            <h4>Reviewer Notes</h4>
            <textarea 
              id="reviewer-notes"
              placeholder="Add your notes about this verification..."
              rows={4}
            ></textarea>
          </div>
          
          <button 
            className="submit-review"
            disabled={reviewedAreas < totalAreas || isLoading}
            onClick={handleSubmitReview}
          >
            {isLoading ? 'Submitting...' : 'Complete Review'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default VerificationReview;
```

## 4. System Integration and Deployment

### 4.1 Docker Configuration

#### Backend Dockerfile

```dockerfile
# backend.Dockerfile
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder stage
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY backend/ .
COPY ml/ /app/ml/

# Run as non-root user
RUN useradd -m appuser
USER appuser

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile

```dockerfile
# frontend.Dockerfile
FROM node:16-alpine AS builder

WORKDIR /app

# Install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source code
COPY frontend/ .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files to nginx
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgis/postgis:13-3.1
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: ${POSTGRES_DB:-carbon_credits}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  backend:
    build:
      context: ..
      dockerfile: docker/backend.Dockerfile
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-postgres}@postgres:5432/${POSTGRES_DB:-carbon_credits}
      SECRET_KEY: ${SECRET_KEY:-your-secret-key}
      ALGORITHM: HS256
      ACCESS_TOKEN_EXPIRE_MINUTES: 30
      BLOCKCHAIN_RPC_URL: ${BLOCKCHAIN_RPC_URL:-https://polygon-rpc.com}
      CONTRACT_ADDRESS: ${CONTRACT_ADDRESS}
      WALLET_ADDRESS: ${WALLET_ADDRESS}
      PRIVATE_KEY: ${PRIVATE_KEY}
      CHAIN_ID: ${CHAIN_ID:-137}
    volumes:
      - satellite_data:/app/data/satellite
      - verification_results:/app/data/verification
      - ml_models:/app/ml/models
    ports:
      - "8000:8000"
  
  frontend:
    build:
      context: ..
      dockerfile: docker/frontend.Dockerfile
    depends_on:
      - backend
    ports:
      - "80:80"
    environment:
      REACT_APP_API_URL: ${API_URL:-http://localhost:8000/api}

volumes:
  postgres_data:
  satellite_data:
  verification_results:
  ml_models:
```

### 4.2 Deployment Script

```bash
#!/bin/bash
# deploy.sh

# Set environment variables
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your-secure-password
export POSTGRES_DB=carbon_credits
export SECRET_KEY=your-secret-key
export BLOCKCHAIN_RPC_URL=https://polygon-rpc.com
export CONTRACT_ADDRESS=0x...
export WALLET_ADDRESS=0x...
export PRIVATE_KEY=your-private-key
export CHAIN_ID=137
export API_URL=https://api.your-domain.com

# Pull latest code
git pull

# Build and start containers
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up -d

# Run database migrations if needed
docker-compose -f docker/docker-compose.yml exec backend python -c "from app.core.database import Base, engine; Base.metadata.create_all(bind=engine)"

echo "Deployment completed successfully!"
```

### 4.3 Monitoring Setup

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'backend'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['backend:8000']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

## 5. Final Steps and Considerations

### 5.1 Security Hardening

#### CORS Configuration

```python
# Update backend/app/core/config.py
from pydantic import BaseSettings, AnyHttpUrl
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Carbon Credit Verification API"
    API_V1_STR: str = "/api"
    
    # CORS
    CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:80",
    ]
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/carbon_credits")
    
    # Blockchain
    BLOCKCHAIN_RPC_URL: str = os.getenv("BLOCKCHAIN_RPC_URL", "https://polygon-rpc.com")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")
    WALLET_ADDRESS: str = os.getenv("WALLET_ADDRESS", "")
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    CHAIN_ID: int = int(os.getenv("CHAIN_ID", "137"))
    
    # ML
    ML_MODEL_DIR: str = os.getenv("ML_MODEL_DIR", "../ml/models")
    SATELLITE_DATA_DIR: str = os.getenv("SATELLITE_DATA_DIR", "../data/satellite")
    VERIFICATION_RESULTS_DIR: str = os.getenv("VERIFICATION_RESULTS_DIR", "../data/verification")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    class Config:
        case_sensitive = True

settings = Settings()
```

#### Secure Headers Middleware

```python
# Add to backend/main.py
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Redirect HTTP to HTTPS in production
if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# Only allow specific hosts
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["your-domain.com", "*.your-domain.com"]
)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
```

### 5.2 Performance Optimization

#### Database Indexing

```sql
-- Run in PostgreSQL
-- Index for projects table
CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_created_at ON projects(created_at);

-- Index for verifications table
CREATE INDEX idx_verifications_project_id ON verifications(project_id);
CREATE INDEX idx_verifications_status ON verifications(status);
CREATE INDEX idx_verifications_created_at ON verifications(created_at);
```

#### Frontend Optimization

```javascript
// Add to frontend/src/index.js
import { lazy, Suspense } from 'react';

// Lazy load components
const Dashboard = lazy(() => import('./pages/Dashboard'));
const ProjectDetail = lazy(() => import('./pages/ProjectDetail'));
const Verification = lazy(() => import('./pages/Verification'));

// In your router
<Suspense fallback={<div>Loading...</div>}>
  <Route path="/dashboard" element={<Dashboard />} />
  <Route path="/projects/:id" element={<ProjectDetail />} />
  <Route path="/verification/:id" element={<Verification />} />
</Suspense>
```

## 6. Implementation Timeline

### Week 1: ML Model Training and Evaluation
- **Days 1-2**: Data preparation and acquisition
- **Days 3-5**: Model training and optimization

### Week 2: Backend Development
- **Days 1-2**: Database and core services setup
- **Days 3-5**: ML integration and API development

### Week 3: Frontend Development
- **Days 1-2**: Core UI components
- **Days 3-5**: Advanced features

### Week 4: Integration and Testing
- **Days 1-2**: System integration
- **Days 3-4**: Testing and optimization
- **Day 5**: Documentation and deployment preparation

### Week 5: Deployment and Launch
- **Days 1-2**: Staging deployment
- **Days 3-4**: Production deployment
- **Day 5**: Launch and handover

## 7. Key Considerations and Next Steps

### Critical Success Factors
1. **Data Quality and Availability**
2. **ML Model Performance**
3. **Scalability Considerations**
4. **Regulatory Compliance**
5. **User Experience**

### Immediate Next Steps
1. **Environment Setup**
2. **Data Acquisition**
3. **Initial ML Model Training**
4. **Backend Development Kickoff**

### Risk Mitigation Strategies
1. **Technical Risks**
2. **Project Management Risks**
3. **Business Risks**

### Long-term Considerations
1. **Continuous Improvement**
2. **Expansion Opportunities**
3. **Sustainability**
