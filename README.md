# Carbon Credit Verification SaaS Application

This README provides an overview of the Carbon Credit Verification application — an **honest
prototype** that verifies forest-related carbon projects from uploaded satellite imagery using
trained machine-learning models, with a human-in-the-loop verification workflow.

> **Status:** This is a working prototype, **not** a production or commercial product. For an
> accurate account of what is real, what is disabled, and the real model metrics, read
> [STATUS.md](./STATUS.md).

## Project Overview

The application lets a reviewer upload Sentinel-2 GeoTIFF imagery, runs genuine forest-cover and
change-detection inference on it, computes a carbon estimate, and records a human-reviewed
verification. Prototype-grade model performance: forest-cover F1 ≈ 0.49, change-detection
F1 ≈ 0.60 (source: `ml/evaluation_results/*.csv`, `ml/inference/production_inference.py`).

Real, working features:
- **Authentication & RBAC** — password login with opaque bearer tokens (server-side token store,
  not JWT) and role-based access control
- **Project management** — create and manage carbon-credit projects
- **Real ML inference on uploaded imagery** — forest-cover segmentation and before/after change
  detection with trained PyTorch U-Net models
- **Carbon estimation** — computed from real model output (IPCC above-ground biomass figure)
- **Human-in-the-loop verification workflow** — verification records are never fabricated
- **Interactive mapping and visualization**

Not yet real / disabled (see [STATUS.md](./STATUS.md) for details):
- **Explainable AI** — disabled in this build (the genuine code is not wired to real inputs)
- **Coordinate-based analysis** — removed (no imagery-from-coordinates pipeline exists)
- **Blockchain certification** — the Solidity contract is real but disabled until configured with
  a deployed address and signing key

## Repository Structure

```
carbon_credit_project/
├── backend/               # FastAPI backend
│   ├── main.py            # Single-file FastAPI application
│   ├── services/          # Business logic services
│   ├── utils/             # Utility functions
│   └── requirements.txt   # Python dependencies
├── frontend/              # React frontend
│   ├── public/            # Static files
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API services
│   │   └── store/         # Redux store
│   └── package.json       # Node.js dependencies
├── ml/                    # Machine learning components
│   ├── data/              # ML data storage
│   │   ├── sentinel2_downloads/  # Raw Sentinel-2 imagery
│   │   ├── hansen_downloads/     # Hansen forest data
│   │   └── prepared/            # Processed data
│   │       ├── change_labels/   # Change detection outputs
│   │       ├── s2_stacks/       # Processed Sentinel-2 stacks
│   │       └── quicklooks/      # Visualization outputs
│   ├── inference/         # Inference scripts
│   ├── models/            # Trained models
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── database/              # Database files
│   └── carbon_credits.db  # SQLite database
├── blockchain/            # Blockchain integration
├── docker/                # Docker configuration
│   ├── docker-compose.yml # Docker Compose configuration
│   ├── backend.Dockerfile # Backend Dockerfile
│   └── frontend.Dockerfile # Frontend Dockerfile
├── documentation/         # Project documentation
├── run_app.sh             # Script to start both frontend and backend
├── stop_app.sh            # Script to stop all services
└── scripts/               # Additional utility scripts
```

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: React with JavaScript
- **Database**: SQLite (development), PostgreSQL with PostGIS (production)
- **Machine Learning**: PyTorch, scikit-learn
- **Explainable AI**: SHAP, LIME, Captum
- **Blockchain**: Polygon (Ethereum L2)
- **Containerization**: Docker
- **Mapping**: Leaflet.js
- **Data Visualization**: D3.js, Recharts

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git
- Python 3.10+
- Node.js 16+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd carbon_credit_project
```

2. Run with Docker (recommended):
```bash
cd docker
docker-compose up --build
```

This will:
- Build and start Docker containers
- Initialize the PostgreSQL database with PostGIS
- Start the application

3. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API documentation: http://localhost:8000/docs

### Local Development without Docker

If you want to run the application locally without Docker:

1. Set up the Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

3. Start both frontend and backend servers:
```bash
chmod +x run_app.sh
./run_app.sh
```

This will:
- Initialize the SQLite database automatically
- Start the backend server on port 8000
- Start the frontend server on port 3000
- Provide a convenient way to stop both servers with Ctrl+C

To stop the servers separately:
```bash
./stop_app.sh
```

## Documentation

For more detailed information, please refer to:
- [User Guide](./user_guide.md) - Instructions for using the application
- [Technical Documentation](./final_documentation.md) - Detailed technical documentation
- [Local Setup Guide](./local_setup_guide.md) - Guide for local development setup

## License

This project is licensed under the MIT License - see the LICENSE file for details.
