# Carbon Credit Verification SaaS Application

This README provides an overview of the Carbon Credit Verification SaaS application, a comprehensive solution for verifying carbon credits using satellite imagery, machine learning, and blockchain technology.

## Project Overview

The Carbon Credit Verification SaaS application is designed to provide transparent, reliable verification of carbon sequestration projects. It combines satellite imagery analysis, machine learning, and blockchain certification to create a trustworthy system for carbon credit verification.

Key features include:
- Forest cover change detection using Sentinel-2 satellite imagery
- Carbon sequestration estimation
- Explainable AI for transparent decision-making
- Blockchain certification for immutable verification records
- Human-in-the-loop verification workflow
- Interactive mapping and visualization

## Repository Structure

```
carbon_credit_project/
├── backend/               # FastAPI backend
│   ├── app/               # Application code
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Core functionality
│   │   ├── models/        # Database models
│   │   ├── schemas/       # Pydantic schemas
│   │   └── services/      # Business logic
│   ├── main.py            # Application entry point
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
│   ├── data/              # Data storage
│   ├── inference/         # Inference scripts
│   ├── models/            # Trained models
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── blockchain/            # Blockchain integration
├── docker/                # Docker configuration
│   ├── docker-compose.yml # Docker Compose configuration
│   ├── backend.Dockerfile # Backend Dockerfile
│   └── frontend.Dockerfile # Frontend Dockerfile
├── documentation/         # Project documentation
├── start_app.sh           # Script to start both frontend and backend
└── start.sh               # Startup script
```

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: React with TypeScript
- **Database**: PostgreSQL with PostGIS
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
cd carbon-credit-verification
```

2. Run the startup script:
```bash
chmod +x start.sh
./start.sh
```

This will:
- Set up the environment
- Build and start Docker containers
- Initialize the database
- Start the application

3. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API documentation: http://localhost:8000/docs

### Local Development without Docker

If you want to run the application locally without Docker:

1. Set up the local development environment:
```bash
chmod +x local_dev_setup.sh
./local_dev_setup.sh
```

2. Start both frontend and backend servers with a single command:
```bash
./start_app.sh
```

This will:
- Initialize the SQLite database if needed
- Start the backend server on port 8000
- Start the frontend server on port 3000
- Provide a convenient way to stop both servers with Ctrl+C

## Documentation

For more detailed information, please refer to:
- [User Guide](./user_guide.md) - Instructions for using the application
- [Technical Documentation](./final_documentation.md) - Detailed technical documentation
- [Local Setup Guide](./local_setup_guide.md) - Guide for local development setup

## License

This project is licensed under the MIT License - see the LICENSE file for details.
