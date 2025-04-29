# Carbon Credit Verification SaaS - Local Setup Guide

This guide will help you set up and run the Carbon Credit Verification SaaS application on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:
- Docker and Docker Compose
- Node.js (v16 or higher)
- npm (v8 or higher)
- Python 3.10+
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/carbon-credit-verification.git
cd carbon-credit-verification
```

### 2. Backend Setup

#### Set up environment variables

```bash
cd backend
cp .env.example .env
```

Edit the `.env` file to configure your database and other settings.

#### Start the backend services

```bash
docker-compose -f ../docker/docker-compose.yml up -d db
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at http://localhost:8000

### 3. Frontend Setup

In a new terminal:

```bash
cd frontend
npm install
npm start
```

The frontend application will be available at http://localhost:3000

## Using the Application

1. Open your browser and navigate to http://localhost:3000
2. Register a new account or use the default credentials:
   - Email: admin@example.com
   - Password: password123
3. Follow the user guide to create projects and verifications

## Development

### Backend Development

- API documentation is available at http://localhost:8000/docs
- Database migrations are handled automatically
- To add new dependencies, update the requirements.txt file

### Frontend Development

- The React application uses Redux for state management
- To add new dependencies, use `npm install --save package-name`
- The application is configured with hot reloading for development

## Troubleshooting

### Common Issues

1. **Database connection errors**: Ensure the PostgreSQL container is running
   ```bash
   docker ps
   ```

2. **Frontend build errors**: Clear node_modules and reinstall
   ```bash
   rm -rf node_modules
   npm install
   ```

3. **CORS errors**: Ensure the backend is running and accessible

4. **Authentication issues**: Check the .env file for correct JWT settings

## Additional Resources

- See the `final_documentation.md` for detailed technical documentation
- See the `user_guide.md` for application usage instructions

## Support

If you encounter any issues not covered in this guide, please contact support at support@carboncreditverification.com
