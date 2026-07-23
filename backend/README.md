# Carbon Credit Verification API

A FastAPI backend for carbon credit verification and management.

## 🚀 Features

- **Professional Architecture**: Clean, modular code structure with separation of concerns
- **Security First**: JWT authentication, password hashing, input validation, SQL injection protection
- **Database Integration**: SQLite with proper connection management and error handling
- **Comprehensive Validation**: Pydantic models with custom validators
- **Error Handling**: Global exception handlers with proper logging
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Logging**: Structured logging with configurable levels
- **Testing Ready**: Pytest integration with async support
- **Deployment Support**: CORS, security middleware, environment configuration

## 🏗️ Architecture

```
backend/
├── app/
│   ├── api/v1/                 # API routes
│   │   ├── endpoints/          # Individual endpoint modules
│   │   └── api.py             # Main API router
│   ├── core/                   # Core functionality
│   │   ├── config.py          # Configuration management
│   │   └── security.py        # Security utilities
│   ├── db/                     # Database layer
│   │   └── database.py        # Database manager
│   ├── models/                 # Pydantic models
│   │   ├── user.py            # User models
│   │   └── project.py         # Project models
│   └── services/               # Business logic
│       ├── auth_service.py    # Authentication service
│       └── project_service.py # Project service
├── main.py                     # Application entry point
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🛠️ Installation

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables** (optional):
   ```bash
   export SECRET_KEY="your-secret-key"
   export DATABASE_URL="sqlite:///./database/carbon_credits.db"
   export LOG_LEVEL="INFO"
   ```

## 🚀 Running the Application

### Development Mode
```bash
python main.py
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### With Custom Configuration
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
```

## 📚 API Documentation

Once running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 🔐 Authentication

The API uses JWT (JSON Web Tokens) for authentication:

1. **Register**: `POST /api/v1/auth/register`
2. **Login**: `POST /api/v1/auth/login`
3. **Get User Info**: `GET /api/v1/auth/me`
4. **Logout**: `POST /api/v1/auth/logout`

### Example Usage

```bash
# Register a new user
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123",
    "full_name": "John Doe",
    "role": "Project Developer"
  }'

# Login
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123"

# Use the returned token for authenticated requests
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/auth/me"
```

## 📊 Project Management

### Endpoints

- `GET /api/v1/projects` - List user's projects (with pagination)
- `POST /api/v1/projects` - Create a new project
- `GET /api/v1/projects/{id}` - Get project details
- `PUT /api/v1/projects/{id}` - Update project
- `DELETE /api/v1/projects/{id}` - Delete project

### Example Project Creation

```bash
curl -X POST "http://localhost:8000/api/v1/projects" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Amazon Reforestation Project",
    "description": "Large-scale reforestation initiative",
    "location_name": "Amazon Basin, Brazil",
    "area_size": 1000.5,
    "project_type": "Reforestation"
  }'
```

## 🔧 Configuration

The application supports configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `your-secret-key-change-in-production` | JWT secret key |
| `DATABASE_URL` | `sqlite:///./database/carbon_credits.db` | Database connection string |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `11520` (8 days) | Token expiration time |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_auth.py
```

## 📝 Logging

The application uses structured logging with the following levels:
- `DEBUG`: Detailed information for debugging
- `INFO`: General information about application flow
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

Logs are written to both console and `app.log` file.

## 🔒 Security Features

- **Password Hashing**: Bcrypt with salt
- **JWT Tokens**: Secure token-based authentication
- **Input Validation**: Comprehensive Pydantic validation
- **SQL Injection Protection**: Parameterized queries
- **CORS Configuration**: Configurable cross-origin requests
- **Rate Limiting Ready**: Middleware support for rate limiting
- **Security Headers**: Trusted host middleware

## 🚀 Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

```bash
SECRET_KEY=your-super-secret-production-key
DATABASE_URL=postgresql://user:pass@localhost/dbname
LOG_LEVEL=WARNING
ENVIRONMENT=production
```

## 📈 Performance

- **Connection Pooling**: Efficient database connection management
- **Async Support**: Full async/await support for high concurrency
- **Caching**: Token caching for improved performance
- **Pagination**: Efficient data pagination for large datasets

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation
4. Follow PEP 8 style guidelines
5. Add proper logging and error handling

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Check the API documentation at `/api/v1/docs`
- Review the logs in `app.log`
- Check the health endpoint at `/health` 