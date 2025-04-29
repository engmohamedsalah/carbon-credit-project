## Error Handling Strategy for Carbon Credit Verification SaaS

A robust error handling strategy is essential for creating a reliable and user-friendly application. This section outlines best practices and specific implementations for handling errors across different components of the Carbon Credit Verification SaaS.

### 1. General Principles

-   **Consistency**: Use consistent error response formats across the API.
-   **Logging**: Log all significant errors with sufficient context (timestamp, user ID, request ID, stack trace) for debugging.
-   **User Feedback**: Provide clear, user-friendly error messages in the frontend, avoiding technical jargon.
-   **Fail Gracefully**: Ensure the application remains functional or provides informative messages even when parts of it fail.
-   **Monitoring**: Integrate error reporting with monitoring tools to track error rates and identify patterns.

### 2. Backend (FastAPI) Error Handling

#### 2.1 Centralized Exception Handling

FastAPI provides `@app.exception_handler` decorators to centralize error handling.

```python
# backend/app/core/exceptions.py
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class CustomAPIException(HTTPException):
    """Base class for custom API exceptions."""
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

# Define specific custom exceptions
class ProjectNotFoundException(CustomAPIException):
    def __init__(self, project_id: int):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, 
                         detail=f"Project with id {project_id} not found",
                         error_code="PROJECT_NOT_FOUND")

class VerificationFailedException(CustomAPIException):
    def __init__(self, verification_id: int, reason: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                         detail=f"Verification {verification_id} failed: {reason}",
                         error_code="VERIFICATION_FAILED")

# Add exception handlers in backend/main.py
from app.core.exceptions import CustomAPIException, ProjectNotFoundException

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(CustomAPIException)
async def custom_api_exception_handler(request: Request, exc: CustomAPIException):
    logger.error(f"Custom API Exception: {exc.status_code} - {exc.detail} (Code: {exc.error_code})", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": exc.error_code},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Catch-all for unexpected errors
    logger.critical(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred."},
    )
```

#### 2.2 Handling Specific Error Types

-   **Database Errors (SQLAlchemy)**:
    -   Wrap database operations in `try...except` blocks.
    -   Catch specific SQLAlchemy exceptions (e.g., `IntegrityError`, `NoResultFound`).
    -   Rollback transactions on error.

    ```python
    # backend/app/services/project_service.py
    from sqlalchemy.orm import Session
    from sqlalchemy.exc import IntegrityError, NoResultFound
    from app.core.exceptions import ProjectNotFoundException
    
    def get_project(db: Session, project_id: int):
        try:
            project = db.query(models.Project).filter(models.Project.id == project_id).one()
            return project
        except NoResultFound:
            raise ProjectNotFoundException(project_id)
        except Exception as e:
            logger.error(f"Database error fetching project {project_id}: {e}", exc_info=True)
            raise CustomAPIException(status_code=500, detail="Database error")
    
    def create_project(db: Session, project_data):
        try:
            db_project = models.Project(**project_data.dict())
            db.add(db_project)
            db.commit()
            db.refresh(db_project)
            return db_project
        except IntegrityError as e:
            db.rollback()
            logger.warning(f"Integrity error creating project: {e}")
            raise CustomAPIException(status_code=400, detail="Project creation failed due to data conflict.")
        except Exception as e:
            db.rollback()
            logger.error(f"Database error creating project: {e}", exc_info=True)
            raise CustomAPIException(status_code=500, detail="Database error")
    ```

-   **ML Service Errors**:
    -   Handle errors during model loading or prediction within the `MLService`.
    -   Update verification status to `FAILED` with an error message.

    ```python
    # backend/app/services/ml_service.py
    def predict_forest_change(self, image_path, output_dir):
        try:
            self.load_model()
            # Check if image_path exists and is valid
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Input image path not found: {image_path}")
            
            results = self.model.process_satellite_image(image_path, output_dir)
            return results
        except FileNotFoundError as e:
            logger.error(f"ML Prediction Error: {e}")
            raise VerificationFailedException(verification_id=-1, reason=str(e)) # Pass verification_id if available
        except Exception as e:
            logger.error(f"Unexpected ML Prediction Error: {e}", exc_info=True)
            raise VerificationFailedException(verification_id=-1, reason="ML model prediction failed")
    ```

-   **Blockchain Service Errors**:
    -   Handle network connection errors, transaction failures (e.g., out of gas, reverted), and contract interaction errors.

    ```python
    # backend/app/services/blockchain_service.py
    from web3.exceptions import TransactionNotFound, ContractLogicError
    
    def issue_certificate(self, verification_id, carbon_impact, metadata_uri):
        try:
            if not self.w3.is_connected():
                raise ConnectionError("Not connected to blockchain network")
            
            # ... build and sign transaction ...
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if tx_receipt["status"] == 0:
                raise ContractLogicError("Blockchain transaction reverted")
                
            return { ... }
        except ConnectionError as e:
            logger.error(f"Blockchain connection error: {e}")
            raise CustomAPIException(status_code=503, detail="Blockchain network unavailable")
        except (TransactionNotFound, ContractLogicError, Exception) as e:
            logger.error(f"Blockchain transaction error: {e}", exc_info=True)
            raise CustomAPIException(status_code=500, detail=f"Blockchain transaction failed: {e}")
    ```

-   **Background Task Errors**: Ensure errors in background tasks (like ML processing) are logged and the status of the related entity (e.g., Verification) is updated appropriately.

### 3. Frontend (React) Error Handling

#### 3.1 Displaying User-Friendly Messages

-   Use a global notification system (e.g., using `react-toastify` or a custom context/provider) to display errors.
-   Map API error codes (if provided) to user-friendly messages.
-   Avoid showing raw error details or stack traces to the user.

```javascript
// src/components/Notifications.js
import React from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

export const showError = (message) => {
  toast.error(message || 'An unexpected error occurred.');
};

export const showSuccess = (message) => {
  toast.success(message);
};

const Notifications = () => {
  return (
    <ToastContainer
      position="top-right"
      autoClose={5000}
      hideProgressBar={false}
      newestOnTop={false}
      closeOnClick
      rtl={false}
      pauseOnFocusLoss
      draggable
      pauseOnHover
    />
  );
};

export default Notifications;

// In App.js
import Notifications from './components/Notifications';

function App() {
  return (
    <Provider store={store}>
      <Router>
        <Layout>
          <Notifications />
          {/* Routes... */}
        </Layout>
      </Router>
    </Provider>
  );
}
```

#### 3.2 Handling API Errors (RTK Query)

RTK Query provides mechanisms to handle errors from API calls.

```javascript
// src/pages/ProjectDetail.js
import React, { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useGetProjectQuery } from '../services/api';
import { showError } from '../components/Notifications';

const ProjectDetail = () => {
  const { id } = useParams();
  const { data: project, error, isLoading, isError } = useGetProjectQuery(id);

  useEffect(() => {
    if (isError && error) {
      // Map error to user-friendly message
      let errorMessage = 'Failed to load project details.';
      if (error.status === 404) {
        errorMessage = 'Project not found.';
      } else if (error.data?.detail) {
        errorMessage = error.data.detail;
      }
      showError(errorMessage);
      // Optionally redirect or show specific error component
    }
  }, [isError, error]);

  if (isLoading) return <div>Loading project...</div>;
  if (isError) return <div>Error loading project. Please try again later.</div>; // Fallback UI
  if (!project) return <div>Project not found.</div>;

  return (
    <div>
      <h1>{project.name}</h1>
      {/* Display project details */}
    </div>
  );
};

export default ProjectDetail;
```

#### 3.3 State Management for Errors

-   Use component state or Redux slices to manage loading and error states for specific actions.
-   Provide visual feedback to the user during loading and error states (e.g., disabling buttons, showing spinners or error messages).

```javascript
// Example using component state
const CreateProjectForm = () => {
  const [createProject, { isLoading, isError, error }] = useCreateProjectMutation();
  const [formData, setFormData] = useState({ name: '', description: '' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await createProject(formData).unwrap();
      showSuccess('Project created successfully!');
      // Redirect or clear form
    } catch (err) {
      showError(err.data?.detail || 'Failed to create project.');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Form fields */}
      {isError && <div className="error-message">{error.data?.detail || 'Creation failed'}</div>}
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Creating...' : 'Create Project'}
      </button>
    </form>
  );
};
```

### 4. ML Component Error Handling

-   **Data Errors**: Implement checks in `data_preparation.py` and `ForestChangeDataset` for missing files, incorrect formats, or inconsistent data (e.g., mismatched CRS, resolution).
-   **Model Loading**: Handle `FileNotFoundError` or corrupted model files gracefully during `ForestChangePredictor` initialization.
-   **Prediction Errors**: Catch exceptions during the `process_satellite_image` method, such as memory errors for large images or issues with specific patches.

### 5. Logging and Monitoring Integration

-   **Structured Logging**: Use a consistent JSON format for logs (as shown in the previous guide) to facilitate parsing by log aggregation tools.
-   **Log Aggregation**: Use tools like Elasticsearch, Logstash, Kibana (ELK Stack) or cloud-based services (AWS CloudWatch Logs, Google Cloud Logging) to collect and search logs from all components.
-   **Error Tracking**: Integrate with error tracking services like Sentry or Rollbar. These services capture unhandled exceptions, group similar errors, and provide context.

```python
# Example Sentry integration in backend/main.py
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    traces_sample_rate=1.0, # Adjust in production
    environment=os.getenv("ENVIRONMENT", "development")
)

# Add Sentry middleware (usually one of the first)
app.add_middleware(SentryAsgiMiddleware)
```

-   **Monitoring Dashboards**: Create dashboards (e.g., in Grafana, Datadog) to visualize error rates, types of errors, and affected services over time.
-   **Alerting**: Set up alerts based on error rates or specific critical errors to notify the development team promptly.

By implementing this multi-faceted error handling strategy, the Carbon Credit Verification SaaS application will be more resilient, easier to debug, and provide a better experience for users when issues inevitably arise.
