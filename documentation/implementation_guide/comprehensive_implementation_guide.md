
---

## 9. Error Handling Strategy

A robust error handling strategy is essential for creating a reliable and user-friendly application. This section outlines best practices and specific implementations for handling errors across different components of the Carbon Credit Verification SaaS.

### 9.1 General Principles

-   **Consistency**: Use consistent error response formats across the API (e.g., `{"detail": "Error message", "error_code": "SPECIFIC_CODE"}`).
-   **Logging**: Log all significant errors with sufficient context (timestamp, user ID, request ID, stack trace, relevant data) for debugging.
-   **User Feedback**: Provide clear, concise, and actionable error messages in the frontend, avoiding technical jargon.
-   **Fail Gracefully**: Ensure the application remains functional or provides informative messages even when parts of it fail (e.g., degraded functionality).
-   **Monitoring**: Integrate error reporting with monitoring tools (e.g., Sentry, Datadog) to track error rates, identify patterns, and alert developers.

### 9.2 Backend (FastAPI) Error Handling

#### 9.2.1 Centralized Exception Handling

FastAPI provides `@app.exception_handler` decorators to centralize error handling.

```python
# backend/app/core/exceptions.py
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class CustomAPIException(HTTPException):
    """Base class for custom API exceptions with optional error codes."""
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

# Define specific custom exceptions inheriting from CustomAPIException
class ProjectNotFoundException(CustomAPIException):
    def __init__(self, project_id: int):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, 
                         detail=f"Project with id {project_id} not found",
                         error_code="PROJECT_NOT_FOUND")

class VerificationFailedException(CustomAPIException):
    def __init__(self, verification_id: int, reason: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 400/422 if user input related
                         detail=f"Verification {verification_id} failed: {reason}",
                         error_code="VERIFICATION_FAILED")

class BlockchainTransactionError(CustomAPIException):
    def __init__(self, detail: str = "Blockchain transaction failed", tx_hash: str = None):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                         detail=detail,
                         error_code="BLOCKCHAIN_TX_ERROR")
        self.tx_hash = tx_hash # Add extra context if needed

# Add exception handlers in backend/main.py
from fastapi import FastAPI
from app.core.exceptions import CustomAPIException, ProjectNotFoundException # Import others as needed

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Log standard HTTP exceptions (like validation errors from FastAPI)
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(CustomAPIException)
async def custom_api_exception_handler(request: Request, exc: CustomAPIException):
    # Log custom application errors
    log_message = f"Custom API Exception: {exc.status_code} - {exc.detail}"
    if exc.error_code:
        log_message += f" (Code: {exc.error_code})"
    logger.error(log_message, exc_info=True) # Include stack trace for errors
    
    content = {"detail": exc.detail}
    if exc.error_code:
        content["error_code"] = exc.error_code
        
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Catch-all for unexpected internal server errors
    logger.critical(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred.", "error_code": "INTERNAL_SERVER_ERROR"},
    )
```

#### 9.2.2 Handling Specific Error Types

-   **Database Errors (SQLAlchemy)**:
    -   Wrap database operations in `try...except` blocks within service functions.
    -   Catch specific SQLAlchemy exceptions (e.g., `IntegrityError` for uniqueness violations, `NoResultFound` for `.one()` calls).
    -   Rollback the session on error to prevent inconsistent state.
    -   Raise appropriate `CustomAPIException` subclasses.

    ```python
    # backend/app/services/project_service.py
    from sqlalchemy.orm import Session
    from sqlalchemy.exc import IntegrityError, NoResultFound
    from app.core.exceptions import ProjectNotFoundException, CustomAPIException
    import logging

    logger = logging.getLogger(__name__)
    
    def get_project(db: Session, project_id: int):
        try:
            # Use .first() instead of .one() to handle not found more gracefully
            project = db.query(models.Project).filter(models.Project.id == project_id).first()
            if not project:
                 raise ProjectNotFoundException(project_id)
            return project
        except Exception as e: # Catch broader exceptions for logging
            logger.error(f"Database error fetching project {project_id}: {e}", exc_info=True)
            # Re-raise a generic server error
            raise CustomAPIException(status_code=500, detail="Database error retrieving project.", error_code="DB_FETCH_ERROR")
    
    def create_project(db: Session, project_in, owner_id):
        try:
            db_project = models.Project(**project_in.dict(), owner_id=owner_id)
            db.add(db_project)
            db.commit()
            db.refresh(db_project)
            return db_project
        except IntegrityError as e:
            db.rollback()
            logger.warning(f"Integrity error creating project: {e.original}") # Log original DB error
            # Check specific constraint violation if possible
            raise CustomAPIException(status_code=409, detail="Project creation failed due to data conflict (e.g., duplicate name).", error_code="DB_CONFLICT")
        except Exception as e:
            db.rollback()
            logger.error(f"Database error creating project: {e}", exc_info=True)
            raise CustomAPIException(status_code=500, detail="Database error during project creation.", error_code="DB_CREATE_ERROR")
    ```

-   **ML Service Errors**:
    -   Handle errors during model loading, data preprocessing, or prediction within the ML service/task.
    -   Update the corresponding `Verification` status to `FAILED` in the database, storing an error message.
    -   Raise `VerificationFailedException` or log appropriately.

    ```python
    # backend/app/services/ml_service.py (or background task)
    def run_ml_verification(verification_id: int, image_path: str):
        db = SessionLocal()
        try:
            verification = db.query(models.Verification).filter(models.Verification.id == verification_id).first()
            if not verification:
                logger.error(f"Verification {verification_id} not found for ML task.")
                return

            verification.status = "PROCESSING"
            db.commit()

            # --- ML Prediction Logic --- 
            predictor = ForestChangePredictor()
            results = predictor.predict_forest_change(image_path)
            # --- End ML Logic --- 

            verification.results = results # Store results
            verification.status = "NEEDS_REVIEW" # Or COMPLETED
            db.commit()
            logger.info(f"ML verification completed for {verification_id}")

        except FileNotFoundError as e:
            logger.error(f"ML Error (Verification {verification_id}): Input file not found - {e}")
            if verification:
                verification.status = "FAILED"
                verification.error_message = f"Input data not found: {e}"
                db.commit()
        except Exception as e:
            logger.error(f"ML Error (Verification {verification_id}): {e}", exc_info=True)
            if verification:
                verification.status = "FAILED"
                verification.error_message = f"ML processing failed: {e}"
                db.commit()
            # Optionally raise to notify task queue manager
            # raise VerificationFailedException(verification_id, str(e))
        finally:
            db.close()
    ```

-   **Blockchain Service Errors**:
    -   Handle network connection errors (`ConnectionError`), transaction failures (e.g., `TransactionNotFound`, `ContractLogicError`, out of gas), and contract interaction errors.
    -   Implement retries for transient network issues where appropriate.
    -   Raise `BlockchainTransactionError` or other specific exceptions.

    ```python
    # backend/app/services/blockchain_service.py
    from web3.exceptions import TransactionNotFound, ContractLogicError, ConnectionError
    from app.core.exceptions import BlockchainTransactionError, CustomAPIException
    import time

    MAX_RETRIES = 3
    RETRY_DELAY = 5 # seconds

    def issue_certificate(self, verification_id, metadata_uri, owner_address):
        for attempt in range(MAX_RETRIES):
            try:
                if not self.w3.is_connected():
                    raise ConnectionError("Not connected to blockchain network")
                
                contract_func = self.contract.functions.issueCertificate(
                    owner_address,
                    verification_id,
                    metadata_uri
                )
                
                # Estimate gas, build transaction
                tx_params = { ... } # nonce, gasPrice/maxFeePerGas, gas, from, chainId
                transaction = contract_func.build_transaction(tx_params)
                
                signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for receipt (consider async or background task for long waits)
                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
                
                if tx_receipt["status"] == 0:
                    # Transaction reverted
                    logger.error(f"Blockchain tx reverted for verification {verification_id}. Hash: {tx_hash.hex()}")
                    raise BlockchainTransactionError(detail="Transaction reverted by contract logic.", tx_hash=tx_hash.hex())
                    
                token_id = self._get_token_id_from_receipt(tx_receipt) # Helper to parse logs
                logger.info(f"Certificate issued for verification {verification_id}. Token ID: {token_id}, TxHash: {tx_hash.hex()}")
                return {"tx_hash": tx_hash.hex(), "token_id": token_id}

            except ConnectionError as e:
                logger.warning(f"Blockchain connection error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise CustomAPIException(status_code=503, detail="Blockchain network unavailable after retries.", error_code="BLOCKCHAIN_CONNECTION_ERROR")
                time.sleep(RETRY_DELAY)
            except (TransactionNotFound, ContractLogicError, ValueError) as e: # ValueError for gas issues etc.
                logger.error(f"Blockchain transaction error for verification {verification_id}: {e}", exc_info=True)
                raise BlockchainTransactionError(detail=f"Transaction failed: {e}")
            except Exception as e:
                logger.critical(f"Unexpected blockchain error for verification {verification_id}: {e}", exc_info=True)
                raise BlockchainTransactionError(detail=f"Unexpected blockchain error: {e}")
        # Should not be reached if exceptions are raised correctly
        raise BlockchainTransactionError(detail="Failed after multiple retries.")

    def _get_token_id_from_receipt(self, tx_receipt):
        # Logic to parse the CertificateIssued event logs from the receipt
        try:
            event_signature_hash = self.w3.keccak(text="CertificateIssued(uint256,uint256,address,string,uint256)").hex()
            for log in tx_receipt["logs"]:
                if log["topics"][0].hex() == event_signature_hash:
                    # Decode log data (adjust based on indexed/non-indexed params)
                    token_id = self.w3.to_int(hexstr=log["topics"][1].hex())
                    return token_id
            return None # Event not found
        except Exception as e:
            logger.error(f"Failed to parse token ID from receipt logs: {e}")
            return None
    ```

-   **Background Task Errors (Celery/RQ)**: Ensure the task queue framework handles exceptions properly. Log errors within the task and update the status of the related entity (e.g., `Verification`) in the database.

### 9.3 Frontend (React) Error Handling

#### 9.3.1 Displaying User-Friendly Messages

-   Use a global notification system (e.g., `react-toastify`, `Notistack`, or a custom context/provider) to display non-blocking error messages.
-   Map API error codes (`error_code` from backend) to user-friendly messages where possible.
-   For critical errors preventing functionality, display inline messages or dedicated error components.

```javascript
// src/components/Notifications.js (using react-toastify)
import React from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Function to map error codes/details to user messages
const getFriendlyErrorMessage = (error) => {
  if (error?.data?.error_code === 'PROJECT_NOT_FOUND') {
    return 'The requested project could not be found.';
  }
  if (error?.data?.error_code === 'DB_CONFLICT') {
    return 'A project with this name might already exist. Please check your input.';
  }
  if (error?.data?.detail) {
    // Use backend detail if specific and user-friendly enough
    return error.data.detail;
  }
  if (error?.status === 401) {
      return 'Authentication failed. Please log in again.';
  }
  if (error?.status === 403) {
      return 'You do not have permission to perform this action.';
  }
  // Generic fallback
  return 'An unexpected error occurred. Please try again later.';
};

export const showErrorToast = (error) => {
  const message = getFriendlyErrorMessage(error);
  toast.error(message);
};

export const showSuccessToast = (message) => {
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
      theme="colored" // Use colored theme for better visibility
    />
  );
};

export default Notifications;

// In App.js
// import Notifications from './components/Notifications';
// ... render <Notifications /> within the main layout ...
```

#### 9.3.2 Handling API Errors (RTK Query)

RTK Query provides `isError`, `error` flags in query/mutation hooks.

```javascript
// src/pages/ProjectDetail.js
import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useGetProjectQuery } from '../services/api';
import { showErrorToast } from '../components/Notifications';
import LoadingSpinner from '../components/LoadingSpinner'; // Assume this exists
import ErrorDisplay from '../components/ErrorDisplay'; // Assume this exists

const ProjectDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { data: project, error, isLoading, isError, refetch } = useGetProjectQuery(id);

  useEffect(() => {
    // Show toast for non-critical errors, but rely on inline display for critical ones
    if (isError && error && error.status !== 404) { // Don't toast for 404, handle inline
      showErrorToast(error);
    }
  }, [isError, error]);

  if (isLoading) return <LoadingSpinner message="Loading project details..." />;
  
  // Handle critical errors preventing display
  if (isError) {
      if (error.status === 404) {
          return <ErrorDisplay title="Project Not Found" message="The project you are looking for does not exist or you do not have permission to view it." onRetry={null} />;
      } else {
          return <ErrorDisplay title="Error Loading Project" message={getFriendlyErrorMessage(error)} onRetry={refetch} />;
      }
  }
  
  // Should not happen if isError is handled, but good practice
  if (!project) return <ErrorDisplay title="Project Not Found" message="Project data is unavailable." />;

  return (
    <div>
      <h1>{project.name}</h1>
      {/* Display project details */}
    </div>
  );
};

export default ProjectDetail;
```

#### 9.3.3 State Management for Errors

-   Use component state (`useState`) or Redux/RTK Query state (`isLoading`, `isError`) to manage loading and error states for specific actions (e.g., form submissions).
-   Provide visual feedback: disable buttons during loading, show inline error messages near form fields, use spinners.

```javascript
// Example form submission
import { useCreateProjectMutation } from '../services/api';
import { showErrorToast, showSuccessToast } from '../components/Notifications';

const CreateProjectForm = () => {
  const [createProject, { isLoading, error: mutationError }] = useCreateProjectMutation();
  const [formErrors, setFormErrors] = useState({}); // For field-specific errors

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormErrors({}); // Clear previous errors
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    // Add geometry data if using map input

    try {
      // Use unwrap to automatically throw error on failure
      const newProject = await createProject(data).unwrap(); 
      showSuccessToast(`Project '${newProject.name}' created successfully!`);
      // Redirect or clear form
    } catch (err) {
      showErrorToast(err); // Show generic toast
      // Optionally set specific form field errors based on err.data
      if (err.status === 422) { // Validation error
          // Assuming err.data.detail is an array of Pydantic errors
          const fieldErrors = {};
          err.data.detail.forEach(detail => {
              if (detail.loc && detail.loc.length > 1) {
                  fieldErrors[detail.loc[1]] = detail.msg;
              }
          });
          setFormErrors(fieldErrors);
      }
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Form fields with potential inline errors */}
      <div>
        <label htmlFor="name">Project Name:</label>
        <input type="text" id="name" name="name" required />
        {formErrors.name && <span className="error-text">{formErrors.name}</span>}
      </div>
      {/* ... other fields ... */}
      
      {/* Display general mutation error if not field-specific */}
      {mutationError && !Object.keys(formErrors).length && (
          <div className="error-message">{getFriendlyErrorMessage(mutationError)}</div>
      )}
      
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Creating...' : 'Create Project'}
      </button>
    </form>
  );
};
```

### 9.4 ML Component Error Handling

-   **Data Errors**: Implement robust checks in `data_preparation.py` and `Dataset` classes for missing files, incorrect formats, corrupted data, inconsistent CRS/resolution. Log errors and skip problematic data points where possible.
-   **Model Loading**: Handle `FileNotFoundError` or corrupted model files gracefully during predictor initialization. Log critical errors.
-   **Prediction Errors**: Catch exceptions during prediction (e.g., memory errors, invalid input dimensions). Log the error with context (image ID, tile coordinates) and potentially mark the specific area as failed.

### 9.5 Logging and Monitoring Integration

-   **Structured Logging**: Use JSON format for logs (e.g., using `python-json-logger`) including request IDs, user IDs, and relevant context.
-   **Log Aggregation**: Centralize logs using ELK Stack, Graylog, Datadog Logs, AWS CloudWatch Logs, or Google Cloud Logging.
-   **Error Tracking**: Integrate with Sentry, Rollbar, or similar services. Configure them to capture unhandled exceptions and log messages above a certain level (e.g., ERROR, CRITICAL).
    -   Set user context (`sentry_sdk.set_user(...)`) in backend requests.
    -   Use tags (`sentry_sdk.set_tag(...)`) for filtering errors (e.g., by component, feature).
-   **Monitoring Dashboards**: Create dashboards (Grafana, Datadog) visualizing API error rates (4xx, 5xx), specific error code occurrences, ML task failure rates, blockchain transaction errors.
-   **Alerting**: Configure alerts in your monitoring/error tracking system for spikes in error rates, specific critical errors (e.g., `INTERNAL_SERVER_ERROR`, `BLOCKCHAIN_CONNECTION_ERROR`), or high failure rates in background tasks.

By implementing this multi-faceted error handling strategy, the application will be more resilient, easier to debug, and provide a better experience for users when issues arise.



---

## 10. Data Lifecycle Management Strategy

Effective data lifecycle management (DLM) is crucial for controlling costs, ensuring compliance, and maintaining the performance of the Carbon Credit Verification SaaS application, especially given the potentially large volumes of satellite imagery and ML-generated data.

### 10.1 Purpose

This document outlines the strategy for managing data from its creation or acquisition through processing, storage, use, archiving, and eventual deletion.

### 10.2 Data Types and Sources

The primary data types involved are:

-   **User Data**: Account information (email, name, hashed password), roles. Stored in PostgreSQL.
-   **Project Data**: Project details (name, description), geographic boundaries (GeoJSON/PostGIS geometry), owner information, status. Stored in PostgreSQL.
-   **Satellite Imagery**: Raw and processed satellite images (e.g., Sentinel-2 GeoTIFFs). Acquired from providers (e.g., Copernicus Hub) or uploaded by users. Stored in object storage (e.g., AWS S3, MinIO).
-   **ML Training Data**: Prepared image patches and corresponding labels (e.g., Hansen data). Stored potentially in object storage or a dedicated file system during training.
-   **ML Models**: Trained model files (e.g., `.pth` files). Stored in object storage or a model registry.
-   **Verification Data**: Verification request details, status, timestamps, error messages. Stored in PostgreSQL.
-   **ML Inference Results**: Output prediction masks (GeoTIFFs), confidence maps, calculated metrics (e.g., forest loss area, carbon impact estimates), XAI visualizations. Stored in object storage, with key metrics potentially also in PostgreSQL.
-   **Blockchain Data**: Transaction hashes, certificate token IDs, links to metadata. Stored in PostgreSQL for quick reference; the source of truth is the blockchain itself.
-   **Metadata**: JSON files describing blockchain certificates. Stored on IPFS or persistent object storage.
-   **Application Logs**: System and error logs. Stored temporarily on disk, then aggregated in a central logging system (e.g., ELK stack, CloudWatch).

### 10.3 Storage Strategy

A tiered storage approach is recommended, balancing access speed and cost:

-   **Hot Storage (Fast Access, Higher Cost)**:
    -   **PostgreSQL Database**: For structured data requiring frequent access and transactional integrity (User, Project, Verification metadata, key results).
    -   **Object Storage (Standard Tier - e.g., S3 Standard, MinIO)**: For recently acquired satellite imagery, active ML models, recent verification results (masks, visualizations), and blockchain metadata needing frequent access.
-   **Warm Storage (Moderate Access, Lower Cost)**:
    -   **Object Storage (Infrequent Access Tier - e.g., S3 Standard-IA)**: For processed satellite imagery from completed verifications (older than ~6 months), older verification results, historical ML models. Data is still readily accessible but at a lower cost.
-   **Cold Storage (Slow Access, Lowest Cost)**:
    -   **Object Storage (Archive Tier - e.g., S3 Glacier Instant Retrieval, S3 Glacier Flexible Retrieval, S3 Glacier Deep Archive)**: For archiving raw satellite data from old projects, very old verification results, or data required for long-term compliance but rarely accessed. Choose tier based on retrieval time needs (Instant vs. minutes/hours).
    -   **IPFS**: For permanent, immutable storage of blockchain certificate metadata JSON files.

### 10.4 Data Flow and Processing

1.  **Acquisition/Upload**: Raw satellite imagery downloaded/uploaded -> Hot Object Storage.
2.  **ML Preprocessing**: Intermediate files -> Temporary storage (local disk or Hot Object Storage), deleted after use.
3.  **ML Inference**: Model predicts changes. Key metrics -> PostgreSQL. Detailed results (masks, maps) -> Hot Object Storage.
4.  **Verification Review**: Reviewers access results from Hot Storage.
5.  **Certificate Issuance**: Metadata JSON -> IPFS/Hot Object Storage. Link -> Blockchain. Tx details -> PostgreSQL.
6.  **Reporting**: Users access data primarily from PostgreSQL and recent results from Hot Object Storage.
7.  **Archiving**: Automated policies move data from Hot -> Warm -> Cold Object Storage based on age/access.

### 10.5 Access Control

-   Access governed by the RBAC system.
-   **PostgreSQL**: Handled by API logic checking roles/ownership.
-   **Object Storage**: Use pre-signed URLs or temporary credentials generated by the backend API for time-limited, specific object access. Avoid direct bucket access for users.
-   **Logging System**: Access restricted to Administrators.

### 10.6 Retention Policies

Define retention periods based on business needs, user agreements, and regulations. Example policy:

-   **User Data**: Retain while account active. Anonymize/delete [X] years after inactivity (e.g., 2 years), subject to legal holds.
-   **Project Data**: Retain while project active. Archive [Y] years after completion (e.g., 1 year). Delete [Z] years after archival (e.g., 7 years total).
-   **Raw Satellite Imagery**: Hot Storage (6 months) -> Warm Storage (2 years) -> Cold Storage (7 years total). Delete after 7 years.
-   **ML Inference Results (Detailed)**: Hot Storage (6 months) -> Warm Storage (2 years) -> Cold Storage (7 years total). Delete after 7 years.
-   **Verification Records (PostgreSQL - Key Metrics)**: Retain for long period (e.g., 10+ years) for audit trails. Anonymize user links after account deletion.
-   **Blockchain Data/Metadata**: Permanent (on-chain/IPFS).
-   **Application Logs**: Aggregate & retain for [T] days (e.g., 90 days) for debugging/security. Archive/delete afterwards.

### 10.7 Archiving Procedures

-   Implement automated scripts or use object storage lifecycle policies (e.g., S3 Lifecycle rules) to transition data between tiers based on age.
-   Ensure PostgreSQL records are updated or queries designed to handle potential retrieval delays if linking to archived files.

### 10.8 Deletion Procedures

-   **User-Initiated**: Soft delete projects/data initially (marked as deleted), allow recovery for a grace period (e.g., 30 days). Hard delete upon request or after grace period.
-   **Admin-Initiated**: Allow admins to hard delete accounts/projects based on policy.
-   **Automated**: Use object storage lifecycle policies for automatic deletion after retention expires.
-   **Secure Deletion**: Use appropriate methods (e.g., cryptographic erasure if applicable).
-   **Database Cleanup**: Implement periodic jobs to hard delete soft-deleted records and clean up orphaned data.

### 10.9 Backup and Recovery

-   **PostgreSQL**: Daily full backups + Point-in-Time Recovery (PITR). Store backups securely off-site/cross-region. Test recovery procedures quarterly.
-   **Object Storage**: Enable versioning. Consider cross-region replication for critical data.
-   **ML Models**: Backup trained models in a separate location/bucket.
-   **Configuration**: Backup configuration files and secrets securely.
-   **Disaster Recovery Plan**: Document the process for restoring the application and data in case of major failure.

### 10.10 Compliance Considerations

-   **GDPR/CCPA**: Design for data subject rights (access, rectification, erasure - "right to be forgotten"). Ensure deletion procedures effectively remove personal data.
-   **Data Sovereignty**: Use cloud provider regions or storage configurations that comply with data residency requirements for specific projects/clients.
-   **Audit Trails**: Maintain immutable logs of data access, modification, and deletion events, especially for sensitive data and administrative actions.

This DLM strategy provides a framework for managing data effectively. Specific retention periods, storage tiers, and implementation details should be finalized based on detailed requirements, cost analysis, and regulatory advice.



---

## 11. UI/UX Refinement Guidelines

While the core functionality is paramount, refining the User Interface (UI) and User Experience (UX) is crucial for user adoption, efficiency, and satisfaction. These guidelines provide recommendations for enhancing the usability, accessibility, and overall feel of the Carbon Credit Verification SaaS application.

### 11.1 General Principles

-   **Consistency**: Maintain consistent layouts, terminology, colors, typography, and interaction patterns across the entire application (including the admin panel).
-   **Clarity**: Use clear and concise language. Avoid jargon where possible, or provide tooltips/definitions. Ensure visual hierarchy guides the user's attention to important elements.
-   **Feedback**: Provide immediate and clear feedback for user actions (e.g., button clicks, form submissions, loading states, errors, success messages).
-   **Efficiency**: Design workflows to minimize user effort and clicks for common tasks. Use sensible defaults and streamline complex processes.
-   **Accessibility**: Design for inclusivity by adhering to accessibility standards (WCAG 2.1 AA as a target).
-   **User Control**: Users should feel in control. Allow undoing actions where feasible, provide clear navigation, and make it easy to exit workflows.

### 11.2 Visual Design

-   **Color Palette**: Define a limited, consistent color palette. Use color purposefully to indicate status (e.g., green for success, red for error, blue for informational), draw attention, and ensure sufficient contrast for readability (check contrast ratios using online tools).
-   **Typography**: Choose readable fonts (e.g., Inter, Lato, Open Sans). Establish a clear typographic hierarchy (headings, subheadings, body text, captions) using size, weight, and spacing.
-   **Iconography**: Use clear, universally understood icons from a consistent set (e.g., Material Icons, Font Awesome, Feather Icons). Ensure icons are accompanied by text labels where ambiguity exists or for critical actions.
-   **Layout and Spacing**: Use a grid system (e.g., 12-column grid) and consistent spacing rules (e.g., multiples of 4px or 8px) for alignment and layout. Employ generous white space to reduce clutter and improve readability.
-   **Branding**: Incorporate project branding (logo, primary colors) subtly and consistently, typically in the header/navigation.

### 11.3 Interaction Design

-   **Navigation**: Implement clear and predictable navigation (e.g., persistent sidebar or top navigation bar). Use breadcrumbs for deep hierarchies. Ensure the current location is always visually indicated.
-   **Forms**: Design clear and easy-to-use forms.
    -   Use clear labels, positioned consistently (e.g., above the field).
    -   Provide helpful placeholder text and input constraints (e.g., date pickers, number inputs with min/max).
    -   Implement real-time validation where helpful (e.g., email format), with clear error messages near the problematic field upon losing focus or submission attempt.
    -   Group related fields logically using fieldsets or visual separation.
    -   Clearly indicate required fields (e.g., with an asterisk).
    -   Use appropriate input types (e.g., dropdowns/selects for limited choices, radio buttons for mutually exclusive options, checkboxes for multiple selections, text areas for longer descriptions).
-   **Data Visualization**: Present data clearly and effectively.
    -   Use appropriate chart types (bar, line, pie, scatter) for the data being displayed and the insight intended.
    -   Label axes, data points, and legends clearly.
    -   Provide interactive tooltips for detailed information on hover.
    -   Ensure charts are responsive and readable on different screen sizes.
    -   Consider accessibility for charts (e.g., using patterns in addition to color, providing data tables as alternatives).
-   **Map Interaction**: Enhance the map component usability.
    -   Provide clear instructions and visual cues for drawing/editing boundaries.
    -   Use distinct visual styles (color, opacity, borders) for different map layers (boundary, satellite imagery, change detection results).
    -   Implement intuitive zoom (mouse wheel, +/- buttons) and pan controls.
    -   Ensure popups/tooltips on map features are informative and easy to interact with/dismiss.
    -   Consider adding a search function for locations (geocoding).
    -   Provide a clear layer control to toggle visibility of different map layers.
-   **Loading States**: Provide immediate visual feedback during loading operations.
    -   Use spinners or loaders for short waits.
    -   Use skeleton screens for content loading to mimic the final layout.
    -   Use progress bars for longer, trackable operations (e.g., file uploads, batch processing).
    -   Disable interactive elements during loading to prevent duplicate actions.
-   **Feedback States**: Use consistent patterns for success, error, warning, and informational messages (e.g., toast notifications for non-blocking feedback, inline messages for contextual feedback).

### 11.4 Accessibility (A11y)

-   **Semantic HTML**: Use appropriate HTML5 elements (`<nav>`, `<main>`, `<aside>`, `<button>`, `<h1>`-`<h6>`, etc.) to provide inherent structure and meaning.
-   **Keyboard Navigation**: Ensure all interactive elements (links, buttons, form fields, custom controls) are focusable and operable using the keyboard alone (Tab, Shift+Tab, Enter, Space). Maintain a logical and predictable focus order.
-   **Screen Reader Support**: Provide descriptive `alt` text for meaningful images. Use ARIA attributes (`aria-label`, `aria-describedby`, `role`, state attributes like `aria-expanded`, `aria-selected`) judiciously where semantic HTML is insufficient, especially for custom components or dynamic content updates.
-   **Color Contrast**: Ensure text and important UI elements (like icons, form borders) have sufficient contrast against their background (WCAG AA requires 4.5:1 for normal text, 3:1 for large text and graphical elements/UI components).
-   **Resizable Text**: Design layouts that allow users to resize text up to 200% using browser settings without loss of content or functionality (use relative units like `rem` or `em` for text and layout where appropriate).
-   **Forms**: Associate labels explicitly with form controls using `for` and `id` attributes, or by wrapping the input within the label. Use `fieldset` and `legend` for groups of related controls (e.g., radio buttons).

### 11.5 Specific Component Refinements

-   **Dashboard**: Prioritize key information using visual hierarchy. Make summary cards easily scannable with clear metrics and trends. Consider allowing basic customization (e.g., rearranging cards).
-   **Project Creation/Editing**: For complex forms, consider breaking them down into logical steps using a stepper component or tabs to reduce cognitive load.
-   **Verification Workflow**: Clearly visualize the current status and history of the verification process (e.g., using a timeline or status indicator). If human review is needed, provide an intuitive interface for comparing AI results with imagery, adding annotations, and submitting decisions.
-   **Map Component**: Ensure map layers are clearly labeled in the layer control. Provide legends for data layers (e.g., explaining colors used in forest change maps). Ensure controls are large enough for touch interaction.
-   **Tables**: Implement client-side or server-side sorting, filtering, and pagination for large datasets. Ensure tables are responsive (e.g., allowing horizontal scrolling or collapsing columns) on smaller screens.

### 11.6 User Feedback Mechanisms

-   Consider adding a simple, non-intrusive feedback mechanism (e.g., a feedback button in the footer or help menu) allowing users to report bugs or suggest improvements directly from the application.

### 11.7 Tools and Testing

-   **Prototyping**: Use tools like Figma, Sketch, or Adobe XD to design and iterate on UI mockups and interactive prototypes before implementation.
-   **Component Libraries**: Leverage established UI component libraries (e.g., Material-UI, Ant Design, Chakra UI, Bootstrap) which often provide accessible, themeable, and consistent components.
-   **Accessibility Testing**: Regularly use browser extensions (e.g., Axe DevTools, WAVE), automated testing tools (e.g., `jest-axe`), and manual keyboard/screen reader testing (e.g., NVDA, VoiceOver) to identify and fix accessibility issues throughout development.
-   **Usability Testing**: Conduct usability tests with representative users (even informal ones with colleagues or peers) to observe how they interact with the application, identify pain points, and gather qualitative feedback.

By incorporating these UI/UX guidelines, the Carbon Credit Verification SaaS application can become not only functional but also intuitive, efficient, and accessible, leading to better user adoption and satisfaction.

