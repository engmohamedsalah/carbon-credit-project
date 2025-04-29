# Troubleshooting Guidance

This guide provides solutions for common issues that might arise during the implementation and execution of the Carbon Credit Verification SaaS application.

## 1. Environment Setup Issues

-   **Issue**: `pip install` fails due to dependency conflicts.
    *   **Solution**: 
        1.  Try creating a fresh virtual environment.
        2.  Install problematic packages one by one to identify the conflict.
        3.  Check for specific version requirements in `requirements.txt` and adjust if necessary.
        4.  Consider using `pipdeptree` to visualize dependencies.
        5.  For geospatial libraries (like `rasterio`, `geopandas`), installation can be tricky. Using `conda` might simplify this, or ensure system-level dependencies (like GDAL) are correctly installed before using `pip`.

-   **Issue**: Docker or Docker Compose commands fail (`command not found`, permission errors).
    *   **Solution**: 
        1.  Verify Docker and Docker Compose are installed correctly.
        2.  Ensure the Docker daemon is running (`sudo systemctl status docker`).
        3.  Add your user to the `docker` group to run commands without `sudo` (`sudo usermod -aG docker $USER`, then log out and back in).

## 2. Data Preparation Issues

-   **Issue**: `sentinelsat` download fails (authentication error, connection timeout).
    *   **Solution**: 
        1.  Double-check Copernicus Hub username/password.
        2.  Verify internet connectivity from the sandbox.
        3.  Check the Copernicus Hub status page for outages.
        4.  Try downloading manually via the web interface to confirm account validity.

-   **Issue**: Errors reading satellite or Hansen data (`rasterio` errors).
    *   **Solution**: 
        1.  Verify data files downloaded completely and are not corrupted.
        2.  Ensure file paths in the script (`ForestChangeDataset`) are correct.
        3.  Check if the file formats are supported by `rasterio` (GeoTIFF, JP2 should be fine).
        4.  Ensure necessary GDAL drivers are available.

-   **Issue**: Coordinate Reference System (CRS) mismatches or resolution differences between Sentinel-2 and Hansen data.
    *   **Solution**: 
        1.  Use `rasterio` or `gdal` utilities (like `gdalwarp`) to reproject one dataset to match the CRS of the other *before* training.
        2.  Resample the lower-resolution dataset (Hansen 30m) to match the higher-resolution dataset (Sentinel-2 10m) using appropriate resampling methods (e.g., nearest neighbor for categorical data).
        3.  Adapt the data loading code to handle reprojection/resampling on the fly if needed (can be computationally expensive).

## 3. Model Training Issues

-   **Issue**: `RuntimeError: CUDA out of memory`.
    *   **Solution**: 
        1.  Reduce `BATCH_SIZE` in `train_forest_change.py`.
        2.  If using a large model, consider a smaller architecture or techniques like gradient accumulation.
        3.  Ensure no other processes are consuming GPU memory.
        4.  Close unnecessary applications.

-   **Issue**: Training is extremely slow.
    *   **Solution**: 
        1.  Verify that training is running on the GPU (`DEVICE` should be `cuda`). Check PyTorch CUDA availability (`torch.cuda.is_available()`).
        2.  Optimize data loading: Increase `num_workers` in `DataLoader` (e.g., 4 or 8 if CPU allows), ensure data preprocessing isn't a bottleneck.
        3.  If on CPU, this is expected. Consider using cloud GPU resources.

-   **Issue**: Loss becomes `NaN` (Not a Number).
    *   **Solution**: 
        1.  Lower the `LEARNING_RATE`.
        2.  Check for numerical instability in data (e.g., division by zero, very large values). Ensure proper normalization.
        3.  Inspect input data and labels for corruption or anomalies.
        4.  Add gradient clipping (`torch.nn.utils.clip_grad_norm_`).

-   **Issue**: Model accuracy/validation loss does not improve or is very poor.
    *   **Solution**: 
        1.  **Data Issue**: Verify data quality and labels. Ensure data augmentation is appropriate. Check for class imbalance.
        2.  **Model Issue**: Try a different model architecture or pre-trained weights. Ensure model complexity matches data complexity.
        3.  **Hyperparameters**: Experiment with `LEARNING_RATE`, optimizer (e.g., AdamW), `BATCH_SIZE`.
        4.  **Training Time**: Train for more `NUM_EPOCHS`.
        5.  **Loss Function**: Ensure the loss function matches the task (e.g., `BCELoss` for binary segmentation with sigmoid output).
        6.  **Overfitting**: If training loss decreases but validation loss increases, add regularization (dropout, weight decay) or use early stopping.

## 4. Docker Deployment Issues

-   **Issue**: `docker-compose build` fails.
    *   **Solution**: 
        1.  Read the error messages carefully. Often indicates missing files (e.g., `requirements.txt`), incorrect paths in Dockerfile `COPY` commands, or failed package installations within the container.
        2.  Ensure the Docker context is correct (usually the directory containing the Dockerfile or specified in `docker-compose.yml`).

-   **Issue**: Containers start but exit immediately.
    *   **Solution**: Check container logs (`docker-compose logs <service_name>`) for the specific error causing the exit (e.g., application crash, configuration error).

-   **Issue**: Backend cannot connect to the database (`db` service).
    *   **Solution**: 
        1.  Ensure the `db` service is running (`docker-compose ps`).
        2.  Verify the `DATABASE_URL` environment variable in the `backend` service definition in `docker-compose.yml` is correct (using the service name `db` as the hostname: `postgresql://postgres:postgres@db:5432/carbon_credits`).
        3.  Check `db` logs (`docker-compose logs db`) to ensure PostgreSQL started correctly.
        4.  Add a `healthcheck` or `depends_on` with condition (`service_healthy`) in `docker-compose.yml` to ensure `db` is ready before `backend` starts.

-   **Issue**: Frontend cannot connect to the backend API (Network Error, CORS error).
    *   **Solution**: 
        1.  Ensure the `backend` service is running (`docker-compose ps`).
        2.  Verify the API base URL configured in the frontend code (e.g., in `frontend/src/services/api.js`) points to the correct backend address (e.g., `http://localhost:8000` or `http://backend:8000` depending on context).
        3.  **CORS**: Ensure the FastAPI backend has CORS middleware configured correctly (in `backend/main.py`) to allow requests from the frontend origin (`http://localhost:3000`).

-   **Issue**: Port conflicts (e.g., `Error starting userland proxy: listen tcp4 0.0.0.0:3000: bind: address already in use`).
    *   **Solution**: 
        1.  Stop any other application using the conflicting port on your host machine.
        2.  Change the host port mapping in `docker-compose.yml` (e.g., change `"3000:3000"` to `"3001:3000"` and access the frontend on port 3001).

-   **Issue**: Backend fails to load the ML model (`FileNotFoundError`).
    *   **Solution**: Verify the model file (`forest_change_unet.pth`) is accessible *inside* the backend container at the path specified in the backend code. Ensure you have either copied it via the Dockerfile or mounted it as a volume in `docker-compose.yml` (See Step 4, Consideration 5).

## 5. Application Runtime Issues

-   **Issue**: API endpoints return 500 Internal Server Error.
    *   **Solution**: Check backend logs (`docker-compose logs backend`) for detailed Python tracebacks indicating the cause of the error.

-   **Issue**: Frontend displays data incorrectly or UI elements are broken.
    *   **Solution**: 
        1.  Check the browser's developer console (usually F12) for JavaScript errors.
        2.  Check the Network tab in developer tools to inspect API responses received from the backend.
        3.  Verify frontend state management (Redux) is updating correctly.

-   **Issue**: ML inference process fails or takes too long.
    *   **Solution**: 
        1.  Check backend logs for errors during inference.
        2.  Ensure input data format matches what the model expects.
        3.  Optimize inference code (e.g., batching, efficient data loading).
        4.  Consider running inference as an asynchronous background task (using Celery, RQ, or FastAPI's background tasks) to avoid blocking API requests.

-   **Issue**: Blockchain transactions fail (reverted, out of gas).
    *   **Solution**: 
        1.  Check backend logs for specific error messages from Web3.py or the blockchain node.
        2.  Ensure the account sending the transaction has sufficient funds (ETH/gas token).
        3.  Increase the gas limit for the transaction.
        4.  Verify the smart contract address and ABI are correct.
        5.  Check the input parameters being sent to the smart contract function.
        6.  Consult the documentation of the specific blockchain network or smart contract.

Remember to always check the logs (`docker-compose logs <service_name>`) as the first step in diagnosing issues with containerized applications.
