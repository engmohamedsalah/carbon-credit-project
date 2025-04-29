# CI/CD Pipeline Configuration (GitHub Actions Example)

This document provides an example configuration for setting up a Continuous Integration and Continuous Deployment (CI/CD) pipeline using GitHub Actions for the Carbon Credit Verification SaaS application. This pipeline automates the building, testing, and deployment processes.

**Assumptions**:
-   Source code is hosted on GitHub.
-   Docker images are pushed to AWS Elastic Container Registry (ECR).
-   Deployment target is AWS (e.g., ECS Fargate, Elastic Beanstalk).
-   AWS credentials (Access Key ID and Secret Access Key) are configured as GitHub Secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
-   Other necessary secrets (e.g., `ECR_REGISTRY`, `AWS_REGION`, `ECS_CLUSTER_NAME`, `ECS_SERVICE_NAME_BACKEND`, `ECS_SERVICE_NAME_FRONTEND`) are also configured in GitHub repository secrets.

## 1. Pipeline Overview

The pipeline consists of workflows defined in YAML files within the `.github/workflows/` directory of your repository.

**Workflow 1: Build and Test (CI)**
-   **Trigger**: On push to `main` and `develop` branches, or on pull requests targeting these branches.
-   **Jobs**:
    -   Lint code.
    -   Run backend unit tests.
    -   Run frontend unit tests.
    -   Build Docker images (backend, frontend).
    -   (Optional) Run integration tests using Docker Compose.

**Workflow 2: Deploy to Staging (CD)**
-   **Trigger**: On push to `develop` branch (after CI succeeds).
-   **Jobs**:
    -   Build Docker images.
    -   Push images to ECR with `develop` tag.
    -   Deploy to staging environment (e.g., update ECS service, deploy to EB staging environment).

**Workflow 3: Deploy to Production (CD)**
-   **Trigger**: On push to `main` branch (after CI succeeds) or manually triggered/on release tag.
-   **Jobs**:
    -   Build Docker images.
    -   Push images to ECR with `latest` tag and commit SHA tag.
    -   Deploy to production environment (e.g., update ECS service, deploy to EB production environment).

## 2. Example Workflow Files

### 2.1 CI Workflow (`.github/workflows/ci.yml`)

```yaml
name: Build and Test CI

on:
  push:
    branches:
      - main
      - develop
  pull_request:
    branches:
      - main
      - develop

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.10'
      - name: Install backend dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install flake8 # Or your preferred linter
      - name: Lint backend
        run: flake8 backend/app
      # Add frontend linting steps (e.g., using ESLint)

  test-backend:
    runs-on: ubuntu-latest
    needs: lint
    services: # Spin up a temporary Postgres DB for testing
      postgres:
        image: postgis/postgis:15-3.3 # Use PostGIS image
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpassword
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.10'
      - name: Install backend dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install pytest pytest-cov # Testing libraries
      - name: Run backend tests
        env:
          DATABASE_URL: "postgresql+psycopg2://testuser:testpassword@localhost:5432/testdb"
          # Add other necessary test env vars
        run: |
          cd backend
          pytest # Or python -m pytest

  test-frontend:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'
      - name: Install frontend dependencies
        run: |
          cd frontend
          npm install
      - name: Run frontend tests
        run: |
          cd frontend
          npm test -- --watchAll=false # Run tests non-interactively

  build-docker:
    runs-on: ubuntu-latest
    needs: [test-backend, test-frontend]
    steps:
      - uses: actions/checkout@v3
      - name: Build Backend Docker image
        run: docker build -t backend-image:${{ github.sha }} -f docker/backend.Dockerfile backend/
      - name: Build Frontend Docker image
        run: docker build -t frontend-image:${{ github.sha }} -f docker/frontend.Dockerfile frontend/
      # You don't push images here, just ensure they build successfully
```

### 2.2 Deploy to Staging Workflow (`.github/workflows/deploy-staging.yml`)

```yaml
name: Deploy to Staging

on:
  push:
    branches:
      - develop

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: staging # Optional: Define a GitHub environment for secrets/rules
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build, tag, and push Backend image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: carbon-credit-backend # Your ECR repo name
          IMAGE_TAG: develop
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -f docker/backend.Dockerfile backend/
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Build, tag, and push Frontend image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: carbon-credit-frontend # Your ECR repo name
          IMAGE_TAG: develop
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -f docker/frontend.Dockerfile frontend/
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      # --- Deployment Step (Example: Update ECS Service) ---
      - name: Download task definition
        run: |
          aws ecs describe-task-definition --task-definition carbon-credit-backend-staging \
            --query taskDefinition > backend-task-definition.json
          aws ecs describe-task-definition --task-definition carbon-credit-frontend-staging \
            --query taskDefinition > frontend-task-definition.json
        # Assumes task definitions named 'carbon-credit-backend-staging', etc. exist

      - name: Fill in the new image ID in the Amazon ECS task definition
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: backend-task-definition.json
          container-name: backend # Name of the container in your task definition
          image: ${{ steps.login-ecr.outputs.registry }}/carbon-credit-backend:develop
      # Repeat for frontend task definition if needed

      - name: Deploy Amazon ECS task definition
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: ${{ secrets.ECS_SERVICE_NAME_BACKEND_STAGING }} # Service name from secrets
          cluster: ${{ secrets.ECS_CLUSTER_NAME_STAGING }} # Cluster name from secrets
          wait-for-service-stability: true
      # Repeat for frontend service if needed

      # --- Alternative Deployment Step (Example: Elastic Beanstalk) ---
      # - name: Deploy to Elastic Beanstalk
      #   uses: einaregilsson/beanstalk-deploy@v20
      #   with:
      #     aws_access_key: ${{ secrets.AWS_ACCESS_KEY_ID }}
      #     aws_secret_key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      #     application_name: carbon-credit-app
      #     environment_name: carbon-credit-app-staging
      #     version_label: develop-${{ github.sha }}
      #     region: ${{ secrets.AWS_REGION }}
      #     deployment_package: Dockerrun.aws.json # Or zip file if needed
```

### 2.3 Deploy to Production Workflow (`.github/workflows/deploy-prod.yml`)

This workflow is very similar to `deploy-staging.yml` but triggers on the `main` branch and uses different tags/environment targets.

```yaml
name: Deploy to Production

on:
  push:
    branches:
      - main
  # Optional: Trigger on release creation
  # release:
  #   types: [published]
  # Optional: Manual trigger
  # workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production # Use GitHub environment for protection rules
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build, tag, and push Backend image to Amazon ECR
        id: build-backend-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: carbon-credit-backend
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -f docker/backend.Dockerfile backend/
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
          echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"

      - name: Build, tag, and push Frontend image to Amazon ECR
        id: build-frontend-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: carbon-credit-frontend
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -f docker/frontend.Dockerfile frontend/
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
          echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"

      # --- Deployment Step (Example: Update ECS Service) ---
      - name: Download task definition
        run: |
          aws ecs describe-task-definition --task-definition carbon-credit-backend-prod \
            --query taskDefinition > backend-task-definition.json
          # Repeat for frontend if needed

      - name: Fill in the new image ID in the Amazon ECS task definition
        id: task-def-backend
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: backend-task-definition.json
          container-name: backend
          image: ${{ steps.build-backend-image.outputs.image }}
      # Repeat for frontend task definition if needed

      - name: Deploy Amazon ECS task definition (Backend)
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: ${{ steps.task-def-backend.outputs.task-definition }}
          service: ${{ secrets.ECS_SERVICE_NAME_BACKEND_PROD }}
          cluster: ${{ secrets.ECS_CLUSTER_NAME_PROD }}
          wait-for-service-stability: true
      # Repeat for frontend service if needed

      # --- Alternative Deployment Step (Example: Elastic Beanstalk) ---
      # - name: Deploy to Elastic Beanstalk Production
      #   uses: einaregilsson/beanstalk-deploy@v20
      #   with:
      #     aws_access_key: ${{ secrets.AWS_ACCESS_KEY_ID }}
      #     aws_secret_key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      #     application_name: carbon-credit-app
      #     environment_name: carbon-credit-app-prod
      #     version_label: prod-${{ github.sha }}
      #     region: ${{ secrets.AWS_REGION }}
      #     deployment_package: Dockerrun.aws.json
```

## 3. Considerations

-   **Secrets Management**: Ensure AWS credentials and other secrets are stored securely in GitHub Secrets and not hardcoded.
-   **Testing**: Enhance the CI workflow with integration tests, end-to-end tests (using tools like Cypress or Playwright), and security scanning (e.g., Docker image scanning).
-   **Database Migrations**: Add steps to handle database migrations (e.g., using Alembic for FastAPI/SQLAlchemy) as part of the deployment process. This often requires careful coordination, potentially running migrations as a separate step before updating the application service.
-   **Rollbacks**: Implement strategies for automatic or manual rollbacks in case of deployment failures.
-   **Environments**: Use GitHub Environments to configure protection rules (e.g., required reviewers) for production deployments.
-   **Infrastructure as Code (IaC)**: For managing AWS resources (ECS clusters, RDS, S3, etc.), consider using tools like Terraform or AWS CloudFormation. The CI/CD pipeline could potentially trigger IaC tools as well.
-   **Cost**: GitHub Actions runners have usage limits on free plans. Complex builds or tests might require self-hosted runners or paid plans.

This example provides a solid starting point for automating your CI/CD process using GitHub Actions. Adapt the specific steps and tools based on your chosen AWS deployment strategy and testing requirements.
