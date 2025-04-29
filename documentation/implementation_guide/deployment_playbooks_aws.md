# Cloud Deployment Playbooks (AWS Example)

This document outlines potential strategies and steps for deploying the Dockerized Carbon Credit Verification SaaS application to Amazon Web Services (AWS). It covers common services like Elastic Beanstalk, ECS, and EKS.

**Disclaimer**: These are high-level playbooks. Actual implementation requires detailed configuration based on specific security, scalability, and cost requirements. AWS service features and interfaces evolve; always refer to the latest AWS documentation.

## 1. Prerequisites

Before deploying, ensure you have:

1.  **AWS Account**: An active AWS account with appropriate permissions.
2.  **IAM Roles/Users**: Properly configured IAM users or roles with necessary permissions for services like ECR, ECS/EKS/Beanstalk, RDS, S3, VPC, IAM, CloudWatch.
3.  **Docker Images**: Built Docker images for the `backend` and `frontend` services, pushed to Amazon Elastic Container Registry (ECR).
    ```bash
    # Example ECR push commands (run after docker build)
    aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
    docker tag backend-image:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/carbon-credit-backend:latest
    docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/carbon-credit-backend:latest
    # Repeat for frontend image
    ```
4.  **VPC Configuration**: A Virtual Private Cloud (VPC) set up with public and private subnets, security groups, NAT Gateway (for private subnets to access the internet), and route tables.
5.  **Database**: An external database provisioned, preferably using Amazon RDS for PostgreSQL with the PostGIS extension enabled. Note the endpoint, username, and password.
6.  **Object Storage**: An S3 bucket created for storing satellite imagery, ML results, etc.
7.  **Secrets Management**: A strategy for managing secrets (database passwords, API keys, JWT secret) - AWS Secrets Manager or Systems Manager Parameter Store are recommended over environment variables for production.
8.  **Domain Name**: (Optional) A registered domain name configured with Route 53 if using a custom domain.

## 2. Option 1: AWS Elastic Beanstalk (Multi-container Docker)

**Best for**: Simplicity, rapid deployment, less operational overhead.

**Pros**:
-   Abstracts away underlying infrastructure management.
-   Handles load balancing, auto-scaling, deployment strategies automatically.
-   Good integration with other AWS services.

**Cons**:
-   Less flexibility compared to ECS/EKS.
-   Can sometimes be opaque ("magic box").
-   Multi-container setup requires specific `Dockerrun.aws.json` configuration.

**Steps**:

1.  **Create `Dockerrun.aws.json`**: Define the containers (backend, frontend, potentially nginx for routing/serving frontend static files) and their links/dependencies in a `Dockerrun.aws.json` file at the root of your deployment package.

    ```json
    {
      "AWSEBDockerrunVersion": 2,
      "containerDefinitions": [
        {
          "name": "backend",
          "image": "<your-account-id>.dkr.ecr.<your-region>.amazonaws.com/carbon-credit-backend:latest",
          "essential": true,
          "memory": 512, // Adjust memory as needed
          "portMappings": [
            { "hostPort": 8000, "containerPort": 8000 }
          ],
          "environment": [
            // Environment variables - Consider using secrets management instead
            {"name": "DATABASE_URL", "value": "postgresql+psycopg2://user:password@rds-endpoint:5432/dbname"},
            {"name": "SECRET_KEY", "value": "your-jwt-secret"},
            {"name": "ALGORITHM", "value": "HS256"},
            {"name": "ACCESS_TOKEN_EXPIRE_MINUTES", "value": "30"}
            // Add other necessary env vars (S3 bucket, blockchain node, etc.)
          ]
          // "links": ["nginx"] // Link if nginx needs to proxy to backend
        },
        {
          "name": "frontend", // Could be nginx serving React build files
          "image": "<your-account-id>.dkr.ecr.<your-region>.amazonaws.com/carbon-credit-frontend:latest", // Assuming frontend image serves itself
          "essential": true,
          "memory": 256,
          "portMappings": [
            { "hostPort": 80, "containerPort": 80 } // Map host 80 to container 80 (or 3000 if dev server)
          ]
          // If using nginx to serve static files and proxy API:
          // - Use an nginx image
          // - Mount React build files into nginx container
          // - Configure nginx reverse proxy to backend:8000
          // - Map hostPort 80 to nginx containerPort 80
        }
      ]
    }
    ```

2.  **Create Elastic Beanstalk Application & Environment**:
    -   Use the EB CLI (`eb create`) or the AWS Management Console.
    -   Choose Platform: `Docker` -> `Multi-container Docker`.
    -   Upload your code bundle (can be just `Dockerrun.aws.json` if images are in ECR, or include source code if building on instance).
    -   Configure environment properties (instance type, VPC, subnets, security groups, load balancer).
    -   **Important**: Ensure the EC2 instances have an IAM role allowing them to pull images from ECR.
3.  **Configure Environment Variables/Secrets**: Securely provide database credentials and other secrets, preferably using integration with Secrets Manager or Parameter Store, or EB environment properties (less secure).
4.  **Database Connection**: Ensure the application security group allows outbound traffic to the RDS security group on port 5432, and the RDS security group allows inbound traffic from the application security group.
5.  **Deploy**: EB will provision resources and deploy your containers based on `Dockerrun.aws.json`.

## 3. Option 2: AWS ECS (Elastic Container Service) with Fargate

**Best for**: More control over container orchestration, serverless container execution (Fargate).

**Pros**:
-   Granular control over tasks, services, scaling.
-   Serverless option with Fargate (no EC2 instance management).
-   Integrates well with other AWS services (ALB, CloudWatch, Secrets Manager).

**Cons**:
-   Steeper learning curve than Elastic Beanstalk.
-   Requires defining Task Definitions, Services, Clusters manually or via IaC.

**Steps**:

1.  **Create ECS Cluster**: Choose the Fargate launch type for serverless or EC2 launch type if you want to manage instances.
2.  **Create Task Definitions**: Define one Task Definition for the `backend` and one for the `frontend` (or a single task definition with both containers if they need tight coupling, though separate is often better for independent scaling).
    -   Specify container image (from ECR), CPU/memory allocation, port mappings.
    -   Configure environment variables or secrets integration (Secrets Manager/Parameter Store).
    -   Define logging configuration (e.g., `awslogs` driver for CloudWatch Logs).
    -   Assign an IAM Task Role (permissions the container needs, e.g., S3 access) and Task Execution Role (permissions ECS agent needs, e.g., pull from ECR, write logs).
3.  **Create Application Load Balancer (ALB)**:
    -   Set up listeners (e.g., HTTP on 80, HTTPS on 443).
    -   Create Target Groups for `backend` (port 8000) and `frontend` (port 80/3000).
    -   Configure health checks for each target group.
    -   Define listener rules to route traffic (e.g., `/api/*` to backend target group, default `/*` to frontend target group).
4.  **Create ECS Services**: Create one service for the `backend` task definition and one for the `frontend` task definition.
    -   Associate each service with the appropriate ALB target group.
    -   Configure desired task count and auto-scaling policies (based on CPU/memory utilization or request count).
    -   Specify cluster, launch type (Fargate/EC2), VPC, subnets, security groups.
5.  **Configure Security Groups**: Ensure ALB security group allows traffic from the internet (port 80/443). Ensure service security groups allow traffic from the ALB. Ensure backend security group allows traffic from frontend (if direct communication needed) and allows outbound to RDS.
6.  **Deploy**: ECS will launch tasks based on the service definitions.

## 4. Option 3: AWS EKS (Elastic Kubernetes Service)

**Best for**: Complex microservices, portability, leveraging the Kubernetes ecosystem.

**Pros**:
-   Most flexible and powerful container orchestration.
-   Vendor-neutral (Kubernetes standard).
-   Large ecosystem of tools and extensions.
-   Fine-grained control over networking, storage, scaling.

**Cons**:
-   Highest complexity and operational overhead.
-   Steep learning curve for Kubernetes.
-   Managing the EKS control plane and worker nodes (or Fargate profiles) adds cost/complexity.

**Steps**:

1.  **Create EKS Cluster**: Use `eksctl` or AWS Console/CloudFormation to provision the EKS control plane and worker nodes (EC2 or Fargate profiles).
2.  **Configure `kubectl`**: Set up `kubectl` to interact with your EKS cluster.
3.  **Create Kubernetes Manifests**: Define Kubernetes resources in YAML files:
    -   **Namespace**: Create a dedicated namespace (e.g., `carbon-credit-app`).
    -   **Secrets**: Create Kubernetes Secrets for database credentials, JWT keys, etc. (Consider using AWS Secrets Manager integration with CSI driver for better security).
    -   **ConfigMaps**: Create ConfigMaps for non-sensitive configuration.
    -   **Deployments**: Define Deployments for `backend` and `frontend` pods, specifying container images (from ECR), replicas, resource requests/limits, environment variables (from ConfigMaps/Secrets), ports.
    -   **Services**: Define Services (e.g., `ClusterIP` or `NodePort`) to expose the Deployments internally within the cluster.
    -   **Ingress**: Define an Ingress resource to manage external access via an AWS Load Balancer (ALB Ingress Controller is commonly used). Configure rules to route traffic to the backend and frontend services.
4.  **Deploy Manifests**: Apply the YAML files using `kubectl apply -f <directory> -n <namespace>`.
5.  **Set up ALB Ingress Controller**: Deploy the AWS Load Balancer Controller to your cluster to manage ALBs based on Ingress resources.
6.  **Database/S3**: Access RDS and S3 typically via their standard endpoints from within the pods. Ensure VPC networking and security groups allow connectivity.

## 5. Database and Storage

-   **Database (RDS)**:
    -   Provision a PostgreSQL instance on RDS.
    -   Enable the PostGIS extension.
    -   Configure security groups to allow access only from your application's security group/VPC.
    -   Use Secrets Manager to store credentials and inject them into your application environment.
-   **Object Storage (S3)**:
    -   Create an S3 bucket.
    -   Configure bucket policies and CORS if direct browser uploads are needed (though backend-mediated uploads are generally safer).
    -   Assign an IAM role to your application tasks/instances allowing necessary S3 actions (GetObject, PutObject, ListBucket).
    -   Configure lifecycle policies for data archiving/deletion.

## 6. General Considerations

-   **CI/CD Integration**: Integrate your chosen deployment strategy with a CI/CD pipeline (GitHub Actions, GitLab CI, AWS CodePipeline) to automate image building, testing, and deployment.
-   **Monitoring & Logging**: Configure CloudWatch Logs (via `awslogs` driver or Fluentd/Fluent Bit) and CloudWatch Metrics. Set up alarms. Consider more advanced monitoring with Prometheus/Grafana or Datadog.
-   **Security**: Follow AWS security best practices: least privilege IAM roles, security group restrictions, secrets management, VPC network segmentation, regular patching (if using EC2), WAF.
-   **Cost Optimization**: Choose appropriate instance types/Fargate configurations, implement auto-scaling, use Reserved Instances or Savings Plans, monitor costs with AWS Cost Explorer.

Choose the deployment option that best balances your team's expertise, operational capacity, flexibility requirements, and budget.
