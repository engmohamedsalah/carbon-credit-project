# Monitoring and Alerting Setup Guide

Effective monitoring and alerting are crucial for maintaining the reliability, performance, and availability of the Carbon Credit Verification SaaS application. This guide outlines key metrics, tools, and strategies for setting up a robust monitoring and alerting system, primarily focusing on AWS CloudWatch.

## 1. Goals of Monitoring

-   **Performance Tracking**: Understand application response times, resource utilization, and throughput.
-   **Availability**: Ensure the application and its dependencies are up and running.
-   **Error Detection**: Quickly identify and diagnose application errors and infrastructure issues.
-   **Resource Optimization**: Identify bottlenecks and opportunities for cost savings.
-   **Security**: Detect suspicious activities or potential security breaches (though dedicated security monitoring tools are often used).

## 2. Key Metrics to Monitor

### 2.1 Infrastructure (EC2/Fargate/EKS Nodes)
-   **CPU Utilization**: Track average and maximum CPU usage.
-   **Memory Utilization**: Monitor available vs. used memory.
-   **Disk I/O**: Track read/write operations and latency.
-   **Network I/O**: Monitor incoming/outgoing network traffic.
-   **Disk Space**: Track available disk space (especially for EC2).

### 2.2 Application Backend (FastAPI)
-   **Request Rate**: Number of requests per minute/second.
-   **Request Latency**: Average, p95, p99 response times for API endpoints.
-   **Error Rate**: Percentage of 4xx and 5xx HTTP status codes.
-   **Task Queue Metrics (if using Celery/RQ)**: Number of tasks pending, task execution time, failure rate.
-   **Resource Usage**: CPU and memory usage per container/task.

### 2.3 Application Frontend (React)
-   **Page Load Time**: Time to interactive (TTI), First Contentful Paint (FCP).
-   **JavaScript Errors**: Number and type of client-side errors.
-   **API Call Latency (from client)**: Time taken for API requests initiated by the frontend.
-   **User Session Metrics**: Number of active users, session duration (requires analytics integration).

### 2.4 Database (RDS PostgreSQL)
-   **CPU Utilization**: Database instance CPU usage.
-   **Memory Utilization**: Freeable memory, swap usage.
-   **Database Connections**: Number of active connections vs. maximum allowed.
-   **Read/Write IOPS**: Disk I/O operations per second.
-   **Disk Queue Depth**: Number of pending I/O requests.
-   **Replication Lag**: If using read replicas.
-   **Slow Queries**: Identify and track long-running queries (requires specific DB configuration/logs).

### 2.5 Object Storage (S3)
-   **Bucket Size**: Total storage used.
-   **Number of Objects**: Total object count.
-   **Request Metrics**: GET/PUT request counts, latency.
-   **Error Rates**: 4xx/5xx errors.

### 2.6 ML Service / Tasks
-   **Task Execution Time**: Duration of ML prediction tasks.
-   **Task Failure Rate**: Percentage of ML tasks failing.
-   **Resource Consumption**: CPU/Memory usage during ML processing.
-   **Model Performance Drift**: (Advanced) Monitor changes in prediction distribution or accuracy over time if ground truth becomes available.

### 2.7 Blockchain Interaction
-   **Transaction Confirmation Time**: Time taken for blockchain transactions to be confirmed.
-   **Transaction Failure Rate**: Percentage of failed/reverted transactions.
-   **Gas Costs**: Monitor gas fees spent.
-   **Node Connectivity**: Ensure the backend can connect to the blockchain node.

## 3. Monitoring Tools (AWS Focus)

-   **Amazon CloudWatch**: The primary AWS monitoring service.
    -   **Metrics**: Collects default metrics from most AWS services (EC2, Fargate, RDS, S3, ALB, etc.). Custom metrics can be published from applications.
    -   **Logs**: CloudWatch Logs for collecting, storing, and searching logs from applications and AWS services.
    -   **Alarms**: Create alarms based on metric thresholds or log patterns.
    -   **Dashboards**: Visualize metrics and log data.
    -   **Container Insights**: Enhanced monitoring for ECS and EKS, providing container-level metrics.
    -   **Synthetics**: Create canaries to monitor endpoints and UI workflows proactively.
-   **AWS X-Ray**: Distributed tracing service to analyze and debug application performance, identifying bottlenecks across services.
-   **Third-Party Tools**: (Optional) Tools like Datadog, Dynatrace, New Relic, Prometheus & Grafana offer alternative or supplementary monitoring capabilities, often with richer visualization and analysis features.
-   **Error Tracking**: Sentry, Rollbar (as mentioned in Error Handling) specifically for application error aggregation and analysis.

## 4. Logging Strategy

-   **Structured Logging**: Use JSON format for application logs (backend, frontend if possible).
-   **Centralized Logging**: Configure containers/instances to send logs to CloudWatch Logs (e.g., using `awslogs` driver for Docker/ECS, CloudWatch Agent for EC2).
-   **Log Groups & Streams**: Organize logs logically (e.g., `/ecs/backend-service`, `/ecs/frontend-service`, `/rds/postgresql/instance-id`).
-   **Log Retention**: Configure appropriate retention periods in CloudWatch Logs based on compliance and debugging needs.
-   **Log Insights**: Use CloudWatch Logs Insights query language to search and analyze log data effectively.

## 5. Alerting Strategy

-   **Define Critical Thresholds**: Determine acceptable ranges for key metrics (e.g., CPU > 80%, Latency > 500ms, Error Rate > 1%).
-   **Create CloudWatch Alarms**: Set up alarms based on these thresholds.
    -   Use appropriate statistics (Average, Maximum, Percentile).
    -   Set evaluation periods and datapoints to avoid flapping (e.g., alarm if CPU > 80% for 3 consecutive 5-minute periods).
    -   Configure actions: Send notifications to Amazon Simple Notification Service (SNS).
-   **SNS Topics**: Create SNS topics for different alert severities or teams (e.g., `critical-alerts`, `warning-alerts`).
-   **Notification Channels**: Subscribe relevant channels to SNS topics:
    -   Email
    -   SMS
    -   AWS Chatbot (for Slack/Chime integration)
    -   PagerDuty, Opsgenie (via HTTPS endpoint subscription)
    -   AWS Lambda (for custom actions)
-   **Log-Based Alarms**: Create CloudWatch Metric Filters to extract metrics from logs (e.g., count specific error codes) and create alarms based on these metrics.
-   **Synthetic Monitoring Alarms**: Create alarms based on the success/failure rate or duration of CloudWatch Synthetics canaries.
-   **Regular Review**: Periodically review alert thresholds and configurations to ensure they remain relevant and effective.

## 6. Setup Examples (CloudWatch)

1.  **Enable Container Insights**: If using ECS/EKS, enable Container Insights for detailed container-level metrics.
2.  **Configure CloudWatch Agent (EC2)**: If using EC2, install and configure the CloudWatch Agent to collect system metrics (memory, disk) and application logs.
3.  **Configure `awslogs` Driver (ECS/Docker)**: Ensure task definitions/`Dockerrun.aws.json` specify the `awslogs` driver, log group, and region.
4.  **Create CloudWatch Dashboards**: Build dashboards visualizing key metrics for infrastructure, backend, frontend, and database.
5.  **Create Alarms**: Use the CloudWatch console or Infrastructure as Code (CloudFormation, Terraform) to define alarms:
    -   **Example CPU Alarm**: Alarm if `CPUUtilization` on ECS Service (Average) > 80% for 15 minutes.
    -   **Example Latency Alarm**: Alarm if `TargetResponseTime` on ALB Target Group (p95) > 1 second for 5 minutes.
    -   **Example Error Alarm**: Alarm if `HTTPCode_Target_5XX_Count` on ALB Target Group (Sum) > 5 over 5 minutes.
    -   **Example DB Connection Alarm**: Alarm if `DatabaseConnections` on RDS instance (Maximum) > 80% of `max_connections` for 10 minutes.
    -   **Example Log Error Alarm**: Create a Metric Filter for `ERROR` or `CRITICAL` log entries, then alarm if the count > 10 in 5 minutes.
6.  **Set up SNS Topics and Subscriptions**: Create topics (e.g., `MySaaSAppCriticalAlerts`) and subscribe your email or other notification endpoints.
7.  **Integrate X-Ray (Optional)**: Instrument your FastAPI application using the AWS X-Ray SDK for Python to enable distributed tracing.

## 7. Summary

A proactive monitoring and alerting strategy using services like CloudWatch is essential for maintaining a healthy SaaS application. Focus on key metrics across all layers of the stack, configure centralized logging, define meaningful alert thresholds, and ensure alerts reach the right people promptly.
