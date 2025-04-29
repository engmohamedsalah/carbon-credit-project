# Security Hardening Checklist

This checklist provides security hardening recommendations for the Carbon Credit Verification SaaS application. It covers various layers, from infrastructure to application code. This is not exhaustive, and a thorough security review by professionals is recommended for production systems.

**Note**: Apply principles of least privilege and defense-in-depth throughout.

## 1. Infrastructure Security (AWS Focus)

-   **[ ] VPC Configuration**: Use private subnets for application instances and databases. Use NAT Gateways/Endpoints for controlled outbound/AWS service access. Avoid using the default VPC for production.
-   **[ ] Security Groups**: Apply strict ingress/egress rules. Only allow necessary ports and protocols from specific sources (e.g., allow RDS access only from application security group, allow ALB access only from 0.0.0.0/0 on port 443).
-   **[ ] Network ACLs**: Use as a stateless firewall layer for subnet-level traffic control (complementary to Security Groups).
-   **[ ] EC2 Instance Hardening (if using EC2)**:
    -   Use hardened AMIs (e.g., CIS Benchmarked AMIs).
    -   Disable direct SSH access or restrict it via Bastion Hosts/Session Manager.
    -   Regularly patch the OS and installed software.
    -   Run instances with minimal IAM permissions attached via instance profiles.
-   **[ ] Fargate Security (if using Fargate)**:
    -   Ensure tasks run with minimal IAM task roles.
    -   Keep Fargate platform versions updated.
-   **[ ] Load Balancer (ALB/ELB)**:
    -   Use HTTPS listeners with AWS Certificate Manager (ACM) certificates.
    -   Configure strong TLS policies (e.g., `ELBSecurityPolicy-TLS-1-2-Ext-2018-06` or newer).
    -   Enable access logs for the load balancer.
    -   Integrate with AWS WAF (Web Application Firewall) for protection against common web exploits (SQL injection, XSS).
-   **[ ] IAM**: Apply least privilege principle for all IAM users, roles, and policies. Use roles for service-to-service communication (e.g., EC2/ECS tasks accessing S3/RDS). Enable MFA for privileged users.

## 2. Application Security (Backend - FastAPI)

-   **[ ] Input Validation**: Rigorously validate and sanitize all user inputs (query parameters, path parameters, request bodies) using Pydantic models. Check data types, lengths, ranges, and formats.
-   **[ ] Authentication**: Use strong password hashing (e.g., `passlib` with bcrypt). Implement secure JWT handling (strong secret key, appropriate algorithm like HS256/RS256, short expiry times, refresh tokens).
-   **[ ] Authorization**: Implement granular authorization checks based on user roles and ownership (e.g., using FastAPI dependencies) for all sensitive endpoints.
-   **[ ] Dependency Security**: Regularly scan dependencies for known vulnerabilities (e.g., using `safety`, GitHub Dependabot, Snyk).
-   **[ ] Rate Limiting**: Implement rate limiting on sensitive endpoints (login, registration, API calls) to prevent brute-force attacks and abuse (e.g., using `slowapi`).
-   **[ ] Security Headers**: Set appropriate HTTP security headers (e.g., `Strict-Transport-Security`, `Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`).
-   **[ ] CORS Configuration**: Configure CORS middleware (`CORSMiddleware`) strictly, allowing only trusted frontend origins.
-   **[ ] Error Handling**: Avoid leaking sensitive information or stack traces in error messages returned to the client (as detailed in Error Handling Strategy).
-   **[ ] Disable Debug Mode**: Ensure `debug=False` in production Uvicorn/FastAPI settings.

## 3. Application Security (Frontend - React)

-   **[ ] Cross-Site Scripting (XSS) Prevention**: React generally protects against XSS via JSX encoding, but be cautious when using `dangerouslySetInnerHTML`. Sanitize user-generated content if displayed directly.
-   **[ ] Dependency Security**: Regularly scan frontend dependencies (npm/yarn) for vulnerabilities (e.g., `npm audit`, Dependabot, Snyk).
-   **[ ] Secure API Communication**: Always use HTTPS for API calls. Handle API keys or tokens securely (avoid storing in `localStorage` if possible; consider `HttpOnly` cookies managed by the backend or in-memory storage).
-   **[ ] Content Security Policy (CSP)**: Implement a strict CSP via meta tags or HTTP headers (set by backend/proxy) to control which resources the browser is allowed to load.
-   **[ ] Third-Party Scripts**: Be cautious when including third-party scripts; ensure they are from trusted sources and consider Subresource Integrity (SRI).

## 4. Data Security

-   **[ ] Encryption in Transit**: Use TLS/HTTPS for all external communication (user-to-ALB, ALB-to-application). Ensure internal communication (application-to-database) is also encrypted (e.g., RDS SSL connections).
-   **[ ] Encryption at Rest**: Enable encryption at rest for RDS databases, S3 buckets (SSE-S3 or SSE-KMS), and EBS volumes (if using EC2).
-   **[ ] Database Security**: Use strong, unique passwords for database users. Grant minimal necessary privileges to the application database user. Avoid using the master database user for the application.
-   **[ ] S3 Bucket Policies**: Configure strict bucket policies and ACLs. Prefer private buckets with access granted via IAM roles and pre-signed URLs generated by the backend.
-   **[ ] Sensitive Data Handling**: Avoid storing unnecessary sensitive data. Anonymize or pseudonymize data where possible. Be mindful of logging sensitive information.

## 5. Secrets Management

-   **[ ] Secure Storage**: Do NOT hardcode secrets (API keys, database passwords, JWT secrets) in code or commit them to version control.
-   **[ ] Use Secrets Manager**: Store secrets in AWS Secrets Manager or Systems Manager Parameter Store (SecureString).
-   **[ ] Secure Injection**: Inject secrets securely into the application environment at runtime (e.g., via ECS task definition integration with Secrets Manager, or mounted volumes in Kubernetes).
-   **[ ] Rotation**: Implement regular rotation for sensitive secrets like database passwords and API keys.

## 6. Container Security

-   **[ ] Base Image Selection**: Use minimal, trusted base images (e.g., official Python slim images, Distroless images).
-   **[ ] Image Scanning**: Integrate container image vulnerability scanning into the CI/CD pipeline (e.g., ECR Enhanced Scanning, Trivy, Snyk).
-   **[ ] Least Privilege**: Run container processes as non-root users. Remove unnecessary tools or libraries from the final image.
-   **[ ] Multi-Stage Builds**: Use multi-stage Docker builds to keep the final image lean and free of build tools/dependencies.

## 7. Logging & Monitoring for Security

-   **[ ] Comprehensive Logging**: Ensure sufficient logging of authentication events, authorization failures, key application actions, and errors.
-   **[ ] Centralized Logging**: Aggregate logs securely (e.g., CloudWatch Logs).
-   **[ ] Log Monitoring & Alerting**: Set up alerts for suspicious activities detected in logs (e.g., multiple failed logins, access attempts from unusual locations, specific error patterns) using CloudWatch Metric Filters and Alarms or dedicated SIEM tools.
-   **[ ] AWS CloudTrail**: Enable CloudTrail to log all AWS API calls for auditing and security analysis.
-   **[ ] AWS GuardDuty**: Enable GuardDuty for intelligent threat detection across your AWS accounts and workloads.

## 8. API Security

-   **[ ] HTTPS Everywhere**: Enforce HTTPS for all API endpoints.
-   **[ ] Strong Authentication**: Use robust authentication mechanisms (e.g., JWT as implemented).
-   **[ ] Granular Authorization**: Ensure proper authorization checks are performed on every request.
-   **[ ] Input Validation**: Protect against injection attacks and malformed requests.
-   **[ ] Rate Limiting**: Prevent abuse and DoS attacks.

## 9. Blockchain Security

-   **[ ] Smart Contract Audits**: Conduct thorough audits of the Solidity smart contract code by reputable third-party auditors before deployment to mainnet.
-   **[ ] Secure Private Key Management**: Store the private key used by the backend service to interact with the blockchain (e.g., issue certificates) extremely securely, ideally using AWS Secrets Manager or KMS. Never hardcode it or commit it.
-   **[ ] Gas Limits & Monitoring**: Implement appropriate gas limits for transactions. Monitor gas costs and transaction success/failure rates.
-   **[ ] Access Control (Contract)**: Ensure smart contract functions have appropriate access control modifiers (e.g., `onlyOwner`, role-based access) if needed.
-   **[ ] Reentrancy Guards**: Use checks-effects-interactions pattern and reentrancy guards if the contract handles Ether or interacts with other contracts.

## 10. Regular Audits & Patching

-   **[ ] Dependency Patching**: Regularly update OS, application framework, and library dependencies.
-   **[ ] Security Audits**: Periodically conduct internal or external security audits and penetration tests.
-   **[ ] Review IAM Policies**: Regularly review and tighten IAM policies.
-   **[ ] Review Security Group Rules**: Regularly review and remove unnecessary rules.

This checklist serves as a starting point for hardening the security posture of the application.
