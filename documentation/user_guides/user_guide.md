# Carbon Credit Verification SaaS Application
## User Guide

This comprehensive user guide provides step-by-step instructions for using the Carbon Credit Verification SaaS application, from account creation to verification certification.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Creating a New Project](#creating-a-new-project)
4. [Uploading Satellite Imagery](#uploading-satellite-imagery)
5. [Running Forest Change Detection](#running-forest-change-detection)
6. [Understanding Analysis Results](#understanding-analysis-results)
7. [Carbon Sequestration Estimation](#carbon-sequestration-estimation)
8. [Verification Workflow](#verification-workflow)
9. [Blockchain Certification](#blockchain-certification)
10. [Reports and Exports](#reports-and-exports)
11. [Account Management](#account-management)
12. [Troubleshooting](#troubleshooting)

## Getting Started

### Creating an Account

1. Navigate to the application URL in your web browser
2. Click the "Register" button in the top-right corner
3. Fill in the registration form with your details:
   - Email address
   - Full name
   - Password (minimum 8 characters)
   - Organization (optional)
4. Click "Create Account"
5. Verify your email address by clicking the link sent to your inbox
6. Log in with your credentials

### Logging In

1. Navigate to the application URL
2. Enter your email and password
3. Click "Log In"
4. If you've forgotten your password, click "Forgot Password" and follow the instructions

### User Interface Overview

The application interface consists of several key areas:

- **Top Navigation Bar**: Access to main sections and account settings
- **Side Menu**: Project navigation and tools
- **Main Content Area**: Primary workspace for current activity
- **Notification Center**: System messages and alerts

## Dashboard Overview

The dashboard provides a comprehensive overview of your projects and activities:

### Dashboard Sections

1. **Project Summary**: Overview of all your projects
   - Project count by status
   - Recent activity timeline
   - Quick access to active projects

2. **Verification Status**: Summary of verification processes
   - Pending verifications
   - Completed verifications
   - Verification success rate

3. **Carbon Impact**: Visualization of carbon sequestration results
   - Total carbon impact across projects
   - Carbon impact trends over time
   - Comparison with baseline

4. **Quick Actions**: Shortcuts to common tasks
   - Create new project
   - Upload satellite imagery
   - Start verification
   - Generate reports

### Filtering and Sorting

- Use the filter dropdown to view specific project types
- Sort projects by name, date, status, or carbon impact
- Use the search bar to find specific projects

## Creating a New Project

### Step 1: Project Setup

1. Click "New Project" on the dashboard or side menu
2. Enter basic project information:
   - Project name
   - Description
   - Project type (e.g., Reforestation, Avoided Deforestation)
   - Start date

### Step 2: Define Project Area

1. On the map interface, define your project area using one of these methods:
   - Draw polygon directly on the map
   - Upload GeoJSON file with project boundaries
   - Enter coordinates manually
   - Search for a location and define radius

2. Adjust the boundaries as needed using the editing tools
3. Click "Save Area" when finished

### Step 3: Project Details

1. Enter additional project details:
   - Project methodology
   - Expected carbon impact
   - Project timeline
   - Stakeholders
   - Additional documentation (optional)

2. Click "Create Project" to finalize

## Uploading Satellite Imagery

### Automatic Satellite Imagery Acquisition

The system can automatically acquire satellite imagery for your project area:

1. Navigate to your project page
2. Click "Acquire Satellite Imagery" tab
3. Select date range for imagery
4. Choose imagery source (default: Sentinel-2)
5. Click "Acquire Images"
6. The system will automatically download appropriate imagery
7. Once complete, you'll receive a notification

### Manual Satellite Imagery Upload

If you have your own satellite imagery:

1. Navigate to your project page
2. Click "Upload Satellite Imagery" tab
3. Select imagery type (Sentinel-2, Landsat, etc.)
4. Upload band files:
   - For Sentinel-2: Upload B02, B03, B04, B08 band files
   - For Landsat: Upload appropriate band files
5. Enter imagery date and metadata
6. Click "Upload"

### Managing Satellite Imagery

- View all uploaded imagery in the "Imagery" tab
- Filter imagery by date, type, or quality
- Preview imagery with different band combinations
- Delete or replace imagery as needed

## Running Forest Change Detection

### Starting Analysis

1. Navigate to your project page
2. Select the "Analysis" tab
3. Choose the satellite images to analyze:
   - For change detection: Select "before" and "after" images
   - For single-point analysis: Select one image
4. Select analysis type:
   - Forest Cover Change Detection
   - Carbon Stock Estimation
   - Custom Analysis (advanced)
5. Click "Start Analysis"

### Monitoring Progress

1. The analysis progress is displayed in real-time
2. You can continue using the application during processing
3. You'll receive a notification when analysis is complete
4. View the analysis queue for multiple running analyses

## Understanding Analysis Results

### Forest Change Detection Results

The results page displays several key elements:

1. **Overview Map**: Visual representation of the project area with detected changes
   - Green: Forest gain
   - Red: Forest loss
   - Gray: No change

2. **Statistics Panel**:
   - Total area analyzed
   - Forest cover percentage
   - Forest loss area (hectares)
   - Forest gain area (hectares)
   - Net change

3. **Confidence Metrics**:
   - Overall confidence score
   - Area-specific confidence heatmap
   - Factors affecting confidence

4. **Explainable AI Visualizations**:
   - Feature importance heatmap
   - Decision explanation for specific areas
   - Comparison with reference data

### Interpreting Visualizations

- **RGB Composite**: Natural color view of the area
- **NDVI Visualization**: Vegetation health indicator
- **Change Detection Map**: Areas of forest loss/gain
- **Confidence Map**: Reliability of predictions
- **Explanation Map**: Factors influencing predictions

### Exploring Results

1. Use the map navigation tools to zoom and pan
2. Click on specific areas to see detailed information
3. Toggle different visualization layers
4. Adjust visualization parameters as needed
5. Export results in various formats

## Carbon Sequestration Estimation

### Running Carbon Estimation

1. Navigate to your project's analysis results
2. Click "Estimate Carbon Impact"
3. Select carbon estimation methodology:
   - Standard IPCC methodology
   - Custom methodology
4. Enter additional parameters if required
5. Click "Calculate"

### Understanding Carbon Results

The carbon estimation results include:

1. **Carbon Stock**:
   - Total carbon stock (tons C)
   - Carbon density (tons C/ha)
   - Uncertainty range

2. **Carbon Change**:
   - Net carbon change (tons C)
   - CO₂ equivalent (tons CO₂e)
   - Annual rate of change

3. **Visualizations**:
   - Carbon stock distribution map
   - Carbon change heatmap
   - Comparison with baseline

4. **Uncertainty Analysis**:
   - Confidence intervals
   - Sensitivity analysis
   - Comparison with ground truth (if available)

## Verification Workflow

### Initiating Verification

1. Navigate to your project's analysis results
2. Click "Start Verification"
3. Select verification type:
   - Standard verification
   - Enhanced verification (with additional review)
   - Custom verification
4. Add supporting documentation (optional)
5. Click "Submit for Verification"

### Human-in-the-Loop Review

The verification process includes expert review:

1. System performs automated checks
2. Results are queued for expert review
3. Expert reviewer examines:
   - Analysis results
   - Methodology compliance
   - Data quality
   - Uncertainty assessment
4. Reviewer may request additional information
5. Reviewer approves or rejects verification

### Verification Status Tracking

Track verification progress in the "Verification" tab:

1. **Pending**: Initial submission
2. **In Review**: Under expert examination
3. **Information Requested**: Additional data needed
4. **Approved**: Verification successful
5. **Rejected**: Verification unsuccessful
6. **Certified**: Blockchain certification complete

## Blockchain Certification

### Creating Certificates

Once verification is approved:

1. Navigate to the verified project
2. Click "Create Certificate"
3. Review certificate details:
   - Project information
   - Verification results
   - Carbon impact
   - Methodology reference
4. Click "Certify on Blockchain"

### Certificate Details

Each blockchain certificate contains:

1. **Project Identifier**: Unique project reference
2. **Verification Timestamp**: Date and time of verification
3. **Results Hash**: Cryptographic hash of detailed results
4. **Verifier Information**: Who performed the verification
5. **Methodology Reference**: Standards followed
6. **Carbon Impact**: Quantified carbon sequestration

### Viewing and Sharing Certificates

1. Access certificates from the "Certificates" tab
2. View certificate details including blockchain transaction
3. Share certificates via:
   - Public URL
   - PDF export
   - Blockchain explorer link
4. Verify certificate authenticity using the verification tool

## Reports and Exports

### Generating Reports

1. Navigate to your project
2. Click "Reports" tab
3. Select report type:
   - Project Summary
   - Verification Report
   - Carbon Impact Report
   - Technical Analysis Report
   - Custom Report
4. Configure report parameters
5. Click "Generate Report"

### Export Formats

Export data in various formats:

1. **Documents**:
   - PDF
   - Word
   - HTML

2. **Data**:
   - CSV
   - Excel
   - JSON
   - GeoJSON

3. **Visualizations**:
   - PNG
   - SVG
   - Interactive HTML

### Scheduling Reports

Set up automatic report generation:

1. Navigate to "Reports" tab
2. Click "Schedule Report"
3. Select report type and parameters
4. Set frequency (daily, weekly, monthly)
5. Add recipients for email delivery
6. Click "Save Schedule"

## Account Management

### Profile Settings

Manage your account settings:

1. Click your username in the top-right corner
2. Select "Profile Settings"
3. Update personal information
4. Change password
5. Configure notification preferences
6. Manage API keys
7. Set up two-factor authentication

### Team Management

For organization accounts:

1. Navigate to "Team" in the side menu
2. View current team members
3. Invite new members:
   - Enter email address
   - Assign role (Admin, Analyst, Viewer)
   - Set project access
4. Manage existing members:
   - Change roles
   - Adjust project access
   - Remove members

### Billing and Subscription

Manage your subscription:

1. Navigate to "Billing" in the side menu
2. View current plan and usage
3. Upgrade or downgrade subscription
4. Update payment information
5. View invoice history
6. Download invoices

## Troubleshooting

### Common Issues

#### Satellite Imagery Upload Failures

**Problem**: Satellite imagery fails to upload or process.

**Solutions**:
1. Ensure imagery is in the correct format (GeoTIFF)
2. Check that all required bands are included
3. Verify file size is within limits (max 500MB per file)
4. Confirm imagery covers the project area
5. Try processing smaller sections if the area is very large

#### Analysis Timeout

**Problem**: Analysis takes too long or times out.

**Solutions**:
1. Reduce the size of the analysis area
2. Use lower resolution imagery for initial analysis
3. Simplify the project boundary geometry
4. Check for server status issues
5. Try again during off-peak hours

#### Verification Rejection

**Problem**: Verification is rejected by the reviewer.

**Solutions**:
1. Review the rejection comments carefully
2. Address all identified issues
3. Provide additional documentation if requested
4. Improve imagery quality if possible
5. Consider adjusting project boundaries to exclude problematic areas

### Getting Help

If you encounter issues not covered in this guide:

1. Check the Knowledge Base for similar issues
2. Use the in-app chat support
3. Email support@carboncreditverification.com
4. Schedule a consultation with our technical team
5. Join our community forum for peer assistance

---

This user guide provides comprehensive instructions for using the Carbon Credit Verification SaaS application. For technical details about the system architecture and implementation, please refer to the technical documentation.
