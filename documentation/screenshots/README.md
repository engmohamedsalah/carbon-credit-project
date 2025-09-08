# Carbon Credit Verification App - Screenshot Guide

This folder contains screenshots of the Carbon Credit Verification SaaS application for documentation purposes.

## Folder Structure

- `/auth` - Authentication related screens
- `/dashboard` - Main dashboard views
- `/projects` - Project management screens
- `/verification` - Verification workflow screens
- `/xai` - Explainable AI dashboard and visualizations
- `/reports` - Reporting and analytics screens
- `/admin` - Admin panel screens
- `/mobile` - Mobile responsive views

## Screenshots to Capture

### Authentication (`/auth`)
1. **login.png** - Login page
2. **register.png** - Registration page
3. **forgot-password.png** - Password reset page
4. **two-factor.png** - Two-factor authentication (if enabled)

### Dashboard (`/dashboard`)
1. **main-dashboard.png** - Main dashboard overview
2. **dashboard-stats.png** - Statistics and key metrics
3. **recent-activity.png** - Recent activity feed
4. **notifications.png** - Notifications panel

### Projects (`/projects`)
1. **projects-list.png** - Projects list/grid view
2. **project-details.png** - Individual project details
3. **project-create.png** - New project creation form
4. **project-map.png** - Project location on map
5. **project-timeline.png** - Project timeline view

### Verification (`/verification`)
1. **verification-workflow.png** - Main verification workflow
2. **satellite-imagery.png** - Satellite imagery viewer
3. **forest-change-detection.png** - Forest change detection results
4. **carbon-estimation.png** - Carbon sequestration estimates
5. **verification-status.png** - Verification status tracking
6. **blockchain-cert.png** - Blockchain certification view

### XAI Dashboard (`/xai`)
1. **xai-overview.png** - XAI dashboard main view
2. **shap-visualization.png** - SHAP explanations
3. **lime-visualization.png** - LIME explanations
4. **integrated-gradients.png** - Integrated gradients view
5. **method-comparison.png** - Comparison of XAI methods
6. **explanation-history.png** - Historical explanations
7. **compliance-report.png** - XAI compliance reporting

### Reports (`/reports`)
1. **analytics-dashboard.png** - Analytics overview
2. **carbon-credits-report.png** - Carbon credits report
3. **verification-report.png** - Verification report
4. **export-options.png** - Export/download options

### Admin Panel (`/admin`)
1. **admin-dashboard.png** - Admin overview
2. **user-management.png** - User management interface
3. **system-settings.png** - System configuration
4. **ml-model-status.png** - ML model monitoring

### Mobile Views (`/mobile`)
1. **mobile-login.png** - Mobile login screen
2. **mobile-dashboard.png** - Mobile dashboard
3. **mobile-project-view.png** - Mobile project view
4. **mobile-verification.png** - Mobile verification workflow

## How to Capture Screenshots

### For Development Environment:

1. Start both frontend and backend:
   ```bash
   cd /Users/msalah/Hull/Dissertation\ project/carbon_credit_project
   ./run_app.sh
   ```

2. Frontend will be available at: http://localhost:3000
3. Backend API docs at: http://localhost:8000/docs

### Capture Tools:
- **Mac**: Use Command + Shift + 4 for area selection
- **Browser DevTools**: Toggle device toolbar for mobile screenshots
- **Full page**: Use browser extensions for full-page captures

### Best Practices:
1. Use consistent window size (e.g., 1920x1080 for desktop)
2. Use sample data that looks realistic
3. Hide sensitive information
4. Capture both light and dark themes if available
5. Show interactive states (hover, active, error states)

### Automated Screenshot Options:

For automated screenshots, you can use the Playwright tests:
```bash
cd tests/e2e
./run_tests.sh --screenshot
```

Or create a dedicated screenshot script using Playwright:
```javascript
// screenshot-generator.js
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  // Set viewport
  await page.setViewportSize({ width: 1920, height: 1080 });
  
  // Login
  await page.goto('http://localhost:3000/login');
  await page.screenshot({ path: 'auth/login.png' });
  
  // Add more pages...
  
  await browser.close();
})();
```

## Screenshot Naming Convention

- Use lowercase with hyphens: `feature-name.png`
- Include state if relevant: `project-list-empty.png`, `project-list-populated.png`
- Mobile screenshots: `mobile-feature-name.png`
- Error states: `feature-name-error.png`
- Loading states: `feature-name-loading.png`
