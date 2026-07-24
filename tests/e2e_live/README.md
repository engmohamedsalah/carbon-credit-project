# Live E2E Tests

Real end-to-end tests against the **deployed** app (Vercel + Turso). Two layers:

| Layer | Tool | File |
|---|---|---|
| API journey (all endpoints) | pytest + requests | `test_api_journey.py` |
| Browser journey (real UI) | Playwright (chromium) | `playwright/user-journey.spec.js` |

Each run uses a **unique throwaway user** (timestamped email) and cleans up created
resources. Logins tolerate the backend's 5/min rate limit with a short backoff.

## Configuration (env vars, defaults are the live deployment)
- `E2E_API_URL` — default `https://carbon-credit-backend-nu.vercel.app/api/v1`
- `E2E_WEB_URL` — default `https://frontend-seven-rust-ndw61u0v8l.vercel.app`

## Run

API journey:
```
pip install -r tests/e2e_live/requirements.txt
pytest tests/e2e_live/test_api_journey.py -v
```

Browser journey:
```
cd tests/e2e_live/playwright
npm install
npx playwright install chromium
npx playwright test
```

## What's covered
- **Auth:** register (privilege-escalation blocked), login, `me`, duplicate-email, invalid
  token, change-password, forgot-password → reset-password → login.
- **Projects:** create, list, get, update, status + status-logs.
- **Verification:** create (asserts `ai_confidence` is null — no fabricated AI score), get,
  list, human-review.
- **IoT:** sensor CRUD + readings + analytics.
- **Analytics/Reports/Settings:** dashboard/performance/carbon-impact (asserts no fabricated
  numbers), report RBAC, settings.
- **Honest-disabled:** ML / XAI / blockchain return an explicit unavailable/503 — never fake data.
- **Browser:** the real UI renders real data (created project appears), no fabricated stats on
  the dashboard, and the forgot-password flow works.

## Notes / findings
- ML inference and XAI are disabled on the live serverless backend (torch not bundled); the
  tests assert the honest "unavailable" state. Real ML inference needs a local run with torch.
- The New Project **form** requires drawing the area on a Leaflet map (`geometry` is required
  client-side) and a `project_type`, whereas the backend accepts a project without geometry.
  The browser test therefore creates the project via the API and verifies the UI renders it.
