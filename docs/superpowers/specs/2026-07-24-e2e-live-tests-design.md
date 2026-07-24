# Live E2E Test Suite — Design

**Date:** 2026-07-24
**Status:** approved, in implementation

## Goal
Real end-to-end tests against the **live deployment** covering the full user journey
(register → login → project → all working features), plus browser-level validation of
the real UI. Find and fix every real issue surfaced.

## Target
- Backend: `https://carbon-credit-backend-nu.vercel.app` (env `E2E_API_URL`)
- Frontend: `https://frontend-seven-rust-ndw61u0v8l.vercel.app` (env `E2E_WEB_URL`)
- Unique throwaway user per run (timestamped email); minimal/spaced logins (login is 5/min).

## Layer 1 — API E2E (pytest + httpx) — `tests/e2e_live/test_api_journey.py`
- **Auth:** register → login → me → change-password → forgot-password (grab demo reset link)
  → reset-password → login w/ new pw → logout. Security: self-assigned `Administrator`
  is downgraded; duplicate email → 400; invalid token → 401.
- **Projects:** create → list → get → update → status-patch → status-logs.
- **Verification:** create → get → list → human-review. Assert `ai_confidence` is null (honest).
- **IoT:** sensor create → list → post reading → readings → analytics → update → delete.
- **Analytics/Reports/Settings:** dashboard/performance/carbon-impact (assert real F1 0.49/0.60,
  no fabricated numbers), reports endpoints, settings get/patch.
- **Honest-disabled:** `/ml/*`, `/xai/*`, `/blockchain/*` return unavailable/503 — never fake data.
- **Cleanup:** delete created project + sensor.

## Layer 2 — Browser E2E (Playwright, headless chromium) — `tests/e2e_live/playwright/`
Real UI: register → login → dashboard → create project → open it → forgot-password flow →
logout. Assert no fabricated stats are rendered.

## Structure
- `tests/e2e_live/` (new): `test_api_journey.py`, `conftest.py`, `playwright/user-journey.spec.js`,
  `playwright.config.js`, `requirements.txt`, `README.md`.
- Obsolete XAI/localhost tests moved to `tests/_archive/`.

## Run + fix
Install deps (pytest, httpx, `npx playwright install chromium`); run both against live; fix
each real defect (backend/frontend) and re-run until green.

## Constraints
Login rate limit (5/min) → few, spaced logins. Turso pollution → throwaway user + cleanup.
ML/XAI/blockchain disabled on live → assert honest state, don't fail on them.
