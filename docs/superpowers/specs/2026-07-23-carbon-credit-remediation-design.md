# Carbon Credit Verification — Remediation & Honest-Launch Design

**Date:** 2026-07-23
**Author:** engineering review + remediation
**Status:** proposed (awaiting owner approval)

## 1. Goal

Take the project from *"a reviewer catches you in five minutes"* to *"a genuinely
impressive, honestly-scoped prototype that is safe to expose publicly."* Two hard
requirements from the owner:

1. **Remove every fake / demo / mock** — the app must not present fabricated data
   as real.
2. **Make it deployable** — frontend to Vercel, backend to a container host; the
   git repository must be safe to be public.

## 2. Scope — the 11 validated issues

All 11 were independently confirmed against the code (see the validation run).
This spec fixes all of them.

| # | Issue | Class | Fix type |
|---|---|---|---|
| 1 | Secrets committed (DB w/ live tokens, creds in docs, `.env`) | Security | Real fix + history purge |
| 2 | Privilege escalation: `register` trusts client `role` | Security | Real fix |
| 3 | Auth tokens never expire | Security | Real fix |
| 4 | Verification endpoint fabricates AI score via `hash()` | Honesty | Remove fabrication |
| 5 | XAI runs SHAP/IG on `np.random` under "REAL PROCESSING" banner | Honesty | Real-on-upload or honest empty state |
| 6 | `analyze_location` is a coordinate → mock formula | Honesty | Remove mock endpoint |
| 7 | Carbon math: unit error + inconsistent constants | Correctness | Real fix (one cited figure) |
| 8 | Session lost on page refresh | Frontend | Real fix |
| 9 | `ErrorBoundary` never mounted | Frontend | Real fix |
| 10 | Docs claim "99.1% / production ready" | Honesty | Rewrite to truth |
| 11 | Contract never deployed; service returns mock tx-ids | Honesty | Capability-gate, no mock |

### Non-goals (explicitly out of scope for this pass)
- Refactoring the 2994-line `main.py` into routers (maintainability, not launch-blocking).
- Wiring the existing test suites into CI.
- Building a real satellite-imagery ingestion pipeline (see §3).
- Migrating SQLite → Postgres (SQLite + a persistent volume is fine for a single-instance pilot).

## 3. The hard constraint that shapes everything

The trained PyTorch models are **real** and run genuine inference — **but only on
manually-supplied GeoTIFF imagery** (`ml_service.analyze_forest_cover` /
`detect_changes` call the real `CarbonCreditVerificationPipeline`). There is **no
pipeline that fetches imagery from a project's coordinates/geometry**. Therefore:

- Any feature that takes only coordinates (or a project id) and claims an AI result
  **cannot be made real** without infrastructure that does not exist. Those features
  are **removed** or reduced to their genuine parts — never faked.
- The genuine capability we keep and surface: **upload satellite imagery → real model
  inference → real carbon estimate → real explanation.**

This is the honest core. Everything else that pretended to be AI is deleted.

## 4. Design per issue

### Security

**#2 — Privilege escalation (trivial).**
`register()` will force the role server-side before `create_user`:
`user_data.role = Roles.PROJECT_DEVELOPER.value`. Client-supplied `role` is ignored.
Elevation happens only through an existing admin-gated path. Add a self-check: a
POST with `role:"Administrator"` yields a non-admin user.

**#3 — Token expiry (small).**
- `store_token(token, user_id, ttl_minutes)` writes `expires_at = datetime('now', '+<ttl>')`.
- `get_user_by_token` gains `AND (t.expires_at IS NULL OR t.expires_at > datetime('now'))`
  (NULL-safe so it never breaks, though a fresh DB has no NULL rows).
- TTL from config (`ACCESS_TOKEN_EXPIRE_MINUTES`, default 7 days for a pilot).

**#1 — Secrets (medium) — two parts:**

*Code / working-tree:*
- Introduce lightweight runtime config (`backend/config.py`, `os.getenv` with defaults):
  `SECRET_KEY`, `DATABASE_PATH`, `CORS_ORIGINS` (comma-separated), `ACCESS_TOKEN_EXPIRE_MINUTES`,
  blockchain vars. Wire `DATABASE_PATH` and CORS in `main.py` from config (removes hardcoding).
- Stop committing runtime data: `git rm --cached database/carbon_credits.db backend/.env
  fresh_token.txt token_cmd.txt USERS.md USER_ACCOUNTS.md`. Delete `backend/update_user_passwords.py`
  (hardcoded creds).
- `.gitignore` += `database/*.db`, `*token*.txt`.
- Runtime seed: `backend/seed_admin.py` (or startup hook) creates a single admin from
  `ADMIN_EMAIL` / `ADMIN_PASSWORD` env vars **only if the users table is empty**. No
  credentials in the repo; the DB is created fresh on first run.
- `backend/.env.example` rewritten to reflect reality (SQLite, no Postgres URL, documented
  blockchain-optional vars).

*History (destructive, owner-approved):*
- `git filter-repo --invert-paths` (or BFG) to strip `database/carbon_credits.db`,
  `backend/.env`, `fresh_token.txt`, `token_cmd.txt`, `USERS.md`, `USER_ACCOUNTS.md`,
  `backend/update_user_passwords.py` from **all** commits.
- Force-push rewritten history to `origin`.
- Rotation is implicit: the leaked DB (750 tokens) and the `admin123`/`password123`
  accounts no longer exist anywhere; a new `SECRET_KEY` and a new seeded admin replace them.

### Honesty / de-mocking

**#6 — `analyze_location` mock (remove).**
Delete the coordinate-based mock body and its API endpoint + the frontend entry that calls
it. Keep the real image-upload analysis (`MLAnalysis` UI + `analyze_forest_cover` /
`detect_changes`). If any nav/route references the removed feature, remove or relabel it.

**#4 — Verification `hash()` (remove fabrication).**
In `create_verification`:
- Delete the `hash()`-based `ai_confidence` and the `hash()`-based `carbon_impact`.
- `ai_confidence` = the real value from an ML analysis of the project's uploaded imagery
  **if one exists**, else `None`.
- `carbon_impact` = the user-supplied estimate (`verification_data.carbon_impact`) if
  provided, else `None` — never fabricated.
- Record stays `status="pending"`, `human_verified=False` — an honest human-in-the-loop record.
- Make `ai_confidence` / `carbon_impact` `Optional` in the response model if not already.

**#5 — XAI on random noise (make real or honest-empty).**
- Delete the false banners ("NO DEMO OR MOCK DATA — REAL PROCESSING ONLY") everywhere.
- Delete all `np.random`-fabricated outputs (global importance, interactions, method
  stability, padded carbon/financial/risk numbers).
- Primary: run the **real** SHAP / Integrated-Gradients explainers on the **real** model and
  the **real** input tensor produced by the image-upload analysis path.
- If no real analysis input exists for the request, the API returns an honest
  "no analysis available — run an image analysis first" state and the UI renders an empty
  state. **No synthetic explanation is ever returned.**
- Honest fallback (only if real wiring proves infeasible in this pass): disable the XAI
  surface rather than ship fabricated explanations. This will be called out explicitly if used.

**#11 — Blockchain (capability-gate, no mock) — best practice.**
- Delete the `demo_tokens` (123/456/789) mock from `verify_certificate`.
- Compute `self.enabled = bool(contract_address and abi and private_key)` in `__init__`.
- When disabled: mint/verify endpoints return `503 "blockchain certification not configured"`;
  the frontend hides/disables the certify button based on a capability flag from the API.
- Default RPC updated off the dead Mumbai endpoint (Polygon Amoy `80002`) — used only when enabled.
- Keep the real Web3 mint/verify code intact. Provide `blockchain/scripts/deploy-amoy` notes so
  that adding a funded testnet key + deployed address later turns the feature on with no code change.

### Correctness & frontend

**#7 — Carbon units (real fix).**
In `ensemble_model.calculate_carbon_impact`: introduce module constants
`CARBON_DENSITY_TC_PER_HA = 150` (IPCC tropical-forest above-ground biomass carbon, cited)
and `CO2_PER_C = 3.67`; output `total_co2e_tons = forest_area_ha * 150 * 3.67` and rename
keys to `*_tco2e`. Update docstring + any consumer key names. (The other two inconsistent
carbon formulas are deleted by #4 and #6.)

**#8 — Session lost on refresh (trivial).**
- `authSlice` initial `loading = !!localStorage.getItem('token')`.
- `index.js`: on boot, if a token exists, `store.dispatch(getCurrentUser())`.
- `ProtectedRoute`: while `loading` is true (auth check in flight), render a loader instead
  of redirecting; redirect only once the check resolves unauthenticated.

**#9 — ErrorBoundary (trivial).**
`index.js`: import `ErrorBoundary` and wrap the tree (inside `StrictMode`, around `Provider`).

**#10 — Docs honesty (small).**
- Remove every "99.1% accuracy", "100% COMPLETE", "PRODUCTION READY", "ENTERPRISE-GRADE".
- State the real metrics that the code itself already records: forest-cover F1 ≈ 0.49,
  change-detection F1 ≈ 0.60 (source: `ml/evaluation_results/*.csv`, `production_inference.py`).
- Consolidate the ~9 inflated `*_COMPLETION_REPORT.md` / status docs into a single honest
  `STATUS.md` describing real vs. not-yet-real. Rewrite `README.md` to match.

## 5. Deployment design

**Topology:** frontend → **Vercel** (static SPA); backend → **container host**
(Railway / Render / Fly) using the existing `docker/backend.Dockerfile`. Vercel serverless
is unsuitable for the backend (PyTorch + ~100 MB weights exceed bundle limits; SQLite writes
need a persistent filesystem).

**Frontend prep:**
- `frontend/vercel.json` with SPA rewrite (`/(.*) → /index.html`) and the build command.
- `REACT_APP_API_URL` supplied as a Vercel env var pointing at the deployed backend.
- Verify `npm run build` succeeds locally before any deploy.

**Backend prep:**
- Container host config (`render.yaml` or Railway settings) with a **persistent volume**
  mounted at the SQLite `DATABASE_PATH`.
- Required env: `SECRET_KEY`, `CORS_ORIGINS` (the Vercel URL), `ADMIN_EMAIL`, `ADMIN_PASSWORD`,
  `ACCESS_TOKEN_EXPIRE_MINUTES`; optional blockchain vars.
- Verify the image builds and the app boots with a fresh DB.

**Actual deploy is deferred** (owner asked to prepare then discuss, and the deploy itself
needs the owner's interactive Vercel/host login). Deliverable: verified config + a
step-by-step runbook.

## 6. Phased execution

- **Phase A — Security:** config module, auth fixes (#2, #3), stop-tracking + `.gitignore` +
  seed + `.env.example` (#1 working-tree). Verify auth end-to-end.
- **Phase B — De-mock:** #6 remove location mock, #4 verification, #5 XAI, #11 blockchain gate.
- **Phase C — Correctness/docs:** #7 carbon, #8 refresh, #9 ErrorBoundary, #10 docs.
- **Phase D — Verify:** backend boots + auth/verification smoke test; `npm run build`;
  run existing fast tests where practical.
- **Phase E — History purge + force-push** (destructive, owner-approved).
- **Phase F — Deploy prep** + runbook; discuss go-live.

## 7. Verification strategy

- Auth: script proves (a) `register role:"Administrator"` → non-admin, (b) a token rejected
  after expiry, (c) a valid token works.
- De-mock: `grep` proves zero `np.random` in served XAI output paths, zero `hash(`-derived
  scores, zero `demo_tokens`, and the false banners are gone.
- Carbon: a unit assertion (`area_ha * 150 * 3.67`) in a small self-check.
- Frontend: production build succeeds; manual note on refresh-persistence.
- Secrets: `git ls-files` shows none of the removed paths; post-rewrite `git log` search
  finds none of them in history.

## 8. Risks & rollback

- **History rewrite is irreversible and rewrites public history.** Mitigation: a full local
  backup branch/bundle of the pre-rewrite repo is taken before `filter-repo`; force-push only
  after the working tree is verified green.
- **XAI real-wiring may be deeper than one pass.** Mitigation: the honest-empty-state fallback
  means we never ship fabricated data even if full real wiring slips.
- **Removing features (location analysis, blockchain-when-unconfigured) reduces the demo
  surface.** Accepted: honesty over surface area, per owner directive.

## 9. Review incorporations (authoritative — supersedes conflicting text above)

A 5-lens adversarial review of this spec against the code found blocking gaps. These
corrections are binding.

### 9.1 Secrets purge — purge by content, not hand-listed paths (#1)
History contains **additional** copies of the leaked database and tokens that the §4 list
missed: `api/carbon_credits.db` (byte-identical, 751 tokens + 289 hashes, commit 40ab9b1),
`data/carbon_credits.db` (commits b7562fb/3bdc914), `tests/temp_token.txt`.
- Purge with globs: `git filter-repo --path-glob '*.db' --path-glob '*token*.txt' --invert-paths`
  plus the named doc/script/.env paths.
- `.gitignore`: broaden to `*.db` (not just `database/*.db`).
- Verification (§7) MUST prove by content: `git log --all --name-only | grep -iE '\.db$|token'`
  returns nothing, and an object sweep finds no `.db` blob — not a re-check of the same list.
- `SECRET_KEY` is not used by auth (opaque tokens, no JWT); keep it in config only as
  future-proofing, and fix the README's false "JWT authentication" claim under #10.

### 9.2 #6 location removal — rewire MLAnalysis, don't just delete
`mlService.formatAnalysisResults` maps **only** location output into `summary`, which drives
the result tiles and `calculateEligibility`. Removing the endpoint alone leaves the "kept"
upload feature broken (empty tiles, garbage eligibility). Scope adds:
- Delete the Location Analysis accordion + coordinates state in `MLAnalysis.js`; drop the
  location step in `runComprehensiveAnalysis`.
- Remap `forestCoverAnalysis` / `changeDetection` output into `formatAnalysisResults.summary`
  and `calculateEligibility` so the real upload path renders correctly.

### 9.3 #5 XAI — DISABLE the fabricated surface this pass (honest fallback chosen)
Real SHAP/IG-on-upload is **not feasible in one pass**: no image tensor is retained or linked
to a project at XAI-request time (`process_single_image` discards the tensor; uploads aren't
recorded per project; the XAI request carries no image); the consumed panels (global beeswarm,
feature interactions, method stability) need a dataset run and cannot come from one image; and
the 15 named "features" (canopy_density, species_diversity…) are **fictional** — the model's
real inputs are 12 Sentinel-2 channels. Therefore:
- Delete the false banners and **all** `np.random`/`hash()` fabrication in
  `backend/services/real_xai_service.py` **and** the dead mock generators in
  `backend/services/ml_service.py:573-763`, including the fabricated `except`-fallbacks
  (they ship fake data on failure).
- The XAI route renders an honest "explainability not available in this build" state; the
  genuine `IntegratedGradients` code stays dormant (not wired to fake inputs).
- Real IG-attribution-heatmap over 12 channels + the required per-project image persistence +
  frontend rewrite is explicit **future work**, not this pass.

### 9.4 #11 blockchain — correct call sites + safe import
- The certify action + mock live in `Verification.js` (not `Blockchain.js`); the capability
  flag must gate the button there. Explorer URL `mumbai.polygonscan.com` at `Verification.js:224`
  and "demo mode" copy at `Blockchain.js:120` also change.
- Add `web3` + `eth_account` to requirements AND wrap the top-level imports in
  `blockchain_service.py` in try/except so disabled mode returns the designed **503**, not a
  **500** ImportError.
- Remove the hardcoded demo recipient wallet at `main.py:697`.

### 9.5 NEW de-mock surfaces (owner said "remove ANY fake/demo/mocks")
- **Analytics dashboard:** `get_dashboard_analytics` / `get_performance_analytics`
  (`main.py:~2281-2359`) return hardcoded ML metrics (`0.8912`, `0.9156`, `247 processed`) and
  `simulate`-generated accuracy trends, rendered by `Analytics.js` / `ChartComponents.js`.
  Replace with the real recorded metrics (F1 0.49/0.60 from `ml/evaluation_results/*.csv`) or an
  honest empty state; delete the simulated trends.
- **IoT frontend:** `IoTAnalytics.js` `generateSamplePerformanceData()` and
  `iotService.js` `generateSimulatedReading()` fabricate telemetry with `Math.random` (the IoT
  **backend** is real, DB-backed). Drive the charts from real `/iot/readings` or empty state;
  no `Math.random` telemetry served.

### 9.6 Deployment — single-image build from repo root (was FLAWED)
- Backend image must build with **context = repo root**, `COPY ml/` + the `.pth` weights, and
  install a merged requirements set that includes `torch, numpy, rasterio, shap, scikit-learn,
  pillow, web3, eth_account`. `backend/requirements.txt` alone yields `PIPELINE_AVAILABLE=False`.
- Fix model-path resolution: pass **absolute** model paths from config into the pipeline and
  remove the `os.chdir(PROJECT_ROOT)` hack (`ml_service.py:27,89`; `production_inference.py:44-47`)
  — `PROJECT_ROOT` resolves to `/` in the container.
- Fix `DATABASE_PATH` default (currently relative `../database/...`) to an absolute, volume-backed
  path from config.
- **Git-LFS:** deploy runbook runs `git lfs install && git lfs pull`; Phase D verifies the `.pth`
  files are real content (not ~130-byte pointers) inside the image.
- Reality check: torch-CPU + weights → multi-GB image; confirm host build-time/image-size/memory
  limits (free tiers are tight) — may need a paid tier or a slim CPU torch wheel.

### 9.7 Verification additions (§7)
- Grep proves absence of `Math.random` telemetry and hardcoded metric literals in
  `frontend/src`, not just `np.random`/`hash(`/`demo_tokens` in backend.
- Phase D exercises image-upload → **real** inference **in the container** (deferred imports hide
  failures; "boots clean" is not proof the ML path works).
