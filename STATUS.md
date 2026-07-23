# Project Status — Honest Prototype

This is an **honest, working prototype** of a carbon-credit verification system, not a
production or commercial product. This document describes exactly what is real, what is
disabled, and what is not yet real. It supersedes the older `*_COMPLETION_REPORT.md` and
status documents (which claimed "100% complete" / "production ready" and have been removed).

## Real model metrics

The trained PyTorch models are real and run genuine inference. Their measured performance —
recorded by the code itself (`ml/evaluation_results/*.csv`, `ml/inference/production_inference.py`):

| Model | Task | F1 |
|---|---|---|
| Forest Cover U-Net | Forest / non-forest segmentation | **≈ 0.49** (0.4911) |
| Change Detection Siamese U-Net | Before/after change segmentation | **≈ 0.60** (0.6006) |

These are prototype-grade results. There is no "99.1% accuracy" — that figure was fabricated
and has been removed everywhere.

## What is REAL

- **Authentication** — email/password login with hashed passwords and **opaque bearer tokens**
  (server-side token store, not JWT). Tokens expire (configurable TTL). Registration is
  role-locked server-side; privilege elevation only via an admin-gated path.
- **Role-based access control** — centralized roles enforced on protected routes/endpoints.
- **Projects CRUD** — create, read, update, list carbon-credit projects, persisted in SQLite.
- **Verification workflow** — human-in-the-loop verification records. Records are created as
  `pending` / `human_verified=false`. AI confidence and carbon impact are only stored when a
  real value exists (from a real image analysis or a user-supplied estimate) — never fabricated.
- **Real ML inference on uploaded imagery** — upload a Sentinel-2 GeoTIFF stack → the real
  Forest Cover U-Net and Change Detection Siamese U-Net run genuine inference → a real carbon
  estimate is computed (`forest_area_ha × 150 tC/ha × 3.67 CO₂/C`, IPCC tropical above-ground
  biomass figure).
- **Real Solidity contract** — the carbon-credit certification smart contract and its Web3
  mint/verify code are real (see `blockchain/`). See "disabled" below for its runtime state.
- **IoT backend** — real, DB-backed sensor-reading endpoints and schema.

## What is DISABLED or NOT YET REAL

- **Explainability / XAI** — **disabled in this build.** The previous XAI surface ran
  SHAP / Integrated Gradients on `np.random` noise behind a "REAL PROCESSING" banner and
  reported fictional feature names; all of that fabrication is removed. The genuine
  Integrated-Gradients code remains in the repo but is **not wired** to real inputs (no
  per-project image tensor is retained at request time). Real attribution over the model's
  12 Sentinel-2 channels is future work. The UI shows an honest "explainability not available
  in this build" state.
- **Coordinate-based analysis** — **removed.** There is no pipeline that fetches imagery from a
  project's coordinates/geometry, so the old "coordinates → AI result" endpoint (which used a
  mock formula) has been deleted. The only real analysis path is **upload imagery → inference**.
- **Blockchain certification** — **disabled until configured.** The contract is not deployed by
  default. Mint/verify endpoints return `503 "blockchain certification not configured"` and the
  certify button is hidden unless a contract address, ABI, and signing key are supplied via
  environment variables. No mock transaction IDs are ever returned.
- **Analytics trends** — simulated accuracy trends and hardcoded ML metrics were removed;
  dashboards show real recorded metrics or an honest empty state.
- **IoT frontend telemetry** — `Math.random`-generated sample readings were removed; charts are
  driven by real `/iot/readings` or an empty state.

## Scope notes

- Database is SQLite (single-instance pilot). A fresh DB is created on first run; the seeded
  admin comes from `ADMIN_EMAIL` / `ADMIN_PASSWORD` env vars — no credentials are committed.
- This is a single-instance prototype intended for demonstration and further development, not
  commercial carbon-credit issuance.
