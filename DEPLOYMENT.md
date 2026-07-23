# Deployment Runbook

Topology: **frontend → Vercel** (static SPA), **backend → a container host** (Render/Railway/Fly).
The backend runs PyTorch + ~100 MB of model weights and writes SQLite, so it needs a real
container with a persistent volume — it cannot run on Vercel serverless.

All deploy steps below require **your** interactive login (Vercel / host account); they can't be
done from an automated session.

## 0. Prerequisites (once)
- The model weights are git-LFS. On any build host, ensure real files, not pointers:
  ```
  git lfs install && git lfs pull
  ```
- Never commit `backend/.env`, `*.db`, or `*token*.txt` (already gitignored).

## 1. Backend → Render (Docker + persistent disk)
Files provided: `Dockerfile` (repo root), `render.yaml`, `backend/requirements-ml.txt`.

1. Push the repo to GitHub (see the history-purge note in `docs/superpowers/specs/`).
2. Render → **New → Blueprint** → pick this repo (`render.yaml` is detected).
3. Set the env vars marked `sync: false`:
   - `CORS_ORIGINS` = your Vercel URL, e.g. `https://carbon-credit.vercel.app`
   - `ADMIN_EMAIL`, `ADMIN_PASSWORD` = the first admin (created once, on an empty DB)
   - `SECRET_KEY` is auto-generated; `DATABASE_PATH=/data/carbon_credits.db` is preset.
4. Deploy. A 1 GB disk mounts at `/data`; the DB is created fresh and the admin is seeded.
5. Health check: `GET https://<backend>/health` → `{"status":"healthy"}`.

Notes:
- The image is multi-GB (torch + weights); the free tier is too small — use the **Starter** plan
  (persistent disk + enough RAM). Same idea on Railway (add a Volume at `/data`) or Fly.
- Blockchain certification stays **disabled** until you set `CONTRACT_ADDRESS`,
  `BLOCKCHAIN_PRIVATE_KEY`, and `POLYGON_RPC_URL` (Amoy) — then it activates with no code change.

## 2. Frontend → Vercel
Files provided: `frontend/vercel.json`.

1. Vercel → **Add New Project** → import this repo → set **Root Directory = `frontend`**.
2. Env var: `REACT_APP_API_URL = https://<your-backend>/api/v1`.
3. Deploy. (`vercel.json` builds CRA with `CI=false` so lint warnings don't fail the build.)
4. After the backend is up, make sure its `CORS_ORIGINS` includes the final Vercel URL.

CLI alternative (needs `vercel login` — interactive):
```
cd frontend
npx vercel            # preview
npx vercel --prod     # production
```

## 3. Post-deploy smoke check
- Register a user → confirm role is "Project Developer" (not admin).
- Log in as the seeded admin.
- Create a project; create a verification (AI score is intentionally null — honest HITL).
- XAI page shows the honest "not available in this build" state.
- Blockchain UI is hidden/disabled (not configured).
