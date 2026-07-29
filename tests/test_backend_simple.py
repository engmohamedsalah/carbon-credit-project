"""Minimal backend smoke test for CI.

Verifies the FastAPI app imports and boots (catching syntax/import/DB-init errors),
that the core routes exist, and that /health responds. Uses a throwaway SQLite DB
(TURSO_URL unset -> sqlite3), so it needs no external services.

Runs both as `python tests/test_backend_simple.py` and under pytest.
"""
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend"))
# Point the DB at a writable temp file BEFORE importing the app (config reads this at import).
os.environ["DATABASE_PATH"] = os.path.join(tempfile.gettempdir(), "ci_carbon_test.db")
os.environ.setdefault("SECRET_KEY", "ci-test-secret-key")

from main import app  # noqa: E402

EXPECTED_ROUTES = {
    "/health",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/projects",
    "/api/v1/projects/{project_id}/carbon-analysis",
    "/api/v1/reports/project/{project_id}",
}


def test_app_imported():
    assert app is not None


def test_core_routes_exist():
    paths = {getattr(r, "path", None) for r in app.routes}
    missing = EXPECTED_ROUTES - paths
    assert not missing, f"missing routes: {missing}"


def test_health_endpoint():
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json().get("status") == "healthy"


if __name__ == "__main__":
    test_app_imported()
    test_core_routes_exist()
    test_health_endpoint()
    print("backend smoke tests passed")
