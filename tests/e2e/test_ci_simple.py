"""Simple CI E2E smoke test.

The CI e2e job runs `npm run build` before this. We verify the production build was
produced and looks like a real CRA bundle. Kept server/browser-free so it's reliable
in CI (no app server is started in the job); the full browser E2E suite lives in
tests/e2e_live and runs against the deployed app.

Runs both as `python tests/e2e/test_ci_simple.py` and under pytest.
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILD = os.path.join(_ROOT, "frontend", "build")


def test_frontend_build_exists():
    index = os.path.join(_BUILD, "index.html")
    assert os.path.isfile(index), f"frontend build missing: {index} (run `npm run build`)"
    html = open(index, encoding="utf-8").read()
    assert 'id="root"' in html, "index.html has no #root mount point"
    assert "/static/js/" in html, "index.html references no JS bundle"


def test_js_bundle_present():
    js_dir = os.path.join(_BUILD, "static", "js")
    assert os.path.isdir(js_dir), f"missing {js_dir}"
    assert any(f.endswith(".js") for f in os.listdir(js_dir)), "no JS bundle emitted"


if __name__ == "__main__":
    test_frontend_build_exists()
    test_js_bundle_present()
    print("CI e2e smoke test passed")
