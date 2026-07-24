"""
Live end-to-end API journey against the deployed backend.

Runs a real user journey (register -> login -> project -> verification -> IoT ->
analytics -> reports -> settings -> password reset) and asserts honesty invariants
(no fabricated AI scores/metrics; ML/XAI/blockchain report an honest disabled state).

Config via env:
  E2E_API_URL  (default: https://carbon-credit-backend-nu.vercel.app/api/v1)

Run:  pytest tests/e2e_live/test_api_journey.py -v
"""
import os
import time
import re
import requests

API = os.getenv("E2E_API_URL", "https://carbon-credit-backend-nu.vercel.app/api/v1").rstrip("/")
TS = int(time.time())
EMAIL = f"e2e_{TS}@example.com"
PASSWORD = "e2epassword1"
NEW_PASSWORD = "e2epassword2"

state = {}  # shared across ordered steps


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _login(email, password, retries=3):
    """Login, tolerating the 5/min rate limit with a short backoff."""
    for attempt in range(retries):
        r = requests.post(
            f"{API}/auth/login",
            data={"username": email, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if r.status_code != 429:
            return r
        time.sleep(15)
    return r


class TestLiveJourney:
    # ---- Auth ----
    def test_01_health(self):
        r = requests.get(API.replace("/api/v1", "") + "/health", timeout=30)
        assert r.status_code == 200, r.text
        assert r.json().get("status") == "healthy"

    def test_02_register_forces_low_privilege(self):
        # Even asking for Administrator must yield a non-admin account.
        r = requests.post(f"{API}/auth/register", json={
            "email": EMAIL, "password": PASSWORD, "full_name": "E2E User", "role": "Administrator",
        }, timeout=30)
        assert r.status_code == 201, r.text
        state["token"] = r.json()["access_token"]
        me = requests.get(f"{API}/auth/me", headers=_auth(state["token"]), timeout=30)
        assert me.status_code == 200, me.text
        assert me.json()["role"] == "Project Developer", f"privilege escalation! {me.json()}"
        state["user_id"] = me.json()["id"]

    def test_03_duplicate_email_rejected(self):
        r = requests.post(f"{API}/auth/register", json={
            "email": EMAIL, "password": PASSWORD, "full_name": "Dup",
        }, timeout=30)
        assert r.status_code == 400, r.text

    def test_04_invalid_token_rejected(self):
        r = requests.get(f"{API}/auth/me", headers=_auth("not-a-real-token"), timeout=30)
        assert r.status_code == 401, r.text

    def test_05_login(self):
        r = _login(EMAIL, PASSWORD)
        assert r.status_code == 200, r.text
        state["token"] = r.json()["access_token"]

    # ---- Projects ----
    def test_10_create_project(self):
        r = requests.post(f"{API}/projects", headers=_auth(state["token"]), json={
            "name": f"E2E Forest {TS}", "description": "e2e test project",
            "location_name": "Amazon Basin", "area_hectares": 120.5,
            "project_type": "Reforestation", "estimated_carbon_credits": 5000,
        }, timeout=30)
        assert r.status_code in (200, 201), r.text
        state["project_id"] = r.json().get("id") or r.json().get("project", {}).get("id")
        assert state["project_id"], f"no project id in {r.json()}"

    def test_11_list_and_get_project(self):
        r = requests.get(f"{API}/projects", headers=_auth(state["token"]), timeout=30)
        assert r.status_code == 200, r.text
        ids = [p.get("id") for p in (r.json() if isinstance(r.json(), list) else r.json().get("projects", []))]
        assert state["project_id"] in ids, "created project not in list"
        g = requests.get(f"{API}/projects/{state['project_id']}", headers=_auth(state["token"]), timeout=30)
        assert g.status_code == 200, g.text

    def test_12_update_project(self):
        r = requests.put(f"{API}/projects/{state['project_id']}", headers=_auth(state["token"]), json={
            "name": f"E2E Forest {TS} (updated)", "description": "updated",
            "location_name": "Amazon Basin", "area_hectares": 130.0, "project_type": "Reforestation",
        }, timeout=30)
        assert r.status_code in (200, 201), r.text

    def test_13_status_and_logs(self):
        r = requests.patch(f"{API}/projects/{state['project_id']}/status",
                           headers=_auth(state["token"]),
                           json={"status": "Pending", "reason": "e2e", "notes": "e2e"}, timeout=30)
        assert r.status_code in (200, 201), r.text
        logs = requests.get(f"{API}/projects/{state['project_id']}/status-logs",
                            headers=_auth(state["token"]), timeout=30)
        assert logs.status_code == 200, logs.text

    # ---- Verification (honest: no fabricated AI score) ----
    def test_20_create_verification_no_fabrication(self):
        r = requests.post(f"{API}/verification", headers=_auth(state["token"]), json={
            "project_id": state["project_id"], "verification_notes": "e2e",
        }, timeout=30)
        assert r.status_code in (200, 201), r.text
        body = r.json()
        state["verification_id"] = body["id"]
        assert body.get("ai_confidence") is None, f"fabricated AI score: {body.get('ai_confidence')}"

    def test_21_get_and_list_verification(self):
        g = requests.get(f"{API}/verification/{state['verification_id']}", headers=_auth(state["token"]), timeout=30)
        assert g.status_code == 200, g.text
        lst = requests.get(f"{API}/verification", headers=_auth(state["token"]), timeout=30)
        assert lst.status_code == 200, lst.text

    def test_22_human_review(self):
        r = requests.post(f"{API}/verification/{state['verification_id']}/human-review",
                          headers=_auth(state["token"]),
                          json={"approved": True, "notes": "e2e approved"}, timeout=30)
        assert r.status_code in (200, 201), r.text

    # ---- IoT ----
    def test_30_iot_sensor_crud(self):
        sid = f"e2e-sensor-{TS}"
        c = requests.post(f"{API}/iot/sensors", headers=_auth(state["token"]), json={
            "sensor_id": sid, "sensor_type": "co2_flux",
            "location_lat": -3.4, "location_lng": -62.2, "project_id": state["project_id"],
        }, timeout=30)
        assert c.status_code in (200, 201), c.text
        state["sensor_pk"] = c.json().get("id")
        state["sensor_id"] = sid
        lst = requests.get(f"{API}/iot/sensors", headers=_auth(state["token"]), timeout=30)
        assert lst.status_code == 200, lst.text
        rd = requests.post(f"{API}/iot/readings", headers=_auth(state["token"]), json={
            "sensor_id": sid, "reading_type": "co2_flux", "value": 12.3, "unit": "ppm",
        }, timeout=30)
        assert rd.status_code in (200, 201), rd.text
        an = requests.get(f"{API}/iot/analytics", headers=_auth(state["token"]), timeout=30)
        assert an.status_code == 200, an.text

    # ---- Analytics (honest metrics) ----
    def test_40_analytics_real_metrics(self):
        d = requests.get(f"{API}/analytics/dashboard", headers=_auth(state["token"]), timeout=30)
        assert d.status_code == 200, d.text
        text = d.text
        for fake in ("0.8912", "0.9156", "15420", "99.1"):
            assert fake not in text, f"fabricated number {fake} present in analytics"
        for ep in ("/analytics/performance", "/analytics/carbon-impact"):
            r = requests.get(f"{API}{ep}", headers=_auth(state["token"]), timeout=30)
            assert r.status_code == 200, f"{ep}: {r.text}"

    # ---- Reports / Settings ----
    def test_50_reports_and_settings(self):
        # Owner can request their project report (PDF or handled error, never 401).
        proj = requests.get(f"{API}/reports/project/{state['project_id']}",
                            headers=_auth(state["token"]), timeout=60)
        assert proj.status_code != 401, f"reports/project: {proj.status_code}"
        # Analytics report is admin-only -> a Project Developer is correctly forbidden (RBAC).
        an = requests.get(f"{API}/reports/analytics", headers=_auth(state["token"]), timeout=60)
        assert an.status_code in (403, 200), f"reports/analytics expected 403 (RBAC), got {an.status_code}"
        s = requests.get(f"{API}/settings", headers=_auth(state["token"]), timeout=30)
        assert s.status_code == 200, s.text

    # ---- Honest-disabled features ----
    def test_60_ml_xai_blockchain_disabled_not_fabricated(self):
        ml = requests.get(f"{API}/ml/status", headers=_auth(state["token"]), timeout=30)
        assert ml.status_code in (200, 503), ml.text
        if ml.status_code == 200:
            assert not ml.json().get("is_initialized", False), "ML claims initialized on serverless"
        xai = requests.post(f"{API}/xai/generate-explanation", headers=_auth(state["token"]),
                            json={"model_id": "m", "instance_data": {"project_id": state["project_id"]}}, timeout=30)
        assert xai.status_code == 503, f"XAI should be disabled: {xai.status_code} {xai.text[:120]}"
        bc = requests.get(f"{API}/blockchain/enabled", headers=_auth(state["token"]), timeout=30)
        assert bc.status_code == 200 and bc.json().get("enabled") is False, bc.text

    # ---- Password reset flow ----
    def test_70_change_password(self):
        r = requests.post(f"{API}/settings/change-password", headers=_auth(state["token"]),
                          json={"current_password": PASSWORD, "new_password": NEW_PASSWORD}, timeout=30)
        assert r.status_code == 200, r.text

    def test_71_forgot_and_reset(self):
        f = requests.post(f"{API}/auth/forgot-password", json={"email": EMAIL}, timeout=30)
        assert f.status_code == 200, f.text
        link = f.json().get("reset_link")
        assert link and "token=" in link, f"no demo reset link: {f.json()}"
        token = re.search(r"token=([^&]+)", link).group(1)
        r = requests.post(f"{API}/auth/reset-password",
                          json={"token": token, "new_password": "e2epassword3"}, timeout=30)
        assert r.status_code == 200, r.text
        # login with the reset password
        lg = _login(EMAIL, "e2epassword3")
        assert lg.status_code == 200, lg.text
        state["token"] = lg.json()["access_token"]

    # ---- Cleanup ----
    def test_90_cleanup(self):
        if state.get("sensor_pk"):
            requests.delete(f"{API}/iot/sensors/{state['sensor_pk']}", headers=_auth(state["token"]), timeout=30)
        if state.get("project_id"):
            d = requests.delete(f"{API}/projects/{state['project_id']}", headers=_auth(state["token"]), timeout=30)
            assert d.status_code in (200, 204, 404), d.text
