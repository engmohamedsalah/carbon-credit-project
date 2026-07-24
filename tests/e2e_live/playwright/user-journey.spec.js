// Live browser E2E: real user journey against the deployed frontend.
// register -> login -> dashboard (no fabricated stats) -> create project -> forgot-password -> logout
const { test, expect } = require('@playwright/test');

const WEB = (process.env.E2E_WEB_URL || 'https://frontend-seven-rust-ndw61u0v8l.vercel.app').replace(/\/$/, '');
const API = (process.env.E2E_API_URL || 'https://carbon-credit-backend-nu.vercel.app/api/v1').replace(/\/$/, '');
const ts = Date.now();
const EMAIL = `e2e_ui_${ts}@example.com`;
const PASSWORD = 'e2epassword1';
const PROJECT_NAME = `E2E UI Project ${ts}`;

test.describe.configure({ mode: 'serial' });

test('register -> login -> dashboard -> create project -> forgot password -> logout', async ({ page }) => {
  test.setTimeout(120000);

  // --- Register via UI (role defaults to "Project Developer"; register redirects to /login) ---
  await page.goto(`${WEB}/register`);
  await page.fill('input[name="fullName"]', 'E2E UI User');
  await page.fill('input[name="email"]', EMAIL);
  await page.fill('input[name="password"]', PASSWORD);
  await page.fill('input[name="confirmPassword"]', PASSWORD);
  await page.getByRole('button', { name: /sign up|create account|register/i }).click();

  // --- Login via UI ---
  await page.waitForURL(/\/login/, { timeout: 30000 }).catch(() => {});
  if (!/\/dashboard/.test(page.url())) {
    await page.goto(`${WEB}/login`);
    await page.fill('input[name="email"]', EMAIL);
    await page.fill('input[name="password"]', PASSWORD);
    await page.getByRole('button', { name: /sign in/i }).click();
    await page.waitForURL(/\/dashboard/, { timeout: 30000 });
  }
  expect(page.url()).toMatch(/\/dashboard/);

  // --- Dashboard shows NO fabricated stats ---
  await page.waitForLoadState('networkidle');
  const body = await page.locator('body').innerText();
  for (const fake of ['15,420', '1,847', '15420', '99.1%']) {
    expect(body, `fabricated stat "${fake}" visible on dashboard`).not.toContain(fake);
  }

  // --- Create a project via API, then verify the UI renders it ---
  // (The New Project form requires drawing the area on a Leaflet map, which isn't
  //  practical to automate; project CRUD is covered fully by the API E2E suite.
  //  Here we assert the real UI reads real backend/Turso data.)
  const token = await page.evaluate(() => localStorage.getItem('token'));
  const created = await page.request.post(`${API}/projects`, {
    headers: { Authorization: `Bearer ${token}` },
    data: { name: PROJECT_NAME, location_name: 'Amazon Basin', area_hectares: 100, project_type: 'Reforestation' },
  });
  expect(created.ok(), `project create failed: ${created.status()}`).toBeTruthy();

  await page.goto(`${WEB}/projects`);
  await page.waitForLoadState('networkidle');
  await expect(page.getByText(PROJECT_NAME).first()).toBeVisible({ timeout: 30000 });

  // The New Project form itself renders (fields present).
  await page.goto(`${WEB}/projects/new`);
  await expect(page.locator('input[name="name"]')).toBeVisible({ timeout: 30000 });

  // --- Forgot-password flow via UI ---
  await page.goto(`${WEB}/forgot-password`);
  await page.fill('input[type="email"]', EMAIL);
  await page.getByRole('button', { name: /send reset link/i }).click();
  await expect(page.getByText(/reset link has been sent/i)).toBeVisible({ timeout: 30000 });

  // --- Logout: clear session and confirm protected route redirects to login ---
  await page.evaluate(() => localStorage.removeItem('token'));
  await page.goto(`${WEB}/dashboard`);
  await page.waitForURL(/\/login/, { timeout: 30000 });
  expect(page.url()).toMatch(/\/login/);
});
