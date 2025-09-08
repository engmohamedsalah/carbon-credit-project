import re
import os
from pathlib import Path
import pytest
from playwright.async_api import expect

pytestmark = pytest.mark.auth

CANDIDATE_FILES = [
    Path(__file__).resolve().parents[2] / 'USER_ACCOUNTS.md',
    Path(__file__).resolve().parents[2] / 'USERS.md',
]


def parse_credentials_from_user_accounts_md(text: str):
    # Prefer explicit admin/test accounts
    # Look for table entries or quick access section
    # Default universal password if declared
    universal_pw = None
    m = re.search(r"All users use password:\s*`([^`]+)`", text, re.IGNORECASE)
    if m:
        universal_pw = m.group(1).strip()

    # Try primary test admin
    for email in [
        'testadmin@example.com',
        'verifier@example.com',
        'scientist@example.com',
        'investor@example.com',
        'alice@example.com',
    ]:
        if email in text and universal_pw:
            return email, universal_pw

    # Fallback: try to capture any backticked email in table
    emails = re.findall(r"`([\w\.-]+@[\w\.-]+)`", text)
    if emails and universal_pw:
        return emails[0], universal_pw

    return None


def parse_credentials_from_users_md(text: str):
    # Prefer Primary Administrator block
    block = re.search(r"Primary Administrator[\s\S]*?(?:Secondary|Notes|$)", text, re.IGNORECASE)
    if block:
        btxt = block.group(0)
        em = re.search(r"Email\s*:\s*([^\s\n]+)", btxt)
        pw = re.search(r"Password\s*:\s*([^\s\n]+)", btxt)
        if em and pw:
            return em.group(1).strip(), pw.group(1).strip()

    # Fallback: table entry with password explicitly present
    m = re.search(r"\|\s*\*\*3\*\*\s*\|[\s\S]*?\|\s*\*\*admin@admin\.com\*\*\s*\|[\s\S]*?\|\s*\*\*Administrator\*\*[\s\S]*?\|\s*\*\*admin123\*\*\s*\|", text)
    if m:
        return 'admin@admin.com', 'admin123'

    # Another explicitly documented admin
    m2 = re.search(r"blockchaintest@example\.com[\s\S]*?password123", text)
    if m2:
        return 'blockchaintest@example.com', 'password123'

    return None


def load_known_credentials():
    for path in CANDIDATE_FILES:
        try:
            if path.exists():
                txt = path.read_text(encoding='utf-8', errors='ignore')
                if path.name == 'USER_ACCOUNTS.md':
                    creds = parse_credentials_from_user_accounts_md(txt)
                else:
                    creds = parse_credentials_from_users_md(txt)
                if creds:
                    return creds
        except Exception:
            continue
    raise RuntimeError('Could not find credentials in USER_ACCOUNTS.md or USERS.md')


@pytest.mark.asyncio
async def test_login_with_known_user(page, servers):
    email, password = load_known_credentials()

    # Go to login page
    await page.goto('http://localhost:3000/login')

    # Fill and submit
    await page.fill('input[name="email"]', email)
    await page.fill('input[name="password"]', password)
    await page.click('button[type="submit"]')

    # Should redirect to dashboard
    await page.wait_for_url('**/dashboard')
    await expect(page.locator('main').first).to_be_visible()

    # Save an explicit screenshot artifact
    results_dir = Path(__file__).parent / 'test-results'
    results_dir.mkdir(parents=True, exist_ok=True)
    shot_path = results_dir / 'login-with-known-user.png'
    await page.screenshot(path=str(shot_path), full_page=True)
