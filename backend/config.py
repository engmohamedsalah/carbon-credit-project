"""
Runtime configuration for the Carbon Credit Verification backend.

All environment-specific values are read from environment variables here so nothing
is hardcoded in the application and no secrets live in the repository. Sensible
local-development defaults are provided; production supplies real values via env.
"""
import os
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    # --- Database (absolute path so it works identically locally and in a container) ---
    DATABASE_PATH: str = os.getenv(
        "DATABASE_PATH", str(_REPO_ROOT / "database" / "carbon_credits.db")
    )

    # --- Auth ---
    # Opaque random tokens are used (not JWT); SECRET_KEY is reserved for future use.
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(7 * 24 * 60))  # 7 days
    )

    # --- CORS (comma-separated origins; the deployed frontend URL must be added) ---
    CORS_ORIGINS: list[str] = _split_csv(
        os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )

    # --- First-run admin seed (created only if the users table is empty) ---
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")
    ADMIN_NAME: str = os.getenv("ADMIN_NAME", "Administrator")

    # --- ML model paths (absolute; overridable in the container) ---
    FOREST_MODEL_PATH: str = os.getenv(
        "FOREST_MODEL_PATH",
        str(_REPO_ROOT / "ml" / "models" / "forest_cover_unet_focal_alpha_0.75_threshold_0.53.pth"),
    )
    CHANGE_MODEL_PATH: str = os.getenv(
        "CHANGE_MODEL_PATH",
        str(_REPO_ROOT / "ml" / "models" / "change_detection_siamese_unet.pth"),
    )
    CONVLSTM_MODEL_PATH: str = os.getenv(
        "CONVLSTM_MODEL_PATH",
        str(_REPO_ROOT / "ml" / "models" / "convlstm_fast_final.pth"),
    )

    # --- Blockchain (feature is disabled unless all three are provided) ---
    POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "https://rpc-amoy.polygon.technology")
    BLOCKCHAIN_PRIVATE_KEY: str = os.getenv("BLOCKCHAIN_PRIVATE_KEY", "")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")


settings = Settings()
