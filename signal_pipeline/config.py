import json
import os
from pathlib import Path
from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load configuration combining `.env` and JSON defaults."""
    env_path = _REPO_ROOT / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    cfg_path = Path(path) if path else _REPO_ROOT / 'config.json'
    defaults: dict = {}
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                defaults = json.load(f)
        except Exception:
            defaults = {}
    config = defaults.copy()
    for key, value in os.environ.items():
        if key.isupper():
            config[key] = value
    return config
