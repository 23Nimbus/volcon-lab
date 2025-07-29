import json
import os
from pathlib import Path
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_LOADED = False


def load_env(env_path: str | os.PathLike | None = None) -> None:
    """Load environment variables from a ``.env`` file once.

    Parameters
    ----------
    env_path:
        Optional custom path to the ``.env`` file. When not provided,
        ``<repo root>/.env`` is used.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_file = Path(env_path) if env_path else _REPO_ROOT / ".env"
    load_dotenv(env_file)
    _ENV_LOADED = True


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Return configuration combining environment and JSON defaults."""
    load_env()
    cfg_path = Path(path) if path else _REPO_ROOT / "config.json"
    defaults: dict[str, object] = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                defaults = json.load(fh)
        except Exception:
            defaults = {}
    config = defaults.copy()
    for key, value in os.environ.items():
        if key.isupper():
            config[key] = value
    return config
  
  _loaded = False

def load_env(env_path: str | None = None) -> None:
    """Load environment variables from a .env file once."""
    global _loaded
    if _loaded:
        return
    if env_path is None:
        root_dir = Path(__file__).resolve().parents[1]
        env_path = root_dir / ".env"
    load_dotenv(env_path)
    _loaded = True