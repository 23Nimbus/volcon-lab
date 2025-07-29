from pathlib import Path
from dotenv import load_dotenv

_loaded = False

def load_env(env_path: str | None = None) -> None:
    """Load environment variables from a .env file once."""
    global _loaded
    if _loaded:
        return
    if env_path is None:
        # default to project root .env
        root_dir = Path(__file__).resolve().parents[1]
        env_path = root_dir / ".env"
    load_dotenv(env_path)
    _loaded = True

