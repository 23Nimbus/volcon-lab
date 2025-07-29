import logging
import os

def setup_logging(log_path: str) -> None:
    """Configure basic logging to the given file path and stdout."""
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
