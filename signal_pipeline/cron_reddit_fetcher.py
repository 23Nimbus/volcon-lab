import schedule
import time
import subprocess
import os
from .config import load_config
import sys
import logging
from datetime import datetime

# --- Configurable schedule time ---
CONFIG = load_config()
SCHEDULE_TIME = CONFIG.get("REDDIT_FETCH_TIME", "08:30")
MAX_RETRIES = int(CONFIG.get("REDDIT_FETCH_MAX_RETRIES", 3))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.FileHandler("cron_reddit_fetcher.log"), logging.StreamHandler()]
)

def notify(msg, level="info"):
    # Placeholder for notification logic (e.g., email, Slack)
    getattr(logging, level)(msg)


def run_scraper():
    for attempt in range(1, MAX_RETRIES + 1):
        logging.info(f"Attempt {attempt}: Running reddit_scraper.py...")
        try:
            result = subprocess.run([sys.executable, "reddit_scraper.py"], capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("reddit_scraper.py completed successfully.")
                notify("Reddit scraper ran successfully.", level="info")
                break
            else:
                logging.error(f"reddit_scraper.py failed (return code {result.returncode}): {result.stderr}")
                notify(f"Reddit scraper failed: {result.stderr}", level="error")
        except Exception as e:
            logging.error(f"Exception running reddit_scraper.py: {e}")
            notify(f"Exception running reddit_scraper.py: {e}", level="error")
        if attempt < MAX_RETRIES:
            time.sleep(30)  # Wait before retry
        else:
            logging.error("Max retries reached. Giving up.")
            notify("Max retries reached for reddit_scraper.py.", level="error")

# Schedule daily at configurable time
schedule.every().day.at(SCHEDULE_TIME).do(run_scraper)

if __name__ == "__main__":
    logging.info(f"Starting Reddit fetcher daemon. Scheduled for {SCHEDULE_TIME} UTC.")
    while True:
        schedule.run_pending()
        time.sleep(60)
