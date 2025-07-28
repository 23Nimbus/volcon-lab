
# VolCon Signal Layer

This folder contains the operational VolCon signal layer for detecting volatility container behavior and potential breakout signals.

## Structure

- `vol_container_score.py`: Computes the Vol Container Score based on IV rank, sentiment, OI concentration, etc.
- `sentiment_score.py`: NLP classification for suppression vs breakdown language.
- `reddit_scraper.py`: Pulls Reddit data from key subreddits.
- `ingest_sentiment.py`: Scores Reddit posts using `classify_sentiment()`.
- `cron_reddit_fetcher.py`: Scheduled fetcher for daily Reddit ingestion.
- `backtest_runner.py`: Streamlit dashboard to visualize score timeline.

## Setup

1. Fill in `.env` with your Reddit API credentials (see `.env.template`).
2. Install dependencies: `pip install -r requirements.txt`
3. Run scoring:
   ```
   python reddit_scraper.py
   python ingest_sentiment.py
   python vol_container_score.py
   ```

## Dashboard

Run with:

```bash
streamlit run backtest_runner.py
```

This will visualize the historical Vol Container Scores.
## Running Tests

Install dependencies and execute the test suite with:

```bash
pip install -r requirements.txt
pytest
```

---

© Internal Derivatives Research Unit – Confidential
