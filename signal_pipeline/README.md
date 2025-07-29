
# VolCon Signal Layer

This folder contains the operational VolCon signal layer for detecting volatility container behavior and potential breakout signals.

## Structure

- `vol_container_score.py`: CLI entry point for scoring.
- `data_ingestion.py`: Helpers for IV, RV, volume and OI metrics.
- `sentiment_processing.py`: Sentiment score calculations.
- `scoring.py`: Core scoring functions used by the CLI.
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

### Custom Weights

You can override feature weights directly from the command line.  For example,
to give IV rank more influence and lower the weight of the RV-IV spread:

```bash
python vol_container_score.py --ticker GME \
    --weight_iv_rank 0.4 --weight_rv_iv_spread 0.1
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
