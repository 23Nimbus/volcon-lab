# VolCon-Lab

VolCon-Lab is a collection of Python tools for detecting "volatility containers" in cult equities. It pulls Reddit discussion, computes sentiment and option metrics and produces a daily Vol Container Score used to gauge breakout risk. The project also includes a Streamlit dashboard for exploring score history.

## Installation

Install Python requirements:

```bash
pip install -r requirements.txt
```

## Environment

Copy the provided template and fill in your API keys:

```bash
cp signal_pipeline/.env.template .env
# edit .env and supply credentials
```

The resulting `.env` file must live in the project root (same directory as this
README) so all scripts can load your environment variables automatically.

Keys are required for Reddit (and optional market data providers).

### Unified Configuration

Runtime settings are loaded via `signal_pipeline.config.load_config()` which
combines variables defined in `.env` with defaults from `config.json` located at
the repository root. Values in the environment always take precedence.

## Quick Start

Run the scoring scripts sequentially:

```bash
python signal_pipeline/reddit_scraper.py
python signal_pipeline/ingest_sentiment.py
python signal_pipeline/vol_container_score.py
```

Visualize historical scores with Streamlit:

```bash
streamlit run signal_pipeline/backtest_runner.py
```

This launches an interactive dashboard to review the Vol Container Score timeline.

## Running Tests

To run the unit tests, first install the dependencies and then execute `pytest`:

```bash
pip install -r requirements.txt
pytest
```
