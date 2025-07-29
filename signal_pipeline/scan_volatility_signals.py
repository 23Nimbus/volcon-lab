"""Utilities for scanning daily volatility signals."""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import datetime
import os
import logging
import argparse
import sys
from typing import List, Dict, Tuple
from .gex_parser import parse_gex_comment
from .config import load_config
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# --- Config ---
def runtime_config(path: str | None = None) -> Dict:
    """Return merged configuration for this module."""
    cfg = load_config(path)
    defaults = {
        "TICKERS": ["GME", "AMC"],
        "ETF_TICKER": "XRT",
        "REGSHO_LIST_URL": "https://example.com/regsho/latest.txt",
        "SENTIMENT_PATH_TEMPLATE": "reddit/{ticker}/{date}/posts.json",
        "ALERTS_DIR": "alerts",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    if isinstance(cfg.get("TICKERS"), str):
        cfg["TICKERS"] = [t.strip() for t in cfg["TICKERS"].split(",") if t.strip()]
    return cfg

def load_sentiment_score(ticker: str = "GME", date: datetime.date = None, sentiment_path_template: str = "reddit/{ticker}/{date}/posts.json") -> float:
    """Load sentiment score for a ticker from JSON file. Uses keyword and TextBlob scoring if available."""
    if date is None:
        date = datetime.date.today()
    path = sentiment_path_template.format(ticker=ticker, date=date)
    try:
        with open(path) as f:
            data = json.load(f)
        scores = []
        positive_keywords = ["moon", "squeeze", "win", "bull", "rocket"]
        negative_keywords = ["bagholding", "loss", "bear", "down", "fail"]
        for post in data:
            title = post.get('title', '').lower()
            score = 0.5
            if any(word in title for word in positive_keywords):
                score += 0.3
            if any(word in title for word in negative_keywords):
                score -= 0.3
            # Advanced: use TextBlob polarity if available
            if TEXTBLOB_AVAILABLE:
                try:
                    tb_score = TextBlob(title).sentiment.polarity
                    score += tb_score * 0.5
                except Exception:
                    pass
            scores.append(np.clip(score, 0, 1))
        if scores:
            return float(np.mean(scores))
        else:
            return 0.5
    except Exception as e:
        logging.warning(f"Sentiment score fallback for {ticker}: {e}")
        return 0.5  # Neutral fallback

def get_iv_and_rv(ticker: str, retries: int = 3, delay: float = 2.0) -> Tuple[float, float]:
    """Calculate implied volatility proxy and realized volatility for a ticker, with retry logic."""
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period="30d")
            high_low_range = hist['High'] - hist['Low']
            iv_proxy = float(np.mean(high_low_range / hist['Close']))
            returns = hist['Close'].pct_change().dropna()
            realized_vol = float(np.std(returns[-10:]))
            return iv_proxy, realized_vol
        except Exception as e:
            logging.error(f"IV/RV calculation failed for {ticker} (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                import time; time.sleep(delay)
    return 0.0, 0.0

def get_xrt_mock_data() -> Dict:
    """Return mock XRT short interest and shares outstanding."""
    # Simulated data, replace with FINRA + ETF.com or Ortex pipeline
    return {
        "short_interest": 24000000,   # Trigger level
        "shares_outstanding": 2.56e6  # Post-redemption state
    }

def check_regsho_etf(url: str, local_path: str = None, retries: int = 3, delay: float = 2.0) -> bool:
    """Check if ETF is present in RegSHO list from URL or local file, with retry logic."""
    for attempt in range(retries):
        try:
            if local_path and os.path.exists(local_path):
                with open(local_path) as f:
                    lines = f.readlines()
            else:
                response = requests.get(url, timeout=10)
                lines = response.text.splitlines()
            return any(etf in line for line in lines for etf in ["XRT", "GMEU"])
        except Exception as e:
            logging.warning(f"RegSHO check failed (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                import time; time.sleep(delay)
    return False

def scan_volatility_signals(
    tickers: List[str],
    etf_ticker: str,
    regsho_list_url: str,
    sentiment_path_template: str,
    alerts_dir: str,
    output_csv: bool = False,
    date: datetime.date = None
) -> List[Dict]:
    """Scan volatility signals for configured tickers and save results."""
    results = []
    if date is None:
        date = datetime.date.today()

    for ticker in tickers:
        iv, rv = get_iv_and_rv(ticker)
        rv_iv_spread = rv - iv
        iv_rank = float(np.random.uniform(0.1, 0.3))  # Placeholder
        sentiment = load_sentiment_score(ticker, date, sentiment_path_template)

        gex_flags = {"gamma_break_near": False, "fragile_containment": False, "macro_risk_overlay": False}
        try:
            with open(sentiment_path_template.format(ticker=ticker, date=date)) as f:
                posts = json.load(f)
            for post in posts:
                comment = (post.get('title', '') + ' ' + post.get('selftext', '')).lower()
                parsed = parse_gex_comment(comment)
                for k, v in parsed.items():
                    if v:
                        gex_flags[k] = True
        except Exception as e:
            logging.warning(f"GEX comment parse failed for {ticker}: {e}")

        regsho_flag = check_regsho_etf(regsho_list_url)

        signal_flags = {
            "ticker": ticker,
            "date": date.isoformat(),
            "iv_rank": iv_rank,
            "rv_iv_spread": rv_iv_spread,
            "sentiment_score": sentiment,
            "regsho_flag": regsho_flag,
            "iv_break": rv_iv_spread > 0.02,
            "sentiment_disruption": sentiment < 0.4,
            "gamma_break_warning": gex_flags["gamma_break_near"],
            "containment_fragile": gex_flags["fragile_containment"],
            "macro_vol_risk": gex_flags["macro_risk_overlay"],
        }

        results.append(signal_flags)

    # XRT data
    xrt = get_xrt_mock_data()
    si_flag = xrt["short_interest"] > 20_000_000
    redemption_flag = xrt["shares_outstanding"] < 3_000_000

    for r in results:
        r["xrt_si_spike"] = si_flag
        r["xrt_redemptions"] = redemption_flag

    os.makedirs(alerts_dir, exist_ok=True)
    alert_path = os.path.join(alerts_dir, f"{date}_vol_signals.json")
    try:
        with open(alert_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Volatility signals saved to {alert_path}")
    except Exception as e:
        logging.error(f"Failed to save alerts: {e}")

    if output_csv:
        import csv
        csv_path = os.path.join(alerts_dir, f"{date}_vol_signals.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logging.info(f"CSV output saved to {csv_path}")
        except Exception as e:
            logging.error(f"Failed to save CSV: {e}")

    # Print summary stats
    logging.info(f"Summary for {date}:")
    for r in results:
        logging.info(
            f"{r['ticker']}: IV Rank={r['iv_rank']:.2f}, RV-IV Spread={r['rv_iv_spread']:.4f}, "
            f"Sentiment={r['sentiment_score']:.2f}, IV Break={r['iv_break']}, "
            f"Sentiment Disruption={r['sentiment_disruption']}, RegSHO={r['regsho_flag']}, "
            f"XRT SI Spike={r['xrt_si_spike']}, XRT Redemptions={r['xrt_redemptions']}, "
            f"Gamma Break={r['gamma_break_warning']}, Fragile={r['containment_fragile']}, "
            f"Macro Risk={r['macro_vol_risk']}"
        )

    return results

# --- CLI Run ---
def main():
    config = runtime_config()
    parser = argparse.ArgumentParser(description="Scan volatility signals for tickers.")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers", default=','.join(config["TICKERS"]))
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format", default=None)
    parser.add_argument("--output-csv", action="store_true", help="Also output CSV file")
    args = parser.parse_args()

    tickers = args.tickers.split(",")
    date = None
    if args.date:
        try:
            date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
        except Exception:
            logging.error("Invalid date format. Use YYYY-MM-DD.")
            sys.exit(1)

    output = scan_volatility_signals(
        tickers=tickers,
        etf_ticker=config["ETF_TICKER"],
        regsho_list_url=config["REGSHO_LIST_URL"],
        sentiment_path_template=config["SENTIMENT_PATH_TEMPLATE"],
        alerts_dir=config["ALERTS_DIR"],
        output_csv=args.output_csv,
        date=date
    )
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
