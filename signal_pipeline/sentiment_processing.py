from sentiment_score import classify_sentiment
import numpy as np
from typing import List

__all__ = ["simulate_sentiment_score", "fetch_news_sentiment"]

def simulate_sentiment_score() -> float:
    """Return a simple average sentiment score from example texts."""
    texts: List[str] = [
        "I'm renting my shares and selling CSPs all month!",
        "bagholding hard, this is rough",
        "laddering puts while everyone panics",
        "why is this dropping again?",
    ]
    scores = []
    for text in texts:
        polarity, label = classify_sentiment(text)
        modifier = 0.05 if label == 'suppressing' else -0.05 if label == 'breaking' else 0.0
        scores.append(polarity + modifier)
    return float(np.mean(scores))


def fetch_news_sentiment(ticker: str, config: dict) -> float | None:
    """Fetch recent news headlines and compute average sentiment polarity."""
    try:
        import requests
        import datetime as _dt

        headlines = []
        if config.get("FINNHUB_API_KEY"):
            key = config["FINNHUB_API_KEY"]
            to_d = _dt.date.today()
            from_d = to_d - _dt.timedelta(days=7)
            url = (
                f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_d}"
                f"&to={to_d}&token={key}"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            articles = resp.json()
            headlines = [a.get("headline", "") for a in articles]
        elif config.get("NEWSAPI_KEY"):
            key = config["NEWSAPI_KEY"]
            url = (
                f"https://newsapi.org/v2/everything?q={ticker}&pageSize=10"
                f"&sortBy=publishedAt&apiKey={key}"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            headlines = [a.get("title", "") for a in data.get("articles", [])]
        else:
            print("No FINNHUB_API_KEY or NEWSAPI_KEY configured.")
            return None

        if not headlines:
            print(f"No headlines fetched for {ticker}.")
            return None

        sentiments = [classify_sentiment(h)[0] for h in headlines[:10]]
        avg_sentiment = float(np.mean(sentiments)) if sentiments else None
        if avg_sentiment is not None:
            print(f"Avg news sentiment for {ticker}: {avg_sentiment}")
        return avg_sentiment
    except Exception as e:
        print(f"News sentiment fetch error: {e}")
        return None
