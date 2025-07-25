import json
import os
import argparse
import logging
from datetime import datetime
from sentiment_score import classify_sentiment
import statistics
from typing import Optional, List, Dict, Any

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.FileHandler("ingest_sentiment.log"), logging.StreamHandler()]
)

def safe_load_json(path: str, retries: int = 3, delay: float = 2.0) -> Optional[List[Dict[str, Any]]]:
    """Safely load JSON file with retry logic."""
    import time
    for attempt in range(retries):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading JSON (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return None

def validate_post(post: Dict[str, Any]) -> bool:
    """Validate that a post has required fields."""
    return 'title' in post and 'selftext' in post

def ingest_sentiment_from_json(
    json_path: str,
    filter_ticker: Optional[str] = None,
    filter_author: Optional[str] = None,
    output_path: Optional[str] = None
) -> float:
    """
    Ingest Reddit sentiment from a JSON file, filter by ticker/author, and output stats/results.
    Returns average sentiment score.
    """
    if not os.path.exists(json_path):
        logging.warning(f"No Reddit dump found at {json_path}")
        return 0.0
    posts = safe_load_json(json_path)
    if posts is None:
        logging.error(f"Failed to load posts from {json_path}")
        return 0.0
    if not isinstance(posts, list):
        logging.error(f"JSON root is not a list: {type(posts)}")
        return 0.0

    scores = []
    meta = []
    for p in posts:
        if not validate_post(p):
            logging.warning(f"Skipping invalid post: {p}")
            continue
        if filter_ticker and p.get('ticker') != filter_ticker:
            continue
        if filter_author and p.get('author') != filter_author:
            continue
        text = p.get('title', '') + ' ' + p.get('selftext', '')
        try:
            polarity, label = classify_sentiment(text)
            modifier = 0.05 if label == 'suppressing' else -0.05 if label == 'breaking' else 0
            score = polarity + modifier
            scores.append(score)
            meta.append({
                'id': p.get('id'),
                'ticker': p.get('ticker'),
                'author': p.get('author'),
                'score': score,
                'label': label,
                'title': p.get('title'),
                'timestamp': p.get('timestamp')
            })
        except Exception as e:
            logging.error(f"Error scoring post {p.get('id')}: {e}")
            continue

    if not scores:
        logging.info("No posts matched filter or no scores computed.")
        return 0.0

    avg = statistics.mean(scores)
    min_score = min(scores)
    max_score = max(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
    logging.info(f"Sentiment stats: avg={avg:.3f}, min={min_score:.3f}, max={max_score:.3f}, std={std_score:.3f}, n={len(scores)}")

    # Optionally write results to file
    if output_path:
        try:
            with open(output_path, 'w') as f:
                json.dump(meta, f, indent=2)
            logging.info(f"Wrote detailed sentiment results to {output_path}")
        except Exception as e:
            logging.error(f"Error writing output file: {e}")

    return avg

def main():
    """CLI entry point for ingesting Reddit sentiment."""
    parser = argparse.ArgumentParser(description="Ingest Reddit sentiment from JSON.")
    parser.add_argument('--date', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='Date (YYYY-MM-DD)')
    parser.add_argument('--input', type=str, help='Path to Reddit posts JSON')
    parser.add_argument('--output', type=str, help='Path to output sentiment results JSON')
    parser.add_argument('--ticker', type=str, help='Filter by ticker')
    parser.add_argument('--author', type=str, help='Filter by author')
    args = parser.parse_args()

    date = args.date
    json_path = args.input or f"reddit/{date}/posts.json"
    score = ingest_sentiment_from_json(json_path, filter_ticker=args.ticker, filter_author=args.author, output_path=args.output)
    print(f"📊 Ingested Reddit sentiment score: {score:.3f}")
    if score == 0.0:
        print("No valid sentiment scores computed. Check logs for details.")

if __name__ == "__main__":
    main()
