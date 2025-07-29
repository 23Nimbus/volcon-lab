import os
import json
import praw
from datetime import datetime
 codex/implement-unified-configuration-loader
=======
from .config import load_env
 main
import logging
import argparse
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import yaml
from .config import load_config
try:
    from sentiment_score import classify_sentiment
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.FileHandler("reddit_scraper.log"), logging.StreamHandler()]
)

codex/implement-unified-configuration-loader
# Load configuration (env values override defaults)
CONFIG = load_config()
=======
# Load environment variables
load_env()
main

DEFAULT_SUBREDDITS = ['wallstreetbets', 'GME', 'Superstonk']
DEFAULT_LIMIT = 100
TICKER_LIST = ['GME', 'AMC', 'TSLA', 'AAPL', 'MSFT', 'NVDA', 'XRT', 'SPY', 'QQQ']  # Example tickers

REDDIT_CLIENT_ID = CONFIG.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = CONFIG.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = CONFIG.get("REDDIT_USER_AGENT")


def validate_credentials(cfg: dict = CONFIG) -> bool:
    """Return True if Reddit credentials are present in cfg."""
    missing = []
    if not cfg.get("REDDIT_CLIENT_ID"):
        missing.append("REDDIT_CLIENT_ID")
    if not cfg.get("REDDIT_CLIENT_SECRET"):
        missing.append("REDDIT_CLIENT_SECRET")
    if not cfg.get("REDDIT_USER_AGENT"):
        missing.append("REDDIT_USER_AGENT")
    if missing:
        logging.error(f"Missing Reddit API credentials: {', '.join(missing)}")
        return False
    return True

def filter_post(post, min_score=0, exclude_stickied=True, exclude_removed=True, keywords=None, tickers=None):
    if exclude_stickied and getattr(post, 'stickied', False):
        return False
    if exclude_removed and (getattr(post, 'removed_by_category', None) is not None):
        return False
    if min_score and getattr(post, 'score', 0) < min_score:
        return False
    text = (post.title or '') + ' ' + (post.selftext or '')
    if keywords and not any(kw.lower() in text.lower() for kw in keywords):
        return False
    if tickers and not any(tkr.lower() in text.lower() for tkr in tickers):
        return False
    return True

def detect_ticker(text: str, tickers: List[str]) -> Optional[str]:
    for tkr in tickers:
        if tkr.lower() in text.lower():
            return tkr
    return None

def enrich_post(data, tickers):
    text = data['title'] + ' ' + data['selftext']
    data['ticker'] = detect_ticker(text, tickers)
    if SENTIMENT_AVAILABLE:
        try:
            polarity, label = classify_sentiment(text)
            data['sentiment_score'] = polarity
            data['sentiment_label'] = label
        except Exception as e:
            data['sentiment_score'] = None
            data['sentiment_label'] = None
    return data

def log_command(args, cfg: dict = CONFIG):
    """Log CLI arguments and active credentials."""
    logging.info(f"Command args: {args}")
    logging.info(
        "Environment: CLIENT_ID=%s, CLIENT_SECRET=%s, USER_AGENT=%s",
        cfg.get("REDDIT_CLIENT_ID"),
        "***" if cfg.get("REDDIT_CLIENT_SECRET") else None,
        cfg.get("REDDIT_USER_AGENT"),
    )

def load_scraper_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    return None

def fetch_posts(
    subreddits: List[str],
    limit: int,
    save_dir: str,
    retries: int = 3,
    delay: float = 5.0,
    min_score: int = 0,
    exclude_stickied: bool = True,
    exclude_removed: bool = True,
    keywords: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
    config: dict = CONFIG,
) -> int:
    """
    Fetch posts from given subreddits, save to JSON, and return number of posts saved.
    """
    if not validate_credentials(config):
        return 0
    try:
        reddit = praw.Reddit(
            client_id=config.get("REDDIT_CLIENT_ID"),
            client_secret=config.get("REDDIT_CLIENT_SECRET"),
            user_agent=config.get("REDDIT_USER_AGENT"),
        )
    except Exception as e:
        logging.error(f"Error initializing PRAW: {e}")
        return 0

    all_posts: Dict[str, Dict[str, Any]] = {}
    summary = {}
    for sub in subreddits:
        posts_this_sub = 0
        for attempt in range(retries):
            try:
                logging.info(f"Fetching from r/{sub} (limit={limit})...")
                posts = list(reddit.subreddit(sub).hot(limit=limit))
                for post in tqdm(posts, desc=f"r/{sub}"):
                    if not filter_post(post, min_score, exclude_stickied, exclude_removed, keywords, tickers):
                        continue
                    data = {
                        'id': post.id,
                        'subreddit': sub,
                        'title': post.title,
                        'selftext': post.selftext,
                        'created_utc': post.created_utc,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'author': str(post.author) if post.author else None,
                        'url': post.url,
                        'permalink': post.permalink,
                        'timestamp': datetime.utcfromtimestamp(post.created_utc).isoformat() + 'Z',
                        'ticker': None
                    }
                    data = enrich_post(data, tickers or TICKER_LIST)
                    all_posts[data['id']] = data  # Deduplication by post id
                    posts_this_sub += 1
                break  # Success, break retry loop
            except Exception as e:
                logging.error(f"Error fetching from r/{sub} (attempt {attempt+1}): {e}")
                time.sleep(delay)
                if attempt == retries - 1:
                    logging.error(f"Max retries reached for r/{sub}. Skipping.")
        summary[sub] = posts_this_sub

    posts_list = list(all_posts.values())
    # Post-fetch validation
    valid_posts = [p for p in posts_list if p.get('title') and p.get('selftext')]
    if not valid_posts:
        logging.warning("No valid posts fetched.")
        return 0

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'posts.json')
    try:
        with open(out_path, 'w') as f:
            json.dump(valid_posts, f, indent=2)
        logging.info(f"Saved {len(valid_posts)} posts to {out_path}")
    except Exception as e:
        logging.error(f"Error saving posts to {out_path}: {e}")
        return 0

    # Summary report
    top_posts = sorted(valid_posts, key=lambda x: x['score'], reverse=True)[:5]
    logging.info(f"Summary: Total={len(valid_posts)}, Per subreddit={summary}")
    for i, tp in enumerate(top_posts, 1):
        logging.info(f"Top {i}: [{tp['subreddit']}] {tp['title']} (score={tp['score']})")
    return len(valid_posts)

def main():
    parser = argparse.ArgumentParser(description="Reddit scraper for finance subreddits.")
    parser.add_argument('--date', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='Date (YYYY-MM-DD)')
    parser.add_argument('--subreddits', type=str, help='Comma-separated list of subreddits', default=','.join(DEFAULT_SUBREDDITS))
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Number of posts per subreddit')
    parser.add_argument('--output', type=str, help='Output directory', default=None)
    parser.add_argument('--min-score', type=int, default=0, help='Minimum score for posts')
    parser.add_argument('--exclude-stickied', action='store_true', help='Exclude stickied posts')
    parser.add_argument('--exclude-removed', action='store_true', help='Exclude removed posts')
    parser.add_argument('--keywords', type=str, help='Comma-separated keywords to filter')
    parser.add_argument('--tickers', type=str, help='Comma-separated tickers to filter')
    parser.add_argument('--config', type=str, help='Path to config file (yaml or json)')
    args = parser.parse_args()

    # Config file support
    config = load_scraper_config(args.config)
    if config:
        date = config.get('date', args.date)
        subreddits = config.get('subreddits', args.subreddits)
        if isinstance(subreddits, str):
            subreddits = [s.strip() for s in subreddits.split(',') if s.strip()]
        limit = config.get('limit', args.limit)
        save_dir = config.get('output', args.output or ('reddit/' + date))
        min_score = config.get('min_score', args.min_score)
        exclude_stickied = config.get('exclude_stickied', args.exclude_stickied)
        exclude_removed = config.get('exclude_removed', args.exclude_removed)
        keywords = config.get('keywords', args.keywords)
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        tickers = config.get('tickers', args.tickers)
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(',') if t.strip()]
    else:
        date = args.date
        subreddits = [s.strip() for s in args.subreddits.split(',') if s.strip()]
        limit = args.limit
        save_dir = args.output or ('reddit/' + date)
        min_score = args.min_score
        exclude_stickied = args.exclude_stickied
        exclude_removed = args.exclude_removed
        keywords = [k.strip() for k in args.keywords.split(',')] if args.keywords else None
        tickers = [t.strip() for t in args.tickers.split(',')] if args.tickers else None

    log_command(args, CONFIG)
    n_posts = fetch_posts(
        subreddits,
        limit,
        save_dir,
        min_score=min_score,
        exclude_stickied=exclude_stickied,
        exclude_removed=exclude_removed,
        keywords=keywords,
        tickers=tickers,
        config=CONFIG,
    )
    print(f"✅ Saved {n_posts} posts to {save_dir}/posts.json")

if __name__ == '__main__':
    main()
