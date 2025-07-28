import json
import datetime
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from signal_pipeline.scan_volatility_signals import load_sentiment_score

def test_load_sentiment_score_per_ticker(tmp_path):
    date = datetime.date(2021, 1, 1)
    template = str(tmp_path / "{ticker}" / "{date}" / "posts.json")

    # create Reddit dumps for two tickers with different sentiment words
    for ticker, word in [("AAA", "moon"), ("BBB", "down")]:
        p = tmp_path / ticker / str(date)
        p.mkdir(parents=True)
        with open(p / "posts.json", "w") as f:
            json.dump([{"title": word}], f)

    score_a = load_sentiment_score("AAA", date, template)
    score_b = load_sentiment_score("BBB", date, template)

    assert score_a > score_b
    assert 0 <= score_a <= 1
    assert 0 <= score_b <= 1

