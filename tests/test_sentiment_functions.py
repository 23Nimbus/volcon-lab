import pandas as pd
from signal_pipeline.sentiment_score import (
    classify_sentiment,
    score_dataframe,
    batch_classify,
    sentiment_summary,
)


def test_classify_sentiment_labels():
    pol, label = classify_sentiment("Slow and steady wins")
    assert label == "suppressing"
    assert pol > 0

    pol2, label2 = classify_sentiment("I'm bagholding again")
    assert label2 == "breaking"
    assert pol2 <= 0


def test_score_dataframe_output():
    df = pd.DataFrame({"title": ["slow and steady", "bagholding"]})
    result = score_dataframe(df)
    assert list(result.columns) == ["text", "polarity", "label"]
    assert len(result) == 2
    assert set(result["label"]) == {"suppressing", "breaking"}


def test_batch_classify_and_summary():
    texts = ["slow and steady", "bagholding"]
    results = batch_classify(texts)
    summary = sentiment_summary(results)
    assert summary["label_counts"].get("suppressing") == 1
    assert summary["label_counts"].get("breaking") == 1
    assert summary["max"] >= summary["min"]
