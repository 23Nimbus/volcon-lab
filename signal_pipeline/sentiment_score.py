from textblob import TextBlob
import pandas as pd
import re
from typing import List, Tuple, Dict, Any, Optional

# Define positive (vol suppression) and breakdown terms
SUPPRESSION_TERMS = ['slow and steady', 'renting my shares', 'laddering CSPs', 'monthly income']
BREAKDOWN_TERMS = ['bagholding', 'margin call', 'why is this dropping', 'manipulating it']


def classify_sentiment(text: str,
                      suppression_terms: Optional[List[str]] = None,
                      breakdown_terms: Optional[List[str]] = None
                      ) -> Tuple[float, str]:
    """
    Classify sentiment of text using TextBlob polarity and custom suppression/breakdown terms.
    Returns (polarity, label).
    """
    suppression_terms = suppression_terms or SUPPRESSION_TERMS
    breakdown_terms = breakdown_terms or BREAKDOWN_TERMS
    try:
        text_lc = text.lower()
        blob = TextBlob(text_lc)
        polarity = blob.sentiment.polarity
        suppression_hits = sum(term in text_lc for term in suppression_terms)
        breakdown_hits = sum(term in text_lc for term in breakdown_terms)
        if suppression_hits > breakdown_hits:
            label = 'suppressing'
        elif breakdown_hits > suppression_hits:
            label = 'breaking'
        else:
            label = 'neutral'
        return polarity, label
    except Exception as e:
        return 0.0, 'error'


def score_dataframe(df: pd.DataFrame,
                    text_column: str = 'title',
                    suppression_terms: Optional[List[str]] = None,
                    breakdown_terms: Optional[List[str]] = None
                   ) -> pd.DataFrame:
    """
    Score a DataFrame of texts for sentiment polarity and label.
    Returns a DataFrame with columns: text, polarity, label.
    """
    scores = []
    for text in df[text_column].astype(str):
        polarity, label = classify_sentiment(text, suppression_terms, breakdown_terms)
        scores.append({'text': text, 'polarity': polarity, 'label': label})
    return pd.DataFrame(scores)


def batch_classify(texts: List[str],
                   suppression_terms: Optional[List[str]] = None,
                   breakdown_terms: Optional[List[str]] = None
                  ) -> List[Dict[str, Any]]:
    """
    Batch classify a list of texts for sentiment.
    Returns a list of dicts with text, polarity, label.
    """
    results = []
    for text in texts:
        polarity, label = classify_sentiment(text, suppression_terms, breakdown_terms)
        results.append({'text': text, 'polarity': polarity, 'label': label})
    return results


def sentiment_summary(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics for a batch of sentiment scores.
    Returns dict with mean, min, max, std, label counts.
    """
    polarities = [s['polarity'] for s in scores if isinstance(s['polarity'], (int, float))]
    labels = [s['label'] for s in scores]
    summary = {
        'mean': float(pd.Series(polarities).mean()) if polarities else 0.0,
        'min': float(pd.Series(polarities).min()) if polarities else 0.0,
        'max': float(pd.Series(polarities).max()) if polarities else 0.0,
        'std': float(pd.Series(polarities).std()) if polarities else 0.0,
        'label_counts': dict(pd.Series(labels).value_counts())
    }
    return summary

# Example usage (uncomment for testing)
# if __name__ == "__main__":
#     texts = ["I'm bagholding GME", "Slow and steady wins the race", "Why is this dropping so much?"]
#     results = batch_classify(texts)
#     print(results)
#     print(sentiment_summary(results))
