import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
import asyncio
from typing import List, Dict

from .data_ingestion import (
    get_iv_proxy,
    get_iv_history_proxy,
    get_realized_vol,
    get_volume_ratio,
    simulate_oi_concentration,
)
from .sentiment_processing import simulate_sentiment_score, fetch_news_sentiment
from .config import load_config

CONFIG = load_config()

__all__ = [
    "calculate_iv_rank",
    "calculate_score",
    "run_score",
    "batch_score",
    "plot_scores",
    "diagnostics",
    "export_excel",
    "load_json_config",
    "validate_config",
    "send_email_alert",
    "generate_pdf_report",
    "fetch_news_sentiment",
]


def calculate_iv_rank(current_iv: float, iv_history: List[float]) -> float:
    return sum(iv < current_iv for iv in iv_history) / len(iv_history)


def calculate_score(
    iv_rank: float,
    oi_concentration: float,
    sentiment_score: float,
    ov_ratio: float,
    rv_iv_spread: float,
) -> float:
    return (
        0.25 * iv_rank +
        0.20 * oi_concentration +
        0.20 * sentiment_score +
        0.15 * ov_ratio +
        0.20 * rv_iv_spread
    )


def run_score(ticker: str = "GME", weights: Dict[str, float] | None = None, config: Dict | None = None) -> pd.DataFrame:
    current_iv = get_iv_proxy(ticker)
    iv_history = get_iv_history_proxy(ticker)
    iv_rank = calculate_iv_rank(current_iv, iv_history) if len(iv_history) > 0 and not np.isnan(current_iv) else np.nan
    rv = get_realized_vol(ticker)
    rv_iv_spread = rv - current_iv if not np.isnan(rv) and not np.isnan(current_iv) else np.nan
    ov_ratio = get_volume_ratio(ticker)
    oi_concentration = simulate_oi_concentration(ticker)
    sentiment_score = simulate_sentiment_score()

    if weights:
        score = (
            weights.get('iv_rank', 0.25) * iv_rank +
            weights.get('oi_concentration', 0.20) * oi_concentration +
            weights.get('sentiment_score', 0.20) * sentiment_score +
            weights.get('ov_ratio', 0.15) * ov_ratio +
            weights.get('rv_iv_spread', 0.20) * rv_iv_spread
        )
    else:
        score = calculate_score(iv_rank, oi_concentration, sentiment_score, ov_ratio, rv_iv_spread)

    percentile = None
    csv_path = f"data/{ticker}_vol_container_score.csv"
    if os.path.exists(csv_path):
        df_hist = pd.read_csv(csv_path)
        if 'vol_container_score' in df_hist.columns and not np.isnan(score):
            percentile = (df_hist['vol_container_score'] < score).mean()

    result = {
        'date': datetime.date.today(),
        'ticker': ticker,
        'iv_rank': iv_rank,
        'oi_concentration': oi_concentration,
        'sentiment_score': sentiment_score,
        'ov_ratio': ov_ratio,
        'rv_iv_spread': rv_iv_spread,
        'vol_container_score': score,
        'score_percentile': percentile if percentile is not None else np.nan
    }

    df = pd.DataFrame([result])
    os.makedirs("data", exist_ok=True)
    if os.path.exists(csv_path):
        df_hist = pd.read_csv(csv_path)
        df_hist = pd.concat([df_hist, df], ignore_index=True)
        df_hist.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)
    return df


async def async_run_score(ticker: str, weights: Dict[str, float] | None = None, config: Dict | None = None) -> pd.DataFrame:
    return run_score(ticker, weights, config)


async def async_batch_score(tickers: List[str], weights: Dict[str, float] | None = None, config: Dict | None = None) -> pd.DataFrame:
    tasks = [async_run_score(t, weights, config) for t in tickers]
    dfs = await asyncio.gather(*tasks)
    batch_df = pd.concat(dfs, ignore_index=True)
    batch_df.to_csv("data/batch_vol_container_scores.csv", index=False)
    stats = batch_df.describe().to_dict()
    with open("data/batch_vol_container_scores_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print("Batch summary stats exported to data/batch_vol_container_scores_stats.json")
    return batch_df


def batch_score(tickers: List[str], weights: Dict[str, float] | None = None, config: Dict | None = None) -> pd.DataFrame:
    try:
        return asyncio.run(async_batch_score(tickers, weights, config))
    except Exception as e:
        print(f"Async batch scoring failed, using sync: {e}")
        dfs = []
        for ticker in tickers:
            try:
                df = run_score(ticker, weights, config)
                dfs.append(df)
            except Exception as exc:
                print(f"Error scoring {ticker}: {exc}")
        batch_df = pd.concat(dfs, ignore_index=True)
        batch_df.to_csv("data/batch_vol_container_scores.csv", index=False)
        stats = batch_df.describe().to_dict()
        with open("data/batch_vol_container_scores_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print("Batch summary stats exported to data/batch_vol_container_scores_stats.json")
        return batch_df


def plot_scores(ticker: str, save: bool = False, annotate_outliers: bool = True, show_corr: bool = True, show_feature_importance: bool = True) -> None:
    csv_path = f"data/{ticker}_vol_container_score.csv"
    if not os.path.exists(csv_path):
        print(f"No score history for {ticker}")
        return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10,6))
    plt.plot(pd.to_datetime(df['date']), df['vol_container_score'], marker='o', label='Vol Container Score')
    plt.plot(pd.to_datetime(df['date']), df['iv_rank'], label='IV Rank')
    plt.plot(pd.to_datetime(df['date']), df['sentiment_score'], label='Sentiment Score')
    plt.plot(pd.to_datetime(df['date']), df['rv_iv_spread'], label='RV-IV Spread')
    if 'vol_container_score' in df.columns:
        df['rolling_mean'] = df['vol_container_score'].rolling(window=5).mean()
        plt.plot(pd.to_datetime(df['date']), df['rolling_mean'], label='Rolling Mean (5)')
        df['rolling_z'] = (df['vol_container_score'] - df['vol_container_score'].rolling(window=5).mean()) / df['vol_container_score'].rolling(window=5).std()
    if annotate_outliers and 'vol_container_score' in df.columns:
        outliers = df[(df['vol_container_score'] < 0) | (df['vol_container_score'] > 1)]
        for _, row in outliers.iterrows():
            plt.annotate('Outlier', (pd.to_datetime(row['date']), row['vol_container_score']), color='red')
    if 'rolling_z' in df.columns:
        regime = np.where(df['rolling_z'] > 1, 'High Vol', np.where(df['rolling_z'] < -1, 'Low Vol', 'Normal'))
        for i, r in enumerate(regime):
            if r != 'Normal':
                plt.axvline(pd.to_datetime(df['date'].iloc[i]), color='orange' if r == 'High Vol' else 'blue', linestyle='--', alpha=0.3)
    plt.legend()
    plt.title(f"{ticker} Vol Container Score & Components")
    plt.xlabel("Date")
    plt.ylabel("Score / Value")
    plt.tight_layout()
    if save:
        plt.savefig(f"data/{ticker}_score_plot.png")
        print(f"Plot saved to data/{ticker}_score_plot.png")
    plt.show()
    if show_corr:
        numeric_cols = ['iv_rank', 'oi_concentration', 'sentiment_score', 'ov_ratio', 'rv_iv_spread', 'vol_container_score']
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(7,5))
        import seaborn as sns
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title(f"{ticker} Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()
    if show_feature_importance:
        try:
            from sklearn.ensemble import RandomForestRegressor
            fi_cols = ['iv_rank', 'oi_concentration', 'sentiment_score', 'ov_ratio', 'rv_iv_spread']
            rf = RandomForestRegressor()
            rf.fit(df[fi_cols].fillna(0), df['vol_container_score'].fillna(0))
            importances = rf.feature_importances_
            plt.figure(figsize=(7,4))
            plt.bar(fi_cols, importances)
            plt.title(f"{ticker} Feature Importances (Random Forest)")
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"Feature importance unavailable: {exc}")


def diagnostics(ticker: str) -> None:
    csv_path = f"data/{ticker}_vol_container_score.csv"
    if not os.path.exists(csv_path):
        print(f"No score history for {ticker}")
        return
    df = pd.read_csv(csv_path)
    print("Missing values per column:")
    print(df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Outliers (score < 0 or > 1):", ((df['vol_container_score'] < 0) | (df['vol_container_score'] > 1)).sum())
    print("Score percentile (latest):", df['score_percentile'].iloc[-1] if 'score_percentile' in df.columns else 'N/A')


def export_excel(ticker: str) -> None:
    csv_path = f"data/{ticker}_vol_container_score.csv"
    if not os.path.exists(csv_path):
        print(f"No score history for {ticker}")
        return
    df = pd.read_csv(csv_path)
    excel_path = f"data/{ticker}_vol_container_score.xlsx"
    df.to_excel(excel_path, index=False)
    stats_path = f"data/{ticker}_vol_container_score_stats.json"
    stats = df.describe().to_dict()
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Exported to {excel_path} and summary stats to {stats_path}")


def load_json_config(config_path: str = 'volcon_config.json') -> Dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def send_email_alert(subject: str, body: str, to_email: str, smtp_server: str, smtp_port: int, smtp_user: str, smtp_pass: str) -> None:
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = to_email
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as s:
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        print(f"Email alert sent to {to_email}")
    except Exception as e:
        print(f"Email alert error: {e}")


def validate_config(config: Dict) -> bool:
    required_keys = ['weights', 'tickers']
    for k in required_keys:
        if k not in config:
            raise ValueError(f"Missing required config key: {k}")
    return True


def generate_pdf_report(ticker: str) -> None:
    try:
        import matplotlib.backends.backend_pdf as pdf_backend
        csv_path = f"data/{ticker}_vol_container_score.csv"
        if not os.path.exists(csv_path):
            print(f"No score history for {ticker}")
            return
        df = pd.read_csv(csv_path)
        pdf = pdf_backend.PdfPages(f"data/{ticker}_report.pdf")
        plt.figure(figsize=(10,6))
        plt.plot(pd.to_datetime(df['date']), df['vol_container_score'], marker='o', label='Vol Container Score')
        plt.legend()
        plt.title(f"{ticker} Vol Container Score")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        numeric_cols = ['iv_rank', 'oi_concentration', 'sentiment_score', 'ov_ratio', 'rv_iv_spread', 'vol_container_score']
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(7,5))
        import seaborn as sns
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title(f"{ticker} Feature Correlation Matrix")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        pdf.close()
        print(f"PDF report generated: data/{ticker}_report.pdf")
    except Exception as exc:
        print(f"PDF report error: {exc}")
