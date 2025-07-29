import yfinance as yf
import time
import smtplib
from email.message import EmailMessage
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import argparse
import json
codex/implement-unified-configuration-loader
from .config import load_config

CONFIG = load_config()
=======
from .config import load_env
main

from sentiment_score import classify_sentiment

load_env()

from functools import lru_cache

@lru_cache(maxsize=32)
def get_iv_proxy(ticker, retries=3, delay=2):
    for attempt in range(retries):
        try:
            option_chain = yf.Ticker(ticker).option_chain()
            calls = option_chain.calls
            puts = option_chain.puts
            atm_strike = yf.Ticker(ticker).info['regularMarketPrice']
            closest_call = calls.iloc[(calls['strike'] - atm_strike).abs().argsort()[:1]]
            closest_put = puts.iloc[(puts['strike'] - atm_strike).abs().argsort()[:1]]
            iv = (closest_call['impliedVolatility'].values[0] + closest_put['impliedVolatility'].values[0]) / 2
            return iv
        except Exception as e:
            print(f"IV proxy error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return np.nan

@lru_cache(maxsize=32)
def get_iv_history_proxy(ticker, days=30, retries=3, delay=2):
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period=f'{days}d')
            daily_range = hist['High'] - hist['Low']
            iv_proxy = daily_range / hist['Close']
            return iv_proxy.values
        except Exception as e:
            print(f"IV history proxy error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return np.array([])

@lru_cache(maxsize=32)
def get_realized_vol(ticker, window=10, retries=3, delay=2, use_garch=False, use_lstm=False, external_data=None):
    """
    Calculate realized volatility. If use_garch is True and arch is available, use GARCH model.
    If use_lstm is True and Keras is available, use LSTM model. If external_data is provided, use it instead of yfinance.
    """
    try:
        from arch import arch_model
    except ImportError:
        arch_model = None
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense
    except ImportError:
        Sequential = None
    for attempt in range(retries):
        try:
            if external_data is not None:
                hist = external_data
            else:
                hist = yf.Ticker(ticker).history(period='30d')
            returns = hist['Close'].pct_change().dropna()
            if use_garch and arch_model:
                am = arch_model(returns, vol='Garch', p=1, q=1)
                res = am.fit(disp='off')
                garch_vol = np.mean(res.conditional_volatility[-window:])
                return garch_vol
            elif use_lstm and Sequential:
                # LSTM volatility modeling (simple demo)
                X = returns.values.reshape(-1,1)
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                X_seq = np.array([X_scaled[i-window:i] for i in range(window, len(X_scaled))])
                y_seq = X_scaled[window:]
                model = Sequential()
                model.add(LSTM(10, input_shape=(window,1)))
                model.add(Dense(1))
                model.compile(loss='mse', optimizer='adam')
                model.fit(X_seq, y_seq, epochs=5, batch_size=1, verbose=0)
                pred = model.predict(X_seq)
                lstm_vol = np.std(pred[-window:])
                return lstm_vol
            else:
                return np.std(returns[-window:])
        except Exception as e:
            print(f"Realized vol error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return np.nan

@lru_cache(maxsize=32)
def get_volume_ratio(ticker, retries=3, delay=2):
    for attempt in range(retries):
        try:
            data = yf.Ticker(ticker).info
            option_vol = data.get('averageDailyVolume10Day', 0)
            equity_vol = data.get('volume', 1)
            return option_vol / equity_vol if equity_vol else 0
        except Exception as e:
            print(f"Volume ratio error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return np.nan

@lru_cache(maxsize=32)
def simulate_oi_concentration(ticker, retries=3, delay=2):
    """Return the percent of open interest within ±5% of spot price.

    The function fetches the current option chain via ``yfinance`` and
    calculates the proportion of total open interest whose strikes lie within
    a 5% band around the latest spot price.  If data cannot be retrieved the
    function returns ``np.nan``.
    """
    for attempt in range(retries):
        try:
            tkr = yf.Ticker(ticker)
            spot = tkr.history(period="1d")['Close'].iloc[-1]
            expirations = tkr.options
            if not expirations:
                return np.nan

            total_oi = 0
            near_oi = 0

            # Use the nearest expiration to approximate current OI structure
            exp = expirations[0]
            chain = tkr.option_chain(exp)
            df = pd.concat([chain.calls, chain.puts], ignore_index=True)
            df['openInterest'] = df['openInterest'].fillna(0)
            total_oi = df['openInterest'].sum()
            band_mask = (df['strike'] >= spot * 0.95) & (df['strike'] <= spot * 1.05)
            near_oi = df.loc[band_mask, 'openInterest'].sum()

            return near_oi / total_oi if total_oi > 0 else np.nan
        except Exception as e:
            print(f"OI concentration error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return np.nan

def simulate_sentiment_score():
    # Load sample WSB posts and simulate scores
    texts = [
        "I'm renting my shares and selling CSPs all month!",
        "bagholding hard, this is rough",
        "laddering puts while everyone panics",
        "why is this dropping again?"
    ]
    scores = []
    for text in texts:
        polarity, label = classify_sentiment(text)
        modifier = 0.05 if label == 'suppressing' else -0.05 if label == 'breaking' else 0
        scores.append(polarity + modifier)
    return np.mean(scores)

def calculate_iv_rank(current_iv, iv_history):
    return sum(iv < current_iv for iv in iv_history) / len(iv_history)

def calculate_score(iv_rank, oi_concentration, sentiment_score, ov_ratio, rv_iv_spread):
    return (
        0.25 * iv_rank +
        0.20 * oi_concentration +
        0.20 * sentiment_score +
        0.15 * ov_ratio +
        0.20 * rv_iv_spread
    )

def run_score(ticker='GME', weights=None, config=None):
    """Calculate and persist a single day's Vol Container Score.

    Parameters
    ----------
    ticker : str
        Symbol to score.
    weights : dict, optional
        Optional override for feature weights.
    config : dict, optional
        Placeholder for future configuration options. Currently unused.
    """

    current_iv = get_iv_proxy(ticker)
    iv_history = get_iv_history_proxy(ticker)
    iv_rank = calculate_iv_rank(current_iv, iv_history) if len(iv_history) > 0 and not np.isnan(current_iv) else np.nan
    rv = get_realized_vol(ticker)
    rv_iv_spread = rv - current_iv if not np.isnan(rv) and not np.isnan(current_iv) else np.nan
    ov_ratio = get_volume_ratio(ticker)
    oi_concentration = simulate_oi_concentration(ticker)
    sentiment_score = simulate_sentiment_score()

    # Use custom weights if provided
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

    # Percentile rank for score
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
    # Append to historical file
    if os.path.exists(csv_path):
        df_hist = pd.read_csv(csv_path)
        df_hist = pd.concat([df_hist, df], ignore_index=True)
        df_hist.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)
    return df

import asyncio
async def async_run_score(ticker, weights=None, config=None):
    return run_score(ticker, weights, config)

async def async_batch_score(tickers, weights=None, config=None):
    tasks = [async_run_score(t, weights, config) for t in tickers]
    dfs = await asyncio.gather(*tasks)
    batch_df = pd.concat(dfs, ignore_index=True)
    batch_df.to_csv("data/batch_vol_container_scores.csv", index=False)
    stats = batch_df.describe().to_dict()
    with open("data/batch_vol_container_scores_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print("Batch summary stats exported to data/batch_vol_container_scores_stats.json")
    return batch_df

def batch_score(tickers, weights=None, config=None):
    # Fallback to sync if not using async
    try:
        return asyncio.run(async_batch_score(tickers, weights, config))
    except Exception as e:
        print(f"Async batch scoring failed, using sync: {e}")
        dfs = []
        for ticker in tickers:
            try:
                df = run_score(ticker, weights, config)
                dfs.append(df)
            except Exception as e:
                print(f"Error scoring {ticker}: {e}")
        batch_df = pd.concat(dfs, ignore_index=True)
        batch_df.to_csv("data/batch_vol_container_scores.csv", index=False)
        stats = batch_df.describe().to_dict()
        with open("data/batch_vol_container_scores_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print("Batch summary stats exported to data/batch_vol_container_scores_stats.json")
        return batch_df

def plot_scores(ticker, save=False, annotate_outliers=True, show_corr=True, show_feature_importance=True):
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
    # Rolling window analytics
    if 'vol_container_score' in df.columns:
        df['rolling_mean'] = df['vol_container_score'].rolling(window=5).mean()
        plt.plot(pd.to_datetime(df['date']), df['rolling_mean'], label='Rolling Mean (5)')
        # Rolling z-score
        df['rolling_z'] = (df['vol_container_score'] - df['vol_container_score'].rolling(window=5).mean()) / df['vol_container_score'].rolling(window=5).std()
    # Outlier annotation
    if annotate_outliers and 'vol_container_score' in df.columns:
        outliers = df[(df['vol_container_score'] < 0) | (df['vol_container_score'] > 1)]
        for idx, row in outliers.iterrows():
            plt.annotate('Outlier', (pd.to_datetime(row['date']), row['vol_container_score']), color='red')
    # Volatility regime detection
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
    # Correlation matrix visualization
    if show_corr:
        numeric_cols = ['iv_rank', 'oi_concentration', 'sentiment_score', 'ov_ratio', 'rv_iv_spread', 'vol_container_score']
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(7,5))
        import seaborn as sns
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title(f"{ticker} Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()
    # Feature importance (Random Forest)
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
        except Exception as e:
            print(f"Feature importance unavailable: {e}")

def diagnostics(ticker):
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

def export_excel(ticker):
    csv_path = f"data/{ticker}_vol_container_score.csv"
    if not os.path.exists(csv_path):
        print(f"No score history for {ticker}")
        return
    df = pd.read_csv(csv_path)
    excel_path = f"data/{ticker}_vol_container_score.xlsx"
    df.to_excel(excel_path, index=False)
    # Export summary statistics
    stats_path = f"data/{ticker}_vol_container_score_stats.json"
    stats = df.describe().to_dict()
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Exported to {excel_path} and summary stats to {stats_path}")

def fetch_news_sentiment(ticker, config: dict = CONFIG):
    """Fetch recent news and compute average sentiment polarity.

    The function attempts to use Finnhub if ``FINNHUB_API_KEY`` is set or
    NewsAPI if ``NEWSAPI_KEY`` is available. Headlines are classified with
    ``classify_sentiment`` and the mean polarity score is returned. ``None`` is
    returned on failure or if no headlines are retrieved.
    """
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

def load_json_config(config_path='volcon_config.json'):
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}

def send_email_alert(subject, body, to_email, smtp_server, smtp_port, smtp_user, smtp_pass):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = to_email
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"Email alert sent to {to_email}")
    except Exception as e:
        print(f"Email alert error: {e}")

def validate_config(config):
    required_keys = ['weights', 'tickers']
    for k in required_keys:
        if k not in config:
            raise ValueError(f"Missing required config key: {k}")
    return True

def generate_pdf_report(ticker):
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
        # Add correlation matrix
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
    except Exception as e:
        print(f"PDF report error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Vol Container Score Runner')
    parser.add_argument('--ticker', type=str, help='Single ticker to score')
    parser.add_argument('--batch', type=str, nargs='+', help='Batch tickers to score')
    parser.add_argument('--plot', type=str, help='Plot score history for ticker')
    parser.add_argument('--plot_all', action='store_true', help='Plot all tickers in batch file')
    parser.add_argument('--diagnostics', type=str, help='Run diagnostics for ticker')
    parser.add_argument('--export', type=str, help='Export score history to Excel for ticker')
    parser.add_argument('--news', type=str, help='Fetch news sentiment for ticker')
    parser.add_argument('--config', type=str, help='Config file for weights/tickers')
    parser.add_argument('--save_plot', action='store_true', help='Save plot to PNG')
    parser.add_argument('--dashboard', action='store_true', help='Launch interactive Streamlit dashboard')
    parser.add_argument('--help_all', action='store_true', help='Show help for all features')
    parser.add_argument('--email_alert', action='store_true', help='Send email alert for high scores')
    parser.add_argument('--pdf_report', type=str, help='Generate PDF report for ticker')
    parser.add_argument('--weight_iv_rank', type=float, help='Override weight for IV rank feature')
    parser.add_argument('--weight_oi_conc', type=float, help='Override weight for open interest concentration')
    parser.add_argument('--weight_sentiment', type=float, help='Override weight for sentiment score')
    parser.add_argument('--weight_ov_ratio', type=float, help='Override weight for option volume ratio')
    parser.add_argument('--weight_rv_iv_spread', type=float, help='Override weight for RV-IV spread')
    args = parser.parse_args()

    if args.help_all:
        print("""
        Vol Container Score Runner - Features:
        --ticker [TICKER] : Score a single ticker
        --batch [TICKER ...] : Score a batch of tickers
        --plot [TICKER] : Plot score history for ticker
        --plot_all : Plot all tickers in config
        --diagnostics [TICKER] : Run diagnostics for ticker
        --export [TICKER] : Export score history to Excel for ticker
        --news [TICKER] : Fetch news sentiment for ticker
        --config [PATH] : Config file for weights/tickers
        --save_plot : Save plot to PNG
        --dashboard : Launch interactive Streamlit dashboard
        --email_alert : Send email alert for high scores
        --pdf_report [TICKER] : Generate PDF report for ticker
        --weight_iv_rank [FLOAT] : Override IV rank weight
        --weight_oi_conc [FLOAT] : Override OI concentration weight
        --weight_sentiment [FLOAT] : Override sentiment score weight
        --weight_ov_ratio [FLOAT] : Override option volume ratio weight
        --weight_rv_iv_spread [FLOAT] : Override RV-IV spread weight
        --help_all : Show this help message
        """)
        return

    weights = None
    tickers = None
    config = None
    if args.config:
        if not os.path.exists(args.config):
            # Auto-generate config if missing
            default_config = {
                "weights": {
                    "iv_rank": 0.25,
                    "oi_concentration": 0.20,
                    "sentiment_score": 0.20,
                    "ov_ratio": 0.15,
                    "rv_iv_spread": 0.20
                },
                "tickers": ["GME", "AMC", "XRT"],
                "metadata": {
                    "GME": {"sector": "Retail", "exchange": "NYSE"},
                    "AMC": {"sector": "Entertainment", "exchange": "NYSE"},
                    "XRT": {"sector": "ETF", "exchange": "NYSEARCA"}
                },
                "api_keys": {
                    "FINNHUB_API_KEY": CONFIG.get('FINNHUB_API_KEY', '')
                }
            }
            with open(args.config, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Auto-generated config file: {args.config}")
        config = load_json_config(args.config)
        validate_config(config)
        weights = config.get('weights')
        tickers = config.get('tickers')
        metadata = config.get('metadata', {})
        api_keys = config.get('api_keys', {})
    else:
        metadata = {}
        api_keys = {}

    # Apply CLI weight overrides if provided
    cli_weights = {}
    if args.weight_iv_rank is not None:
        cli_weights['iv_rank'] = args.weight_iv_rank
    if args.weight_oi_conc is not None:
        cli_weights['oi_concentration'] = args.weight_oi_conc
    if args.weight_sentiment is not None:
        cli_weights['sentiment_score'] = args.weight_sentiment
    if args.weight_ov_ratio is not None:
        cli_weights['ov_ratio'] = args.weight_ov_ratio
    if args.weight_rv_iv_spread is not None:
        cli_weights['rv_iv_spread'] = args.weight_rv_iv_spread
    if cli_weights:
        weights = weights.copy() if weights else {}
        weights.update(cli_weights)

    if args.dashboard:
        try:
            import streamlit.web.cli as stcli
            import sys
            sys.argv = ["streamlit", "run", __file__]
            sys.exit(stcli.main())
        except Exception as e:
            print(f"Streamlit dashboard launch error: {e}")
            return

    if args.ticker:
        result = run_score(args.ticker, weights)
        print(result)
        # Email alert for high score
        if args.email_alert and result['vol_container_score'].iloc[0] > 0.8:
            send_email_alert(
                subject=f"High Vol Container Score Alert: {args.ticker}",
                body=str(result),
                to_email=CONFIG.get('ALERT_EMAIL', ''),
                smtp_server=CONFIG.get('SMTP_SERVER', ''),
                smtp_port=int(CONFIG.get('SMTP_PORT', 465)),
                smtp_user=CONFIG.get('SMTP_USER', ''),
                smtp_pass=CONFIG.get('SMTP_PASS', '')
            )
    if args.batch:
        print(batch_score(args.batch, weights))
    if tickers:
        print(batch_score(tickers, weights))
    if args.plot:
        plot_scores(args.plot, save=args.save_plot)
    if args.plot_all and tickers:
        for t in tickers:
            plot_scores(t, save=args.save_plot)
    if args.diagnostics:
        diagnostics(args.diagnostics)
    if args.export:
        export_excel(args.export)
    if args.news:
        fetch_news_sentiment(args.news)
    if args.pdf_report:
        generate_pdf_report(args.pdf_report)

if __name__ == '__main__':
    main()
