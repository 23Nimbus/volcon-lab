
# vol_signal_layer.py - Real Signal Layer Logic

import pandas as pd
import os
from datetime import datetime
import logging
from .utils import setup_logging


SIGNAL_DIR = "data"
DEFAULT_THRESHOLDS = {"high": 0.75, "low": 0.4}
LOG_PATH = "logs/vol_signal_layer.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
setup_logging(LOG_PATH)

def load_latest_score(ticker='GME'):
    try:
        files = sorted([f for f in os.listdir(SIGNAL_DIR) if f.startswith(ticker) and f.endswith(".csv")], reverse=True)
        if not files:
            logging.warning(f"No score file for {ticker}")
            return None
        df = pd.read_csv(os.path.join(SIGNAL_DIR, files[0]))
        return df.iloc[-1].to_dict()
    except Exception as e:
        logging.error(f"Error loading score for {ticker}: {e}")
        return None

def evaluate_signal(score_data, thresholds=None):
    if not score_data:
        return {"score": None, "alerts": ["No score data"], "details": {}}
    score = score_data.get('vol_container_score', None)
    alerts = []
    thresholds = thresholds or DEFAULT_THRESHOLDS
    try:
        if score is None:
            alerts.append("No score available")
        elif score > thresholds["high"]:
            alerts.append("⚠️ High containment: short-vol setup ideal")
        elif score < thresholds["low"]:
            alerts.append("📈 Breakdown risk: monitor for long-vol setup")
        else:
            alerts.append("📊 Neutral: hold or prepare")
    except Exception as e:
        alerts.append(f"Error evaluating signal: {e}")
        logging.error(f"Signal evaluation error: {e}")
    return {
        "score": score,
        "alerts": alerts,
        "details": score_data
    }

if __name__ == "__main__":
    result = evaluate_signal(load_latest_score())
    print(result)

import argparse
import json

def batch_evaluate(tickers, thresholds=None):
    results = {}
    for t in tickers:
        score_data = load_latest_score(t)
        results[t] = evaluate_signal(score_data, thresholds)
    return results

def export_alerts(ticker, out_path=None, thresholds=None, fmt="json"):
    score_data = load_latest_score(ticker)
    if not score_data:
        logging.warning(f"No score data for {ticker}")
        print(f"No score data for {ticker}")
        return
    result = evaluate_signal(score_data, thresholds)
    os.makedirs("alerts", exist_ok=True)
    if fmt == "json":
        out_path = out_path or f"alerts/{datetime.now().date()}_{ticker}_alerts.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    elif fmt == "excel":
        import pandas as pd
        out_path = out_path or f"alerts/{datetime.now().date()}_{ticker}_alerts.xlsx"
        df = pd.DataFrame([result["details"]])
        df["score"] = result["score"]
        df["alerts"] = ", ".join(result["alerts"])
        df.to_excel(out_path, index=False)
    elif fmt == "html":
        out_path = out_path or f"alerts/{datetime.now().date()}_{ticker}_alerts.html"
        html = f"<h2>Vol Signal Alert for {ticker}</h2><p>Score: {result['score']}</p><ul>" + "".join([f"<li>{a}</li>" for a in result["alerts"]]) + "</ul>"
        with open(out_path, "w") as f:
            f.write(html)
    elif fmt == "pdf":
        out_path = out_path or f"alerts/{datetime.now().date()}_{ticker}_alerts.pdf"
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            fig, ax = plt.subplots()
            ax.text(0.1, 0.8, f"Vol Signal Alert for {ticker}", fontsize=16)
            ax.text(0.1, 0.6, f"Score: {result['score']}", fontsize=12)
            ax.text(0.1, 0.4, "Alerts: " + ", ".join(result["alerts"]), fontsize=12)
            ax.axis('off')
            with PdfPages(out_path) as pdf:
                pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print(f"PDF export error: {e}")
    logging.info(f"Exported alerts to {out_path}")
    print(f"Exported alerts to {out_path}")
def run_unit_tests():
    print("Running unit tests...")
    # Test load_latest_score
    assert load_latest_score("GME") is None or isinstance(load_latest_score("GME"), dict)
    # Test evaluate_signal
    dummy = {"vol_container_score": 0.8}
    res = evaluate_signal(dummy)
    assert "score" in res and "alerts" in res
    # Test batch_evaluate
    batch = batch_evaluate(["GME", "AMC"])
    assert isinstance(batch, dict)
    print("All basic tests passed.")
def run_api_server(host="127.0.0.1", port=5000):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Run 'pip install flask' to use API.")
        return
    app = Flask(__name__)

    @app.route("/api/evaluate", methods=["GET"])
    def api_evaluate():
        ticker = request.args.get("ticker", "GME")
        thresholds = {
            "high": float(request.args.get("high", DEFAULT_THRESHOLDS["high"])),
            "low": float(request.args.get("low", DEFAULT_THRESHOLDS["low"]))
        }
        score_data = load_latest_score(ticker)
        result = evaluate_signal(score_data, thresholds)
        return jsonify(result)

    @app.route("/api/diagnostics", methods=["GET"])
    def api_diagnostics():
        ticker = request.args.get("ticker", "GME")
        score_data = load_latest_score(ticker)
        return jsonify(score_data)

    print(f"API server running at http://{host}:{port}/api/evaluate")
    app.run(host=host, port=port)

def diagnostics(ticker):
    score_data = load_latest_score(ticker)
    if not score_data:
        logging.warning(f"No score data for {ticker}")
        print(f"No score data for {ticker}")
        return
    print("Diagnostics for", ticker)
    print(json.dumps(score_data, indent=2, default=str))
    # Historical analysis
    try:
        files = sorted([f for f in os.listdir(SIGNAL_DIR) if f.startswith(ticker) and f.endswith(".csv")], reverse=True)
        if files:
            df = pd.read_csv(os.path.join(SIGNAL_DIR, files[0]))
            print("Historical stats:")
            print(df.describe())
            print("Score trend:")
            print(df['vol_container_score'].tail(10))
    except Exception as e:
        logging.error(f"Diagnostics error: {e}")
def send_notification(alerts, ticker):
    # Stub for notification integration (email, Slack, etc.)
    if any("High containment" in a for a in alerts):
        logging.info(f"Notification: High containment alert for {ticker}")
        print(f"[NOTIFY] High containment alert for {ticker}")

def launch_dashboard(ticker=None, date_range=None):
    try:
        import streamlit.web.cli as stcli
        import sys
        dashboard_path = os.path.join(os.path.dirname(__file__), "streamlit_signal_dashboard.py")
        args = ["streamlit", "run", dashboard_path]
        if ticker:
            args += ["--", f"--ticker={ticker}"]
        if date_range:
            args += [f"--date_range={date_range}"]
        sys.argv = args
        sys.exit(stcli.main())
    except Exception as e:
        print(f"Dashboard launch error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vol Signal Layer CLI")
    parser.add_argument("--ticker", type=str, help="Ticker to evaluate")
    parser.add_argument("--batch", nargs='+', help="Batch tickers to evaluate")
    parser.add_argument("--export_alerts", type=str, help="Export alerts for ticker")
    parser.add_argument("--export_format", type=str, choices=["json", "excel", "pdf", "html"], default="json", help="Export format")
    parser.add_argument("--diagnostics", type=str, help="Diagnostics for ticker")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--dashboard_ticker", type=str, help="Dashboard ticker")
    parser.add_argument("--dashboard_date_range", type=str, help="Dashboard date range")
    parser.add_argument("--high_threshold", type=float, help="Custom high containment threshold")
    parser.add_argument("--low_threshold", type=float, help="Custom breakdown threshold")
    parser.add_argument("--run_tests", action="store_true", help="Run unit tests")
    parser.add_argument("--api", action="store_true", help="Run REST API server")
    args = parser.parse_args()

    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.high_threshold is not None:
        thresholds["high"] = args.high_threshold
    if args.low_threshold is not None:
        thresholds["low"] = args.low_threshold

    if args.run_tests:
        run_unit_tests()
    elif args.api:
        run_api_server()
    elif args.dashboard:
        launch_dashboard(ticker=args.dashboard_ticker, date_range=args.dashboard_date_range)
    elif args.ticker:
        result = evaluate_signal(load_latest_score(args.ticker), thresholds)
        print(json.dumps(result, indent=2, default=str))
        send_notification(result["alerts"], args.ticker)
    elif args.batch:
        results = batch_evaluate(args.batch, thresholds)
        print(json.dumps(results, indent=2, default=str))
        for t, res in results.items():
            send_notification(res["alerts"], t)
    elif args.export_alerts:
        export_alerts(args.export_alerts, thresholds=thresholds, fmt=args.export_format)
    elif args.diagnostics:
        diagnostics(args.diagnostics)
    else:
        # Default: evaluate GME
        result = evaluate_signal(load_latest_score(), thresholds)
        print(json.dumps(result, indent=2, default=str))
        send_notification(result["alerts"], "GME")
