"""Command line interface for Vol Container Score calculations."""
import argparse
from .scoring import (
    run_score,
    batch_score,
    plot_scores,
    diagnostics,
    export_excel,
    fetch_news_sentiment,
    load_json_config,
    validate_config,
    send_email_alert,
    generate_pdf_report,
    calculate_iv_rank,
    calculate_score,
)
from .scoring import CONFIG


def main() -> None:
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
        config = load_json_config(args.config)
        validate_config(config)
        weights = config.get('weights')
        tickers = config.get('tickers')
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
        fetch_news_sentiment(args.news, CONFIG)
    if args.pdf_report:
        generate_pdf_report(args.pdf_report)

if __name__ == '__main__':
    main()
