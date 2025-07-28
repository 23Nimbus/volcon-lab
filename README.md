# VolCon Lab

This repository contains tools for detecting and monitoring "volatility containers".

## Environment configuration

The `signal_pipeline/.env.template` file lists the environment variables used by the
pipeline. Copy this template to `.env` and fill in your secrets:

| Variable | Description |
| -------- | ----------- |
| `REDDIT_CLIENT_ID` | OAuth client ID for Reddit API access. |
| `REDDIT_CLIENT_SECRET` | OAuth client secret for Reddit API access. |
| `REDDIT_USER_AGENT` | User agent string identifying your Reddit app. |
| `FINNHUB_API_KEY` | API key for retrieving market data from Finnhub. |

For security, never commit your populated `.env` file to version control.
The repository's `.gitignore` already excludes `.env`.
