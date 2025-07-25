# streamlit_app.py - Real Signal Dashboard

import streamlit as st
import pandas as pd
from vol_signal_layer import load_latest_score, evaluate_signal

st.set_page_config(layout="wide")
st.title("📊 VolCon Signal Layer Dashboard")

with st.sidebar:
    st.header("Controls & Info")
    st.markdown("""
    **Instructions:**
    - View latest Vol Container Score and alerts
    - Filter alerts by keyword
    - Download raw score data
    - View historical score chart
    """)
    alert_filter = st.text_input("Filter alerts by keyword", "")
    st.markdown("---")
    st.markdown("**Analytics Options:**")
    show_stats = st.checkbox("Show Advanced Stats", value=True)
    show_chart = st.checkbox("Show Historical Chart", value=True)
    st.markdown("---")
    st.markdown("**Export Options:**")
    export_alerts = st.button("Export Filtered Alerts")
    st.markdown("---")
    st.markdown("**Feedback:**")
    feedback = st.text_area("Share feedback or suggestions:")
    if st.button("Submit Feedback") and feedback:
        st.success("Thank you for your feedback!")

# Load latest score data
score_data = load_latest_score()

if score_data:
    result = evaluate_signal(score_data)

    st.metric("Vol Container Score", round(result['score'], 3))
    st.subheader("🚨 Alerts")
    filtered_alerts = [a for a in result['alerts'] if alert_filter.lower() in a.lower()] if alert_filter else result['alerts']
    for alert in filtered_alerts:
        st.warning(alert)
    if not filtered_alerts:
        st.success("No alerts match filter.")

    # Export filtered alerts
    if export_alerts and filtered_alerts:
        st.download_button("Download Filtered Alerts", '\n'.join(filtered_alerts), file_name="filtered_alerts.txt", mime="text/plain")

    st.subheader("🔎 Raw Score Data")
    st.json(result['details'])

    # Summary statistics
    details_df = pd.DataFrame([result['details']]) if isinstance(result['details'], dict) else pd.DataFrame(result['details'])
    if show_stats:
        st.subheader("📈 Score Summary Stats")
        if 'score' in details_df.columns:
            st.write(details_df['score'].describe())
        else:
            st.write(details_df.describe())
        # Advanced analytics
        st.markdown("**Advanced Analytics:**")
        if 'score' in details_df.columns:
            mean = details_df['score'].mean()
            std = details_df['score'].std()
            min_ = details_df['score'].min()
            max_ = details_df['score'].max()
            st.write(f"Mean: {mean:.3f}")
            st.write(f"Std Dev (Volatility): {std:.3f}")
            st.write(f"Min: {min_:.3f}")
            st.write(f"Max: {max_:.3f}")
            # Daily change and percent change
            if 'date' in details_df.columns:
                details_df['date'] = pd.to_datetime(details_df['date'])
                details_df = details_df.sort_values('date')
                details_df['daily_change'] = details_df['score'].diff()
                details_df['pct_change'] = details_df['score'].pct_change() * 100
                st.write("**Daily Change (Score):**")
                st.line_chart(details_df.set_index('date')['daily_change'])
                st.write("**Percent Change (Score %):**")
                st.line_chart(details_df.set_index('date')['pct_change'])
                # Rolling volatility
                vol_window = st.number_input("Rolling Volatility Window (days)", min_value=2, max_value=30, value=5)
                details_df['rolling_vol'] = details_df['score'].rolling(window=vol_window).std()
                st.write("**Rolling Volatility:**")
                st.line_chart(details_df.set_index('date')['rolling_vol'])
                # Z-score
                details_df['zscore'] = (details_df['score'] - mean) / std if std != 0 else 0
                st.write("**Z-Score (Standardized Score):**")
                st.line_chart(details_df.set_index('date')['zscore'])
                # Outlier detection
                outlier_thresh = st.number_input("Outlier Z-Score Threshold", min_value=1.0, max_value=5.0, value=2.0)
                outliers = details_df[details_df['zscore'].abs() > outlier_thresh]
                st.write(f"Outliers (|z| > {outlier_thresh}): {len(outliers)}")
                if not outliers.empty:
                    st.dataframe(outliers[['date', 'score', 'zscore']])
                # Quantile stats
                st.write("**Quantile Stats:**")
                quantiles = details_df['score'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                st.write(quantiles)
                # Distribution plot
                st.write("**Score Distribution (Interactive):**")
                import numpy as np
                import matplotlib.pyplot as plt
                bins = st.slider("Histogram Bins", min_value=5, max_value=100, value=20)
                fig, ax = plt.subplots()
                ax.hist(details_df['score'].dropna(), bins=bins, color='skyblue', edgecolor='black')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                # Autocorrelation
                st.write("**Autocorrelation (Score):**")
                from pandas.plotting import autocorrelation_plot
                fig2, ax2 = plt.subplots()
                autocorrelation_plot(details_df['score'].dropna(), ax=ax2)
                st.pyplot(fig2)
                # Seasonality decomposition
                st.write("**Seasonality Decomposition (Score):**")
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    freq = st.number_input("Decomposition Period (days)", min_value=2, max_value=30, value=7)
                    decomposition = seasonal_decompose(details_df['score'].dropna(), period=freq, model='additive')
                    fig3, axs = plt.subplots(4, 1, figsize=(8, 8))
                    axs[0].plot(decomposition.observed)
                    axs[0].set_title('Observed')
                    axs[1].plot(decomposition.trend)
                    axs[1].set_title('Trend')
                    axs[2].plot(decomposition.seasonal)
                    axs[2].set_title('Seasonal')
                    axs[3].plot(decomposition.resid)
                    axs[3].set_title('Residual')
                    plt.tight_layout()
                    st.pyplot(fig3)
                except Exception as e:
                    st.info(f"Seasonality decomposition unavailable: {e}")
                # Score trend regression
                st.write("**Score Trend Regression:**")
                try:
                    from sklearn.linear_model import LinearRegression
                    x = np.arange(len(details_df['score'].dropna())).reshape(-1, 1)
                    y = details_df['score'].dropna().values
                    model = LinearRegression().fit(x, y)
                    trend = model.predict(x)
                    fig4, ax4 = plt.subplots()
                    ax4.plot(details_df['date'].iloc[:len(trend)], y, label='Score')
                    ax4.plot(details_df['date'].iloc[:len(trend)], trend, label='Trend', color='red')
                    ax4.legend()
                    ax4.set_title('Score & Trend')
                    st.pyplot(fig4)
                except Exception as e:
                    st.info(f"Trend regression unavailable: {e}")
            # Correlation with other columns
            other_cols = [col for col in details_df.columns if col not in ['score', 'date', 'daily_change', 'pct_change', 'MA'] and pd.api.types.is_numeric_dtype(details_df[col])]
            if other_cols:
                st.write("**Correlation with Other Metrics:**")
                corr = details_df[['score'] + other_cols].corr()
                st.dataframe(corr)
        if 'alerts' in result:
            st.write(f"Total Alerts: {len(result['alerts'])}")
            st.write(f"Filtered Alerts: {len(filtered_alerts)}")
            # Alert frequency by day
            if 'date' in details_df.columns:
                alert_dates = []
                for alert in result['alerts']:
                    # Try to extract date from alert string if present
                    import re
                    found = re.findall(r'\d{4}-\d{2}-\d{2}', alert)
                    if found:
                        alert_dates.extend(found)
                if alert_dates:
                    alert_freq = pd.Series(alert_dates).value_counts().sort_index()
                    st.write("**Alert Frequency by Day:**")
                    st.bar_chart(alert_freq)

    # Download button for raw data
    st.download_button("Download Raw Score Data", details_df.to_json(orient='records', indent=2), file_name="volcon_score_details.json", mime="application/json")

    # Historical chart (if available)
    if show_chart and 'date' in details_df.columns and 'score' in details_df.columns:
        st.subheader("📊 Historical Vol Container Score")
        chart_df = details_df.sort_values('date')
        # Date range selection
        min_date, max_date = chart_df['date'].min(), chart_df['date'].max()
        date_range = st.slider("Select Date Range", min_value=pd.to_datetime(min_date), max_value=pd.to_datetime(max_date), value=(pd.to_datetime(min_date), pd.to_datetime(max_date)), format="YYYY-MM-DD")
        chart_df['date'] = pd.to_datetime(chart_df['date'])
        chart_df = chart_df[(chart_df['date'] >= date_range[0]) & (chart_df['date'] <= date_range[1])]
        chart_type = st.selectbox("Chart Type", ["Line", "Area", "Bar"])
        # Moving average overlay
        ma_window = st.number_input("Moving Average Window (days)", min_value=1, max_value=30, value=3)
        chart_df['MA'] = chart_df['score'].rolling(window=ma_window).mean()
        chart_data = chart_df.set_index('date')[['score', 'MA']]
        if chart_type == "Line":
            st.line_chart(chart_data)
        elif chart_type == "Area":
            st.area_chart(chart_data[['score']])
        elif chart_type == "Bar":
            st.bar_chart(chart_data[['score']])
        st.markdown("*Tip: Use sidebar to toggle chart visibility*")
        # Chart data download
        st.download_button("Download Chart Data", chart_df.to_csv(index=False), file_name="volcon_score_chart.csv", mime="text/csv")
else:
    st.error("No signal data found. Run vol_container_score.py first.")
