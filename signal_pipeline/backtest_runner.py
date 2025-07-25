import pandas as pd
import streamlit as st
import os
import glob
import io
from datetime import timedelta
import altair as alt

st.set_page_config(page_title="VolCon Score Backtest", layout="wide")

with st.sidebar:
    st.title("Controls")
    st.markdown("""
    **Instructions:**
    - Filter by ticker and date range
    - Select chart column and overlays
    - Download filtered data as CSV or Excel
    - Group by week/month for trend analysis
    """)

data_dir = "data"
all_files = sorted(glob.glob(os.path.join(data_dir, "*_vol_container_score.csv")))

st.title("📈 Vol Container Score Timeline")

if not all_files:
    st.warning("No score data found. Run vol_container_score.py first.")
else:
    # Robust CSV loading
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if "date" not in df.columns or "vol_container_score" not in df.columns:
                st.error(f"File {f} missing required columns.")
                continue
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {f}: {e}")
    if not df_list:
        st.warning("No valid score data found.")
    else:
        df = pd.concat(df_list)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Multi-ticker selection (if present)
        ticker_col = None
        for col in df.columns:
            if col.lower() == "ticker":
                ticker_col = col
                break
        if ticker_col:
            tickers = sorted(df[ticker_col].dropna().unique())
            selected_tickers = st.sidebar.multiselect("Select tickers", tickers, default=tickers)
            if selected_tickers:
                df = df[df[ticker_col].isin(selected_tickers)]

        # Date range filter
        min_date, max_date = df["date"].min(), df["date"].max()
        date_range = st.sidebar.date_input("Date range", [min_date.date(), max_date.date()])
        if len(date_range) == 2:
            df = df[(df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))]

        # Aggregation/grouping
        group_option = st.sidebar.selectbox("Group by", ["None", "Week", "Month"])
        if group_option == "Week":
            df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
            agg_df = df.groupby("week").mean(numeric_only=True).reset_index().rename(columns={"week": "date"})
        elif group_option == "Month":
            df["month"] = df["date"].dt.to_period("M").apply(lambda r: r.start_time)
            agg_df = df.groupby("month").mean(numeric_only=True).reset_index().rename(columns={"month": "date"})
        else:
            agg_df = df.copy()

        # Raw vs. aggregated data toggle
        show_raw = st.sidebar.checkbox("Show raw data (no aggregation)", value=False)
        plot_df = df if show_raw else agg_df

        # Interactive chart column selection
        chart_cols = [col for col in plot_df.columns if plot_df[col].dtype in ["float64", "int64"] and col != "date"]
        selected_chart_col = st.sidebar.selectbox("Select column to plot", chart_cols, index=chart_cols.index("vol_container_score") if "vol_container_score" in chart_cols else 0)

        # Moving average and volatility overlays
        ma_window = st.sidebar.slider("Moving Average Window (days)", 1, 30, 7)
        show_volatility = st.sidebar.checkbox("Show Volatility Overlay", value=True)
        y = plot_df[selected_chart_col]
        ma = y.rolling(window=ma_window).mean()
        vol = y.rolling(window=ma_window).std() if show_volatility else None

        # Custom alert threshold
        alert_thresh = st.sidebar.number_input("Custom alert threshold", value=float(y.mean() + 2 * y.std()))

        # Altair chart with tooltips and outlier highlight
        base = alt.Chart(plot_df.reset_index() if plot_df.index.name == "date" else plot_df).mark_line().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{selected_chart_col}:Q", title=selected_chart_col),
            tooltip=["date:T", selected_chart_col]
        )
        ma_chart = alt.Chart(plot_df.reset_index() if plot_df.index.name == "date" else plot_df).mark_line(color='orange').encode(
            x="date:T",
            y=alt.Y(ma, title=f"{selected_chart_col} MA"),
            tooltip=["date:T"]
        )
        chart = base
        if show_volatility:
            vol_chart = alt.Chart(plot_df.reset_index() if plot_df.index.name == "date" else plot_df).mark_line(color='purple').encode(
                x="date:T",
                y=alt.Y(vol, title=f"{selected_chart_col} Volatility"),
                tooltip=["date:T"]
            )
            chart = chart + ma_chart + vol_chart
        else:
            chart = chart + ma_chart
        # Outlier points
        outliers = plot_df[plot_df[selected_chart_col] > alert_thresh]
        if not outliers.empty:
            outlier_chart = alt.Chart(outliers.reset_index() if outliers.index.name == "date" else outliers).mark_point(color='red', size=60).encode(
                x="date:T",
                y=alt.Y(f"{selected_chart_col}:Q"),
                tooltip=["date:T", selected_chart_col]
            )
            chart = chart + outlier_chart
        st.altair_chart(chart, use_container_width=True)

        # Color-coded alerts for extreme scores
        st.subheader("Extreme Score Alerts")
        extreme = plot_df[plot_df[selected_chart_col] > alert_thresh]
        if not extreme.empty:
            st.error(f"Extreme {selected_chart_col} values detected:")
            st.dataframe(extreme)
        else:
            st.success("No extreme values detected.")

        # Correlation matrix
        st.subheader("Correlation Matrix (numeric columns)")
        st.write(plot_df[chart_cols].corr())

        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(plot_df[selected_chart_col].describe())

        # Download buttons for combined DataFrame
        csv_buffer = io.StringIO()
        plot_df.to_csv(csv_buffer, index=False)
        st.download_button("Download filtered data as CSV", csv_buffer.getvalue(), file_name="filtered_vol_container_score.csv", mime="text/csv")

        # Export filtered tickers only
        if ticker_col:
            tickers_csv = io.StringIO()
            pd.DataFrame({"ticker": plot_df[ticker_col].unique()}).to_csv(tickers_csv, index=False)
            st.download_button("Download filtered tickers as CSV", tickers_csv.getvalue(), file_name="filtered_tickers.csv", mime="text/csv")

        try:
            import xlsxwriter
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                plot_df.to_excel(writer, index=False)
            st.download_button("Download filtered data as Excel", excel_buffer.getvalue(), file_name="filtered_vol_container_score.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except ImportError:
            st.info("Install xlsxwriter for Excel export support.")

        st.dataframe(plot_df, use_container_width=True)
