import streamlit as st
import pandas as pd
import json
import os
from datetime import date
import matplotlib.pyplot as plt

# Load today's alert file
def load_alerts():
    path = f"alerts/{date.today()}_vol_signals.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        return []

# UI Start
st.set_page_config(page_title="VolCon-Lab: Signal Dashboard", layout="wide")
st.title("📡 VolCon-Lab: Live Volatility Signal Layer")
st.caption("Real-time monitoring for retail-coordinated breakout potential and volatility suppression breakdown.")

alerts = load_alerts()

with st.sidebar:
    st.header("Controls & Info")
    st.markdown("""
    **Instructions:**
    - View today's VolCon signals
    - Filter by ticker or alert level
    - Download table or filtered alerts
    - View analytics and charts
    """)
    filter_ticker = st.text_input("Filter by Ticker", "")
    filter_level = st.selectbox("Filter by Alert Level", ["All", "🟢 Watch", "🟡 Tension", "🔴 Breakout Potential"])
    show_analytics = st.checkbox("Show Analytics", value=True)
    show_chart = st.checkbox("Show Score Chart", value=True)
    export_table = st.button("Export Table CSV")
    export_alerts = st.button("Export Filtered Alerts")
    st.markdown("---")
    feedback = st.text_area("Feedback / Suggestions:")
    if st.button("Submit Feedback") and feedback:
        st.success("Thank you for your feedback!")

if not alerts:
    st.warning("No signals found for today. Run `vol_signal_layer.py` first.")
else:
    df = pd.DataFrame(alerts)

    # Score logic
    def compute_threat_level(row):
        score = 0
        if row.get('iv_break'): score += 1
        if row.get('sentiment_disruption'): score += 1
        if row.get('xrt_si_spike'): score += 1
        if row.get('xrt_redemptions'): score += 1
        if row.get('regsho_flag'): score += 1
        return score

    df["signal_score"] = df.apply(compute_threat_level, axis=1)
    df["alert_level"] = df["signal_score"].apply(
        lambda x: "🟢 Watch" if x <= 1 else "🟡 Tension" if x == 2 else "🔴 Breakout Potential"
    )

    # Filtering
    filtered_df = df.copy()
    if filter_ticker:
        filtered_df = filtered_df[filtered_df['ticker'].str.contains(filter_ticker, case=False, na=False)]
    if filter_level != "All":
        filtered_df = filtered_df[filtered_df['alert_level'] == filter_level]

    # Display table
    st.dataframe(
        filtered_df[[
            "ticker", "iv_rank", "rv_iv_spread", "sentiment_score",
            "regsho_flag", "xrt_si_spike", "xrt_redemptions", "alert_level"
        ]],
        height=400,
        use_container_width=True
    )

    # Export table
    if export_table:
        st.download_button("Download Table CSV", filtered_df.to_csv(index=False), file_name="volcon_signals.csv", mime="text/csv")
    # Export filtered alerts
    if export_alerts:
        st.download_button("Download Filtered Alerts", filtered_df.to_json(orient='records', indent=2), file_name="filtered_alerts.json", mime="application/json")

    # Highlight breakout candidates
    breakout_df = filtered_df[filtered_df["alert_level"] == "🔴 Breakout Potential"]
    if not breakout_df.empty:
        st.subheader("🚨 High-Signal Candidates")
        for _, row in breakout_df.iterrows():
            st.markdown(f"""
            - **{row['ticker']}** shows multiple containment breach triggers.
            - 📈 IV-RV spread: `{row['rv_iv_spread']:.3f}`
            - 🧠 Sentiment Score: `{row['sentiment_score']:.2f}`
            - 🔻 RegSHO: `{row['regsho_flag']}`, ETF Compression: `{row['xrt_si_spike']}` / `{row['xrt_redemptions']}`
            """)

    # Analytics
    if show_analytics:
        st.subheader("📈 Signal Analytics")
        st.write(filtered_df.describe())
        st.write("**Alert Level Counts:**")
        st.bar_chart(filtered_df['alert_level'].value_counts())
        st.write("**IV-RV Spread Distribution:**")
        fig_iv, ax_iv = plt.subplots()
        ax_iv.hist(filtered_df['rv_iv_spread'].dropna(), bins=20, color='skyblue', edgecolor='black')
        ax_iv.set_xlabel('RV-IV Spread')
        ax_iv.set_ylabel('Frequency')
        st.pyplot(fig_iv)

        st.write("**Sentiment Score Distribution:**")
        fig_sent, ax_sent = plt.subplots()
        ax_sent.hist(filtered_df['sentiment_score'].dropna(), bins=20, color='salmon', edgecolor='black')
        ax_sent.set_xlabel('Sentiment Score')
        ax_sent.set_ylabel('Frequency')
        st.pyplot(fig_sent)
        # Rolling volatility
        if 'signal_score' in filtered_df.columns:
            vol_window = st.number_input("Rolling Volatility Window (days)", min_value=2, max_value=30, value=5)
            filtered_df['rolling_vol'] = filtered_df['signal_score'].rolling(window=vol_window).std()
            st.write("**Rolling Volatility:**")
            st.line_chart(filtered_df['rolling_vol'])
            # Z-score
            mean = filtered_df['signal_score'].mean()
            std = filtered_df['signal_score'].std()
            filtered_df['zscore'] = (filtered_df['signal_score'] - mean) / std if std != 0 else 0
            st.write("**Z-Score (Standardized Score):**")
            st.line_chart(filtered_df['zscore'])
            # Outlier detection
            outlier_thresh = st.number_input("Outlier Z-Score Threshold", min_value=1.0, max_value=5.0, value=2.0)
            outliers = filtered_df[filtered_df['zscore'].abs() > outlier_thresh]
            st.write(f"Outliers (|z| > {outlier_thresh}): {len(outliers)}")
            if not outliers.empty:
                st.dataframe(outliers[['ticker', 'signal_score', 'zscore']])
            # Quantile stats
            st.write("**Quantile Stats:**")
            quantiles = filtered_df['signal_score'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            st.write(quantiles)
            # Interactive histogram
            st.write("**Signal Score Distribution (Interactive):**")
            import numpy as np
            import matplotlib.pyplot as plt
            bins = st.slider("Histogram Bins", min_value=5, max_value=100, value=20)
            fig, ax = plt.subplots()
            ax.hist(filtered_df['signal_score'].dropna(), bins=bins, color='skyblue', edgecolor='black')
            ax.set_xlabel('Signal Score')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            # Correlation matrix
            numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 1:
                st.write("**Correlation Matrix:**")
                st.dataframe(filtered_df[numeric_cols].corr())
            # Autocorrelation
            st.write("**Autocorrelation (Signal Score):**")
            from pandas.plotting import autocorrelation_plot
            fig2, ax2 = plt.subplots()
            autocorrelation_plot(filtered_df['signal_score'].dropna(), ax=ax2)
            st.pyplot(fig2)
            # Seasonality decomposition
            st.write("**Seasonality Decomposition (Signal Score):**")
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                freq = st.number_input("Decomposition Period (days)", min_value=2, max_value=30, value=7)
                decomposition = seasonal_decompose(filtered_df['signal_score'].dropna(), period=freq, model='additive')
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
            # Trend regression
            st.write("**Signal Score Trend Regression:**")
            try:
                from sklearn.linear_model import LinearRegression
                x = np.arange(len(filtered_df['signal_score'].dropna())).reshape(-1, 1)
                y = filtered_df['signal_score'].dropna().values
                model = LinearRegression().fit(x, y)
                trend = model.predict(x)
                fig4, ax4 = plt.subplots()
                ax4.plot(filtered_df['signal_score'].index[:len(trend)], y, label='Score')
                ax4.plot(filtered_df['signal_score'].index[:len(trend)], trend, label='Trend', color='red')
                ax4.legend()
                ax4.set_title('Signal Score & Trend')
                st.pyplot(fig4)
            except Exception as e:
                st.info(f"Trend regression unavailable: {e}")
            # PCA for dimensionality reduction
            st.write("**PCA (Dimensionality Reduction):**")
            try:
                from sklearn.decomposition import PCA
                pca_cols = [col for col in numeric_cols if col != 'signal_score']
                if len(pca_cols) >= 2:
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(filtered_df[pca_cols].dropna())
                    fig_pca, ax_pca = plt.subplots()
                    ax_pca.scatter(pca_result[:,0], pca_result[:,1], c=filtered_df['signal_score'].dropna(), cmap='viridis')
                    ax_pca.set_xlabel('PC1')
                    ax_pca.set_ylabel('PC2')
                    ax_pca.set_title('PCA of Features')
                    st.pyplot(fig_pca)
                else:
                    st.info("Not enough numeric columns for PCA.")
            except Exception as e:
                st.info(f"PCA unavailable: {e}")
            # Clustering
            st.write("**KMeans Clustering (Signal Score):**")
            try:
                from sklearn.cluster import KMeans
                if len(filtered_df) >= 3:
                    kmeans = KMeans(n_clusters=3)
                    clusters = kmeans.fit_predict(filtered_df[['signal_score']].fillna(0))
                    filtered_df['cluster'] = clusters
                    st.write(filtered_df[['ticker', 'signal_score', 'cluster']])
                    st.bar_chart(filtered_df['cluster'].value_counts())
                else:
                    st.info("Not enough data for clustering.")
            except Exception as e:
                st.info(f"Clustering unavailable: {e}")
            # Anomaly detection
            st.write("**Isolation Forest Anomaly Detection:**")
            try:
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination=0.1)
                filtered_df['anomaly'] = iso.fit_predict(filtered_df[['signal_score']].fillna(0))
                st.write(filtered_df[['ticker', 'signal_score', 'anomaly']])
                st.bar_chart(filtered_df['anomaly'].value_counts())
            except Exception as e:
                st.info(f"Anomaly detection unavailable: {e}")
            # t-SNE visualization
            st.write("**t-SNE Visualization:**")
            try:
                from sklearn.manifold import TSNE
                tsne_cols = [col for col in numeric_cols if col != 'signal_score']
                if len(tsne_cols) >= 2:
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(filtered_df[tsne_cols].dropna())
                    fig_tsne, ax_tsne = plt.subplots()
                    ax_tsne.scatter(tsne_result[:,0], tsne_result[:,1], c=filtered_df['signal_score'].dropna(), cmap='plasma')
                    ax_tsne.set_xlabel('TSNE-1')
                    ax_tsne.set_ylabel('TSNE-2')
                    ax_tsne.set_title('t-SNE of Features')
                    st.pyplot(fig_tsne)
                else:
                    st.info("Not enough numeric columns for t-SNE.")
            except Exception as e:
                st.info(f"t-SNE unavailable: {e}")
            # DBSCAN clustering
            st.write("**DBSCAN Clustering (Signal Score):**")
            try:
                from sklearn.cluster import DBSCAN
                if len(filtered_df) >= 3:
                    dbscan = DBSCAN(eps=0.5, min_samples=2)
                    db_labels = dbscan.fit_predict(filtered_df[['signal_score']].fillna(0))
                    filtered_df['dbscan_cluster'] = db_labels
                    st.write(filtered_df[['ticker', 'signal_score', 'dbscan_cluster']])
                    st.bar_chart(filtered_df['dbscan_cluster'].value_counts())
                else:
                    st.info("Not enough data for DBSCAN clustering.")
            except Exception as e:
                st.info(f"DBSCAN clustering unavailable: {e}")
            # Feature importance (Random Forest)
            st.write("**Feature Importance (Random Forest):**")
            try:
                from sklearn.ensemble import RandomForestRegressor
                fi_cols = [col for col in numeric_cols if col != 'signal_score']
                if len(fi_cols) >= 1:
                    rf = RandomForestRegressor()
                    rf.fit(filtered_df[fi_cols].fillna(0), filtered_df['signal_score'].fillna(0))
                    importances = rf.feature_importances_
                    fi_df = pd.DataFrame({'feature': fi_cols, 'importance': importances})
                    st.bar_chart(fi_df.set_index('feature')['importance'])
                    # SHAP feature explanation
                    st.write("**Feature Explanation (SHAP):**")
                    try:
                        import shap
                        explainer = shap.TreeExplainer(rf)
                        shap_values = explainer.shap_values(filtered_df[fi_cols].fillna(0))
                        fig_shap = shap.summary_plot(shap_values, filtered_df[fi_cols].fillna(0), show=False)
                        st.pyplot(fig_shap)
                    except Exception as e:
                        st.info(f"SHAP explanation unavailable: {e}")
                else:
                    st.info("Not enough features for importance analysis.")
            except Exception as e:
                st.info(f"Feature importance unavailable: {e}")
            # Time series forecasting (Prophet)
            # UMAP embedding
            st.write("**UMAP Embedding:**")
            try:
                import umap
                umap_cols = [col for col in numeric_cols if col != 'signal_score']
                if len(umap_cols) >= 2:
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    umap_result = reducer.fit_transform(filtered_df[umap_cols].dropna())
                    fig_umap, ax_umap = plt.subplots()
                    ax_umap.scatter(umap_result[:,0], umap_result[:,1], c=filtered_df['signal_score'].dropna(), cmap='cool')
                    ax_umap.set_xlabel('UMAP-1')
                    ax_umap.set_ylabel('UMAP-2')
                    ax_umap.set_title('UMAP Embedding of Features')
                    st.pyplot(fig_umap)
                else:
                    st.info("Not enough numeric columns for UMAP.")
            except Exception as e:
                st.info(f"UMAP unavailable: {e}")
            # Hierarchical clustering
            st.write("**Hierarchical Clustering (Signal Score):**")
            try:
                from scipy.cluster.hierarchy import linkage, dendrogram
                if len(filtered_df) >= 3:
                    Z = linkage(filtered_df[['signal_score']].fillna(0), method='ward')
                    fig_hc, ax_hc = plt.subplots(figsize=(8, 4))
                    dendrogram(Z, ax=ax_hc)
                    ax_hc.set_title('Hierarchical Clustering Dendrogram')
                    st.pyplot(fig_hc)
                else:
                    st.info("Not enough data for hierarchical clustering.")
            except Exception as e:
                st.info(f"Hierarchical clustering unavailable: {e}")
            # Granger causality test
            # Heatmap of correlations
            st.write("**Correlation Heatmap:**")
            try:
                import seaborn as sns
                if len(numeric_cols) > 1:
                    fig_heat, ax_heat = plt.subplots()
                    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax_heat)
                    ax_heat.set_title('Correlation Heatmap')
                    st.pyplot(fig_heat)
                else:
                    st.info("Not enough numeric columns for heatmap.")
            except Exception as e:
                st.info(f"Heatmap unavailable: {e}")
            # Pairplot
            st.write("**Pairplot of Numeric Features:**")
            try:
                import seaborn as sns
                if len(numeric_cols) > 1:
                    fig_pair = sns.pairplot(filtered_df[numeric_cols].dropna())
                    st.pyplot(fig_pair)
                else:
                    st.info("Not enough numeric columns for pairplot.")
            except Exception as e:
                st.info(f"Pairplot unavailable: {e}")
            # Parallel coordinates plot
            st.write("**Parallel Coordinates Plot (Feature Comparison):**")
            try:
                import matplotlib.pyplot as plt
                from pandas.plotting import parallel_coordinates
                pc_cols = [col for col in numeric_cols if col != 'signal_score']
                if len(pc_cols) >= 2 and 'alert_level' in filtered_df.columns:
                    pc_df = filtered_df[pc_cols + ['alert_level']].dropna()
                    fig_pc, ax_pc = plt.subplots(figsize=(8, 4))
                    parallel_coordinates(pc_df, 'alert_level', ax=ax_pc, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    ax_pc.set_title('Parallel Coordinates by Alert Level')
                    st.pyplot(fig_pc)
                else:
                    st.info("Not enough numeric columns for parallel coordinates plot.")
            except Exception as e:
                st.info(f"Parallel coordinates plot unavailable: {e}")

            # Feature selection tool for custom analytics
            st.write("**Custom Feature Analytics:**")
            selected_features = st.multiselect("Select features for analysis", numeric_cols, default=numeric_cols[:2])
            if len(selected_features) >= 2:
                try:
                    import seaborn as sns
                    fig_custom, ax_custom = plt.subplots()
                    sns.pairplot(filtered_df[selected_features].dropna())
                    st.pyplot(fig_custom)
                except Exception as e:
                    st.info(f"Custom feature analytics unavailable: {e}")
            else:
                st.info("Select at least two features for custom analytics.")

            # SHAP force plot for model interpretability
            st.write("**SHAP Force Plot (Random Forest):**")
            try:
                import shap
                fi_cols = [col for col in numeric_cols if col != 'signal_score']
                if len(fi_cols) >= 1:
                    from sklearn.ensemble import RandomForestRegressor
                    rf = RandomForestRegressor()
                    rf.fit(filtered_df[fi_cols].fillna(0), filtered_df['signal_score'].fillna(0))
                    explainer = shap.TreeExplainer(rf)
                    shap_values = explainer.shap_values(filtered_df[fi_cols].fillna(0))
                    fig_force = shap.force_plot(explainer.expected_value, shap_values, filtered_df[fi_cols].fillna(0), matplotlib=True, show=False)
                    st.pyplot(fig_force)
                else:
                    st.info("Not enough features for SHAP force plot.")
            except Exception as e:
                st.info(f"SHAP force plot unavailable: {e}")

            # Volatility clustering visualization
            st.write("**Volatility Clustering Visualization:**")
            try:
                import matplotlib.pyplot as plt
                if 'signal_score' in filtered_df.columns:
                    rolling_std = filtered_df['signal_score'].rolling(window=5).std()
                    fig_volc, ax_volc = plt.subplots()
                    ax_volc.plot(filtered_df.index, rolling_std, label='Rolling Std (Volatility)')
                    ax_volc.set_title('Volatility Clustering')
                    ax_volc.set_xlabel('Index')
                    ax_volc.set_ylabel('Rolling Std')
                    ax_volc.legend()
                    st.pyplot(fig_volc)
                else:
                    st.info("No signal_score column for volatility clustering.")
            except Exception as e:
                st.info(f"Volatility clustering unavailable: {e}")

            # Macroeconomic overlay chart
            st.write("**Macroeconomic Overlay Chart:**")
            try:
                macro_col = st.selectbox("Select macro column to overlay", [col for col in filtered_df.columns if 'macro' in col.lower()] + ['None'])
                if macro_col != 'None' and macro_col in filtered_df.columns:
                    fig_macro, ax_macro = plt.subplots()
                    ax_macro.plot(filtered_df.index, filtered_df['signal_score'], label='Signal Score')
                    ax_macro.plot(filtered_df.index, filtered_df[macro_col], label=macro_col)
                    ax_macro.set_title('Signal Score & Macro Overlay')
                    ax_macro.legend()
                    st.pyplot(fig_macro)
                else:
                    st.info("No macro column selected or available.")
            except Exception as e:
                st.info(f"Macro overlay unavailable: {e}")

            # News sentiment timeline
            st.write("**News Sentiment Timeline:**")
            try:
                if 'news_sentiment' in filtered_df.columns and 'date' in filtered_df.columns:
                    news_df = filtered_df[['date', 'news_sentiment']].dropna().copy()
                    news_df['date'] = pd.to_datetime(news_df['date'])
                    fig_news, ax_news = plt.subplots()
                    ax_news.plot(news_df['date'], news_df['news_sentiment'], marker='o')
                    ax_news.set_title('News Sentiment Over Time')
                    ax_news.set_xlabel('Date')
                    ax_news.set_ylabel('Sentiment')
                    st.pyplot(fig_news)
                else:
                    st.info("No news_sentiment or date column for timeline.")
            except Exception as e:
                st.info(f"News sentiment timeline unavailable: {e}")

            # Custom alert builder
            st.write("**Custom Alert Builder:**")
            try:
                custom_score = st.slider("Custom Signal Score Threshold", min_value=0, max_value=5, value=3)
                custom_alerts = filtered_df[filtered_df['signal_score'] >= custom_score]
                st.write(f"Custom alerts (score >= {custom_score}): {len(custom_alerts)}")
                if not custom_alerts.empty:
                    st.dataframe(custom_alerts[['ticker', 'signal_score', 'alert_level']])
            except Exception as e:
                st.info(f"Custom alert builder unavailable: {e}")

            # Data quality diagnostics
            st.write("**Data Quality Diagnostics:**")
            try:
                missing = filtered_df.isnull().sum()
                st.write("Missing values per column:")
                st.write(missing)
                st.write("Duplicate rows:", filtered_df.duplicated().sum())
                st.write("Rows with out-of-range signal_score:", (filtered_df['signal_score'] < 0).sum() + (filtered_df['signal_score'] > 5).sum())
            except Exception as e:
                st.info(f"Data quality diagnostics unavailable: {e}")
            # Interactive feature correlation explorer
            st.write("**Interactive Feature Correlation Explorer:**")
            try:
                corr_features = st.multiselect("Select features for correlation", numeric_cols, default=numeric_cols[:2])
                if len(corr_features) >= 2:
                    corr_matrix = filtered_df[corr_features].corr()
                    import seaborn as sns
                    fig_corr, ax_corr = plt.subplots()
                    sns.heatmap(corr_matrix, annot=True, cmap='Spectral', ax=ax_corr)
                    ax_corr.set_title('Selected Feature Correlations')
                    st.pyplot(fig_corr)
                else:
                    st.info("Select at least two features for correlation explorer.")
            except Exception as e:
                st.info(f"Correlation explorer unavailable: {e}")

            # Anomaly timeline plot
            st.write("**Anomaly Timeline Plot:**")
            try:
                if 'anomaly' in filtered_df.columns and 'date' in filtered_df.columns:
                    anomaly_df = filtered_df[['date', 'anomaly']].dropna().copy()
                    anomaly_df['date'] = pd.to_datetime(anomaly_df['date'])
                    fig_anom, ax_anom = plt.subplots()
                    ax_anom.plot(anomaly_df['date'], anomaly_df['anomaly'], marker='o', linestyle='-')
                    ax_anom.set_title('Anomaly Timeline')
                    ax_anom.set_xlabel('Date')
                    ax_anom.set_ylabel('Anomaly')
                    st.pyplot(fig_anom)
                else:
                    st.info("No anomaly or date column for timeline plot.")
            except Exception as e:
                st.info(f"Anomaly timeline unavailable: {e}")

            # Rolling quantile chart
            st.write("**Rolling Quantile Chart (Signal Score):**")
            try:
                quantile_window = st.number_input("Rolling Quantile Window (days)", min_value=2, max_value=30, value=7)
                q25 = filtered_df['signal_score'].rolling(window=quantile_window).quantile(0.25)
                q50 = filtered_df['signal_score'].rolling(window=quantile_window).quantile(0.5)
                q75 = filtered_df['signal_score'].rolling(window=quantile_window).quantile(0.75)
                fig_q, ax_q = plt.subplots()
                ax_q.plot(filtered_df.index, q25, label='25th Quantile')
                ax_q.plot(filtered_df.index, q50, label='50th Quantile')
                ax_q.plot(filtered_df.index, q75, label='75th Quantile')
                ax_q.set_title('Rolling Quantiles of Signal Score')
                ax_q.legend()
                st.pyplot(fig_q)
            except Exception as e:
                st.info(f"Rolling quantile chart unavailable: {e}")

            # Custom clustering controls
            st.write("**Custom Clustering Controls:**")
            try:
                cluster_method = st.selectbox("Clustering Method", ["KMeans", "DBSCAN", "Agglomerative"])
                n_clusters = st.number_input("Number of Clusters (KMeans/Agglomerative)", min_value=2, max_value=10, value=3)
                if cluster_method == "KMeans":
                    from sklearn.cluster import KMeans
                    clusters = KMeans(n_clusters=n_clusters).fit_predict(filtered_df[['signal_score']].fillna(0))
                elif cluster_method == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    clusters = DBSCAN(eps=0.5, min_samples=2).fit_predict(filtered_df[['signal_score']].fillna(0))
                else:
                    from sklearn.cluster import AgglomerativeClustering
                    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(filtered_df[['signal_score']].fillna(0))
                filtered_df['custom_cluster'] = clusters
                st.write(filtered_df[['ticker', 'signal_score', 'custom_cluster']])
                st.bar_chart(filtered_df['custom_cluster'].value_counts())
            except Exception as e:
                st.info(f"Custom clustering unavailable: {e}")

            # Advanced export options
            st.write("**Advanced Export Options:**")
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
            if st.button("Export Custom Alerts"):
                try:
                    if export_format == "CSV":
                        st.download_button("Download Custom Alerts CSV", filtered_df.to_csv(index=False), file_name="custom_alerts.csv", mime="text/csv")
                    elif export_format == "JSON":
                        st.download_button("Download Custom Alerts JSON", filtered_df.to_json(orient='records', indent=2), file_name="custom_alerts.json", mime="application/json")
                    else:
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            filtered_df.to_excel(writer, index=False)
                        st.download_button("Download Custom Alerts Excel", output.getvalue(), file_name="custom_alerts.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.info(f"Export unavailable: {e}")

            # Model comparison (Random Forest vs. XGBoost)
            st.write("**Model Comparison: Random Forest vs. XGBoost:**")
            try:
                fi_cols = [col for col in numeric_cols if col != 'signal_score']
                if len(fi_cols) >= 1:
                    from sklearn.ensemble import RandomForestRegressor
                    rf = RandomForestRegressor()
                    rf.fit(filtered_df[fi_cols].fillna(0), filtered_df['signal_score'].fillna(0))
                    rf_score = rf.score(filtered_df[fi_cols].fillna(0), filtered_df['signal_score'].fillna(0))
                    try:
                        from xgboost import XGBRegressor
                        xgb = XGBRegressor()
                        xgb.fit(filtered_df[fi_cols].fillna(0), filtered_df['signal_score'].fillna(0))
                        xgb_score = xgb.score(filtered_df[fi_cols].fillna(0), filtered_df['signal_score'].fillna(0))
                        st.write(f"Random Forest R^2: {rf_score:.3f}")
                        st.write(f"XGBoost R^2: {xgb_score:.3f}")
                    except Exception as e:
                        st.info(f"XGBoost unavailable: {e}")
                else:
                    st.info("Not enough features for model comparison.")
            except Exception as e:
                st.info(f"Model comparison unavailable: {e}")

            # Explainable AI dashboard section
            st.write("**Explainable AI Dashboard:**")
            try:
                import shap
                fi_cols = [col for col in numeric_cols if col != 'signal_score']
                if len(fi_cols) >= 1:
                    from sklearn.ensemble import RandomForestRegressor
                    rf = RandomForestRegressor()
                    rf.fit(filtered_df[fi_cols].fillna(0), filtered_df['signal_score'].fillna(0))
                    explainer = shap.TreeExplainer(rf)
                    shap_values = explainer.shap_values(filtered_df[fi_cols].fillna(0))
                    st.write("SHAP Summary Plot:")
                    fig_shap = shap.summary_plot(shap_values, filtered_df[fi_cols].fillna(0), show=False)
                    st.pyplot(fig_shap)
                else:
                    st.info("Not enough features for explainable AI.")
            except Exception as e:
                st.info(f"Explainable AI dashboard unavailable: {e}")

            # Interactive dashboard theme switcher
            st.write("**Dashboard Theme Switcher:**")
            theme = st.selectbox("Select Dashboard Theme", ["Light", "Dark", "Solarized"])
            st.write(f"Current theme: {theme}")
            # (Note: Streamlit theme switching is limited, but you can use st.markdown for custom CSS)
            if theme == "Dark":
                st.markdown("""
                    <style>
                    .stApp { background-color: #222; color: #eee; }
                    </style>
                """, unsafe_allow_html=True)
            elif theme == "Solarized":
                st.markdown("""
                    <style>
                    .stApp { background-color: #fdf6e3; color: #657b83; }
                    </style>
                """, unsafe_allow_html=True)

            # Real-time alert notification panel
            st.write("**Real-Time Alert Notification Panel:**")
            try:
                if not breakout_df.empty:
                    st.success(f"🚨 {len(breakout_df)} high-signal alerts! Check candidates above.")
                else:
                    st.info("No high-signal alerts at this time.")
            except Exception as e:
                st.info(f"Notification panel unavailable: {e}")

            # API health check diagnostics
            st.write("**API Health Check Diagnostics:**")
            try:
                api_status = {}
                import requests
                # Finnhub
                if finnhub_api_key:
                    resp = requests.get(f"https://finnhub.io/api/v1/quote?symbol=XRT&token={finnhub_api_key}")
                    api_status['Finnhub'] = resp.status_code
                # Alpha Vantage
                if alpha_vantage_key:
                    resp = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=XRT&apikey={alpha_vantage_key}")
                    api_status['Alpha Vantage'] = resp.status_code
                # Quandl
                if quandl_key:
                    try:
                        import quandl
                        quandl.ApiConfig.api_key = quandl_key
                        data = quandl.get("EOD/XRT", rows=1)
                        api_status['Quandl'] = "OK" if not data.empty else "No Data"
                    except Exception as e:
                        api_status['Quandl'] = f"Error: {e}"
                st.write(api_status)
            except Exception as e:
                st.info(f"API health check unavailable: {e}")
            try:
                import seaborn as sns
                if 'alert_level' in filtered_df.columns:
                    fig_violin, ax_violin = plt.subplots()
                    sns.violinplot(x='alert_level', y='signal_score', data=filtered_df, ax=ax_violin)
                    ax_violin.set_title('Signal Score by Alert Level')
                    st.pyplot(fig_violin)
                else:
                    st.info("No alert_level column for violin plot.")
            except Exception as e:
                st.info(f"Violin plot unavailable: {e}")
            # Box plot
            st.write("**Box Plot (Signal Score by Alert Level):**")
            try:
                import seaborn as sns
                if 'alert_level' in filtered_df.columns:
                    fig_box, ax_box = plt.subplots()
                    sns.boxplot(x='alert_level', y='signal_score', data=filtered_df, ax=ax_box)
                    ax_box.set_title('Signal Score by Alert Level')
                    st.pyplot(fig_box)
                else:
                    st.info("No alert_level column for box plot.")
            except Exception as e:
                st.info(f"Box plot unavailable: {e}")
            # Autocorrelation heatmap
            st.write("**Autocorrelation Heatmap (Signal Score):**")
            try:
                import seaborn as sns
                from statsmodels.tsa.stattools import acf
                acf_vals = acf(filtered_df['signal_score'].dropna(), nlags=20)
                fig_acf, ax_acf = plt.subplots()
                sns.heatmap([acf_vals], annot=True, cmap='YlGnBu', ax=ax_acf)
                ax_acf.set_title('Autocorrelation Heatmap')
                st.pyplot(fig_acf)
            except Exception as e:
                st.info(f"Autocorrelation heatmap unavailable: {e}")
            st.write("**Lag Plot (Signal Score):**")
            try:
                from pandas.plotting import lag_plot
                fig_lag, ax_lag = plt.subplots()
                lag_plot(filtered_df['signal_score'].dropna(), ax=ax_lag)
                ax_lag.set_title('Lag Plot of Signal Score')
                st.pyplot(fig_lag)
            except Exception as e:
                st.info(f"Lag plot unavailable: {e}")
            st.write("**Granger Causality Test:**")
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                if 'date' in filtered_df.columns and 'sentiment_score' in filtered_df.columns:
                    granger_df = filtered_df[['date', 'signal_score', 'sentiment_score']].dropna().copy()
                    granger_df['date'] = pd.to_datetime(granger_df['date'])
                    granger_df = granger_df.sort_values('date')
                    test_result = grangercausalitytests(granger_df[['signal_score', 'sentiment_score']], maxlag=2, verbose=False)
                    st.write("Granger causality test (signal_score causes sentiment_score):")
                    st.write(test_result[1][0]['ssr_ftest'])
                else:
                    st.info("Need date and sentiment_score columns for Granger test.")
            except Exception as e:
                st.info(f"Granger causality test unavailable: {e}")
            st.write("**Signal Score Forecasting (Prophet):**")
            try:
                from prophet import Prophet
                if 'date' in filtered_df.columns:
                    ts_df = filtered_df[['date', 'signal_score']].dropna().copy()
                    ts_df['date'] = pd.to_datetime(ts_df['date'])
                    ts_df = ts_df.rename(columns={'date': 'ds', 'signal_score': 'y'})
                    m = Prophet()
                    m.fit(ts_df)
                    future = m.make_future_dataframe(periods=7)
                    forecast = m.predict(future)
                    import matplotlib.pyplot as plt
                    fig_prophet, ax_prophet = plt.subplots()
                    ax_prophet.plot(ts_df['ds'], ts_df['y'], label='Actual')
                    ax_prophet.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
                    ax_prophet.legend()
                    ax_prophet.set_title('Signal Score Forecast (Prophet)')
                    st.pyplot(fig_prophet)
                else:
                    st.info("No date column for forecasting.")
            except Exception as e:
                st.info(f"Prophet forecasting unavailable: {e}")
        # Optional: Finnhub integration for real-time price and news
        # Google Trends integration for search interest
        with st.expander("Google Trends Search Interest"):
            trends_query = st.text_input("Google Trends Query", value="XRT")
            if st.button("Fetch Google Trends Data"):
                try:
                    from pytrends.request import TrendReq
                    pytrends = TrendReq()
                    pytrends.build_payload([trends_query], timeframe='today 3-m')
                    trends_data = pytrends.interest_over_time()
                    if not trends_data.empty:
                        import matplotlib.pyplot as plt
                        fig_trends, ax_trends = plt.subplots()
                        ax_trends.plot(trends_data.index, trends_data[trends_query])
                        ax_trends.set_title(f"Google Trends: {trends_query}")
                        ax_trends.set_xlabel('Date')
                        ax_trends.set_ylabel('Search Interest')
                        plt.xticks(rotation=45)
                        st.pyplot(fig_trends)
                    else:
                        st.write("No Google Trends data found.")
                except Exception as e:
                    st.error(f"Could not fetch Google Trends data: {e}")
        # Twitter API integration for sentiment/news
        with st.expander("Twitter Sentiment & News Integration"):
            twitter_bearer = st.text_input("Twitter Bearer Token", value="")
            twitter_query = st.text_input("Twitter Search Query", value="$XRT")
            if st.button("Fetch Twitter News & Sentiment") and twitter_bearer:
                try:
                    import requests
                    headers = {"Authorization": f"Bearer {twitter_bearer}"}
                    url = f"https://api.twitter.com/2/tweets/search/recent?query={twitter_query}&max_results=10&tweet.fields=created_at,text"
                    resp = requests.get(url, headers=headers)
                    tweets = resp.json().get('data', [])
                    if tweets:
                        st.write("**Recent Tweets:**")
                        for tweet in tweets:
                            st.markdown(f"- {tweet['created_at']}: {tweet['text']}")
                    else:
                        st.write("No tweets found.")
                except Exception as e:
                    st.error(f"Could not fetch Twitter data: {e}")
        st.markdown("---")
        st.subheader("📉 Price, News & Macro Integrations")
        ticker_for_price = st.text_input("Enter ticker for price/news (Finnhub/Alpha Vantage/Quandl/Polygon.io)", value="XRT")
        finnhub_api_key = st.text_input("Finnhub API Key (for price/news)", value="")
        alpha_vantage_key = st.text_input("Alpha Vantage API Key (for price)", value="")
        quandl_key = st.text_input("Quandl API Key (for price)", value="")
        polygon_key = st.text_input("Polygon.io API Key (for price)", value="")
        fred_key = st.text_input("FRED API Key (for macro)", value="")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("Show Finnhub Price & News") and finnhub_api_key:
                try:
                    import requests
                    # Price
                    url = f"https://finnhub.io/api/v1/quote?symbol={ticker_for_price}&token={finnhub_api_key}"
                    resp = requests.get(url)
                    price_data = resp.json()
                    st.write(f"Current Price: {price_data.get('c', 'N/A')}")
                    # News
                    news_url = f"https://finnhub.io/api/v1/company-news?symbol={ticker_for_price}&from=2025-07-01&to=2025-07-25&token={finnhub_api_key}"
                    news_resp = requests.get(news_url)
                    news_items = news_resp.json()
                    if news_items:
                        st.write("**Recent News:**")
                        for item in news_items[:5]:
                            st.markdown(f"- [{item.get('headline')}]({item.get('url')}) ({item.get('datetime')})")
                    else:
                        st.write("No recent news found.")
                except Exception as e:
                    st.error(f"Could not fetch Finnhub data: {e}")
        with col2:
            if st.button("Show Alpha Vantage Price Chart") and alpha_vantage_key:
                try:
                    import requests
                    av_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker_for_price}&apikey={alpha_vantage_key}&outputsize=compact"
                    av_resp = requests.get(av_url)
                    av_data = av_resp.json()
                    ts = av_data.get('Time Series (Daily)', {})
                    if ts:
                        import matplotlib.pyplot as plt
                        dates = list(ts.keys())[:30][::-1]
                        closes = [float(ts[d]['4. close']) for d in dates]
                        fig_av, ax_av = plt.subplots()
                        ax_av.plot(dates, closes)
                        ax_av.set_title(f"Alpha Vantage Close Price: {ticker_for_price}")
                        ax_av.set_xlabel('Date')
                        ax_av.set_ylabel('Close Price')
                        plt.xticks(rotation=45)
                        st.pyplot(fig_av)
                    else:
                        st.write("No price data found.")
                except Exception as e:
                    st.error(f"Could not fetch Alpha Vantage data: {e}")
        with col3:
            if st.button("Show Quandl Price Chart") and quandl_key:
                try:
                    import quandl
                    quandl.ApiConfig.api_key = quandl_key
                    # Example: WIKI/{ticker} (for US stocks, deprecated, but for demo)
                    # Use a more current Quandl database if available
                    data = quandl.get(f"EOD/{ticker_for_price}", rows=30)
                    if not data.empty:
                        import matplotlib.pyplot as plt
                        fig_q, ax_q = plt.subplots()
                        ax_q.plot(data.index, data['Adj_Close'])
                        ax_q.set_title(f"Quandl Adj Close: {ticker_for_price}")
                        ax_q.set_xlabel('Date')
                        ax_q.set_ylabel('Adj Close Price')
                        plt.xticks(rotation=45)
                        st.pyplot(fig_q)
                    else:
                        st.write("No price data found.")
                except Exception as e:
                    st.error(f"Could not fetch Quandl data: {e}")
        with col4:
            if st.button("Show Polygon.io Price Chart") and polygon_key:
                try:
                    import requests
                    poly_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker_for_price}/range/1/day/2025-06-25/2025-07-25?adjusted=true&sort=desc&apiKey={polygon_key}"
                    poly_resp = requests.get(poly_url)
                    poly_data = poly_resp.json()
                    results = poly_data.get('results', [])
                    if results:
                        import matplotlib.pyplot as plt
                        dates = [pd.to_datetime(r['t'], unit='ms').strftime('%Y-%m-%d') for r in results][::-1]
                        closes = [r['c'] for r in results][::-1]
                        fig_poly, ax_poly = plt.subplots()
                        ax_poly.plot(dates, closes)
                        ax_poly.set_title(f"Polygon.io Close Price: {ticker_for_price}")
                        ax_poly.set_xlabel('Date')
                        ax_poly.set_ylabel('Close Price')
                        plt.xticks(rotation=45)
                        st.pyplot(fig_poly)
                    else:
                        st.write("No price data found.")
                except Exception as e:
                    st.error(f"Could not fetch Polygon.io data: {e}")
        with col5:
            if st.button("Show FRED Macro Chart") and fred_key:
                try:
                    import requests
                    fred_series = st.text_input("FRED Series ID (e.g. GDP, CPIAUCSL)", value="GDP")
                    fred_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={fred_series}&api_key={fred_key}&file_type=json"
                    fred_resp = requests.get(fred_url)
                    fred_data = fred_resp.json()
                    observations = fred_data.get('observations', [])
                    if observations:
                        import matplotlib.pyplot as plt
                        dates = [obs['date'] for obs in observations[-30:]]
                        values = [float(obs['value']) if obs['value'] not in ['.', None] else None for obs in observations[-30:]]
                        fig_fred, ax_fred = plt.subplots()
                        ax_fred.plot(dates, values)
                        ax_fred.set_title(f"FRED {fred_series} (Last 30 obs)")
                        ax_fred.set_xlabel('Date')
                        ax_fred.set_ylabel(fred_series)
                        plt.xticks(rotation=45)
                        st.pyplot(fig_fred)
                    else:
                        st.write("No macro data found.")
                except Exception as e:
                    st.error(f"Could not fetch FRED data: {e}")

    # Score chart
    if show_chart:
        st.subheader("📊 Signal Score Chart")
        if 'date' in filtered_df.columns:
            chart_df = filtered_df.sort_values('date')
            chart_df['date'] = pd.to_datetime(chart_df['date'])
            st.line_chart(chart_df.set_index('date')['signal_score'])
        else:
            st.line_chart(filtered_df['signal_score'])

    st.markdown("---")
    st.caption("Use this dashboard to coordinate long-vol entries and sentiment-aware exits based on VolCon disruption signals.")
