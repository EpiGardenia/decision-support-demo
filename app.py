# app.py
# Run:
#   pip install streamlit pandas numpy scikit-learn
#   streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Decision Support Demo (Rule vs ML)", layout="wide")

# -----------------------------
# Data generation (synthetic)
# -----------------------------
@st.cache_data
def generate_data(seed: int = 42, days: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regions = ["North", "South", "East", "West"]
    indicators = ["PM2.5", "NO2", "CO2"]
    start = pd.Timestamp("2025-01-01")
    dates = pd.date_range(start, periods=days, freq="D")

    rows = []
    for region in regions:
        for ind in indicators:
            base = rng.normal(50, 5)  # base level
            trend = rng.normal(0.02, 0.01)  # small drift

            # Seasonality-ish pattern
            t = np.arange(days)
            seasonal = 8.0 * np.sin(2 * np.pi * t / 60.0)

            noise = rng.normal(0, 3.0, size=days)
            values = base + trend * t + seasonal + noise

            # Inject anomalies (spikes/dips)
            anomaly_idx = rng.choice(days, size=max(3, days // 40), replace=False)
            for idx in anomaly_idx:
                values[idx] += rng.normal(20, 8) * (1 if rng.random() > 0.5 else -1)

            for d, v in zip(dates, values):
                rows.append((d, region, ind, float(v)))

    df = pd.DataFrame(rows, columns=["date", "region", "indicator", "value"])
    return df


df = generate_data()

# -----------------------------
# Feature engineering
# -----------------------------
def build_features(ts: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    ts: DataFrame with columns ['date', 'value'] sorted by date.
    """
    x = ts.copy()
    x = x.sort_values("date").reset_index(drop=True)

    x["delta"] = x["value"].diff()
    x["roll_mean"] = x["value"].rolling(window=window, min_periods=max(5, window // 3)).mean()
    x["roll_std"] = x["value"].rolling(window=window, min_periods=max(5, window // 3)).std()
    x["residual"] = x["value"] - x["roll_mean"]

    # Replace problematic std values
    eps = 1e-6
    x["roll_std"] = x["roll_std"].fillna(0.0)
    x.loc[x["roll_std"] < eps, "roll_std"] = eps

    # Fill NaNs in features
    x["delta"] = x["delta"].fillna(0.0)
    x["roll_mean"] = x["roll_mean"].fillna(x["value"].expanding().mean())
    x["residual"] = x["residual"].fillna(0.0)

    return x


# -----------------------------
# Rule-based anomaly detection
# -----------------------------
def rule_based_anomaly(feat: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    out = feat.copy()
    out["z_score"] = (out["value"] - out["roll_mean"]) / out["roll_std"]
    out["rule_anomaly"] = out["z_score"].abs() > z_threshold
    return out


# -----------------------------
# ML-based anomaly detection (Isolation Forest)
# -----------------------------
def ml_anomaly_isolation_forest(feat: pd.DataFrame, contamination: float, random_state: int = 7) -> pd.DataFrame:
    out = feat.copy()

    # Feature set for ML
    X = out[["value", "delta", "roll_mean", "roll_std", "residual"]].to_numpy()

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # predict: 1 = normal, -1 = anomaly
    pred = model.predict(X)
    score = model.decision_function(X)  # higher = more normal

    out["ml_anomaly"] = pred == -1
    out["ml_score"] = score
    return out


# -----------------------------
# UI
# -----------------------------
st.title("Rule-based vs ML-based Anomaly Detection (BI/Visualization Prototype)")

with st.sidebar:
    st.header("Data selection")
    region = st.selectbox("Region", sorted(df["region"].unique()))
    indicator = st.selectbox("Indicator", sorted(df["indicator"].unique()))

    st.header("Method")
    mode = st.radio("View", ["Rule", "ML", "Compare"], index=2)

    st.header("Parameters")
    window = st.slider("Rolling window", min_value=7, max_value=60, value=21, step=1)
    z_threshold = st.slider("Rule z-threshold", min_value=1.5, max_value=5.0, value=3.0, step=0.1)

    contamination = st.slider("ML contamination", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

# Filter time series
ts = df[(df["region"] == region) & (df["indicator"] == indicator)][["date", "value"]].copy()
ts = ts.sort_values("date")

feat = build_features(ts, window=window)
rule = rule_based_anomaly(feat, z_threshold=z_threshold)
ml = ml_anomaly_isolation_forest(feat, contamination=contamination)

# Merge outputs
merged = rule[["date", "value", "roll_mean", "roll_std", "z_score", "rule_anomaly"]].merge(
    ml[["date", "ml_anomaly", "ml_score"]],
    on="date",
    how="left",
)

# -----------------------------
# Visualization (simple + clear)
# -----------------------------
st.subheader(f"Time series: {region} / {indicator}")

chart_df = merged.set_index("date")[["value"]]
st.line_chart(chart_df)

col1, col2 = st.columns(2)

def anomalies_table(df_in: pd.DataFrame, flag_col: str, score_col: str | None = None) -> pd.DataFrame:
    x = df_in[df_in[flag_col]].copy()
    x = x.sort_values("date")
    cols = ["date", "value"]
    if score_col:
        cols.append(score_col)
    cols += ["roll_mean", "roll_std"]
    return x[cols].reset_index(drop=True)

if mode in ["Rule", "Compare"]:
    with col1:
        st.markdown("### Rule-based anomalies")
        st.caption("Rolling z-score based. Highly interpretable.")
        st.dataframe(anomalies_table(merged, "rule_anomaly", "z_score"), use_container_width=True)

if mode in ["ML", "Compare"]:
    with col2:
        st.markdown("### ML-based anomalies")
        st.caption("Isolation Forest on engineered features.")
        st.dataframe(anomalies_table(merged, "ml_anomaly", "ml_score"), use_container_width=True)

if mode == "Compare":
    st.markdown("### Comparison summary")
    both = (merged["rule_anomaly"] & merged["ml_anomaly"]).sum()
    rule_only = (merged["rule_anomaly"] & ~merged["ml_anomaly"]).sum()
    ml_only = (~merged["rule_anomaly"] & merged["ml_anomaly"]).sum()

    s1, s2, s3 = st.columns(3)
    s1.metric("Overlap (Rule ∩ ML)", int(both))
    s2.metric("Rule-only", int(rule_only))
    s3.metric("ML-only", int(ml_only))

    st.markdown("### Raw table (for debugging)")
    st.dataframe(merged, use_container_width=True)

