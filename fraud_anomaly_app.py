"""
fraud_anomaly_app.py

Streamlit app for anomaly detection on financial transactions.

Features:
- Use synthetic sample data or upload your own CSV.
- Preprocessing: datetime parsing, encoding categorical features, scaling.
- Models: IsolationForest, LocalOutlierFactor, Autoencoder (if tensorflow available).
- Visualizations: PCA scatter, time series of amounts with anomalies highlighted, distribution of anomaly scores.
- Download anomalies CSV.

Run:
    streamlit run fraud_anomaly_app.py
"""

import io
import math
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Optional tensorflow autoencoder
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, losses

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Anomaly Detection — Financial Transactions", layout="wide"
)


# -------------------------
# Utility: synthetic dataset
# -------------------------
def generate_synthetic_transactions(n=3000, start_date="2024-01-01"):
    rng = np.random.RandomState(42)
    start = pd.to_datetime(start_date)
    records = []
    merchants = [
        "Amazon",
        "Walmart",
        "Target",
        "Starbucks",
        "Uber",
        "Stripe",
        "Apple",
        "Shell",
        "LocalMarket",
    ]
    categories = [
        "grocery",
        "fuel",
        "restaurant",
        "transport",
        "shopping",
        "subscription",
        "tech",
    ]
    for i in range(n):
        # more transactions during daytime, random timestamps over 180 days
        t = start + pd.to_timedelta(rng.randint(0, 180 * 24 * 60), unit="m")
        merchant = rng.choice(
            merchants, p=[0.15, 0.15, 0.12, 0.12, 0.1, 0.12, 0.1, 0.08, 0.06]
        )
        category = rng.choice(categories)
        # Amount pattern: per category base + noise
        base = {
            "grocery": 40,
            "fuel": 60,
            "restaurant": 30,
            "transport": 15,
            "shopping": 80,
            "subscription": 12,
            "tech": 150,
        }[category]
        amount = max(1, rng.normal(base, base * 0.4))
        # Add occasional customer-specific larger transactions
        if rng.rand() < 0.02:
            amount *= rng.uniform(3, 10)
        # create label: 0 normal, 1 anomaly (hidden)
        label = 0
        # inject anomalies: weird large transactions, odd merchants, or at weird times
        if rng.rand() < 0.01:
            label = 1
            # either extremely large or negative/refund or unusual merchant
            if rng.rand() < 0.6:
                amount *= rng.uniform(10, 50)
            else:
                merchant = "UnknownVendor"
        records.append(
            {
                "transaction_id": f"tx_{i:06d}",
                "timestamp": t,
                "merchant": merchant,
                "category": category,
                "amount": round(float(amount), 2),
                "user_id": rng.randint(1, 500),  # multiple users
            }
        )
    df = pd.DataFrame.from_records(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# -------------------------
# Preprocessing utilities
# -------------------------
def preprocess(
    df,
    datetime_col="timestamp",
    numeric_cols=None,
    categorical_cols=None,
    drop_cols=None,
):
    df = df.copy()
    # Parse datetime
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    else:
        # if no timestamp, create artificial index-based timestamp
        df[datetime_col] = pd.date_range(start="2024-01-01", periods=len(df), freq="T")

    # Drop specified columns
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

    # Numeric selection
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Categorical selection
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # Remove datetime/numeric from categorical if present
    categorical_cols = [
        c for c in categorical_cols if c not in numeric_cols and c != datetime_col
    ]

    # Fillna
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    # Create time-based features
    df["_hour"] = df[datetime_col].dt.hour
    df["_dayofweek"] = df[datetime_col].dt.weekday
    df["_day"] = df[datetime_col].dt.day
    df["_month"] = df[datetime_col].dt.month

    # Build feature matrix:
    features = []
    # include numeric raw features
    for c in numeric_cols:
        if c in df.columns:
            features.append(c)
    # include engineered time features
    features += ["_hour", "_dayofweek", "_day", "_month"]

    # One-hot encode categorical small-cardinality columns
    encoded = pd.DataFrame(index=df.index)
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    # Only encode categorical columns with limited distinct values
    for c in categorical_cols:
        if c in df.columns:
            if df[c].nunique() <= 30 and df[c].nunique() > 1:
                vals = ohe.fit_transform(df[[c]])
                cols = [f"{c}__{cat}" for cat in ohe.categories_[0][1:]]
                encoded_tmp = pd.DataFrame(vals, columns=cols, index=df.index)
                encoded = pd.concat([encoded, encoded_tmp], axis=1)
            else:
                # for high-cardinality categories, skip or hash; we'll skip to keep things simple
                pass

    feature_df = pd.concat(
        [df[features].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1
    )
    # Keep scaler outside to allow inverse transform if needed
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.fillna(0))

    return df.reset_index(drop=True), feature_df.columns.tolist(), X, scaler


# -------------------------
# Model wrappers
# -------------------------
def run_isolation_forest(X, contamination=0.05, random_state=42, n_estimators=200):
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    labels = model.predict(X)
    scores = model.decision_function(X)
    return labels, scores, model


def run_lof(X, contamination=0.01, n_neighbors=20):
    # LOF returns -1 for outliers
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, novelty=False
    )
    labels = lof.fit_predict(X)
    # LOF negative_outlier_factor_ : lower means more abnormal, we invert
    scores = -lof.negative_outlier_factor_
    labels_binary = (labels == -1).astype(int)
    return labels_binary, scores, lof


def build_autoencoder(input_dim, encoding_dim=8):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(max(encoding_dim * 4, 32), activation="relu"),
            layers.Dense(max(encoding_dim * 2, 16), activation="relu"),
            layers.Dense(encoding_dim, activation="relu", name="bottleneck"),
            layers.Dense(max(encoding_dim * 2, 16), activation="relu"),
            layers.Dense(max(encoding_dim * 4, 32), activation="relu"),
            layers.Dense(input_dim, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def run_autoencoder(X_train, X_all, epochs=30, batch_size=128, encoding_dim=8):
    input_dim = X_train.shape[1]
    ae = build_autoencoder(input_dim, encoding_dim=encoding_dim)
    ae.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.05,
    )
    # reconstruction error as anomaly score
    recon = ae.predict(X_all)
    mse = np.mean(np.square(X_all - recon), axis=1)
    # Larger mse => more anomalous
    # label by threshold (e.g. top contamination fraction)
    return mse, ae


# -------------------------
# Streamlit UI
# -------------------------
st.title("🔍 Anomaly Detection for Financial Transactions")

st.markdown(
    """
Upload a CSV of transactions or use the synthetic dataset.  
Expected (recommended) columns: `transaction_id`, `timestamp` (ISO), `amount`, `merchant`, `category`, `user_id`.
The app will create features, run detectors, and highlight transactions flagged as anomalies.
"""
)

with st.sidebar:
    st.header("Options")
    use_sample = st.checkbox("Use synthetic sample dataset", value=True)
    uploaded = st.file_uploader("Or upload transactions CSV", type=["csv"])
    datetime_col = st.text_input("Datetime column name", value="timestamp")
    numeric_cols_input = st.text_input(
        "Numeric columns (comma separated) — leave blank to auto-detect", value="amount"
    )
    categorical_cols_input = st.text_input(
        "Categorical columns (comma separated)", value="merchant,category"
    )
    drop_cols_input = st.text_input(
        "Columns to drop (comma separated)", value="transaction_id"
    )
    model_choice = st.selectbox(
        "Detector",
        options=["IsolationForest", "LocalOutlierFactor"]
        + (["Autoencoder"] if TF_AVAILABLE else []),
    )
    contamination = st.slider(
        "Contamination (expected fraction of anomalies)",
        min_value=0.001,
        max_value=0.2,
        value=0.01,
        step=0.001,
    )
    run_button = st.button("Run detection")

# Load data
if use_sample and uploaded is None:
    df = generate_synthetic_transactions(n=3000)
    st.success("Loaded synthetic dataset (3k transactions).")
    st.dataframe(df.head(10))
else:
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df)} rows from uploaded CSV.")
            st.dataframe(df.head(8))
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()
    else:
        df = None
        st.info("Upload a CSV or toggle 'Use synthetic sample dataset'.")

# Parse inputs
numeric_cols = (
    [c.strip() for c in numeric_cols_input.split(",") if c.strip()]
    if numeric_cols_input
    else None
)
categorical_cols = (
    [c.strip() for c in categorical_cols_input.split(",") if c.strip()]
    if categorical_cols_input
    else None
)
drop_cols = (
    [c.strip() for c in drop_cols_input.split(",") if c.strip()]
    if drop_cols_input
    else None
)

# Run detection
if df is not None and run_button:
    st.subheader("Preprocessing")
    df0 = df.copy()
    df0, feature_names, X, scaler = preprocess(
        df0,
        datetime_col=datetime_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        drop_cols=drop_cols,
    )
    st.write("Feature columns used:", feature_names)
    st.write("Feature matrix shape:", X.shape)

    # Split for autoencoder training (train on presumed normal data)
    X_train_for_ae = None
    if model_choice == "Autoencoder":
        # Train AE on a subset (80%) assuming most are normal; for small contamination use train_test_split
        X_train_for_ae, X_val = train_test_split(X, test_size=0.2, random_state=42)

    st.subheader("Running detector")
    if model_choice == "IsolationForest":
        labels, scores, model = run_isolation_forest(
            X, contamination=contamination, random_state=42, n_estimators=200
        )
    elif model_choice == "LocalOutlierFactor":
        labels, scores, model = run_lof(X, contamination=contamination, n_neighbors=20)
    else:  # Autoencoder
        if not TF_AVAILABLE:
            st.error(
                "TensorFlow not available — install tensorflow to use Autoencoder."
            )
            st.stop()
        scores, ae = run_autoencoder(
            X_train_for_ae, X, epochs=30, batch_size=128, encoding_dim=8
        )
        # determine threshold by contamination quantile
        thr = np.quantile(scores, 1 - contamination)
        labels = (scores > thr).astype(int)

    # Attach results to dataframe
    df_results = df0.copy()
    df_results["anomaly_score"] = scores
    df_results["anomaly_label"] = labels  # 1 => anomaly

    n_anom = int(df_results["anomaly_label"].sum())
    st.success(
        f"Detection finished — flagged {n_anom} anomalies (out of {len(df_results)} rows)."
    )

    # Show top anomalies sorted by score
    st.subheader("Top anomalies")
    topk = st.number_input(
        "How many top anomalies to show", min_value=5, max_value=500, value=20, step=5
    )
    top_anoms = df_results.sort_values("anomaly_score", ascending=False).head(topk)
    st.dataframe(top_anoms.drop(columns=[]).reset_index(drop=True))

    # Visualization: PCA scatter of features colored by anomaly
    st.subheader("PCA scatter (2D) of feature space")
    try:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = {0: "#2ca02c", 1: "#d62728"}
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=[cmap[int(l)] for l in df_results["anomaly_label"]],
            alpha=0.6,
            s=20,
        )
        ax.set_title("PCA 2D projection (green=normal, red=anomaly)")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"PCA visualization failed: {e}")

    # Time-series plot of amounts with anomalies highlighted (if timestamp & amount available)
    st.subheader("Time series: amount with anomalies highlighted")
    if "amount" in df_results.columns and datetime_col in df_results.columns:
        ts_df = df_results[
            [datetime_col, "amount", "anomaly_label", "anomaly_score"]
        ].sort_values(datetime_col)
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(ts_df[datetime_col], ts_df["amount"], label="amount", alpha=0.6)
        anom_points = ts_df[ts_df["anomaly_label"] == 1]
        ax2.scatter(
            anom_points[datetime_col],
            anom_points["amount"],
            color="red",
            label="anomaly",
            s=30,
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amount")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("Time series plot requires columns 'amount' and the datetime column.")

    # Distribution of anomaly scores
    st.subheader("Anomaly score distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    sns.histplot(df_results["anomaly_score"], bins=60, ax=ax3, kde=False)
    ax3.axvline(
        np.quantile(df_results["anomaly_score"], 1 - contamination),
        color="red",
        linestyle="--",
        label="threshold",
    )
    ax3.set_title("Anomaly scores")
    ax3.legend()
    st.pyplot(fig3)

    # Allow download of anomalies
    buf = io.StringIO()
    out_df = df_results[df_results["anomaly_label"] == 1].copy()
    out_df.to_csv(buf, index=False)
    b = buf.getvalue().encode()
    st.download_button(
        "Download flagged anomalies (CSV)",
        data=b,
        file_name="anomalies.csv",
        mime="text/csv",
    )

    # Show a couple of example rows and explanation
    st.markdown("### Example flagged anomalies (first 5)")
    st.table(out_df.head(5))

    # Optional: show explanation for a selected anomaly row (feature values)
    st.subheader("Inspect a flagged transaction")
    if len(out_df) > 0:
        selection = st.selectbox(
            "Pick an anomaly to inspect (index)", options=out_df.index.tolist()
        )
        row = df_results.loc[selection]
        st.json(row.to_dict())
    else:
        st.info(
            "No anomalies found — try increasing contamination or use a different detector."
        )

    st.markdown("---")
    st.markdown(
        """
    **Notes & next steps**
    - This app shows unsupervised anomaly detection; flagged items require human review.
    - For production: build per-user models, add domain rules, use ensemble detectors, and incorporate behavioral features.
    - Consider adding explainability (SHAP) or rule-based filters to reduce false positives.
    """
    )
else:
    st.info("Press 'Run detection' after selecting data and options.")
