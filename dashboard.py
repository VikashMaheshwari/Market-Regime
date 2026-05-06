import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM
from scipy.stats import f_oneway

warnings.filterwarnings("ignore")

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Market Regime Dashboard",
    layout="wide",
    page_icon="📈"
)

st.title("📈 AI Market Regime Detection Dashboard")
st.markdown("Gaussian Mixture Models (GMM) + Hidden Markov Models (HMM)")

# =====================================================
# CONFIG
# =====================================================

TRAIN_START = "2010-01-01"
TRAIN_END = "2024-12-31"

FEATURE_COLS = [
    "Log_Return",
    "Volatility",
    "MA_Crossover",
    "VIX_Change",
    "term_spread"
]

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["GMM", "HMM"]
)

show_features = st.sidebar.checkbox(
    "Show Feature Charts",
    True
)

show_backtest = st.sidebar.checkbox(
    "Show Backtest",
    True
)

show_statistics = st.sidebar.checkbox(
    "Show Statistics",
    True
)

# =====================================================
# DATA FUNCTIONS
# =====================================================

@st.cache_data
def yf_close(ticker, start, end, name=None):

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"].iloc[:, 0]
    else:
        s = df["Close"]

    s.index = pd.to_datetime(s.index).tz_localize(None)

    return s.rename(name or ticker)

# =====================================================
# FRED FIX
# =====================================================

@st.cache_data
def fetch_fred(series, start, end):

    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    )

    df = pd.read_csv(url)

    df.columns = ["Date", series]

    df["Date"] = pd.to_datetime(df["Date"])

    df = df[
        (df["Date"] >= pd.to_datetime(start)) &
        (df["Date"] <= pd.to_datetime(end))
    ]

    df.set_index("Date", inplace=True)

    return df[series]

# =====================================================
# BUILD DATASET
# =====================================================

@st.cache_data
def build_dataset(start, end):

    spy = yf.download(
        "^GSPC",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)

    spy = spy[[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]]

    vix = yf_close(
        "^VIX",
        start,
        end,
        name="VIX"
    )

    spy["VIX"] = vix

    weekly = spy.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
        "VIX": "last"
    }).dropna()

    term_spread = fetch_fred(
        "T10Y2Y",
        start,
        end
    )

    term_spread = (
        term_spread
        .resample("W-FRI")
        .last()
    )

    weekly["term_spread"] = term_spread

    weekly = weekly.dropna()

    return weekly

# =====================================================
# FEATURE ENGINEERING
# =====================================================

@st.cache_data
def compute_features(df):

    data = df.copy()

    data["Log_Return"] = np.log(
        data["Close"] / data["Close"].shift(1)
    )

    data["Volatility"] = (
        data["Log_Return"]
        .rolling(4)
        .std()
    )

    ma10 = data["Close"].rolling(10).mean()
    ma40 = data["Close"].rolling(40).mean()

    data["MA_Crossover"] = (
        (ma10 - ma40) / data["Close"]
    )

    data["VIX_Change"] = (
        data["VIX"]
        .pct_change()
    )

    data = data.dropna()

    return data

# =====================================================
# PREPROCESSOR
# =====================================================

class Preprocessor:

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):

        df = X.copy()

        lower = df.quantile(0.01)
        upper = df.quantile(0.99)

        df = df.clip(
            lower,
            upper,
            axis=1
        )

        df = (
            df
            .rolling(3)
            .mean()
            .dropna()
        )

        scaled = self.scaler.fit_transform(df)

        return pd.DataFrame(
            scaled,
            index=df.index,
            columns=df.columns
        )

# =====================================================
# LOAD DATA
# =====================================================

with st.spinner("Loading Market Data..."):

    train_df = build_dataset(
        TRAIN_START,
        TRAIN_END
    )

    features_df = compute_features(train_df)

    prep = Preprocessor()

    X_train = prep.fit_transform(
        features_df[FEATURE_COLS]
    )

# =====================================================
# TRAIN OPTIMIZED GMM
# =====================================================

with st.spinner("Training Optimized GMM..."):

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        reg_covar=0.0001,
        n_init=15,
        tol=0.0001,
        max_iter=500,
        random_state=42
    )

    gmm.fit(X_train.values)

    gmm_states = gmm.predict(X_train.values)

    gmm_probs = gmm.predict_proba(
        X_train.values
    )

# =====================================================
# GMM LABELING
# =====================================================

gmm_regime_stats = pd.DataFrame({

    "mean_return": (
        features_df.loc[
            X_train.index,
            "Log_Return"
        ]
        .groupby(gmm_states)
        .mean()
    ),

    "mean_vol": (
        features_df.loc[
            X_train.index,
            "Volatility"
        ]
        .groupby(gmm_states)
        .mean()
    )

})

gmm_regime_stats["score"] = (
    gmm_regime_stats["mean_return"] -
    gmm_regime_stats["mean_vol"]
)

sorted_gmm_clusters = (
    gmm_regime_stats["score"]
    .sort_values()
    .index
    .tolist()
)

gmm_label_map = {

    sorted_gmm_clusters[0]: "Bear",
    sorted_gmm_clusters[1]: "Neutral",
    sorted_gmm_clusters[2]: "Bull"

}

gmm_regimes = pd.Series(
    gmm_states,
    index=X_train.index
).map(gmm_label_map)

# =====================================================
# TRAIN OPTIMIZED HMM
# =====================================================

with st.spinner("Training Optimized HMM..."):

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        tol=0.0001,
        random_state=42
    )

    hmm.fit(X_train.values)

    hmm_states = hmm.predict(
        X_train.values
    )

    hmm_probs = hmm.predict_proba(
        X_train.values
    )

# =====================================================
# HMM LABELING
# =====================================================

hmm_regime_stats = pd.DataFrame({

    "mean_return": (
        features_df.loc[
            X_train.index,
            "Log_Return"
        ]
        .groupby(hmm_states)
        .mean()
    ),

    "mean_vol": (
        features_df.loc[
            X_train.index,
            "Volatility"
        ]
        .groupby(hmm_states)
        .mean()
    )

})

hmm_regime_stats["score"] = (
    hmm_regime_stats["mean_return"] -
    hmm_regime_stats["mean_vol"]
)

sorted_hmm_clusters = (
    hmm_regime_stats["score"]
    .sort_values()
    .index
    .tolist()
)

hmm_label_map = {

    sorted_hmm_clusters[0]: "Bear",
    sorted_hmm_clusters[1]: "Neutral",
    sorted_hmm_clusters[2]: "Bull"

}

hmm_regimes = pd.Series(
    hmm_states,
    index=X_train.index
).map(hmm_label_map)

# =====================================================
# MODEL SELECTION
# =====================================================

if model_choice == "GMM":

    current_regime = gmm_regimes
    current_probs = gmm_probs
    current_states = gmm_states

else:

    current_regime = hmm_regimes
    current_probs = hmm_probs
    current_states = hmm_states

# =====================================================
# MAIN METRICS
# =====================================================

latest_regime = current_regime.iloc[-1]

latest_close = (
    features_df
    .loc[current_regime.index, "Close"]
    .iloc[-1]
)

latest_vix = (
    train_df
    .loc[current_regime.index, "VIX"]
    .iloc[-1]
)

latest_confidence = (
    current_probs
    .max(axis=1)[-1]
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current Regime",
    latest_regime
)

c2.metric(
    "Confidence",
    f"{latest_confidence:.2%}"
)

c3.metric(
    "S&P 500",
    f"{latest_close:.2f}"
)

c4.metric(
    "VIX",
    f"{latest_vix:.2f}"
)

# =====================================================
# REGIME OVERLAY
# =====================================================

st.subheader("Market Regime Overlay")

price_data = (
    features_df
    .loc[current_regime.index, "Close"]
)

colors = {
    "Bull": "green",
    "Neutral": "orange",
    "Bear": "red"
}

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=price_data.index,
        y=price_data,
        mode="lines",
        line=dict(
            color="black",
            width=2
        ),
        name="S&P 500"
    )
)

for regime in ["Bull", "Neutral", "Bear"]:

    idx = current_regime[
        current_regime == regime
    ].index

    fig.add_trace(
        go.Scatter(
            x=idx,
            y=price_data.loc[idx],
            mode="markers",
            marker=dict(
                color=colors[regime],
                size=7
            ),
            name=regime
        )
    )

fig.update_layout(
    template="plotly_white",
    height=600,
    xaxis_title="Date",
    yaxis_title="S&P 500"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# FEATURE ANALYSIS
# =====================================================

if show_features:

    st.subheader("Feature Analysis")

    selected_feature = st.selectbox(
        "Choose Feature",
        FEATURE_COLS
    )

    feature_fig = px.line(
        features_df.loc[current_regime.index],
        x=features_df.loc[current_regime.index].index,
        y=selected_feature,
        title=selected_feature
    )

    st.plotly_chart(
        feature_fig,
        use_container_width=True
    )

# =====================================================
# MODEL METRICS
# =====================================================

st.subheader(f"{model_choice} Model Metrics")

m1, m2, m3, m4 = st.columns(4)

if model_choice == "GMM":

    silhouette = silhouette_score(
        X_train.values,
        gmm_states
    )

    m1.metric(
        "Silhouette",
        f"{silhouette:.4f}"
    )

    m2.metric(
        "BIC",
        f"{gmm.bic(X_train.values):.0f}"
    )

    m3.metric(
        "AIC",
        f"{gmm.aic(X_train.values):.0f}"
    )

    m4.metric(
        "Avg Confidence",
        f"{gmm_probs.max(axis=1).mean():.2%}"
    )

else:

    stickiness = np.mean(
        np.diag(hmm.transmat_)
    )

    m1.metric(
        "Stickiness",
        f"{stickiness:.3f}"
    )

    m2.metric(
        "Log Likelihood",
        f"{hmm.score(X_train.values):.0f}"
    )

    m3.metric(
        "Avg Confidence",
        f"{hmm_probs.max(axis=1).mean():.2%}"
    )

    m4.metric(
        "States",
        "3"
    )

# =====================================================
# HMM TRANSITION MATRIX
# =====================================================

if model_choice == "HMM":

    st.subheader("HMM Transition Matrix")

    fig2, ax = plt.subplots(
        figsize=(6, 4)
    )

    sns.heatmap(
        hmm.transmat_,
        annot=True,
        cmap="Blues",
        fmt=".3f",
        ax=ax
    )

    st.pyplot(fig2)

# =====================================================
# BACKTEST
# =====================================================

if show_backtest:

    st.subheader("Strategy Backtest")

    returns = (
        features_df
        .loc[current_regime.index, "Log_Return"]
    )

    allocation = current_regime.map({

        "Bull": 1.0,
        "Neutral": 0.5,
        "Bear": 0.0

    })

    strategy_returns = (
        allocation.shift(1) * returns
    )

    cum_strategy = np.exp(
        strategy_returns.cumsum()
    )

    cum_market = np.exp(
        returns.cumsum()
    )

    backtest_fig = go.Figure()

    backtest_fig.add_trace(
        go.Scatter(
            x=cum_market.index,
            y=cum_market,
            name="Buy & Hold"
        )
    )

    backtest_fig.add_trace(
        go.Scatter(
            x=cum_strategy.index,
            y=cum_strategy,
            name="Strategy"
        )
    )

    backtest_fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        backtest_fig,
        use_container_width=True
    )

# =====================================================
# STATISTICS
# =====================================================

if show_statistics:

    st.subheader("Statistical Validation")

    returns = (
        features_df
        .loc[current_regime.index, "Log_Return"]
    )

    bull = returns[current_regime == "Bull"]
    neutral = returns[current_regime == "Neutral"]
    bear = returns[current_regime == "Bear"]

    f_stat, p_val = f_oneway(
        bull,
        neutral,
        bear
    )

    s1, s2 = st.columns(2)

    s1.metric(
        "ANOVA F-Statistic",
        f"{f_stat:.4f}"
    )

    s2.metric(
        "P-Value",
        f"{p_val:.6f}"
    )

# =====================================================
# RECENT PREDICTIONS
# =====================================================

st.subheader("Recent Predictions")

recent = pd.DataFrame({

    "Close": price_data,
    "Regime": current_regime,
    "Confidence": current_probs.max(axis=1)

}).tail(10)

st.dataframe(
    recent,
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(
    "Built using Streamlit + Optimized GMM + Optimized HMM"
)
