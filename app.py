"""
Streamlit dashboard for the Capstone market-regime project.
GMM + HMM regime detection on weekly S&P 500 data.
Run with:  streamlit run app.py
"""
import warnings
from datetime import date
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from hmmlearn.hmm import GaussianHMM
from scipy.stats import f_oneway, levene

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
np.random.seed(42)

st.set_page_config(page_title="Market Regime Dashboard", layout="wide")

FEATURE_COLS = ["Log_Return", "Volatility", "MA_Crossover", "VIX_Change", "term_spread"]
COLORS = {"Bear": "#D32F2F", "Neutral": "#F57C00", "Bull": "#388E3C"}


# ----------------------------- Data fetch helpers -----------------------------
def yf_close(ticker, start, end, name=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"].iloc[:, 0]
    else:
        s = df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(name or ticker)


def fetch_fred(series, start, end):
    s = pdr.DataReader(series, "fred", start, end).iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(series)


def merge_backward(base, other, name, tolerance="7 days"):
    base = base.copy(); other = other.copy()
    idx_name = base.index.name or "Date"
    base.index.name = idx_name; other.index.name = idx_name
    a = base.reset_index().sort_values(idx_name)
    b = other.reset_index().sort_values(idx_name)
    b.columns = [idx_name, name]
    out = pd.merge_asof(a, b, on=idx_name, direction="backward",
                        tolerance=pd.Timedelta(tolerance))
    return out.set_index(idx_name)


@st.cache_data(show_spinner="Fetching market + macro data…", ttl=60 * 60)
def build_dataset(start, end):
    spy = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    spy = spy[["Open", "High", "Low", "Close", "Volume"]].copy()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)

    vix = yf_close("^VIX", start, end, name="VIX")
    spy = spy.join(vix, how="inner")

    weekly = spy.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum", "VIX": "last"
    }).dropna()
    weekly.index.name = "Date"

    try:
        term_spread = fetch_fred("T10Y2Y", start, end).rename("term_spread")
    except Exception:
        t10 = yf_close("^TNX", start, end, name="T10")
        t3m = yf_close("^IRX", start, end, name="T3M")
        term_spread = (t10 - t3m).rename("term_spread")

    weekly = merge_backward(weekly, term_spread, "term_spread")
    weekly = weekly.replace([np.inf, -np.inf], np.nan).dropna()
    return weekly


def compute_features(weekly):
    out = weekly.copy()
    out["Log_Return"]   = np.log(out["Close"] / out["Close"].shift(1))
    out["Volatility"]   = out["Log_Return"].rolling(4).std()
    ma10 = out["Close"].rolling(10).mean()
    ma40 = out["Close"].rolling(40).mean()
    out["MA_Crossover"] = (ma10 - ma40) / out["Close"]
    out["VIX_Change"]   = out["VIX"].pct_change()
    cols = FEATURE_COLS + ["Close"]
    return out[cols].replace([np.inf, -np.inf], np.nan).dropna()


# -------------------------------- Preprocessor --------------------------------
class Preprocessor:
    def __init__(self, lower_q=0.01, upper_q=0.99, smooth_window=3):
        self.lower_q = lower_q; self.upper_q = upper_q
        self.smooth_window = smooth_window
        self.scaler = StandardScaler()

    def fit(self, X):
        df = X[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
        self.lower_ = df.quantile(self.lower_q)
        self.upper_ = df.quantile(self.upper_q)
        self.feature_names_ = list(df.columns)
        clipped = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window,
                                   min_periods=self.smooth_window).mean().dropna()
        self.scaler.fit(smoothed.values)
        return self

    def transform(self, X):
        df = X[self.feature_names_].replace([np.inf, -np.inf], np.nan)
        clipped = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window,
                                   min_periods=self.smooth_window).mean().dropna()
        scaled = self.scaler.transform(smoothed.values)
        return pd.DataFrame(scaled, index=smoothed.index, columns=self.feature_names_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# -------------------------------- Model training -------------------------------
@st.cache_resource(show_spinner="Training GMM…")
def train_gmm(X_values, _key):
    K = 3
    grid = {
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "reg_covar":       [1e-6, 1e-5, 1e-4, 1e-3],
        "n_init":          [10],
        "tol":             [1e-3, 1e-4],
        "max_iter":        [200],
    }
    best = None; best_bic = np.inf
    for params in product(*grid.values()):
        cfg = dict(zip(grid.keys(), params))
        try:
            m = GaussianMixture(n_components=K, random_state=42, **cfg).fit(X_values)
            b = m.bic(X_values)
            if b < best_bic:
                best_bic, best = b, (m, cfg)
        except Exception:
            continue
    return best


def bic_hmm(model, X, cov_type):
    K, d = model.n_components, X.shape[1]
    n_cov = {"full": K*d*(d+1)//2, "tied": d*(d+1)//2,
             "diag": K*d, "spherical": K}[cov_type]
    n_params = K*(K-1) + K*d + n_cov
    return -2 * model.score(X) + n_params * np.log(len(X))


@st.cache_resource(show_spinner="Training HMM…")
def train_hmm(X_values, _key):
    K = 3
    grid = {"covariance_type": ["full", "tied", "diag", "spherical"],
            "n_iter": [200]}
    best_overall = None; best_bic = np.inf
    for values in product(*grid.values()):
        cfg = dict(zip(grid.keys(), values))
        best_ll, best_m = -np.inf, None
        for seed in range(10):
            try:
                m = GaussianHMM(n_components=K, covariance_type=cfg["covariance_type"],
                                n_iter=cfg["n_iter"], tol=1e-4, random_state=seed).fit(X_values)
                ll = m.score(X_values)
                if ll > best_ll:
                    best_ll, best_m = ll, m
            except Exception:
                continue
        if best_m is None: continue
        b = bic_hmm(best_m, X_values, cfg["covariance_type"])
        if b < best_bic:
            best_bic, best_overall = b, (best_m, cfg)
    return best_overall


def label_clusters(states, train_features, X_index):
    stats = pd.DataFrame({
        "mean_return": train_features.loc[X_index, "Log_Return"].groupby(states).mean(),
        "mean_vol":    train_features.loc[X_index, "Volatility"].groupby(states).mean(),
    })
    stats["score"] = stats["mean_return"] - stats["mean_vol"]
    sorted_clusters = stats["score"].sort_values().index.tolist()
    return {sorted_clusters[0]: "Bear",
            sorted_clusters[1]: "Neutral",
            sorted_clusters[2]: "Bull"}


# -------------------------------- Backtest ------------------------------------
COST_BPS = 5 / 10_000
ALLOC_K3 = {"Bull": 1.00, "Neutral": 0.50, "Bear": 0.00}


def run_backtest(regime_series, alloc_map, weekly_returns):
    df = pd.DataFrame({"log_ret": weekly_returns, "regime": regime_series}).dropna()
    df["signal"] = df["regime"].shift(1)
    df["allocation"] = df["signal"].map(alloc_map).fillna(0)
    df["strat_ret"] = df["allocation"] * df["log_ret"]
    df["switched"] = (df["signal"] != df["signal"].shift(1)).astype(int)
    df["strat_ret"] -= df["switched"] * COST_BPS
    df["cum_bnh"]   = df["log_ret"].cumsum().apply(np.exp)
    df["cum_strat"] = df["strat_ret"].cumsum().apply(np.exp)
    return df


def metrics_row(bt):
    years = len(bt) / 52
    cagr_s = bt["cum_strat"].iloc[-1] ** (1 / years) - 1
    cagr_b = bt["cum_bnh"].iloc[-1]   ** (1 / years) - 1
    vol_s  = bt["strat_ret"].std() * np.sqrt(52)
    vol_b  = bt["log_ret"].std()   * np.sqrt(52)
    sh_s   = cagr_s / vol_s if vol_s > 0 else 0
    sh_b   = cagr_b / vol_b if vol_b > 0 else 0
    mdd_s  = ((bt["cum_strat"] - bt["cum_strat"].cummax()) / bt["cum_strat"].cummax()).min()
    mdd_b  = ((bt["cum_bnh"]   - bt["cum_bnh"].cummax())   / bt["cum_bnh"].cummax()).min()
    return {
        "CAGR (Strat)":   f"{cagr_s:.2%}",   "CAGR (B&H)": f"{cagr_b:.2%}",
        "Vol (Strat)":    f"{vol_s:.2%}",    "Vol (B&H)":  f"{vol_b:.2%}",
        "Sharpe (Strat)": f"{sh_s:.3f}",     "Sharpe (B&H)": f"{sh_b:.3f}",
        "MaxDD (Strat)":  f"{mdd_s:.2%}",    "MaxDD (B&H)":  f"{mdd_b:.2%}",
        "Switches":       int(bt["switched"].sum()),
        "Final $1 (Strat)": f"${bt['cum_strat'].iloc[-1]:.3f}",
        "Final $1 (B&H)":   f"${bt['cum_bnh'].iloc[-1]:.3f}",
    }


# ============================ STREAMLIT LAYOUT ================================
st.title("S&P 500 Market Regime Dashboard")
st.caption("GMM + HMM regime detection on weekly S&P 500 data. Based on Capstone notebook.")

with st.sidebar:
    st.header("Configuration")
    train_start = st.date_input("Training start", date(2010, 1, 1))
    train_end   = st.date_input("Training end",   date(2024, 12, 31))
    test_start  = st.date_input("Test (OOS) start", date(2025, 1, 1))
    test_end    = st.date_input("Test (OOS) end",   date.today())
    st.markdown("---")
    st.markdown("**Allocation per regime**")
    alloc_bull = st.slider("Bull",    0.0, 1.0, 1.00, 0.05)
    alloc_neu  = st.slider("Neutral", 0.0, 1.0, 0.50, 0.05)
    alloc_bear = st.slider("Bear",    0.0, 1.0, 0.00, 0.05)
    ALLOC_K3 = {"Bull": alloc_bull, "Neutral": alloc_neu, "Bear": alloc_bear}
    st.markdown("---")
    run = st.button("Run pipeline", type="primary", use_container_width=True)

if not run:
    st.info("Set the windows in the sidebar and click **Run pipeline** to fetch data and train both models.")
    st.stop()

TRAIN_START = train_start.strftime("%Y-%m-%d")
TRAIN_END   = train_end.strftime("%Y-%m-%d")
TEST_START  = test_start.strftime("%Y-%m-%d")
TEST_END    = test_end.strftime("%Y-%m-%d")
PRED_START  = (pd.Timestamp(TEST_START) - relativedelta(months=12)).strftime("%Y-%m-%d")

# Fetch & prep
train_df = build_dataset(TRAIN_START, TRAIN_END)
train_features = compute_features(train_df)
prep = Preprocessor()
X_train = prep.fit_transform(train_features[FEATURE_COLS])

pred_df = build_dataset(PRED_START, TEST_END)
pred_features = compute_features(pred_df)
X_pred = prep.transform(pred_features[FEATURE_COLS])
X_pred = X_pred[X_pred.index >= TEST_START]

# Train both models
gmm_final, gmm_cfg = train_gmm(X_train.values, TRAIN_START + TRAIN_END)
hmm_final, hmm_cfg = train_hmm(X_train.values, TRAIN_START + TRAIN_END)

gmm_states_train = gmm_final.predict(X_train.values)
hmm_states_train = hmm_final.predict(X_train.values)
gmm_label_map = label_clusters(gmm_states_train, train_features, X_train.index)
hmm_label_map = label_clusters(hmm_states_train, train_features, X_train.index)

train_gmm_regime = pd.Series(gmm_states_train, index=X_train.index).map(gmm_label_map)
train_hmm_regime = pd.Series(hmm_states_train, index=X_train.index).map(hmm_label_map)

oos_gmm_regime = pd.Series(gmm_final.predict(X_pred.values), index=X_pred.index).map(gmm_label_map)
oos_hmm_regime = pd.Series(hmm_final.predict(X_pred.values), index=X_pred.index).map(hmm_label_map)
oos_gmm_conf = gmm_final.predict_proba(X_pred.values).max(axis=1)
oos_hmm_conf = hmm_final.predict_proba(X_pred.values).max(axis=1)

# ---- Header KPIs ----
latest_date = X_pred.index[-1]
latest_gmm = oos_gmm_regime.iloc[-1]
latest_hmm = oos_hmm_regime.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training rows",  f"{len(X_train):,}")
c2.metric("OOS rows",        f"{len(X_pred):,}")
c3.metric(f"Latest GMM call ({latest_date.date()})",
          latest_gmm, f"conf {oos_gmm_conf[-1]:.2f}")
c4.metric(f"Latest HMM call ({latest_date.date()})",
          latest_hmm, f"conf {oos_hmm_conf[-1]:.2f}")

# ---- Tabs ----
tab_eda, tab_feat, tab_models, tab_oos, tab_bt, tab_stats = st.tabs(
    ["EDA", "Features", "Models", "OOS Regimes", "Backtest", "Stat Tests"]
)

# ---- EDA ----
with tab_eda:
    st.subheader("S&P 500 Weekly Close & Volume (training window)")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(train_df.index, train_df["Close"], color="#2196F3", linewidth=1)
    axes[0].set_ylabel("Index Level")
    axes[1].bar(train_df.index, train_df["Volume"], color="#FF9800", alpha=0.6, width=5)
    axes[1].set_ylabel("Volume")
    plt.tight_layout(); st.pyplot(fig)

    weekly_lr = np.log(train_df["Close"] / train_df["Close"].shift(1)).dropna()
    st.subheader("Weekly Log Returns")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(weekly_lr.index, weekly_lr, linewidth=0.8, color="#FF5722")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout(); st.pyplot(fig)

    a, b, c, d = st.columns(4)
    a.metric("Mean",     f"{weekly_lr.mean():.5f}")
    b.metric("Std Dev",  f"{weekly_lr.std():.5f}")
    c.metric("Skew",     f"{weekly_lr.skew():.3f}")
    d.metric("Kurtosis", f"{weekly_lr.kurtosis():.3f}")

# ---- Features ----
with tab_feat:
    st.subheader("Feature time series (scaled)")
    fig, axes = plt.subplots(len(FEATURE_COLS), 1,
                             figsize=(14, 2.4 * len(FEATURE_COLS)), sharex=True)
    for ax, col in zip(axes, FEATURE_COLS):
        ax.plot(X_train.index, X_train[col], linewidth=0.8)
        ax.set_title(col, fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Correlation matrix")
    corr = train_features[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

# ---- Models ----
with tab_models:
    col_g, col_h = st.columns(2)
    with col_g:
        st.subheader("GMM K=3 (in-sample)")
        st.json(gmm_cfg)
        st.write({
            "BIC":        round(gmm_final.bic(X_train.values), 2),
            "AIC":        round(gmm_final.aic(X_train.values), 2),
            "Silhouette": round(silhouette_score(X_train.values, gmm_states_train), 4),
            "DB":         round(davies_bouldin_score(X_train.values, gmm_states_train), 4),
            "CH":         round(calinski_harabasz_score(X_train.values, gmm_states_train), 2),
        })
        st.write("**Regime distribution**")
        st.write(train_gmm_regime.value_counts())

    with col_h:
        st.subheader("HMM K=3 (in-sample)")
        st.json(hmm_cfg)
        st.write({
            "BIC":         round(bic_hmm(hmm_final, X_train.values, hmm_cfg["covariance_type"]), 2),
            "Stickiness":  round(float(np.mean(np.diag(hmm_final.transmat_))), 4),
            "Silhouette":  round(silhouette_score(X_train.values, hmm_states_train), 4),
        })
        st.write("**Regime distribution**")
        st.write(train_hmm_regime.value_counts())

        order = sorted(hmm_label_map,
                       key=lambda s: ["Bear", "Neutral", "Bull"].index(hmm_label_map[s]))
        trans = hmm_final.transmat_[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(trans, annot=True, fmt=".3f", cmap="Blues",
                    xticklabels=["→Bear", "→Neutral", "→Bull"],
                    yticklabels=["Bear", "Neutral", "Bull"], ax=ax)
        ax.set_title("HMM transition matrix")
        st.pyplot(fig)

# ---- OOS regimes ----
with tab_oos:
    st.subheader(f"Out-of-sample regimes ({TEST_START} → {TEST_END})")
    price_oos = pred_features.loc[X_pred.index, "Close"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, regimes, name in [(axes[0], oos_gmm_regime, "GMM"),
                              (axes[1], oos_hmm_regime, "HMM")]:
        ax.plot(price_oos.index, price_oos, color="black", linewidth=1.6, zorder=2)
        for lbl, grp in regimes.groupby(regimes):
            ax.scatter(grp.index, price_oos.loc[grp.index],
                       c=COLORS[lbl], s=60, alpha=0.85, label=lbl, zorder=3)
        ax.set_title(f"{name} — OOS Regimes", fontweight="bold")
        ax.set_ylabel("S&P 500"); ax.legend(loc="upper left")
    plt.tight_layout(); st.pyplot(fig)

    st.write("**Recent regime calls**")
    recent = pd.DataFrame({
        "Close":      price_oos.round(2),
        "GMM":        oos_gmm_regime,
        "GMM_conf":   oos_gmm_conf.round(3),
        "HMM":        oos_hmm_regime,
        "HMM_conf":   oos_hmm_conf.round(3),
    }).tail(15)
    st.dataframe(recent, use_container_width=True)

# ---- Backtest ----
with tab_bt:
    train_ret = train_features.loc[X_train.index, "Log_Return"]
    oos_ret   = pred_features.loc[X_pred.index, "Log_Return"]

    bt_is_g  = run_backtest(train_gmm_regime, ALLOC_K3, train_ret)
    bt_is_h  = run_backtest(train_hmm_regime, ALLOC_K3, train_ret)
    bt_oos_g = run_backtest(oos_gmm_regime,   ALLOC_K3, oos_ret)
    bt_oos_h = run_backtest(oos_hmm_regime,   ALLOC_K3, oos_ret)

    st.subheader("Strategy vs Buy & Hold")
    summary = pd.DataFrame({
        "GMM In-Sample":  metrics_row(bt_is_g),
        "HMM In-Sample":  metrics_row(bt_is_h),
        "GMM OOS":        metrics_row(bt_oos_g),
        "HMM OOS":        metrics_row(bt_oos_h),
    })
    st.dataframe(summary, use_container_width=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    combos = [
        (axes[0][0], bt_is_g,  train_gmm_regime, "GMM In-Sample",  "#1565C0"),
        (axes[0][1], bt_oos_g, oos_gmm_regime,   "GMM OOS",        "#1565C0"),
        (axes[1][0], bt_is_h,  train_hmm_regime, "HMM In-Sample",  "#E65100"),
        (axes[1][1], bt_oos_h, oos_hmm_regime,   "HMM OOS",        "#E65100"),
    ]
    for ax, bt, regimes, title, clr in combos:
        prev_d, prev_r = bt.index[0], regimes.iloc[0]
        for i in range(1, len(regimes)):
            if regimes.iloc[i] != prev_r or i == len(regimes) - 1:
                ax.axvspan(prev_d, bt.index[i], alpha=0.15,
                           color=COLORS[prev_r], linewidth=0)
                prev_d, prev_r = bt.index[i], regimes.iloc[i]
        ax.plot(bt.index, bt["cum_strat"], color=clr, linewidth=2, label="Strategy")
        ax.plot(bt.index, bt["cum_bnh"], color="black", linewidth=1.3,
                linestyle="--", label="Buy & Hold")
        ax.set_title(title, fontweight="bold"); ax.legend(loc="upper left")
        ax.grid(alpha=0.2)
    plt.tight_layout(); st.pyplot(fig)

# ---- Stat tests ----
with tab_stats:
    returns = train_features.loc[X_train.index, "Log_Return"]
    rows = []
    for name, regimes in [("GMM", train_gmm_regime), ("HMM", train_hmm_regime)]:
        f, p = f_oneway(returns[regimes == "Bull"],
                        returns[regimes == "Neutral"],
                        returns[regimes == "Bear"])
        lf, lp = levene(returns[regimes == "Bull"],
                        returns[regimes == "Neutral"],
                        returns[regimes == "Bear"])
        rows.append({"Model": name,
                     "ANOVA F": round(f, 3),  "ANOVA p": f"{p:.2e}",
                     "Levene F": round(lf, 3), "Levene p": f"{lp:.2e}"})
    st.subheader("ANOVA (mean returns differ?) and Levene (variances differ?)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Per-feature ANOVA across GMM regimes")
    feat_rows = []
    for col in FEATURE_COLS:
        bull = X_train.loc[train_gmm_regime == "Bull", col]
        neu  = X_train.loc[train_gmm_regime == "Neutral", col]
        bear = X_train.loc[train_gmm_regime == "Bear", col]
        f, p = f_oneway(bull, neu, bear)
        feat_rows.append({"Feature": col, "F": round(f, 2), "p": f"{p:.2e}"})
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)
