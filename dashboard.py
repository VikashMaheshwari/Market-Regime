"""
Market Regime Detection Dashboard
==================================
Run with:  streamlit run dashboard.py
Requires:  pip install streamlit plotly yfinance hmmlearn pandas-datareader scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
from itertools import product
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from hmmlearn.hmm import GaussianHMM
import joblib, os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 13px; color: #8b949e; font-weight: 500; }
    .metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
    .metric-delta { font-size: 12px; margin-top: 4px; }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        padding: 10px 0 5px 0;
        border-bottom: 2px solid #30363d;
        margin-bottom: 20px;
    }
    .regime-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 8px;
        color: #8b949e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb;
        color: #ffffff;
    }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
TRAIN_START = "2010-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = date.today().strftime("%Y-%m-%d")
PRED_START  = (pd.Timestamp(TEST_START) - relativedelta(months=12)).strftime("%Y-%m-%d")
PRED_END    = TEST_END

FEATURE_COLS = ["Log_Return", "Volatility", "MA_Crossover", "VIX_Change", "term_spread"]

COLORS_K2 = {"Risk-ON": "#00c853", "Risk-OFF": "#ff1744"}
COLORS_K3 = {"Bull": "#00c853", "Neutral": "#ffa726", "Bear": "#ff1744"}
COST_BPS  = 5 / 10_000


# ─────────────────────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────────────────────
def yf_close(ticker, start, end, name=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    s = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(name or ticker)


def fetch_fred(series, start, end):
    """Fetch a FRED series via the public CSV endpoint (no API key, no extra deps)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df.columns = ["Date", series]
    df[series] = pd.to_numeric(df[series], errors="coerce")
    df = df.dropna().set_index("Date")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return df[series].rename(series)


def merge_backward(base, other, name):
    idx = base.index.name or "Date"
    base.index.name = idx; other.index.name = idx
    a = base.reset_index().sort_values(idx)
    b = other.reset_index().sort_values(idx)
    b.columns = [idx, name]
    return pd.merge_asof(a, b, on=idx, direction="backward",
                         tolerance=pd.Timedelta("7 days")).set_index(idx)


@st.cache_data(show_spinner="Fetching market data...")
def build_dataset(start, end):
    spy = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    spy = spy[["Open", "High", "Low", "Close", "Volume"]].copy()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    vix = yf_close("^VIX", start, end, "VIX")
    spy = spy.join(vix, how="inner")
    weekly = spy.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum", "VIX": "last"
    }).dropna()
    weekly.index.name = "Date"
    try:
        ts = fetch_fred("T10Y2Y", start, end).rename("term_spread")
    except Exception:
        t10 = yf_close("^TNX", start, end, "T10")
        t3m = yf_close("^IRX", start, end, "T3M")
        ts  = (t10 - t3m).rename("term_spread")
    hyg    = yf_close("HYG", start, end, "HYG")
    lqd    = yf_close("LQD", start, end, "LQD")
    credit = (lqd.pct_change(20) - hyg.pct_change(20)).rename("credit_spread")
    try:
        dxy_chg = yf_close("DX-Y.NYB", start, end, "DXY").pct_change(5).rename("dxy_change")
    except Exception:
        dxy_chg = pd.Series(0.0, index=weekly.index, name="dxy_change")
    weekly = merge_backward(weekly, ts,      "term_spread")
    weekly = merge_backward(weekly, credit,  "credit_spread")
    weekly = merge_backward(weekly, dxy_chg, "dxy_change")
    return weekly.replace([np.inf, -np.inf], np.nan).dropna()


def compute_features(weekly):
    out = weekly.copy()
    out["Log_Return"]   = np.log(out["Close"] / out["Close"].shift(1))
    out["Volatility"]   = out["Log_Return"].rolling(4).std()
    ma10 = out["Close"].rolling(10).mean()
    ma40 = out["Close"].rolling(40).mean()
    out["MA_Crossover"] = (ma10 - ma40) / out["Close"]
    out["VIX_Change"]   = out["VIX"].pct_change()
    return out[FEATURE_COLS + ["Close"]].replace([np.inf, -np.inf], np.nan).dropna()


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
        clipped  = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window, min_periods=self.smooth_window).mean().dropna()
        self.scaler.fit(smoothed.values)
        return self

    def transform(self, X):
        df = X[self.feature_names_].replace([np.inf, -np.inf], np.nan)
        clipped  = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window, min_periods=self.smooth_window).mean().dropna()
        scaled   = self.scaler.transform(smoothed.values)
        return pd.DataFrame(scaled, index=smoothed.index, columns=self.feature_names_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def bic_hmm(model, X, cov_type):
    K, d = model.n_components, X.shape[1]
    n_cov = {"full": K*d*(d+1)//2, "tied": d*(d+1)//2,
             "diag": K*d, "spherical": K}[cov_type]
    n_params = K*(K-1) + K*d + n_cov
    return -2*model.score(X) + n_params*np.log(len(X))


# ─────────────────────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models — please wait...")
def train_all_models():
    train_df      = build_dataset(TRAIN_START, TRAIN_END)
    train_features = compute_features(train_df)
    prep           = Preprocessor()
    X_train        = prep.fit_transform(train_features[FEATURE_COLS])
    X              = X_train.values
    ret            = train_features.loc[X_train.index, "Log_Return"]

    # ── K selection sweep (lightweight; single fit per K) ─────
    k_metrics = []
    for K in range(2, 6):
        m = GaussianMixture(n_components=K, covariance_type="full",
                            reg_covar=1e-4, n_init=3, max_iter=200,
                            tol=1e-3, random_state=42).fit(X)
        labels = m.predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
        k_metrics.append({"K": K, "BIC": m.bic(X), "AIC": m.aic(X),
                          "Silhouette": sil,
                          "AvgConf": m.predict_proba(X).max(axis=1).mean()})

    # ── GMM K=2 ───────────────────────────────────────────────
    gmm2 = GaussianMixture(n_components=2, covariance_type="full",
                            reg_covar=1e-4, n_init=5, max_iter=200,
                            tol=1e-3, random_state=42).fit(X)
    s2   = gmm2.predict(X)
    sm2  = ret.groupby(s2).mean().sort_values()
    glm2 = {s: l for s, l in zip(sm2.index, ["Risk-OFF", "Risk-ON"])}
    gdf2 = pd.DataFrame({"Regime": pd.Series(s2, index=X_train.index).map(glm2),
                          "Confidence": gmm2.predict_proba(X).max(axis=1)},
                         index=X_train.index)

    # ── GMM K=3 ───────────────────────────────────────────────
    # Notebook's BIC-optimal config
    gmm3 = GaussianMixture(n_components=3, covariance_type="full",
                            reg_covar=1e-4, n_init=15, max_iter=500,
                            tol=1e-4, random_state=42).fit(X)
    s3   = gmm3.predict(X)
    sm3  = ret.groupby(s3).mean().sort_values()
    glm3 = {s: l for s, l in zip(sm3.index, ["Bear", "Neutral", "Bull"])}
    gdf3 = pd.DataFrame({"Regime": pd.Series(s3, index=X_train.index).map(glm3),
                          "Confidence": gmm3.predict_proba(X).max(axis=1)},
                         index=X_train.index)

    # ── HMM K=2 ───────────────────────────────────────────────
    best_ll2, hmm2 = -np.inf, None
    for seed in range(3):
        try:
            m  = GaussianHMM(n_components=2, covariance_type="full",
                             n_iter=100, tol=1e-3, random_state=seed).fit(X)
            ll = m.score(X)
            if ll > best_ll2:
                best_ll2, hmm2 = ll, m
        except Exception:
            continue
    hs2  = hmm2.predict(X)
    hsm2 = ret.groupby(hs2).mean().sort_values()
    hlm2 = {s: l for s, l in zip(hsm2.index, ["Risk-OFF", "Risk-ON"])}
    hdf2 = pd.DataFrame({"Regime": pd.Series(hs2, index=X_train.index).map(hlm2),
                          "Confidence": hmm2.predict_proba(X).max(axis=1)},
                         index=X_train.index)

    # ── HMM K=3 (notebook's BIC-optimal config, single deterministic fit) ─
    hmm3 = GaussianHMM(n_components=3, covariance_type="full",
                       n_iter=100, tol=1e-4, random_state=42).fit(X)
    hs3  = hmm3.predict(X)
    hsm3 = ret.groupby(hs3).mean().sort_values()
    hlm3 = {s: l for s, l in zip(hsm3.index, ["Bear", "Neutral", "Bull"])}
    hdf3 = pd.DataFrame({"Regime": pd.Series(hs3, index=X_train.index).map(hlm3),
                          "Confidence": hmm3.predict_proba(X).max(axis=1)},
                         index=X_train.index)

    return {
        "train_df": train_df, "train_features": train_features,
        "prep": prep, "X_train": X_train, "ret": ret,
        "k_metrics": pd.DataFrame(k_metrics),
        "gmm2": gmm2, "glm2": glm2, "gdf2": gdf2,
        "gmm3": gmm3, "glm3": glm3, "gdf3": gdf3,
        "hmm2": hmm2, "hlm2": hlm2, "hdf2": hdf2,
        "hmm3": hmm3, "hlm3": hlm3, "hdf3": hdf3,
    }


@st.cache_data(show_spinner="Generating predictions...")
def get_predictions(_models):
    prep = _models["prep"]
    pred_df       = build_dataset(PRED_START, PRED_END)
    pred_features = compute_features(pred_df)
    X_pred        = prep.transform(pred_features[FEATURE_COLS])
    X_pred        = X_pred[X_pred.index >= TEST_START]
    Xp            = X_pred.values

    def pred_df_fn(model, lmap):
        states = model.predict(Xp)
        proba  = model.predict_proba(Xp)
        return pd.DataFrame({
            "Regime":     pd.Series(states, index=X_pred.index).map(lmap),
            "Confidence": proba.max(axis=1),
        }, index=X_pred.index)

    return {
        "X_pred": X_pred,
        "pred_features": pred_features,
        "gmm2": pred_df_fn(_models["gmm2"], _models["glm2"]),
        "gmm3": pred_df_fn(_models["gmm3"], _models["glm3"]),
        "hmm2": pred_df_fn(_models["hmm2"], _models["hlm2"]),
        "hmm3": pred_df_fn(_models["hmm3"], _models["hlm3"]),
    }


# ─────────────────────────────────────────────────────────────
# BACKTEST FUNCTION
# ─────────────────────────────────────────────────────────────
def run_backtest(regime_series, alloc_map, weekly_returns):
    df = pd.DataFrame({"log_ret": weekly_returns,
                        "regime":  regime_series}).dropna()
    df["signal"]     = df["regime"].shift(1)
    df["allocation"] = df["signal"].map(alloc_map).fillna(0)
    df["strat_ret"]  = df["allocation"] * df["log_ret"]
    df["switched"]   = (df["signal"] != df["signal"].shift(1)).astype(int)
    df["strat_ret"] -= df["switched"] * COST_BPS
    df["cum_bnh"]    = df["log_ret"].cumsum().apply(np.exp)
    df["cum_strat"]  = df["strat_ret"].cumsum().apply(np.exp)
    return df


def calc_metrics(df):
    years = len(df) / 52
    s_ret = df["strat_ret"]; b_ret = df["log_ret"]
    cagr_s = df["cum_strat"].iloc[-1]**(1/years) - 1
    cagr_b = df["cum_bnh"].iloc[-1]**(1/years)   - 1
    vol_s  = s_ret.std() * np.sqrt(52)
    vol_b  = b_ret.std() * np.sqrt(52)
    sh_s   = cagr_s / vol_s if vol_s > 0 else 0
    sh_b   = cagr_b / vol_b if vol_b > 0 else 0

    def mdd(c):
        dd = (c - c.cummax()) / c.cummax()
        return dd.min()

    mdd_s = mdd(df["cum_strat"]); mdd_b = mdd(df["cum_bnh"])
    cal_s = cagr_s / abs(mdd_s) if mdd_s != 0 else 0

    return {"CAGR": cagr_s, "CAGR_BnH": cagr_b,
            "Vol": vol_s, "Sharpe": sh_s, "Sharpe_BnH": sh_b,
            "MaxDD": mdd_s, "MaxDD_BnH": mdd_b,
            "Calmar": cal_s, "Switches": int(df["switched"].sum()),
            "Final": df["cum_strat"].iloc[-1], "Final_BnH": df["cum_bnh"].iloc[-1]}


# ─────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
)


def regime_price_fig(price, regime_series, colors, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines",
                              line=dict(color="#58a6ff", width=1.5),
                              name="S&P 500", hovertemplate="%{x}<br>$%{y:,.0f}"))
    for regime, clr in colors.items():
        idx = regime_series == regime
        if idx.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=price.index[idx], y=price.values[idx],
            mode="markers", name=regime,
            marker=dict(color=clr, size=5, opacity=0.85),
            hovertemplate=f"<b>{regime}</b><br>%{{x}}<br>$%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(title=title, height=420, **PLOTLY_THEME)
    return fig


def cum_ret_fig(bt_list, labels, colors_list):
    fig = go.Figure()
    for bt, lbl, clr in zip(bt_list, labels, colors_list):
        fig.add_trace(go.Scatter(x=bt.index, y=bt["cum_strat"], mode="lines",
                                  name=lbl, line=dict(color=clr, width=2.5),
                                  hovertemplate=f"<b>{lbl}</b><br>%{{x}}<br>$%{{y:.3f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=bt_list[0].index, y=bt_list[0]["cum_bnh"],
                              mode="lines", name="Buy & Hold",
                              line=dict(color="#8b949e", width=1.8, dash="dash"),
                              hovertemplate="<b>Buy & Hold</b><br>%{x}<br>$%{y:.3f}<extra></extra>"))
    fig.update_layout(title="Cumulative Returns ($1 invested)",
                       yaxis_title="Portfolio Value", height=420, **PLOTLY_THEME)
    return fig


def drawdown_fig(bt_list, labels, colors_list):
    fig = go.Figure()
    for bt, lbl, clr in zip(bt_list, labels, colors_list):
        dd = (bt["cum_strat"] - bt["cum_strat"].cummax()) / bt["cum_strat"].cummax() * 100
        fig.add_trace(go.Scatter(x=bt.index, y=dd, mode="lines", name=lbl,
                                  line=dict(color=clr, width=2),
                                  fill="tozeroy", fillcolor=clr.replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in clr else clr,
                                  hovertemplate=f"<b>{lbl}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>"))
    dd_b = (bt_list[0]["cum_bnh"] - bt_list[0]["cum_bnh"].cummax()) / bt_list[0]["cum_bnh"].cummax() * 100
    fig.add_trace(go.Scatter(x=bt_list[0].index, y=dd_b, mode="lines",
                              name="Buy & Hold",
                              line=dict(color="#8b949e", width=1.5, dash="dash")))
    fig.update_layout(title="Drawdown (%)", yaxis_title="Drawdown (%)",
                       height=380, **PLOTLY_THEME)
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Market Regime Detection")
    st.markdown("*GMM & HMM — Probabilistic Models*")
    st.divider()

    page = st.radio("Navigation", [
        "🏠  Overview",
        "📊  Data & Features",
        "🔢  K Selection",
        "🔵  GMM Model",
        "🟣  HMM Model",
        "💰  Backtest",
        "🔮  Live Predictions",
    ], label_visibility="collapsed")

    st.divider()

    k_choice = st.selectbox("Regime Mode", ["K=2  (Risk-ON / Risk-OFF)",
                                              "K=3  (Bull / Neutral / Bear)"])
    K = 2 if "K=2" in k_choice else 3

    model_choice = st.selectbox("Model", ["GMM", "HMM", "Both"])
    st.divider()
    st.caption(f"Train  : {TRAIN_START[:4]} – {TRAIN_END[:4]}")
    st.caption(f"Test   : {TEST_START[:4]} – Today")
    st.caption(f"Author : Capstone Project")


# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    models = train_all_models()
    preds  = get_predictions(models)

X_train        = models["X_train"]
train_features = models["train_features"]
ret            = models["ret"]
price_train    = train_features.loc[X_train.index, "Close"]
price_oos      = preds["pred_features"].loc[preds["X_pred"].index, "Close"]
oos_ret        = preds["pred_features"].loc[preds["X_pred"].index, "Log_Return"]

ALLOC_K2 = {"Risk-ON": 1.0, "Risk-OFF": 0.0}
ALLOC_K3 = {"Bull": 1.0, "Neutral": 0.5, "Bear": 0.0}
colors   = COLORS_K2 if K == 2 else COLORS_K3
alloc    = ALLOC_K2  if K == 2 else ALLOC_K3


# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────
if "Overview" in page:

    st.markdown("# Market Regime Detection")
    st.markdown("### Probabilistic Models — GMM & HMM | S&P 500 Weekly (2010–2024)")
    st.divider()

    # Current signal
    latest_gmm = preds[f"gmm{K}"]["Regime"].iloc[-1]
    latest_hmm = preds[f"hmm{K}"]["Regime"].iloc[-1]
    latest_conf_gmm = preds[f"gmm{K}"]["Confidence"].iloc[-1]
    latest_conf_hmm = preds[f"hmm{K}"]["Confidence"].iloc[-1]
    latest_date = preds["X_pred"].index[-1].strftime("%b %d, %Y")
    agree = latest_gmm == latest_hmm

    st.markdown(f"#### Current Market Signal  —  week ending {latest_date}")

    c1, c2, c3, c4 = st.columns(4)
    regime_color = COLORS_K2.get(latest_gmm, COLORS_K3.get(latest_gmm, "#fff"))

    with c1:
        st.metric("GMM Signal", latest_gmm,
                  delta=f"{latest_conf_gmm:.1%} confident")
    with c2:
        st.metric("HMM Signal", latest_hmm,
                  delta=f"{latest_conf_hmm:.1%} confident")
    with c3:
        weeks_in_regime = 0
        for r in preds[f"gmm{K}"]["Regime"].iloc[::-1]:
            if r == latest_gmm:
                weeks_in_regime += 1
            else:
                break
        st.metric("Weeks in Current Regime", f"{weeks_in_regime}w")
    with c4:
        st.metric("Model Agreement", "✅ Yes" if agree else "⚠️ No",
                  delta="Both models aligned" if agree else "Signals diverge")

    st.divider()

    # Training period summary
    st.markdown("#### Training Period Overview  (2010–2024)")

    bt_gmm = run_backtest(models[f"gdf{K}"]["Regime"], alloc, ret)
    bt_hmm = run_backtest(models[f"hdf{K}"]["Regime"], alloc, ret)
    m_gmm  = calc_metrics(bt_gmm)
    m_hmm  = calc_metrics(bt_hmm)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("GMM Sharpe",  f"{m_gmm['Sharpe']:.3f}",  delta=f"BnH: {m_gmm['Sharpe_BnH']:.3f}")
    with c2: st.metric("HMM Sharpe",  f"{m_hmm['Sharpe']:.3f}",  delta=f"BnH: {m_hmm['Sharpe_BnH']:.3f}")
    with c3: st.metric("GMM Max DD",  f"{m_gmm['MaxDD']:.1%}",   delta=f"BnH: {m_gmm['MaxDD_BnH']:.1%}", delta_color="inverse")
    with c4: st.metric("GMM CAGR",    f"{m_gmm['CAGR']:.2%}",    delta=f"BnH: {m_gmm['CAGR_BnH']:.2%}")
    with c5: st.metric("Training Weeks", len(X_train))

    st.divider()

    # Regime distribution pie
    gdf = models[f"gdf{K}"]
    hdf = models[f"hdf{K}"]

    col1, col2 = st.columns(2)

    with col1:
        dist = gdf["Regime"].value_counts()
        fig  = go.Figure(go.Pie(
            labels=dist.index, values=dist.values,
            marker_colors=[colors[r] for r in dist.index],
            hole=0.45, textfont_size=14,
            hovertemplate="<b>%{label}</b><br>%{value} weeks (%{percent})<extra></extra>",
        ))
        fig.update_layout(title=f"GMM K={K} — Regime Distribution",
                           height=350, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dist2 = hdf["Regime"].value_counts()
        fig2  = go.Figure(go.Pie(
            labels=dist2.index, values=dist2.values,
            marker_colors=[colors[r] for r in dist2.index],
            hole=0.45, textfont_size=14,
        ))
        fig2.update_layout(title=f"HMM K={K} — Regime Distribution",
                            height=350, **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    # Price with regimes
    fig3 = regime_price_fig(price_train, gdf["Regime"], colors,
                             f"GMM K={K} — Detected Regimes on S&P 500 (2010–2024)")
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: DATA & FEATURES
# ─────────────────────────────────────────────────────────────
elif "Data" in page:

    st.markdown("# Data & Feature Engineering")
    st.divider()

    train_df = models["train_df"]

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Feature Distributions", "Correlation Matrix"])

    with tab1:
        st.markdown("#### Weekly S&P 500 Data  (2010–2024)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Weeks",    len(train_df))
        c2.metric("Training Years", "15 years")
        c3.metric("Features Used",  len(FEATURE_COLS))

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=train_df.index, y=train_df["Close"],
                                  mode="lines", name="S&P 500",
                                  line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
        fig.add_trace(go.Bar(x=train_df.index, y=train_df["Volume"],
                              name="Volume", marker_color="#1f6feb", opacity=0.6), row=2, col=1)
        fig.update_layout(title="S&P 500 Weekly Close & Volume",
                           height=500, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Raw Data Sample")
        st.dataframe(train_df.tail(10).style.format("{:.4f}"), use_container_width=True)

    with tab2:
        st.markdown("#### Feature Distributions per Regime")
        gdf = models[f"gdf{K}"]
        feat_df = pd.DataFrame(X_train.values, columns=FEATURE_COLS, index=X_train.index)
        feat_df["Regime"] = gdf["Regime"]

        sel_feat = st.selectbox("Select Feature", FEATURE_COLS)

        fig = go.Figure()
        for regime, clr in colors.items():
            data = feat_df.loc[feat_df["Regime"] == regime, sel_feat].dropna()
            fig.add_trace(go.Violin(y=data, name=regime,
                                     fillcolor=clr, opacity=0.7,
                                     line_color=clr, box_visible=True,
                                     meanline_visible=True))
        fig.update_layout(title=f"{sel_feat} Distribution per Regime (GMM K={K})",
                           height=450, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

        # Feature time series
        fig2 = go.Figure()
        ts = train_features.loc[X_train.index, sel_feat] if sel_feat in train_features.columns else X_train[sel_feat]
        fig2.add_trace(go.Scatter(x=X_train.index, y=X_train[sel_feat],
                                   mode="lines", name=sel_feat,
                                   line=dict(color="#f0883e", width=1.2)))
        fig2.update_layout(title=f"{sel_feat} Over Time (standardised)",
                            height=300, **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        corr = pd.DataFrame(X_train.values, columns=FEATURE_COLS).corr()
        fig  = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(title="Feature Correlation Matrix (Standardised)",
                           height=450, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: K SELECTION
# ─────────────────────────────────────────────────────────────
elif "K Selection" in page:

    st.markdown("# K Selection — BIC / AIC Sweep")
    st.markdown("Quantitative justification for choosing the number of regimes.")
    st.divider()

    km = models["k_metrics"]

    best_K_bic = km.loc[km["BIC"].idxmin(), "K"]
    best_K_sil = km.loc[km["Silhouette"].idxmax(), "K"]

    c1, c2, c3 = st.columns(3)
    c1.metric("BIC-Optimal K",        int(best_K_bic))
    c2.metric("Silhouette-Optimal K", int(best_K_sil))
    c3.metric("Domain Choice",        "K=2 or K=3")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=km["K"], y=km["BIC"], mode="lines+markers+text",
                                  text=km["BIC"].round(0).astype(int),
                                  textposition="top center",
                                  line=dict(color="#58a6ff", width=2.5),
                                  marker=dict(size=10, color="#58a6ff"),
                                  name="BIC"))
        fig.add_trace(go.Scatter(x=km["K"], y=km["AIC"], mode="lines+markers",
                                  line=dict(color="#bc8cff", width=2.5, dash="dash"),
                                  marker=dict(size=10, color="#bc8cff"),
                                  name="AIC"))
        fig.add_vline(x=int(best_K_bic), line_dash="dot",
                      line_color="#ff7b72", annotation_text=f"Optimal K={int(best_K_bic)}")
        fig.update_layout(title="BIC & AIC vs Number of Regimes (lower = better)",
                           xaxis_title="K", yaxis_title="Score",
                           xaxis=dict(tickvals=list(range(2, 6))),
                           height=400, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=km["K"], y=km["Silhouette"],
                               marker_color=["#1f6feb" if k != int(best_K_sil) else "#3fb950"
                                             for k in km["K"]],
                               text=km["Silhouette"].round(3),
                               textposition="outside", name="Silhouette"))
        fig2.update_layout(title="Silhouette Score vs K (higher = better)",
                            xaxis_title="K", yaxis_title="Silhouette",
                            xaxis=dict(tickvals=list(range(2, 6))),
                            height=400, **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Full Metrics Table")
    st.dataframe(km.set_index("K").style.format("{:.4f}").highlight_min(
        subset=["BIC", "AIC"], color="#1a3a1a").highlight_max(
        subset=["Silhouette", "AvgConf"], color="#1a3a1a"),
        use_container_width=True)

    st.info(f"""
    **K Selection Decision:**
    BIC selects K={int(best_K_bic)} as statistically optimal.
    We offer both **K=2** (binary risk signal — simpler, actionable for a dashboard)
    and **K=3** (Bull/Neutral/Bear — richer economic interpretation).
    Literature: Hamilton (1989) used K=2; Ang & Bekaert (2002) extended to K=3.
    """)


# ─────────────────────────────────────────────────────────────
# PAGE: GMM MODEL
# ─────────────────────────────────────────────────────────────
elif "GMM" in page:

    st.markdown(f"# GMM Model  —  K={K}")
    st.divider()

    gdf   = models[f"gdf{K}"]
    gmm   = models[f"gmm{K}"]
    X     = X_train.values

    # Metrics
    states = gmm.predict(X)
    sil    = silhouette_score(X, states)
    db     = davies_bouldin_score(X, states)
    ch     = calinski_harabasz_score(X, states)
    conf   = gmm.predict_proba(X).max(axis=1).mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BIC",                f"{gmm.bic(X):,.0f}")
    c2.metric("AIC",                f"{gmm.aic(X):,.0f}")
    c3.metric("Silhouette",         f"{sil:.4f}")
    c4.metric("Avg Confidence",     f"{conf:.4f}")
    c5.metric("Converged",          "Yes" if gmm.converged_ else "No")

    tab1, tab2, tab3 = st.tabs(["Regime Map", "Regime Statistics", "Confidence"])

    with tab1:
        fig = regime_price_fig(price_train, gdf["Regime"], colors,
                                f"GMM K={K} — Regimes on S&P 500 (2010–2024)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        feat_df = pd.DataFrame(X, columns=FEATURE_COLS, index=X_train.index)
        feat_df["Regime"] = gdf["Regime"]
        stats = feat_df.groupby("Regime")[FEATURE_COLS].mean().round(4)
        stats["Count"]   = feat_df["Regime"].value_counts()
        stats["Pct %"]   = (feat_df["Regime"].value_counts(normalize=True)*100).round(1)
        st.markdown("#### Mean Feature Values per Regime")
        st.dataframe(stats.style.background_gradient(cmap="RdYlGn", axis=0),
                     use_container_width=True)

        # Bar chart of regime returns
        ret_by_regime = train_features.loc[X_train.index, "Log_Return"].copy()
        ret_by_regime.name = "Log_Return"
        rdf = pd.DataFrame({"Log_Return": ret_by_regime, "Regime": gdf["Regime"]})
        mean_ret = rdf.groupby("Regime")["Log_Return"].mean() * 100

        fig2 = go.Figure(go.Bar(
            x=mean_ret.index, y=mean_ret.values,
            marker_color=[colors[r] for r in mean_ret.index],
            text=mean_ret.round(3).astype(str) + "%",
            textposition="outside",
        ))
        fig2.update_layout(title="Mean Weekly Log Return per Regime (%)",
                            height=380, **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        conf_s = pd.Series(gmm.predict_proba(X).max(axis=1), index=X_train.index)
        fig3   = go.Figure()
        fig3.add_trace(go.Scatter(x=conf_s.index, y=conf_s.values,
                                   mode="lines", fill="tozeroy",
                                   line=dict(color="#58a6ff", width=1),
                                   fillcolor="rgba(88,166,255,0.15)",
                                   name="Confidence"))
        fig3.add_hline(y=0.8, line_dash="dash", line_color="#ffa726",
                       annotation_text="80% threshold")
        fig3.update_layout(title="GMM Confidence Over Time",
                            yaxis_title="Max Confidence", yaxis_range=[0, 1.05],
                            height=380, **PLOTLY_THEME)
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: HMM MODEL
# ─────────────────────────────────────────────────────────────
elif "HMM" in page:

    st.markdown(f"# HMM Model  —  K={K}")
    st.divider()

    hdf = models[f"hdf{K}"]
    hmm = models[f"hmm{K}"]
    X   = X_train.values

    states = hmm.predict(X)
    sil    = silhouette_score(X, states)
    conf   = hmm.predict_proba(X).max(axis=1).mean()
    stick  = np.mean(np.diag(hmm.transmat_))
    bic    = bic_hmm(hmm, X, "full")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BIC",              f"{bic:,.0f}")
    c2.metric("Silhouette",       f"{sil:.4f}")
    c3.metric("Avg Confidence",   f"{conf:.4f}")
    c4.metric("Stickiness",       f"{stick:.4f}",
              delta="Healthy (0.85–0.95)" if 0.85 <= stick <= 0.95 else "Check value")

    tab1, tab2, tab3 = st.tabs(["Regime Map", "Transition Matrix", "Dwell Time"])

    with tab1:
        fig = regime_price_fig(price_train, hdf["Regime"], colors,
                                f"HMM K={K} — Regimes on S&P 500 (2010–2024)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        labels_ordered = (["Risk-OFF", "Risk-ON"] if K == 2
                          else ["Bear", "Neutral", "Bull"])
        hlm = models[f"hlm{K}"]
        order = sorted(hlm, key=lambda s: labels_ordered.index(hlm[s]))
        trans = hmm.transmat_[np.ix_(order, order)]

        fig2 = go.Figure(go.Heatmap(
            z=trans, x=[f"→{l}" for l in labels_ordered],
            y=labels_ordered, colorscale="Blues",
            zmin=0, zmax=1,
            text=np.round(trans, 3), texttemplate="%{text}",
            hovertemplate="%{y} → %{x}<br>P = %{z:.3f}<extra></extra>",
        ))
        fig2.update_layout(title="HMM Transition Matrix  (diagonal = stickiness)",
                            height=400, **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Interpretation:** High diagonal values mean the model stays "
                    "in each regime for many weeks — realistic for financial markets.")

    with tab3:
        diag   = np.diag(trans)
        dwells = [1 / (1 - p + 1e-10) for p in diag]

        fig3 = go.Figure(go.Bar(
            x=labels_ordered, y=dwells,
            marker_color=[colors[l] for l in labels_ordered],
            text=[f"{d:.1f} wks" for d in dwells],
            textposition="outside",
        ))
        fig3.update_layout(title="Expected Dwell Time per Regime (weeks)",
                            yaxis_title="Weeks", height=380, **PLOTLY_THEME)
        st.plotly_chart(fig3, use_container_width=True)

        st.caption("Dwell time = 1 / (1 − p_ii).  "
                   "Healthy range: Bear 8–20 wks, Neutral 5–15 wks, Bull 15–40 wks.")


# ─────────────────────────────────────────────────────────────
# PAGE: BACKTEST
# ─────────────────────────────────────────────────────────────
elif "Backtest" in page:

    st.markdown("# Regime-Aware Backtest")
    st.markdown(f"**Strategy:** K={K}  |  5 bps cost per switch  |  1-week signal lag")
    st.divider()

    tab_is, tab_oos, tab_cmp = st.tabs(["In-Sample (2010–2024)",
                                         "Out-of-Sample (2025→Today)",
                                         "Full Comparison"])

    # ── In-sample ─────────────────────────────────────────────
    with tab_is:
        bt_g = run_backtest(models[f"gdf{K}"]["Regime"], alloc, ret)
        bt_h = run_backtest(models[f"hdf{K}"]["Regime"], alloc, ret)
        mg   = calc_metrics(bt_g)
        mh   = calc_metrics(bt_h)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GMM Sharpe",  f"{mg['Sharpe']:.3f}",  delta=f"BnH {mg['Sharpe_BnH']:.3f}")
        c2.metric("HMM Sharpe",  f"{mh['Sharpe']:.3f}",  delta=f"BnH {mh['Sharpe_BnH']:.3f}")
        c3.metric("GMM Max DD",  f"{mg['MaxDD']:.1%}",   delta=f"BnH {mg['MaxDD_BnH']:.1%}", delta_color="inverse")
        c4.metric("GMM CAGR",    f"{mg['CAGR']:.2%}",    delta=f"BnH {mg['CAGR_BnH']:.2%}")

        fig = cum_ret_fig([bt_g, bt_h], [f"GMM K={K}", f"HMM K={K}"],
                          ["#58a6ff", "#bc8cff"])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = drawdown_fig([bt_g, bt_h], [f"GMM K={K}", f"HMM K={K}"],
                             ["#58a6ff", "#bc8cff"])
        st.plotly_chart(fig2, use_container_width=True)

        # Annual returns
        ann_g = bt_g["strat_ret"].resample("YE").sum().apply(lambda x: (np.exp(x)-1)*100)
        ann_b = bt_g["log_ret"].resample("YE").sum().apply(lambda x: (np.exp(x)-1)*100)
        years = ann_g.index.year

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=years, y=ann_g.values, name=f"GMM K={K}",
                               marker_color=["#3fb950" if v >= 0 else "#f85149"
                                             for v in ann_g.values],
                               opacity=0.9))
        fig3.add_trace(go.Bar(x=years, y=ann_b.values, name="Buy & Hold",
                               marker_color="#8b949e", opacity=0.5))
        fig3.update_layout(title="Annual Returns — Strategy vs Buy & Hold",
                            barmode="group", height=380,
                            yaxis_ticksuffix="%", **PLOTLY_THEME)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Out-of-sample ──────────────────────────────────────────
    with tab_oos:
        oos_g = run_backtest(preds[f"gmm{K}"]["Regime"], alloc, oos_ret)
        oos_h = run_backtest(preds[f"hmm{K}"]["Regime"], alloc, oos_ret)
        og    = calc_metrics(oos_g)
        oh    = calc_metrics(oos_h)

        st.caption(f"True unseen data — {TEST_START} → {TEST_END}  "
                   f"({len(preds['X_pred'])} weeks)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GMM Sharpe (OOS)", f"{og['Sharpe']:.3f}",
                  delta=f"BnH {og['Sharpe_BnH']:.3f}")
        c2.metric("HMM Sharpe (OOS)", f"{oh['Sharpe']:.3f}",
                  delta=f"BnH {oh['Sharpe_BnH']:.3f}")
        c3.metric("GMM Max DD (OOS)", f"{og['MaxDD']:.1%}",
                  delta=f"BnH {og['MaxDD_BnH']:.1%}", delta_color="inverse")
        c4.metric("GMM Final $1",     f"${og['Final']:.3f}",
                  delta=f"BnH ${og['Final_BnH']:.3f}")

        fig = cum_ret_fig([oos_g, oos_h], [f"GMM K={K} OOS", f"HMM K={K} OOS"],
                          ["#58a6ff", "#bc8cff"])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = drawdown_fig([oos_g, oos_h], [f"GMM K={K} OOS", f"HMM K={K} OOS"],
                             ["#58a6ff", "#bc8cff"])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Full comparison ────────────────────────────────────────
    with tab_cmp:
        rows = []
        for k in [2, 3]:
            al = ALLOC_K2 if k == 2 else ALLOC_K3
            for name, reg_is, reg_oos in [
                (f"GMM K={k}", models[f"gdf{k}"]["Regime"], preds[f"gmm{k}"]["Regime"]),
                (f"HMM K={k}", models[f"hdf{k}"]["Regime"], preds[f"hmm{k}"]["Regime"]),
            ]:
                m_is  = calc_metrics(run_backtest(reg_is,  al, ret))
                m_oos = calc_metrics(run_backtest(reg_oos, al, oos_ret))
                rows.append({
                    "Model":        name,
                    "IS CAGR":      f"{m_is['CAGR']:.2%}",
                    "IS Sharpe":    f"{m_is['Sharpe']:.3f}",
                    "IS Max DD":    f"{m_is['MaxDD']:.2%}",
                    "OOS CAGR":     f"{m_oos['CAGR']:.2%}",
                    "OOS Sharpe":   f"{m_oos['Sharpe']:.3f}",
                    "OOS Max DD":   f"{m_oos['MaxDD']:.2%}",
                    "OOS Final $1": f"${m_oos['Final']:.3f}",
                })

        bnh_is  = calc_metrics(run_backtest(models["gdf2"]["Regime"],
                                             {"Risk-ON":1,"Risk-OFF":1}, ret))
        bnh_oos = calc_metrics(run_backtest(preds["gmm2"]["Regime"],
                                             {"Risk-ON":1,"Risk-OFF":1}, oos_ret))
        rows.append({
            "Model":        "Buy & Hold",
            "IS CAGR":      f"{bnh_is['CAGR_BnH']:.2%}",
            "IS Sharpe":    f"{bnh_is['Sharpe_BnH']:.3f}",
            "IS Max DD":    f"{bnh_is['MaxDD_BnH']:.2%}",
            "OOS CAGR":     f"{bnh_oos['CAGR_BnH']:.2%}",
            "OOS Sharpe":   f"{bnh_oos['Sharpe_BnH']:.3f}",
            "OOS Max DD":   f"{bnh_oos['MaxDD_BnH']:.2%}",
            "OOS Final $1": f"${bnh_oos['Final_BnH']:.3f}",
        })

        cmp_df = pd.DataFrame(rows)
        st.markdown("#### All Models — In-Sample vs Out-of-Sample")
        st.dataframe(cmp_df.set_index("Model"), use_container_width=True)
        st.caption("IS = In-Sample (2010–2024) | OOS = Out-of-Sample (2025→Today) | "
                   "5 bps cost per switch | 1-week lag")


# ─────────────────────────────────────────────────────────────
# PAGE: LIVE PREDICTIONS
# ─────────────────────────────────────────────────────────────
elif "Predictions" in page:

    st.markdown("# Live Predictions  —  2025 → Today")
    st.markdown("Model trained on 2010–2024.  These weeks were **never seen during training.**")
    st.divider()

    gp = preds[f"gmm{K}"]
    hp = preds[f"hmm{K}"]

    # Current signal banner
    cur_gmm   = gp["Regime"].iloc[-1]
    cur_hmm   = hp["Regime"].iloc[-1]
    cur_date  = gp.index[-1].strftime("%B %d, %Y")
    cur_conf_g = gp["Confidence"].iloc[-1]
    cur_conf_h = hp["Confidence"].iloc[-1]

    clr_g = colors.get(cur_gmm, "#fff")
    clr_h = colors.get(cur_hmm, "#fff")

    st.markdown(f"### Signal as of {cur_date}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card' style='border-color:{clr_g}'>
            <div class='metric-label'>GMM Signal</div>
            <div class='metric-value' style='color:{clr_g}'>{cur_gmm}</div>
            <div class='metric-delta'>{cur_conf_g:.1%} confident</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card' style='border-color:{clr_h}'>
            <div class='metric-label'>HMM Signal</div>
            <div class='metric-value' style='color:{clr_h}'>{cur_hmm}</div>
            <div class='metric-delta'>{cur_conf_h:.1%} confident</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        agree = cur_gmm == cur_hmm
        st.markdown(f"""
        <div class='metric-card' style='border-color:{"#3fb950" if agree else "#ffa726"}'>
            <div class='metric-label'>Model Agreement</div>
            <div class='metric-value'>{"✅ Aligned" if agree else "⚠️ Diverge"}</div>
            <div class='metric-delta'>{"High conviction signal" if agree else "Elevated uncertainty"}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Regime Chart", "Weekly Table", "Confidence"])

    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.65, 0.35],
                             subplot_titles=[f"GMM K={K}", f"HMM K={K}"])
        for row, (df, lbl) in enumerate([(gp, "GMM"), (hp, "HMM")], start=1):
            fig.add_trace(go.Scatter(x=price_oos.index, y=price_oos.values,
                                      mode="lines", name="S&P 500",
                                      line=dict(color="#58a6ff", width=1.5),
                                      showlegend=(row==1)), row=row, col=1)
            for regime, clr in colors.items():
                idx = df["Regime"] == regime
                if idx.sum() == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=price_oos.index[idx], y=price_oos.values[idx],
                    mode="markers", name=f"{lbl} {regime}",
                    marker=dict(color=clr, size=8),
                    showlegend=(row==1),
                ), row=row, col=1)
        fig.update_layout(height=600, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        table = pd.DataFrame({
            "Price":       price_oos.round(2),
            "GMM Regime":  gp["Regime"],
            "GMM Conf":    gp["Confidence"].round(3),
            "HMM Regime":  hp["Regime"],
            "HMM Conf":    hp["Confidence"].round(3),
            "Agreement":   (gp["Regime"] == hp["Regime"]).map({True: "✅", False: "⚠️"}),
        })

        def color_regime(val):
            c = COLORS_K2.get(val, COLORS_K3.get(val, ""))
            if c:
                return f"color: {c}; font-weight: bold"
            return ""

        st.dataframe(
            table.sort_index(ascending=False)
                 .style.applymap(color_regime, subset=["GMM Regime", "HMM Regime"]),
            use_container_width=True, height=450,
        )

    with tab3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=gp.index, y=gp["Confidence"],
                                   mode="lines+markers", name="GMM",
                                   line=dict(color="#58a6ff", width=2),
                                   marker=dict(size=5)))
        fig3.add_trace(go.Scatter(x=hp.index, y=hp["Confidence"],
                                   mode="lines+markers", name="HMM",
                                   line=dict(color="#bc8cff", width=2),
                                   marker=dict(size=5)))
        fig3.add_hline(y=0.8, line_dash="dash", line_color="#ffa726",
                       annotation_text="80% confidence threshold")
        fig3.update_layout(title="Prediction Confidence Over Time  "
                                  "(low = model uncertain about current regime)",
                            yaxis_title="Confidence", yaxis_range=[0, 1.05],
                            height=400, **PLOTLY_THEME)
        st.plotly_chart(fig3, use_container_width=True)

        # Sanity check
        st.markdown("#### Sanity Check — Known Risk-OFF / Bear Periods")
        known = {
            "COVID Crash (2020)":   ("2020-02-01", "2020-04-30"),
            "2022 Bear Market":     ("2022-01-01", "2022-10-31"),
            "2018 Q4 Selloff":      ("2018-10-01", "2018-12-31"),
            "2011 Debt Crisis":     ("2011-07-01", "2011-10-31"),
        }
        target = "Risk-OFF" if K == 2 else "Bear"
        gmm_full = models[f"gdf{K}"]["Regime"]

        rows = []
        for event, (s, e) in known.items():
            mask = (gmm_full.index >= s) & (gmm_full.index <= e)
            if mask.sum() == 0:
                continue
            pct = (gmm_full[mask] == target).mean() * 100
            rows.append({"Event": event, f"GMM {target} %": f"{pct:.0f}%",
                          "Result": "✅ Aligned" if pct >= 50 else "⚠️ Weak"})

        st.dataframe(pd.DataFrame(rows).set_index("Event"), use_container_width=True)
        st.caption("≥ 50% weeks labeled as risk-off during known crashes = model is economically sensible. "
                   "Source: Davis et al. NBER w28320; Baker, Bloom & Davis NBER w21633.")
