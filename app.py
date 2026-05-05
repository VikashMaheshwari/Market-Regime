"""
S&P 500 Market Regime Dashboard — mirrors the capstone notebook end-to-end.

Sections (tabs) are aligned with the notebook:
  1. Current State          → live regime call + S&P with regime shading
  2. EDA                    → price/volume, returns vs normal, feature corr
  3. Model Selection        → GMM vs HMM (BIC, LL, transition matrix)
  4. Backtest               → IS + OOS equity curve, drawdown, metrics
  5. Statistical Validation → ANOVA, Levene
  6. Predictions            → table + CSV download

Self-contained — fetches data and retrains models on first load (~60 s).
Deploy to Streamlit Community Cloud with only app.py + requirements.txt.
"""
import warnings
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from hmmlearn.hmm import GaussianHMM
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import f_oneway, levene
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config — same as notebook ──────────────────────────────────────────────────
TRAIN_START  = "2010-01-01"
TRAIN_END    = "2024-12-31"
TEST_START   = "2025-01-01"
TEST_END     = date.today().strftime("%Y-%m-%d")
PRED_START   = (pd.Timestamp(TEST_START) - relativedelta(months=12)).strftime("%Y-%m-%d")
FEATURE_COLS = ["Log_Return", "Volatility", "MA_Crossover", "VIX_Change", "term_spread"]
COST_BPS     = 5 / 10_000
ALLOC        = {"Bull": 1.0, "Neutral": 0.5, "Bear": 0.0}

# ── Theme — Bloomberg-style ────────────────────────────────────────────────────
BG       = "#0B1929"
SURFACE  = "#152844"
SURFACE2 = "#1B3355"
RULE     = "#27406B"
TEXT     = "#E8EEF6"
SUBTEXT  = "#8FA4BF"
GOLD     = "#E5B53D"
CYAN     = "#4FC3F7"
GREEN    = "#4CAF50"
ORANGE   = "#FFB74D"
RED      = "#EF5350"
COLORS   = {"Bear": RED, "Neutral": ORANGE, "Bull": GREEN}

st.set_page_config(
    page_title="Market Regime Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    .stApp > header {{ background-color: {BG}; border-bottom: 3px solid {GOLD}; }}
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {RULE};
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
    h1, h2, h3, h4 {{ color: {TEXT} !important; font-family: "Calibri", sans-serif; }}
    .stCaption, p, .stMarkdown {{ color: {SUBTEXT}; }}
    div[data-testid="stMetric"] {{
        background-color: {SURFACE}; border: 1px solid {RULE};
        border-left: 4px solid {GOLD}; padding: 14px 18px; border-radius: 6px;
    }}
    div[data-testid="stMetric"] label {{
        color: {SUBTEXT} !important; font-size: 10px !important;
        font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    div[data-testid="stMetricValue"] {{
        color: {GOLD} !important; font-size: 26px !important; font-weight: 700 !important;
    }}
    div[data-testid="stMetricDelta"] {{ color: {CYAN} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {RULE}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; color: {SUBTEXT};
        border-radius: 0; padding: 12px 22px; font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        color: {GOLD} !important; border-bottom: 3px solid {GOLD} !important;
        background-color: {SURFACE};
    }}
    .stButton button {{
        background-color: {GOLD}; color: {BG};
        border: none; font-weight: 700; border-radius: 4px;
    }}
    .stButton button:hover {{ background-color: #c89a25; color: {BG}; }}
    .stDataFrame {{ background-color: {SURFACE}; border: 1px solid {RULE}; border-radius: 4px; }}
    div[data-testid="stAlert"] {{
        background-color: {SURFACE2}; border-left: 4px solid {CYAN}; color: {TEXT};
    }}
    .stRadio label {{ color: {TEXT} !important; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
pio.templates["bloomberg"] = dict(layout=dict(
    paper_bgcolor=BG, plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Calibri"),
    xaxis=dict(gridcolor=RULE, zerolinecolor=RULE, color=SUBTEXT),
    yaxis=dict(gridcolor=RULE, zerolinecolor=RULE, color=SUBTEXT),
    legend=dict(bgcolor=SURFACE, bordercolor=RULE, borderwidth=1, font=dict(color=TEXT)),
    hoverlabel=dict(bgcolor=SURFACE2, font=dict(color=TEXT)),
    margin=dict(l=60, r=20, t=60, b=50),
))
pio.templates.default = "bloomberg"


# ═════════════════════════════════════════════════════════════════════════════
# DATA + FEATURES — same logic as the notebook
# ═════════════════════════════════════════════════════════════════════════════
def yf_close(ticker, start, end, name=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    s = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(name or ticker)


def fetch_fred(series, start, end):
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={series}&observation_start={start}&observation_end={end}")
    resp = requests.get(url, timeout=30); resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
    return s.rename(series)


def merge_backward(base, other, name, tolerance="7 days"):
    """Time-asof join — never pulls future data."""
    base, other = base.copy(), other.copy()
    idx = base.index.name or "Date"
    base.index.name = other.index.name = idx
    a = base.reset_index().sort_values(idx)
    b = other.reset_index().sort_values(idx)
    b.columns = [idx, name]
    out = pd.merge_asof(a, b, on=idx, direction="backward",
                        tolerance=pd.Timedelta(tolerance))
    return out.set_index(idx)


@st.cache_data(ttl=86400, show_spinner=False)
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
        "Close": "last", "Volume": "sum", "VIX": "last",
    }).dropna()
    weekly.index.name = "Date"
    try:
        term = fetch_fred("T10Y2Y", start, end).rename("term_spread")
    except Exception:
        t10 = yf_close("^TNX", start, end)
        t3m = yf_close("^IRX", start, end)
        term = (t10 - t3m).rename("term_spread")
    weekly = merge_backward(weekly, term, "term_spread")
    return weekly.replace([np.inf, -np.inf], np.nan).dropna()


def compute_features(weekly):
    out = weekly.copy()
    out["Log_Return"]   = np.log(out["Close"] / out["Close"].shift(1))
    out["Volatility"]   = out["Log_Return"].rolling(4).std()
    ma10 = out["Close"].rolling(10).mean()
    ma40 = out["Close"].rolling(40).mean()
    out["MA_Crossover"] = (ma10 - ma40) / out["Close"]
    out["VIX_Change"]   = out["VIX"].pct_change()
    return out[FEATURE_COLS + ["Close", "VIX"]].replace([np.inf, -np.inf], np.nan).dropna()


class Preprocessor:
    """Winsorize → smooth → scale (fit on train, freeze for prediction)."""
    def __init__(self, lower_q=0.01, upper_q=0.99, smooth_window=3):
        self.lower_q, self.upper_q, self.smooth_window = lower_q, upper_q, smooth_window
        self.scaler = StandardScaler()

    @staticmethod
    def _clean(X):
        """Force numeric float64 and drop bad rows. sklearn 1.5+ on Py3.14 is strict."""
        df = X[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df.astype(np.float64)

    def fit(self, X):
        df = self._clean(X)
        if df.empty:
            raise ValueError(
                f"Preprocessor.fit: empty after cleaning. "
                f"Input rows={len(X)}, cols={list(X.columns)}. "
                f"Likely missing term_spread (FRED unreachable) or all-NaN VIX_Change."
            )
        self.lower_ = df.quantile(self.lower_q)
        self.upper_ = df.quantile(self.upper_q)
        self.feature_names_ = list(df.columns)
        clipped  = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window,
                                    min_periods=self.smooth_window).mean().dropna()
        if smoothed.empty:
            raise ValueError(
                f"Preprocessor.fit: smoothed dataframe is empty "
                f"(input had {len(df)} rows; smooth_window={self.smooth_window})."
            )
        arr = np.ascontiguousarray(smoothed.values, dtype=np.float64)
        self.scaler.fit(arr)
        return self

    def transform(self, X):
        df = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        clipped  = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window,
                                    min_periods=self.smooth_window).mean().dropna()
        smoothed = smoothed.astype(np.float64)
        arr = np.ascontiguousarray(smoothed.values, dtype=np.float64)
        return pd.DataFrame(self.scaler.transform(arr),
                            index=smoothed.index, columns=self.feature_names_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ═════════════════════════════════════════════════════════════════════════════
# MODELS — slim grid search, full-cov, BIC selection (same as notebook)
# ═════════════════════════════════════════════════════════════════════════════
def bic_hmm(model, X, cov_type="full"):
    """Same formula as notebook."""
    K, d = model.n_components, X.shape[1]
    n_cov = {"full": K * d * (d + 1) // 2, "tied": d * (d + 1) // 2,
             "diag": K * d, "spherical": K}[cov_type]
    n_params = K * (K - 1) + K * d + n_cov
    return -2 * model.score(X) + n_params * np.log(len(X))


@st.cache_resource(show_spinner="Fetching data and training models — only happens once per session…")
def train_models():
    train_df       = build_dataset(TRAIN_START, TRAIN_END)
    train_features = compute_features(train_df)
    prep           = Preprocessor()
    X_train        = prep.fit_transform(train_features[FEATURE_COLS])
    Xv             = X_train.values

    # ── GMM K=3 — same grid as notebook §8 ────────────────────────────────────
    best_gmm, best_gmm_bic, best_gmm_cov = None, np.inf, "full"
    for cov in ("full", "tied", "diag", "spherical"):
        for reg in (1e-6, 1e-5, 1e-4):
            try:
                m = GaussianMixture(
                    n_components=3, covariance_type=cov,
                    reg_covar=reg, n_init=15, tol=1e-3,
                    max_iter=500, random_state=42,
                ).fit(Xv)
                if m.bic(Xv) < best_gmm_bic:
                    best_gmm, best_gmm_bic, best_gmm_cov = m, m.bic(Xv), cov
            except Exception:
                continue

    # ── HMM K=3 — two-stage, exactly like notebook §9 + §9.1 ──────────────────
    # Stage 1: grid search across (cov × n_iter) with 15 seeds per combo,
    #          keep best LL per combo, then pick lowest BIC across combos.
    SEEDS_GRID  = 15
    best_combo  = None  # (cov_type, n_iter, BIC)
    best_combo_bic = np.inf
    for cov in ("full", "tied", "diag", "spherical"):
        for n_iter in (100, 200, 300):
            best_ll_combo, best_m_combo = -np.inf, None
            for seed in range(SEEDS_GRID):
                try:
                    m = GaussianHMM(
                        n_components=3, covariance_type=cov,
                        n_iter=n_iter, tol=1e-4, random_state=seed,
                    ).fit(Xv)
                    ll = m.score(Xv)
                    if ll > best_ll_combo:
                        best_ll_combo, best_m_combo = ll, m
                except Exception:
                    continue
            if best_m_combo is not None:
                bic = bic_hmm(best_m_combo, Xv, cov)
                if bic < best_combo_bic:
                    best_combo_bic = bic
                    best_combo     = (cov, n_iter)

    # Stage 2: refit BIC-winning config across 10 seeds, keep best LL.
    hmm_cov, hmm_n_iter = best_combo
    best_hmm, best_total_ll = None, -np.inf
    for seed in range(10):
        try:
            m = GaussianHMM(
                n_components=3, covariance_type=hmm_cov,
                n_iter=hmm_n_iter, tol=1e-4, random_state=seed,
            ).fit(Xv)
            ll = m.score(Xv)
            if ll > best_total_ll:
                best_total_ll, best_hmm = ll, m
        except Exception:
            continue
    best_hmm_bic = bic_hmm(best_hmm, Xv, hmm_cov)
    best_hmm_cov = hmm_cov

    ret = train_features.loc[X_train.index, "Log_Return"]
    vol = train_features.loc[X_train.index, "Volatility"]

    def label_map(states):
        """Order clusters by (mean_return − mean_vol) → Bear/Neutral/Bull."""
        rs = pd.DataFrame({
            "ret": ret.groupby(states).mean(),
            "vol": vol.groupby(states).mean(),
        })
        rs["score"] = rs["ret"] - rs["vol"]
        order = rs["score"].sort_values().index.tolist()
        return {order[0]: "Bear", order[1]: "Neutral", order[2]: "Bull"}

    gmm_states = best_gmm.predict(Xv)
    hmm_states = best_hmm.predict(Xv)

    return {
        "prep": prep, "X_train": X_train, "train_df": train_df,
        "train_features": train_features,
        "gmm": best_gmm, "gmm_bic": best_gmm_bic,
        "gmm_ll": best_gmm.score(Xv) * len(Xv),
        "gmm_states": gmm_states,
        "gmm_map": label_map(gmm_states),
        "hmm": best_hmm, "hmm_bic": best_hmm_bic, "hmm_cov": best_hmm_cov,
        "hmm_ll": best_hmm.score(Xv),
        "hmm_states": hmm_states,
        "hmm_map": label_map(hmm_states),
    }


@st.cache_data(ttl=86400, show_spinner="Loading out-of-sample predictions…")
def get_predictions():
    M             = train_models()
    pred_df       = build_dataset(PRED_START, TEST_END)
    pred_features = compute_features(pred_df)
    X_pred        = M["prep"].transform(pred_features[FEATURE_COLS])
    X_pred        = X_pred[X_pred.index >= TEST_START]

    out = {
        "price":         pred_features.loc[X_pred.index, "Close"],
        "log_ret":       pred_features.loc[X_pred.index, "Log_Return"],
        "X_pred":        X_pred,
    }
    for key, model, lblmap in (("gmm", M["gmm"], M["gmm_map"]),
                                ("hmm", M["hmm"], M["hmm_map"])):
        states = model.predict(X_pred.values)
        proba  = model.predict_proba(X_pred.values)
        out[key] = {
            "regime":     pd.Series(states, index=X_pred.index).map(lblmap),
            "confidence": pd.Series(proba.max(axis=1), index=X_pred.index),
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST — same rule as notebook (1-week lag, 5 bps, Bull/Neutral/Bear alloc)
# ═════════════════════════════════════════════════════════════════════════════
def run_backtest(regime_series, weekly_returns):
    df = pd.DataFrame({"log_ret": weekly_returns, "regime": regime_series}).dropna()
    df["signal"]     = df["regime"].shift(1)
    df["allocation"] = df["signal"].map(ALLOC).fillna(0)
    df["strat_ret"]  = df["allocation"] * df["log_ret"]
    df["switched"]   = (df["signal"] != df["signal"].shift(1)).astype(int)
    df["strat_ret"] -= df["switched"] * COST_BPS
    df["cum_bnh"]    = df["log_ret"].cumsum().apply(np.exp)
    df["cum_strat"]  = df["strat_ret"].cumsum().apply(np.exp)
    return df


def perf_metrics(df):
    yrs    = max(len(df) / 52, 1e-9)
    cagr_s = df["cum_strat"].iloc[-1] ** (1 / yrs) - 1
    cagr_b = df["cum_bnh"].iloc[-1]   ** (1 / yrs) - 1
    vol_s  = df["strat_ret"].std() * np.sqrt(52)
    vol_b  = df["log_ret"].std()   * np.sqrt(52)
    def mdd(c): return ((c - c.cummax()) / c.cummax()).min()
    mdd_s, mdd_b = mdd(df["cum_strat"]), mdd(df["cum_bnh"])
    win_rate = (df["strat_ret"] > df["log_ret"]).mean()
    return {
        "cagr_s": cagr_s, "cagr_b": cagr_b,
        "vol_s":  vol_s,  "vol_b":  vol_b,
        "shrp_s": cagr_s / vol_s if vol_s > 0 else 0,
        "shrp_b": cagr_b / vol_b if vol_b > 0 else 0,
        "mdd_s":  mdd_s,  "mdd_b":  mdd_b,
        "calm_s": cagr_s / abs(mdd_s) if mdd_s != 0 else 0,
        "calm_b": cagr_b / abs(mdd_b) if mdd_b != 0 else 0,
        "win":    float(win_rate),
        "switches": int(df["switched"].sum()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div style="padding: 6px 0 14px 0;">
      <div style="color:{GOLD}; font-size:11px; font-weight:700; letter-spacing:1px;
                  text-transform:uppercase;">
        MARKET REGIME DETECTION  ·  CAPSTONE LIVE DASHBOARD
      </div>
      <div style="color:{TEXT}; font-size:34px; font-weight:700;
                  border-left: 5px solid {GOLD}; padding-left:14px; margin-top:6px;">
        S&P 500 · Regime Dashboard
      </div>
      <div style="color:{SUBTEXT}; font-size:13px; margin-top:6px;">
        Trained {TRAIN_START} → {TRAIN_END}  ·  Out-of-sample {TEST_START} → {TEST_END}
      </div>
    </div>
    <hr style="border:none; border-top:1px solid {RULE}; margin: 4px 0 18px 0;">
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        f"<div style='color:{GOLD}; font-weight:700; font-size:11px;"
        f" letter-spacing:1px;'>CONTROLS</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<h3 style='color:{TEXT}; margin-top:0;'>Settings</h3>",
                unsafe_allow_html=True)
    model_choice = st.radio("Model", ["GMM", "HMM"], horizontal=True, index=1)
    st.markdown(
        f"<div style='color:{SUBTEXT}; font-size:11px; margin-top:6px;'>"
        f"Regimes: <b style='color:{GOLD};'>K = 3</b>  ·  Bull / Neutral / Bear"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{RULE}'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{GOLD}; font-weight:700; font-size:10px;"
        f" letter-spacing:1px;'>FEATURES</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style='color:{TEXT}; font-size:12.5px; line-height:1.7;'>
        ▪ Log Return<br>
        ▪ 4-week Volatility<br>
        ▪ MA Crossover (10w − 40w)<br>
        ▪ VIX Change<br>
        ▪ Term spread (10y − 2y)
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{RULE}'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{GOLD}; font-weight:700; font-size:10px;"
        f" letter-spacing:1px;'>STRATEGY RULE</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style='color:{TEXT}; font-size:12px; line-height:1.6;'>
        Bull&nbsp;&nbsp;&nbsp;→ 100%<br>
        Neutral → 50%<br>
        Bear&nbsp;&nbsp;→ 0%<br>
        <span style='color:{SUBTEXT};'>1-week lag · 5 bps cost</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{RULE}'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh data & retrain"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

M      = train_models()
preds  = get_predictions()
key    = "gmm" if model_choice == "GMM" else "hmm"
regime = preds[key]["regime"]
conf   = preds[key]["confidence"]

# ── KPI strip ──────────────────────────────────────────────────────────────────
latest_regime = regime.iloc[-1]
latest_conf   = conf.iloc[-1]
weeks_in = 1
for r in regime.iloc[::-1][1:]:
    if r == latest_regime: weeks_in += 1
    else: break

bt_oos  = run_backtest(regime, preds["log_ret"])
m_oos   = perf_metrics(bt_oos)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Regime",  latest_regime)
c2.metric("Confidence",      f"{latest_conf:.1%}")
c3.metric("Weeks in regime", weeks_in)
c4.metric(
    "Strategy CAGR (OOS)",
    f"{m_oos['cagr_s']:.2%}",
    delta=f"{(m_oos['cagr_s'] - m_oos['cagr_b']) * 100:+.2f}pp vs B&H",
)

tabs = st.tabs([
    "📊 Current State",
    "🔍 EDA",
    "🤖 Model Selection",
    "💰 Backtest",
    "🧪 Validation",
    "📋 Predictions",
])
tab_state, tab_eda, tab_model, tab_bt, tab_val, tab_table = tabs


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Current State
# ═════════════════════════════════════════════════════════════════════════════
with tab_state:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.08,
        subplot_titles=("S&P 500 with Regime Shading (OOS)",
                        "Model Confidence (per week)"),
    )
    # regime shading
    prev_d, prev_r = preds["price"].index[0], regime.iloc[0]
    for i in range(1, len(regime)):
        if regime.iloc[i] != prev_r or i == len(regime) - 1:
            fig.add_vrect(x0=prev_d, x1=preds["price"].index[i],
                          fillcolor=COLORS[prev_r], opacity=0.18,
                          layer="below", line_width=0, row=1, col=1)
            prev_d, prev_r = preds["price"].index[i], regime.iloc[i]
    fig.add_trace(go.Scatter(x=preds["price"].index, y=preds["price"].values,
                             mode="lines", name="S&P 500",
                             line=dict(color=GOLD, width=2)), row=1, col=1)
    for r, c in COLORS.items():
        idx = regime == r
        if idx.any():
            fig.add_trace(go.Scatter(
                x=preds["price"].index[idx], y=preds["price"].values[idx],
                mode="markers", name=r,
                marker=dict(color=c, size=9, line=dict(width=0.6, color=BG))),
                row=1, col=1)
    fig.add_trace(go.Bar(x=conf.index, y=conf.values,
                         marker=dict(color=[COLORS[r] for r in regime]),
                         showlegend=False, name="Confidence"),
                  row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color=SUBTEXT, row=2, col=1)
    fig.update_layout(height=700, hovermode="x unified",
                      legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(title="S&P 500",    row=1, col=1)
    fig.update_yaxes(title="Confidence", range=[0, 1.05], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""<div style="background:{SURFACE2}; border-left:4px solid {CYAN};
                       padding:14px 18px; border-radius:4px; color:{TEXT};">
          <b style="color:{CYAN};">CURRENT CALL</b> &nbsp; · &nbsp;
          {model_choice} (K=3) classifies the market as
          <b style="color:{COLORS[latest_regime]};">{latest_regime}</b>
          at <b>{latest_conf:.1%}</b> confidence,
          holding for <b>{weeks_in}</b> consecutive week(s).
        </div>""",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA (mirrors notebook §4)
# ═════════════════════════════════════════════════════════════════════════════
with tab_eda:
    train_df = M["train_df"]; train_feat = M["train_features"]
    weekly_lr = train_feat["Log_Return"].dropna()

    # Price + Volume
    st.markdown(f"<h3 style='color:{GOLD};'>S&P 500 — Training Window</h3>",
                unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.06,
                        subplot_titles=("Weekly Close (2010–2024)", "Weekly Volume"))
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df["Close"],
                             line=dict(color=GOLD, width=1.5),
                             name="S&P 500"), row=1, col=1)
    fig.add_trace(go.Bar(x=train_df.index, y=train_df["Volume"],
                         marker=dict(color=CYAN), opacity=0.6,
                         name="Volume", showlegend=False), row=2, col=1)
    fig.update_layout(height=480, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Returns vs Normal — mirrors notebook §4.3
    st.markdown(f"<h3 style='color:{GOLD};'>Returns vs Normal — Fat Tails</h3>",
                unsafe_allow_html=True)
    colA, colB = st.columns([3, 1])
    with colA:
        x = np.linspace(weekly_lr.min(), weekly_lr.max(), 250)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=weekly_lr, nbinsx=70, histnorm="probability density",
                                    marker_color=CYAN, opacity=0.85, name="Empirical"))
        fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, weekly_lr.mean(), weekly_lr.std()),
                                 line=dict(color=GOLD, width=2.6),
                                 name="Normal fit"))
        fig.update_layout(height=380, xaxis_title="Log Return",
                          yaxis_title="Density",
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.metric("Std. dev.",        f"{weekly_lr.std():.4f}")
        st.metric("Skewness",         f"{weekly_lr.skew():.3f}")
        st.metric("Excess kurtosis",  f"{weekly_lr.kurtosis():.2f}")
        st.markdown(
            f"<div style='color:{SUBTEXT}; font-size:11px; margin-top:8px;'>"
            f"Excess kurtosis ≫ 0 → fat tails → mixture model justified."
            f"</div>",
            unsafe_allow_html=True,
        )

    # Feature correlation
    st.markdown(f"<h3 style='color:{GOLD};'>Feature Correlation</h3>",
                unsafe_allow_html=True)
    corr = train_feat[FEATURE_COLS].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, RED], [0.5, SURFACE], [1, GOLD]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont={"color": TEXT, "size": 12},
        colorbar=dict(tickfont=dict(color=SUBTEXT)),
    ))
    fig.update_layout(height=420, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Selection (mirrors notebook §8 + §9 + §13)
# ═════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown(f"<h3 style='color:{GOLD};'>GMM K=3 vs HMM K=3</h3>",
                unsafe_allow_html=True)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Bar(
            x=["GMM K=3", "HMM K=3"], y=[M["gmm_bic"], M["hmm_bic"]],
            marker_color=[CYAN, GOLD], width=[0.45, 0.45],
            text=[f"{M['gmm_bic']:,.0f}", f"{M['hmm_bic']:,.0f}"],
            textposition="outside", textfont=dict(color=TEXT),
        ))
        winner = "GMM K=3" if M["gmm_bic"] < M["hmm_bic"] else "HMM K=3"
        fig.update_layout(title=dict(
            text=f"<b>BIC — lower wins</b>  ·  winner: <span style='color:{GOLD};'>{winner}</span>",
            font=dict(color=GOLD)),
            height=360, yaxis_title="BIC")
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        fig = go.Figure(go.Bar(
            x=["GMM K=3", "HMM K=3"], y=[M["gmm_ll"], M["hmm_ll"]],
            marker_color=[CYAN, GOLD], width=[0.45, 0.45],
            text=[f"{M['gmm_ll']:,.0f}", f"{M['hmm_ll']:,.0f}"],
            textposition="outside", textfont=dict(color=TEXT),
        ))
        winner = "GMM K=3" if M["gmm_ll"] > M["hmm_ll"] else "HMM K=3"
        fig.update_layout(title=dict(
            text=f"<b>Log-Likelihood — higher wins</b>  ·  winner: <span style='color:{GOLD};'>{winner}</span>",
            font=dict(color=GOLD)),
            height=360, yaxis_title="Total LL")
        st.plotly_chart(fig, use_container_width=True)

    # HMM transition matrix (only meaningful for HMM)
    st.markdown(f"<h3 style='color:{GOLD};'>HMM Transition Matrix</h3>",
                unsafe_allow_html=True)
    hmm = M["hmm"]; lblmap = M["hmm_map"]
    order_labels = ["Bear", "Neutral", "Bull"]
    order_idx = sorted(lblmap, key=lambda s: order_labels.index(lblmap[s]))
    trans = hmm.transmat_[np.ix_(order_idx, order_idx)]

    cL, cR = st.columns([2, 1])
    with cL:
        fig = go.Figure(data=go.Heatmap(
            z=trans, x=[f"→ {l}" for l in order_labels], y=order_labels,
            colorscale=[[0, SURFACE], [0.5, "#3D7BB8"], [1, GOLD]],
            zmin=0, zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in trans],
            texttemplate="%{text}",
            textfont={"size": 16, "color": TEXT, "family": "Calibri"},
            colorbar=dict(tickfont=dict(color=SUBTEXT)),
        ))
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"),
                          title=dict(text="<b>P(next state | this state)</b>",
                                     font=dict(color=GOLD)))
        st.plotly_chart(fig, use_container_width=True)
    with cR:
        for lbl, p in zip(order_labels, np.diag(trans)):
            st.metric(f"Dwell · {lbl}", f"{1 / (1 - p + 1e-10):.1f} weeks")
        stickiness = float(np.mean(np.diag(trans)))
        st.markdown(
            f"<div style='background:{SURFACE2}; border-left:4px solid {GOLD};"
            f" padding:10px 14px; border-radius:4px; color:{TEXT}; "
            f" font-size:11.5px; margin-top:10px;'>"
            f"<b style='color:{GOLD};'>STICKINESS</b> = {stickiness:.3f}<br>"
            f"Healthy weekly-equity range: 0.85 – 0.97."
            f"</div>",
            unsafe_allow_html=True,
        )

    # Mean feature values per regime (notebook §8.1)
    st.markdown(f"<h3 style='color:{GOLD};'>Mean Feature Values per Regime (Training)</h3>",
                unsafe_allow_html=True)
    states = M[f"{key}_states"]
    lblmap_active = M[f"{key}_map"]
    train_regime = pd.Series(states, index=M["X_train"].index).map(lblmap_active)
    feat_means   = (M["X_train"].assign(Regime=train_regime)
                     .groupby("Regime")[FEATURE_COLS].mean().round(3)
                     .reindex(["Bear", "Neutral", "Bull"]))
    st.dataframe(feat_means, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Backtest (mirrors notebook §12)
# ═════════════════════════════════════════════════════════════════════════════
with tab_bt:
    # In-sample regime + returns
    train_states = M[f"{key}_states"]
    train_regime = pd.Series(train_states, index=M["X_train"].index).map(M[f"{key}_map"])
    train_ret    = M["train_features"].loc[M["X_train"].index, "Log_Return"]
    bt_is = run_backtest(train_regime, train_ret)
    m_is  = perf_metrics(bt_is)

    view = st.radio("View", ["Out-of-sample (2025+)", "In-sample (2010-2024)"],
                    horizontal=True, index=0)
    bt = bt_oos if view.startswith("Out") else bt_is
    m  = m_oos  if view.startswith("Out") else m_is
    title_window = (f"OOS · {TEST_START} → {TEST_END}" if view.startswith("Out")
                    else f"IS · {TRAIN_START} → {TRAIN_END}")

    # Equity curve with regime shading
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    reg_series = bt["regime"]
    prev_d, prev_r = bt.index[0], reg_series.iloc[0]
    for i in range(1, len(reg_series)):
        cur = reg_series.iloc[i]
        if cur != prev_r or i == len(reg_series) - 1:
            fig.add_vrect(x0=str(prev_d), x1=str(bt.index[i]),
                          fillcolor=COLORS[prev_r], opacity=0.16,
                          layer="below", line_width=0)
            prev_d, prev_r = bt.index[i], cur

    fig.add_trace(go.Scatter(x=bt.index, y=bt["cum_strat"], name="Strategy",
                             line=dict(color=GOLD, width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["cum_bnh"], name="Buy & Hold",
                             line=dict(color=SUBTEXT, width=2, dash="dash")),
                  secondary_y=False)

    fig.add_annotation(x=bt.index[-1], y=bt["cum_bnh"].iloc[-1],
                       text=f"${bt['cum_bnh'].iloc[-1]:.3f}",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(color=SUBTEXT, size=12))
    fig.add_annotation(x=bt.index[-1], y=bt["cum_strat"].iloc[-1],
                       text=f"${bt['cum_strat'].iloc[-1]:.3f}",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(color=GOLD, size=12))

    for lbl, c in COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=c, symbol="square", opacity=0.5),
                                 name=lbl, showlegend=True))

    fig.update_layout(
        title=dict(text=f"<b>Equity Curve</b>  ·  {model_choice} K=3  ·  {title_window}",
                   font=dict(color=GOLD)),
        height=520, hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_yaxes(title_text="Portfolio Value ($1 invested)", secondary_y=False)
    fig.update_yaxes(visible=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    dd_s = (bt["cum_strat"] - bt["cum_strat"].cummax()) / bt["cum_strat"].cummax()
    dd_b = (bt["cum_bnh"]   - bt["cum_bnh"].cummax())   / bt["cum_bnh"].cummax()
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_s.index, y=dd_s * 100, name="Strategy",
                                line=dict(color=GOLD, width=2),
                                fill="tozeroy", fillcolor="rgba(229,181,61,0.2)"))
    fig_dd.add_trace(go.Scatter(x=dd_b.index, y=dd_b * 100, name="Buy & Hold",
                                line=dict(color=SUBTEXT, dash="dash")))
    fig_dd.update_layout(title=dict(text="<b>Drawdown (%)</b>", font=dict(color=GOLD)),
                         yaxis_title="Drawdown (%)", height=300, hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Metrics table
    st.markdown(f"<h3 style='color:{GOLD};'>Performance — {title_window}</h3>",
                unsafe_allow_html=True)
    rows = [
        ["CAGR",                  f"{m['cagr_s']:.2%}",  f"{m['cagr_b']:.2%}"],
        ["Annualised Volatility", f"{m['vol_s']:.2%}",   f"{m['vol_b']:.2%}"],
        ["Sharpe Ratio",          f"{m['shrp_s']:.3f}",  f"{m['shrp_b']:.3f}"],
        ["Max Drawdown",          f"{m['mdd_s']:.2%}",   f"{m['mdd_b']:.2%}"],
        ["Calmar Ratio",          f"{m['calm_s']:.3f}",  f"{m['calm_b']:.3f}"],
        ["Win Rate vs B&H",       f"{m['win']:.2%}",     "—"],
        ["Regime Switches",       f"{m['switches']}",     "—"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Strategy", "Buy & Hold"]),
                 use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Statistical Validation (mirrors notebook §11, §12.1, §12.2)
# ═════════════════════════════════════════════════════════════════════════════
with tab_val:
    train_ret    = M["train_features"].loc[M["X_train"].index, "Log_Return"]
    gmm_reg_tr   = pd.Series(M["gmm_states"], index=M["X_train"].index).map(M["gmm_map"])
    hmm_reg_tr   = pd.Series(M["hmm_states"], index=M["X_train"].index).map(M["hmm_map"])

    f_g, p_g = f_oneway(
        train_ret[gmm_reg_tr == "Bull"],
        train_ret[gmm_reg_tr == "Neutral"],
        train_ret[gmm_reg_tr == "Bear"],
    )
    f_h, p_h = f_oneway(
        train_ret[hmm_reg_tr == "Bull"],
        train_ret[hmm_reg_tr == "Neutral"],
        train_ret[hmm_reg_tr == "Bear"],
    )
    lv_g, lp_g = levene(
        train_ret[gmm_reg_tr == "Bull"],
        train_ret[gmm_reg_tr == "Neutral"],
        train_ret[gmm_reg_tr == "Bear"],
    )
    lv_h, lp_h = levene(
        train_ret[hmm_reg_tr == "Bull"],
        train_ret[hmm_reg_tr == "Neutral"],
        train_ret[hmm_reg_tr == "Bear"],
    )

    cL, cR = st.columns(2)
    with cL:
        st.markdown(
            f"""<div style='background:{SURFACE}; border-left:4px solid {GOLD};
                          padding:14px 18px; border-radius:6px;'>
              <div style='color:{GOLD}; font-weight:700; font-size:11px;
                          letter-spacing:1px;'>ONE-WAY ANOVA · mean returns differ?</div>
              <div style='font-family:Consolas, monospace; color:{TEXT};
                          font-size:14px; margin-top:10px; line-height:1.7;'>
                GMM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;F = {f_g:.2f}&nbsp;&nbsp;&nbsp;&nbsp;p = {p_g:.4g}<br>
                HMM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;F = {f_h:.2f}&nbsp;&nbsp;&nbsp;&nbsp;p = {p_h:.4g}
              </div>
              <div style='color:{SUBTEXT}; font-size:11px; margin-top:10px;'>
                GMM separates strongly by mean return; HMM groups more by volatility regime than direction.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cR:
        st.markdown(
            f"""<div style='background:{SURFACE}; border-left:4px solid {CYAN};
                          padding:14px 18px; border-radius:6px;'>
              <div style='color:{CYAN}; font-weight:700; font-size:11px;
                          letter-spacing:1px;'>LEVENE'S TEST · variances differ?</div>
              <div style='font-family:Consolas, monospace; color:{TEXT};
                          font-size:14px; margin-top:10px; line-height:1.7;'>
                GMM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;F = {lv_g:.2f}&nbsp;&nbsp;&nbsp;&nbsp;p = {lp_g:.4g}<br>
                HMM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;F = {lv_h:.2f}&nbsp;&nbsp;&nbsp;&nbsp;p = {lp_h:.4g}
              </div>
              <div style='color:{SUBTEXT}; font-size:11px; margin-top:10px;'>
                Both models cleanly separate volatility regimes — the more useful result for risk management.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Regime distribution OOS
    st.markdown(f"<h3 style='color:{GOLD};'>Regime Distribution (OOS)</h3>",
                unsafe_allow_html=True)
    dist = regime.value_counts(normalize=True).mul(100).round(1).reindex(
        ["Bear", "Neutral", "Bull"]).fillna(0)
    fig = go.Figure(go.Bar(
        x=dist.index, y=dist.values,
        marker_color=[COLORS[r] for r in dist.index],
        marker_line_color=BG, marker_line_width=2,
        text=[f"{v}%" for v in dist.values], textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(yaxis_title="% of weeks", height=350,
                      yaxis=dict(range=[0, max(dist.values) * 1.25 if dist.values.max() else 100]))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — Predictions
# ═════════════════════════════════════════════════════════════════════════════
with tab_table:
    n = st.slider("Show last N weeks", 5, len(regime), min(20, len(regime)))
    table = pd.DataFrame({
        "Price":         preds["price"].round(2),
        "Regime":        regime,
        "Confidence":    conf.round(3),
        "Weekly Return": preds["log_ret"].apply(lambda x: f"{x:+.2%}"),
    }).tail(n).iloc[::-1]
    st.dataframe(table, use_container_width=True)
    csv = table.to_csv().encode("utf-8")
    st.download_button("📥 Download CSV", csv,
                       file_name=f"regime_predictions_{model_choice}_K3.csv",
                       mime="text/csv")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <hr style="border:none; border-top:1px solid {RULE}; margin-top:32px;">
    <div style="display:flex; justify-content:space-between; padding:8px 0;
                color:{SUBTEXT}; font-size:10px; font-weight:700;
                letter-spacing:0.6px;">
      <div>MARKET REGIME DETECTION  ·  CAPSTONE LIVE DASHBOARD</div>
      <div style="color:{GOLD};">FIRST LOAD ~60s  ·  CACHED PER SESSION</div>
    </div>
    """,
    unsafe_allow_html=True,
)
