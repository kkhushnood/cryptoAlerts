# -*- coding: utf-8 -*-
"""
Crypto .618 / .786 Touch + 15m Running Strong Reversal — Pakistan Time — ATR-adaptive
FIX: h1_last_touch_level (and h1_touched_level) chooses level by relevant candle extreme (LOW for up-leg, HIGH for down-leg)
Output: Single Excel file with 2 sheets -> Sheet1: Summary, Sheet2: Detail

ADDED: Reclaim (0.618/0.786) + next-1H reversal + 15m confirmation (>=2 closes + EMA21 alignment)
Entry/TP/SL logic integrated; cc_entry/cc_sl/cc_tp1/cc_tp2 are filled when strategy fires.

UPDATED: Adds NEW_TP and NEW_SL in Summary (strategy TP/SL per your rules)
"""

import os, time, math, requests, numpy as np, pandas as pd, ccxt
from datetime import datetime, timezone

# ----------------------------- Config -----------------------------
VS_CURRENCY = "usd"
TOP_N = 100

EXCHANGE_ID = "binance"
PAIR_QUOTE = "USDT"

TIMEFRAME = "1h"        # core TF
M15_TIMEFRAME = "15m"   # confirmation TF

CANDLES_LIMIT = 1200
M15_LIMIT = 1000

PIVOT_LEFT = 10
PIVOT_RIGHT = 10

GOLDEN_RATIO = 0.618
RATIO_786 = 0.786
NEAR_THRESHOLD_PCT = 0.005  # 0.5%

REQUESTS_TIMEOUT = 20
USE_RUNNING_CANDLE = False  # 1H will use last CLOSED candle (as required)

import os
SAVE_DIR = os.environ.get("SAVE_DIR", "./out")  # portable: env var ya local ./out

FILE_PREFIX = "golden_pocket_1h"

PK_TZ = "Asia/Karachi"

EXCLUDE_STABLE_SYMBOLS = {"usdt","usdc","busd","tusd","usde","dai","fdusd","susd","lusd","gusd"}

TREND_TIMEFRAMES = {"1D": "1d", "4H": "4h", "1H": "1h"}
TREND_CANDLES_LIMIT = 300

# .618 band controls (display/risk only)
GZ_BAND_PCT_BASE = 0.002
ATR_MULT_GZ = 0.75
MIN_GZ_WIDTH_ABS = 0.0

# Base tolerances (fraction of price)
SR_MIN_TOUCHES = 3
SR_TOL_PCT_BASE = 0.004
EMA_TOL_PCT_BASE = 0.004
TL_TOL_PCT_BASE = 0.005
VP_TOL_PCT_BASE = 0.005

# ATR scaling multipliers
ATR_MULT_SR = 1.00
ATR_MULT_EMA = 0.75
ATR_MULT_TL = 1.20
ATR_MULT_VP = 1.20

# Recommendation (risk model)
ATR_SL_MULT = 0.50
WIDTH_BUF_MULT = 0.15
MIN_SL_PCT = 0.003
TP1_R_MULT = 1.0
TP2_R_MULT = 2.0
ENTRY_MODE = "close"  # "close" or "mid"

LOG_ERRORS = True
MAX_RETRIES = 3
RETRY_BACKOFF = 0.75

# ----------------------------- Reclaim+Reversal strategy params -----------------------------
TOUCH_TOL_FRAC = 0.0015          # 0.15% tolerance for level touch/cross
RECLAIM_REQUIRE_CROSS = True     # wick must cross beyond level
REVERSAL_MIN_RETRACE = 0.50      # next-1H body >=50% retrace or strong wick
M15_CLOSINGS_REQUIRED = 2        # >= N closes on 15m in the hour window
USE_M15_EMA21_FILTER = True      # require some EMA21 alignment on 15m
ATR_BUFFER_MULT = 0.50           # SL buffer = this * ATR(14)
TARGET_FOR_618 = 0.5             # if reclaim=0.618 -> TP at 0.5
SL_MODE = "leg_100"              # 'leg_100' => invalidation at leg extreme (± ATR buffer)

# ----------------------------- Confidence weights -----------------------------
CONF_WEIGHTS = {
    "proximity": 0.40,
    "confluences": 0.30,
    "candle": 0.10,
    "trend_agree": 0.10,
    "momentum": 0.10
}
PROX_SOFTEN_MULT = 2.0

# ----------------------------- SUMMARY columns (order locked) -----------------------------
SUMMARY_COLUMNS = [
    "pair","timeframe","recommendation","customConfirmation",
    "swing_low","swing_high","currentPrice","goldenZone",
    "trend_1D","trend_4H","trend_1H","trend","structu",
    "Price Action candle confirmation","ema21","rsi14","confluence_king_score",
    "cc_entry","cc_sl","cc_tp1","cc_tp2",
    # ---- Added columns ----
    "NEW_TP","NEW_SL",               # <--- NEW columns here
    "HitMinuts",
    "h1_touched_level",
    "h1_last_touch_level",
    "m15_running_reversal",
    "m15_reversal_level",
    "m15_reversal_signal"
]

# ----------------------------- Preferred console order (optional) -----------------------------
COLUMN_ORDER = [
    "pair","timeframe","run_remaining_hms","currentPrice",
    "fibStart","fibEnd","goldenZone","where_vs_gz","nearest_gz_bound",
    "Fibonacci zone","Price Action candle confirmation","Volume/Orderflow","Momentum/Divergence","Micro-structure trigger",
    "structu","trend","trend_1D","trend_4H","trend_1H",
    "candle_signal","candle_score",
    "ema21","ema21_slope","rsi14","rsi_bias","conf_ema21_trend","conf_rsi_trend",
    "momentum_bias","momentum_score",
    "confluence_king_score","confluence_king_flags","rsi_divergence",
    "conf_sr_match","conf_ema_21_50","conf_trendline_channel","conf_vp_hvn_lvn",
    "order_block_value","order_block_confirm",
    "recommendation","confidence_level","atr14",
    "swing_low","swing_high",
    "strongSupport","strongSupportTouches","strongSupportLevel",
    "strongResistance","strongResistanceTouches","strongResistanceLevel",
    "triggerTime_min",
    # 15m visibility fields
    "m15_touch_time","m15_touch_high","m15_touch_low","m15_low_vs_ema21","m15_high_vs_ema21","m15_rsi14",
    "customConfirmation",
    # TP/SL derived from customConfirmation
    "cc_entry","cc_sl","cc_tp1","cc_tp2",
    "HitMinuts",
    # Hard-filter columns for this strategy
    "h1_touched_level",
    "h1_last_touch_level",
    "m15_running_reversal","m15_reversal_level","m15_reversal_signal",
    # Convenience
    "NEW_TP","NEW_SL"
]

# ----------------------------- Helpers -----------------------------
def parse_tf_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}")

TF_MIN = parse_tf_minutes(TIMEFRAME)
M15_MIN = parse_tf_minutes(M15_TIMEFRAME)

def fetch_with_retry(func, *args, **kwargs):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if LOG_ERRORS:
                print(f"[retry {attempt}/{MAX_RETRIES}] {func.__name__}: {e}")
            time.sleep((RETRY_BACKOFF ** attempt) + 0.05 * attempt)
    raise last_err

def get_top_coins_by_mcap(n=TOP_N, vs_currency="usd"):
    per_page = max(100, min(250, n * 3))
    url = ("https://api.coingecko.com/api/v3/coins/markets"
           f"?vs_currency={vs_currency}&order=market_cap_desc&per_page={per_page}&page=1&sparkline=false")
    r = fetch_with_retry(requests.get, url, timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    cleaned = [c for c in data if c.get("symbol","").lower() not in EXCLUDE_STABLE_SYMBOLS]
    return cleaned[:n]

def load_exchange(exchange_id=EXCHANGE_ID):
    ex_class = getattr(ccxt, exchange_id)
    exchange = ex_class({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    exchange.load_markets()
    return exchange

def map_symbol_to_usdt(exchange, cg_symbol: str):
    target = cg_symbol.upper()
    direct = f"{target}/{PAIR_QUOTE}"
    if direct in exchange.markets:
        return direct
    for sym, m in exchange.markets.items():
        if m.get("quote") == PAIR_QUOTE and m.get("base","") == target:
            return sym
    return None

def fetch_ohlcv(exchange, symbol, timeframe=TIMEFRAME, limit=CANDLES_LIMIT):
    def _call():
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    ohlcv = fetch_with_retry(_call)
    if not ohlcv or len(ohlcv) < (PIVOT_LEFT + PIVOT_RIGHT + 5):
        return None
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df

# -------- STRICT (no ties) pivot detector --------
def find_pivots(df: pd.DataFrame, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    highs_idx, lows_idx = [], []
    highs = df["high"].values; lows = df["low"].values
    for i in range(left, len(df) - right):
        win_hi_left = highs[i-left:i]; win_hi_right = highs[i+1:i+right+1]
        win_lo_left = lows[i-left:i]; win_lo_right = lows[i+1:i+right+1]
        if highs[i] > win_hi_left.max() and highs[i] > win_hi_right.max():
            highs_idx.append(i)
        if lows[i] < win_lo_left.min() and lows[i] < win_lo_right.min():
            lows_idx.append(i)
    return highs_idx, lows_idx

def last_two(items):
    if not items: return None, None
    if len(items) == 1: return items[-1], None
    return items[-1], items[-2]

def classify_structure(df: pd.DataFrame, ph_idx: list, pl_idx: list):
    last_hi_i, prev_hi_i = last_two(ph_idx)
    last_lo_i, prev_lo_i = last_two(pl_idx)
    hi_lbl = lo_lbl = None
    if last_hi_i is not None and prev_hi_i is not None:
        hi_lbl = "HH" if df["high"].iloc[last_hi_i] > df["high"].iloc[prev_hi_i] else "LH"
    if last_lo_i is not None and prev_lo_i is not None:
        lo_lbl = "HL" if df["low"].iloc[last_lo_i] > df["low"].iloc[prev_lo_i] else "LL"
    trend = "Sideways"
    if hi_lbl == "HH" and lo_lbl == "HL": trend = "Uptrend"
    elif hi_lbl == "LH" and lo_lbl == "LL": trend = "Downtrend"
    structure = f"{hi_lbl}+{lo_lbl}" if hi_lbl and lo_lbl else None
    return hi_lbl, lo_lbl, structure, trend

def level_from_leg(L: float, H: float, ratio: float, direction_up: bool) -> float:
    if direction_up:
        return float(H - (H - L) * ratio)
    else:
        return float(L + (H - L) * ratio)

def active_leg_smart(df: pd.DataFrame, ph_idx: list, pl_idx: list,
                     stable_idx: int, left: int = 10, right: int = 10):
    if not ph_idx or not pl_idx:
        return None, None, None, None
    last_hi_i = ph_idx[-1]
    last_lo_i = pl_idx[-1]
    if last_lo_i < last_hi_i:
        lows_before_hi = [i for i in pl_idx if i < last_hi_i]
        if not lows_before_hi:
            return None, None, None, None
        start_low_i = lows_before_hi[-1]
        L = float(df["low"].iloc[start_low_i])
        H = float(df["high"].iloc[last_hi_i])
        return True, L, H, (start_low_i, last_hi_i)
    highs_before_lo = [i for i in ph_idx if i < last_lo_i]
    if not highs_before_lo:
        return None, None, None, None
    start_hi_i = highs_before_lo[-1]
    H = float(df["high"].iloc[start_hi_i])
    L = float(df["low"].iloc[last_lo_i])
    return False, L, H, (start_hi_i, last_lo_i)

def distance_to_zone(price: float, z_low: float, z_high: float):
    if z_low is None or z_high is None: return None, None, None, None
    if z_low <= price <= z_high:
        d_low = price - z_low; d_high = z_high - price
        nearest = z_low if d_low <= d_high else z_high
        return 0.0, 0.0, "inside", nearest
    if price > z_high:
        d = price - z_high
        return d, d / price, "above", z_high
    d = z_low - price
    return d, d / price, "below", z_low

def fmt_hms(seconds: float) -> str:
    if seconds is None: return ""
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600; m = (seconds % 3600) // 60; s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------------- Technical indicators ----------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = series_close.shift(1)
    tr = pd.concat([
        (series_high - series_low).abs(),
        (series_high - prev_close).abs(),
        (series_low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def rsi(series_close: pd.Series, period: int = 14) -> pd.Series:
    delta = series_close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def within_tol(a: float, b: float, pct: float) -> bool:
    if a is None or b is None: return False
    return abs(a - b) <= max(a, b) * pct

# ---------------- Confluence helpers (boolean) ----------------
def sr_clusters_from_pivots(df: pd.DataFrame, ph_idx: list, pl_idx: list, tol_pct: float):
    prices = [float(df["high"].iloc[i]) for i in ph_idx] + [float(df["low"].iloc[i]) for i in pl_idx]
    if not prices: return []
    prices.sort(); clusters = []; current = [prices[0]]
    for p in prices[1:]:
        mean = float(np.mean(current)); tol_abs = mean * tol_pct
        if abs(p - mean) <= tol_abs:
            current.append(p)
        else:
            clusters.append((float(np.mean(current)), len(current))); current = [p]
    clusters.append((float(np.mean(current)), len(current)))
    return clusters

def sr_strong_levels(df, ph_idx, pl_idx, min_touches: int, tol_pct: float):
    clusters = sr_clusters_from_pivots(df, ph_idx, pl_idx, tol_pct)
    return [(lvl, cnt) for (lvl, cnt) in clusters if cnt >= min_touches]

def conf_sr_match_bool(gz_prices, strong_levels, tol_pct: float) -> bool:
    for g in gz_prices:
        for lvl, _ in strong_levels:
            if within_tol(g, lvl, tol_pct):
                return True
    return False

def fit_trendline_time(xs_minutes: np.ndarray, ys: np.ndarray):
    if len(xs_minutes) < 2: return None, None
    m, c = np.polyfit(xs_minutes, ys, 1)
    return float(m), float(c)

def conf_trendline_channel_bool(df: pd.DataFrame, ph_idx: list, pl_idx: list, check_time_min: float, gz_prices, tol_pct: float) -> bool:
    def check_line(piv_idx, col):
        if len(piv_idx) >= 3:
            sel = piv_idx[-7:]
            xs = np.array([(df["time"].iloc[i].value // 10**9) / 60.0 for i in sel], dtype=float)
            ys = np.array([df[col].iloc[i] for i in sel], dtype=float)
            m, c = fit_trendline_time(xs, ys)
            if m is not None:
                y_at = m * check_time_min + c
                for g in gz_prices:
                    if within_tol(y_at, g, tol_pct):
                        return True
        return False
    return check_line(ph_idx, "high") or check_line(pl_idx, "low")

def conf_ema_21_50_bool(df: pd.DataFrame, idx: int, gz_prices, tol_pct: float) -> bool:
    e21 = float(ema(df["close"], 21).iloc[idx])
    e50 = float(ema(df["close"], 50).iloc[idx])
    for val in (e21, e50):
        for g in gz_prices:
            if within_tol(val, g, tol_pct):
                return True
    return False

def conf_vp_hvn_lvn_bool(df: pd.DataFrame, lookback: int, bins: int, gz_prices, tol_pct: float):
    n = len(df)
    n = min(n, lookback)
    if n < 50: return False
    slice_df = df.iloc[-n:].copy()
    price_arr = ((slice_df["high"].values + slice_df["low"].values) / 2.0).astype(float)
    vol_arr = slice_df["volume"].values.astype(float)
    pmin, pmax = float(slice_df["low"].min()), float(slice_df["high"].max())
    if pmin <= 0 or pmax <= 0 or pmax <= pmin:
        return False
    hist, edges = np.histogram(price_arr, bins=bins, range=(pmin, pmax), weights=vol_arr)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if len(hist) < 3: return False
    hvn_idxs, lvn_idxs = [], []
    for i in range(1, len(hist)-1):
        if hist[i] >= hist[i-1] and hist[i] >= hist[i+1]: hvn_idxs.append(i)
        if hist[i] <= hist[i-1] and hist[i] <= hist[i+1]: lvn_idxs.append(i)
    def near(nodes):
        if not nodes: return False
        for g in gz_prices:
            idx = int(np.argmin(np.abs(centers[nodes] - g)))
            c = float(centers[nodes][idx])
            if within_tol(c, g, tol_pct): return True
        return False
    return near(hvn_idxs) or near(lvn_idxs)

# ---------------- Order Block (lightweight) ----------------
def detect_order_block(df: pd.DataFrame, lookback=60):
    n = len(df)
    L = max(2, min(lookback, n-5))
    ob_value, ob_confirm = None, False
    for i in range(n-L-1, n-5):
        o = float(df["open"].iloc[i]); c = float(df["close"].iloc[i])
        h = float(df["high"].iloc[i]); l = float(df["low"].iloc[i])
        body = abs(c-o); rng = max(h-l, 1e-12)
        if rng <= 0: continue
        if c < o and body/rng >= 0.35:
            h1 = float(df["high"].iloc[i+1]); h2 = float(df["high"].iloc[i+2]); h3 = float(df["high"].iloc[i+3]); c3 = float(df["close"].iloc[i+3])
            if (h1 > h) and (h2 > h1) and (h3 > h2) and (c3 > h):
                ob_value = (o + c) / 2.0; ob_confirm = True; break
        if c > o and body/rng >= 0.35:
            l1 = float(df["low"].iloc[i+1]); l2 = float(df["low"].iloc[i+2]); l3 = float(df["low"].iloc[i+3]); c3 = float(df["close"].iloc[i+3])
            if (l1 < l) and (l2 < l1) and (l3 < l2) and (c3 < l):
                ob_value = (o + c) / 2.0; ob_confirm = True; break
    if ob_value is None and n >= 3:
        io = n-3
        ob_value = (float(df["open"].iloc[io]) + float(df["close"].iloc[io]))/2.0
        ob_confirm = False
    return (round(float(ob_value), 6) if ob_value is not None else None), bool(ob_confirm)

# ---------------- RSI Divergence ----------------
def rsi_divergence(df: pd.DataFrame, rsi_series: pd.Series, lookback=40):
    n = len(df)
    if n < max(PIVOT_LEFT+PIVOT_RIGHT+5, lookback): return "none"
    ph_idx, pl_idx = find_pivots(df)
    if len(ph_idx) < 2 or len(pl_idx) < 2: return "none"
    h1, h2 = ph_idx[-1], ph_idx[-2]
    l1, l2 = pl_idx[-1], pl_idx[-2]
    price_HH = df["high"].iloc[h1] > df["high"].iloc[h2]
    price_LL = df["low"].iloc[l1] < df["low"].iloc[l2]
    rsi_HL = rsi_series.iloc[l1] > rsi_series.iloc[l2]
    rsi_LH = rsi_series.iloc[h1] < rsi_series.iloc[h2]
    if price_LL and rsi_HL: return "bullish"
    if price_HH and rsi_LH: return "bearish"
    return "none"

# ---------------- Candle signal aggregator (short) ----------------
def body_ok_vs_atr(cur_o, cur_c, atr_val, min_body_atr=0.15):
    body = abs(cur_c - cur_o)
    return atr_val is not None and atr_val > 0 and body >= (min_body_atr * atr_val)

def detect_bullish_engulfing(prev_o, prev_c, cur_o, cur_c, atr_val=None) -> bool:
    cond = (prev_c < prev_o) and (cur_c > cur_o) and (cur_o <= prev_c) and (cur_c >= prev_o)
    if atr_val is not None: cond = cond and body_ok_vs_atr(cur_o, cur_c, atr_val)
    return cond
def detect_bearish_engulfing(prev_o, prev_c, cur_o, cur_c, atr_val=None) -> bool:
    cond = (prev_c > prev_o) and (cur_c < cur_o) and (cur_o >= prev_c) and (cur_c <= prev_o)
    if atr_val is not None: cond = cond and body_ok_vs_atr(cur_o, cur_c, atr_val)
    return cond
def detect_hammer(cur_o, cur_c, cur_h, cur_l, atr_val=None) -> bool:
    body = abs(cur_c - cur_o); upper = cur_h - max(cur_o, cur_c); lower = min(cur_o, cur_c) - cur_l
    rng = cur_h - cur_l
    cond = (rng > 0) and (lower >= 2 * body) and (upper <= body) and (body > 0)
    if atr_val is not None: cond = cond and body_ok_vs_atr(cur_o, cur_c, atr_val, 0.10)
    return cond
def detect_shooting_star(cur_o, cur_c, cur_h, cur_l, atr_val=None) -> bool:
    body = abs(cur_c - cur_o); upper = cur_h - max(cur_o, cur_c); lower = min(cur_o, cur_c) - cur_l
    rng = cur_h - cur_l
    cond = (rng > 0) and (upper >= 2 * body) and (lower <= body) and (body > 0)
    if atr_val is not None: cond = cond and body_ok_vs_atr(cur_o, cur_c, atr_val, 0.10)
    return cond
def detect_pinbar_lower(cur_o, cur_c, cur_h, cur_l, atr_val=None) -> bool:
    body = abs(cur_c - cur_o); upper = cur_h - max(cur_o, cur_c); lower = min(cur_o, cur_c) - cur_l
    rng = cur_h - cur_l
    cond = (rng > 0) and (lower >= 2 * body) and (lower >= upper) and (body > 0)
    if atr_val is not None: cond = cond and body_ok_vs_atr(cur_o, cur_c, atr_val, 0.08)
    return cond
def detect_pinbar_upper(cur_o, cur_c, cur_h, cur_l, atr_val=None) -> bool:
    body = abs(cur_c - cur_o); upper = cur_h - max(cur_o, cur_c); lower = min(cur_o, cur_c) - cur_l
    rng = cur_h - cur_l
    cond = (rng > 0) and (upper >= 2 * body) and (upper >= lower) and (body > 0)
    if atr_val is not None: cond = cond and body_ok_vs_atr(cur_o, cur_c, atr_val, 0.08)
    return cond

def detect_marubozu(cur_o, cur_c, cur_h, cur_l, wick_ratio=0.15):
    rng = cur_h - cur_l
    if rng <= 0: return None
    upper_w = cur_h - max(cur_o, cur_c)
    lower_w = min(cur_o, cur_c) - cur_l
    if upper_w < 0 or lower_w < 0: return None
    if (upper_w + lower_w) / rng <= wick_ratio:
        return "bull" if cur_c > cur_o else "bear"
    return None

def detect_piercing_line(prev_o, prev_c, cur_o, cur_c):
    prev_bear = prev_c < prev_o
    if not prev_bear: return False
    prev_mid = (prev_o + prev_c) / 2.0
    return (cur_c > cur_o) and (cur_c >= prev_mid)

def detect_dark_cloud(prev_o, prev_c, cur_o, cur_c):
    prev_bull = prev_c > prev_o
    if not prev_bull: return False
    prev_mid = (prev_o + prev_c) / 2.0
    return (cur_c < cur_o) and (cur_c <= prev_mid)

def detect_three_white_soldiers(o2,c2,o1,c1,o0,c0):
    return (c2>o2) and (c1>o1) and (c0>o0) and (c2<c1<c0)

def detect_three_black_crows(o2,c2,o1,c1,o0,c0):
    return (c2<o2) and (c1<o1) and (c0<o0) and (c2>c1>c0)

def detect_morning_star(o2,c2,o1,c1,o0,c0):
    prev_bear = c2 < o2
    small_mid = abs(c1 - o1) <= abs(c2 - o2)*0.6
    pierce = c0 > o0 and c0 >= (o2 + c2)/2.0
    return prev_bear and small_mid and pierce

def detect_evening_star(o2,c2,o1,c1,o0,c0):
    prev_bull = c2 > o2
    small_mid = abs(c1 - o1) <= abs(c2 - o2)*0.6
    cover = c0 < o0 and c0 <= (o2 + c2)/2.0
    return prev_bull and small_mid and cover

def derive_candle_signal(trend_tf: str, where: str, dist_pct_price: float, df: pd.DataFrame, stable_idx: int, atr_val: float):
    in_gz = (where == "inside") or (dist_pct_price is not None and dist_pct_price <= NEAR_THRESHOLD_PCT)
    n = len(df)
    if n < 2: return "Normal", 0.0, "none"
    i0 = stable_idx
    if i0 - 1 < -n: return "Normal", 0.0, "none"
    cur_o = float(df["open"].iloc[i0]); cur_c = float(df["close"].iloc[i0])
    cur_h = float(df["high"].iloc[i0]); cur_l = float(df["low"].iloc[i0])
    prev_o = float(df["open"].iloc[i0-1]); prev_c = float(df["close"].iloc[i0-1])

    bull_signals = []; bear_signals = []
    if detect_bullish_engulfing(prev_o, prev_c, cur_o, cur_c, atr_val): bull_signals.append("Bullish Engulfing")
    if detect_hammer(cur_o, cur_c, cur_h, cur_l, atr_val): bull_signals.append("Hammer")
    if detect_pinbar_lower(cur_o, cur_c, cur_h, cur_l, atr_val): bull_signals.append("Pinbar Lower")
    maru = detect_marubozu(cur_o, cur_c, cur_h, cur_l)
    if maru == "bull": bull_signals.append("Marubozu")
    if detect_piercing_line(prev_o, prev_c, cur_o, cur_c): bull_signals.append("Piercing Line")

    if detect_bearish_engulfing(prev_o, prev_c, cur_o, cur_c, atr_val): bear_signals.append("Bearish Engulfing")
    if detect_shooting_star(cur_o, cur_c, cur_h, cur_l, atr_val): bear_signals.append("Shooting Star")
    if detect_pinbar_upper(cur_o, cur_c, cur_h, cur_l, atr_val): bear_signals.append("Pinbar Upper")
    if maru == "bear": bear_signals.append("Marubozu")
    if detect_dark_cloud(prev_o, prev_c, cur_o, cur_c): bear_signals.append("Dark Cloud Cover")

    if i0 - 2 >= -n:
        o2 = float(df["open"].iloc[i0-2]); c2 = float(df["close"].iloc[i0-2])
        o1 = prev_o; c1 = prev_c
        o0 = cur_o; c0 = cur_c
        if detect_three_white_soldiers(o2,c2,o1,c1,o0,c0): bull_signals.append("Three White Soldiers")
        if detect_three_black_crows(o2,c2,o1,c1,o0,c0): bear_signals.append("Three Black Crows")
        if detect_morning_star(o2,c2,o1,c1,o0,c0): bull_signals.append("Morning Star")
        if detect_evening_star(o2,c2,o1,c1,o0,c0): bear_signals.append("Evening Star")

    bull_priority = ["Three White Soldiers","Morning Star","Bullish Engulfing","Piercing Line","Hammer","Pinbar Lower","Marubozu"]
    bear_priority = ["Three Black Crows","Evening Star","Bearish Engulfing","Dark Cloud Cover","Shooting Star","Pinbar Upper","Marubozu"]

    def pick(sig_list, order):
        for t in order:
            if t in sig_list: return t
        return sig_list[0] if sig_list else None

    top_bull = pick(bull_signals, bull_priority)
    top_bear = pick(bear_signals, bear_priority)

    if not in_gz:
        if top_bull and top_bear: return f"Mixed (Outside GZ): {top_bull} / {top_bear}", 0.0, "none"
        if top_bull: return f"Info (Outside GZ): {top_bull}", 0.0, "bull"
        if top_bear: return f"Info (Outside GZ): {top_bear}", 0.0, "bear"
        return "Normal", 0.0, "none"

    if trend_tf == "Uptrend":
        if top_bull: return f"Strong Bullish: {top_bull}", 1.0, "bull"
        if top_bear: return f"Mixed: {top_bear} (against trend)", 0.4, "bear"
        return "Normal", 0.0, "none"

    if trend_tf == "Downtrend":
        if top_bear: return f"Strong Bearish: {top_bear}", 1.0, "bear"
        if top_bull: return f"Mixed: {top_bull} (against trend)", 0.4, "bull"
        return "Normal", 0.0, "none"

    if top_bull and top_bear: return f"Mixed: {top_bull} / {top_bear}", 0.4, "none"
    if top_bull: return f"Bullish: {top_bull}", 0.4, "bull"
    if top_bear: return f"Bearish: {top_bear}", 0.4, "bear"
    return "Normal", 0.0, "none"

# ---------------- Momentum helpers (EMA21 + RSI) ----------------
def ema21_confirm_for_trend(close_now: float, ema21_now: float, ema21_prev: float, trend_tf: str) -> bool:
    slope = ema21_now - ema21_prev
    if trend_tf == "Uptrend": return (close_now >= ema21_now) and (slope >= 0)
    if trend_tf == "Downtrend": return (close_now <= ema21_now) and (slope <= 0)
    return abs(close_now - ema21_now) / max(close_now, 1e-12) <= 0.002

def rsi_bias_from_value(rsi_val: float) -> str:
    if rsi_val is None: return "Neutral"
    if rsi_val >= 55.0: return "Long"
    if rsi_val <= 45.0: return "Short"
    return "Neutral"

# ---------------- Confidence computation ----------------
def compute_confidence_level(close_price, gz_level, conf_sr, conf_ema, conf_tl, conf_vp, candle_score, trend_tf, leg_up, momentum_score) -> int:
    if close_price is None or gz_level is None or close_price <= 0: return 0
    dist_frac = abs(close_price - gz_level) / close_price
    prox_full = NEAR_THRESHOLD_PCT * PROX_SOFTEN_MULT
    prox_score = max(0.0, 1.0 - (dist_frac / prox_full)) if prox_full > 0 else 0.0
    conf_score = (
        (1.0 if conf_sr else 0.0) * 0.30 +
        (1.0 if conf_ema else 0.0) * 0.30 +
        (1.0 if conf_tl else 0.0) * 0.20 +
        (1.0 if conf_vp else 0.0) * 0.20
    )
    conf_score = min(1.0, conf_score)
    trend_score = 1.0 if (leg_up and trend_tf=="Uptrend") or ((not leg_up) and trend_tf=="Downtrend") else 0.0
    mom_score = max(0.0, min(1.0, momentum_score))
    total = (
        CONF_WEIGHTS["proximity"] * prox_score +
        CONF_WEIGHTS["confluences"] * conf_score +
        CONF_WEIGHTS["candle"] * candle_score +
        CONF_WEIGHTS["trend_agree"] * trend_score +
        CONF_WEIGHTS["momentum"] * mom_score
    )
    return int(round(max(0.0, min(1.0, total)) * 100))

# ---------------- Recommendation builder ----------------
def build_recommendation(direction: str, close_stab: float, gz_low: float, gz_high: float, atr_val: float):
    width = max(gz_high - gz_low, 1e-12)
    min_price = max(close_stab, 1e-12)
    atr_buf = (ATR_SL_MULT * atr_val) if (atr_val is not None and atr_val > 0) else 0.0
    width_buf = WIDTH_BUF_MULT * width
    pct_buf = MIN_SL_PCT * min_price
    buffer_abs = max(atr_buf, width_buf, pct_buf)
    entry = close_stab if ENTRY_MODE == "close" else (gz_low + gz_high)/2.0
    entry = min(max(entry, gz_low), gz_high)
    if direction == "LONG":
        sl = gz_low - buffer_abs
        risk = max(entry - sl, 1e-12)
        tp1 = entry + TP1_R_MULT * risk
        tp2 = entry + TP2_R_MULT * risk
        tag = "LONG"
    else:
        sl = gz_high + buffer_abs
        risk = max(sl - entry, 1e-12)
        tp1 = entry - TP1_R_MULT * risk
        tp2 = entry - TP2_R_MULT * risk
        tag = "SHORT"
    r6 = lambda x: round(float(x), 6)
    return f"{tag} | entry: {r6(entry)} | SL: {r6(sl)} | TP1: {r6(tp1)} | TP2: {r6(tp2)}"

# ------------- TP/SL factory for customConfirmation (generic zone model) -------------
def build_tp_sl_from_zone(direction: str, close_stab: float, gz_low: float, gz_high: float, atr_val: float):
    width = max(gz_high - gz_low, 1e-12)
    min_price = max(close_stab, 1e-12)
    atr_buf = (ATR_SL_MULT * atr_val) if (atr_val is not None and atr_val > 0) else 0.0
    width_buf = WIDTH_BUF_MULT * width
    pct_buf = MIN_SL_PCT * min_price
    buffer_abs = max(atr_buf, width_buf, pct_buf)
    entry = close_stab if ENTRY_MODE == "close" else (gz_low + gz_high)/2.0
    entry = min(max(entry, gz_low), gz_high)
    if direction.upper() == "LONG":
        sl = gz_low - buffer_abs
        risk = max(entry - sl, 1e-12)
        tp1 = entry + TP1_R_MULT * risk
        tp2 = entry + TP2_R_MULT * risk
    else:
        sl = gz_high + buffer_abs
        risk = max(sl - entry, 1e-12)
        tp1 = entry - TP1_R_MULT * risk
        tp2 = entry - TP2_R_MULT * risk
    r6 = lambda x: round(float(x), 6)
    return r6(entry), r6(sl), r6(tp1), r6(tp2)

# ---------------- Confluence King score ----------------
def confluence_king_score_builder(conf_sr, conf_ema, conf_tl, conf_vp, ob_confirm, rsi_div):
    w_sr, w_ema, w_tl, w_vp, w_ob, w_rsi = 0.22, 0.18, 0.18, 0.18, 0.16, 0.08
    score = 0.0; flags = []
    if conf_sr: score += w_sr; flags.append("SR")
    if conf_ema: score += w_ema; flags.append("EMA")
    if conf_tl: score += w_tl; flags.append("TL/Channel")
    if conf_vp: score += w_vp; flags.append("HVN/LVN")
    if ob_confirm: score += w_ob; flags.append("OB")
    if rsi_div in ("bullish","bearish"): score += w_rsi; flags.append("RSIdiv")
    return int(round(score*100)), "+".join(flags) if flags else ""

# ---------------- customConfirmation (display-only) ----------------
def build_custom_confirmation(row: dict) -> str:
    try:
        zone_ok = str(row.get("Fibonacci zone","")).lower() in {"inside","near"} or str(row.get("where_vs_gz",""))=="inside"

        pac  = str(row.get("Price Action candle confirmation","")).lower()
        vol  = str(row.get("Volume/Orderflow","")).lower()
        momd = str(row.get("Momentum/Divergence","")).lower()
        trend = row.get("trend","")

        cks = float(row.get("confluence_king_score", 0) or 0)
        conf_cnt = int(bool(row.get("conf_sr_match")) + bool(row.get("conf_ema_21_50"))
                       + bool(row.get("conf_trendline_channel")) + bool(row.get("conf_vp_hvn_lvn")))
        conf_ok = (cks >= 55) and (conf_cnt >= 1)

        conf_level_ok = int(row.get("confidence_level", 0) or 0) >= 65

        pa_ok  = ("confirm" in pac) or ("strong" in str(row.get("candle_signal","")).lower())
        vol_ok = ("rising" in vol) or ("normal" in vol)
        mom_ok = ("momentum:aligned" in momd) or ("momentum:partial" in momd)

        is_long  = (trend=="Uptrend"   and zone_ok and conf_ok and conf_level_ok and pa_ok and vol_ok and mom_ok)
        is_short = (trend=="Downtrend" and zone_ok and conf_ok and conf_level_ok and pa_ok and vol_ok and mom_ok)

        if is_long:  return "STRONG_LONG"
        if is_short: return "STRONG_SHORT"
        return "INFO"
    except Exception:
        return "INFO"

# ---------------- SR details (display-only) ----------------
def _cluster_prices(prices: list, tol_pct: float):
    if not prices: return []
    prices = sorted([float(x) for x in prices])
    clusters, cur = [], [prices[0]]
    for p in prices[1:]:
        mean = float(np.mean(cur)); tol_abs = mean * tol_pct
        if abs(p - mean) <= tol_abs:
            cur.append(p)
        else:
            clusters.append((float(np.mean(cur)), len(cur))); cur = [p]
    clusters.append((float(np.mean(cur)), len(cur)))
    return clusters

def strong_sr_details(df: pd.DataFrame, ph_idx: list, pl_idx: list, min_touches: int, tol_pct: float, ref_price: float):
    high_prices = [float(df["high"].iloc[i]) for i in ph_idx]
    low_prices  = [float(df["low"].iloc[i])  for i in pl_idx]
    high_clusters = [(lvl,cnt) for (lvl,cnt) in _cluster_prices(high_prices, tol_pct) if cnt >= min_touches]
    low_clusters  = [(lvl,cnt) for (lvl,cnt) in _cluster_prices(low_prices,  tol_pct) if cnt >= min_touches]

    sup_candidates = [(lvl,cnt,abs(ref_price-lvl)) for (lvl,cnt) in low_clusters  if lvl <= ref_price]
    res_candidates = [(lvl,cnt,abs(ref_price-lvl)) for (lvl,cnt) in high_clusters if lvl >= ref_price]

    sup_lvl = sup_cnt = None
    res_lvl = res_cnt = None

    if sup_candidates:
        lvl, cnt, _ = min(sup_candidates, key=lambda x: x[2])
        sup_lvl, sup_cnt = float(lvl), int(cnt)
    elif low_clusters:
        lvl, cnt, _ = min([(lvl,cnt,abs(ref_price-lvl)) for (lvl,cnt) in low_clusters], key=lambda x: x[2])
        sup_lvl, sup_cnt = float(lvl), int(cnt)

    if res_candidates:
        lvl, cnt, _ = min(res_candidates, key=lambda x: x[2])
        res_lvl, res_cnt = float(lvl), int(cnt)
    elif high_clusters:
        lvl, cnt, _ = min([(lvl,cnt,abs(ref_price-lvl)) for (lvl,cnt) in high_clusters], key=lambda x: x[2])
        res_lvl, res_cnt = float(lvl), int(cnt)

    return sup_lvl, sup_cnt, res_lvl, res_cnt

# ---------------- 15m helpers ----------------
def get_last_closed_index(df: pd.DataFrame, tf_minutes: int) -> int:
    if df is None or df.empty: return -2
    now = pd.Timestamp.now(tz="UTC")
    last_open = df["time"].iloc[-1]
    last_close = last_open + pd.Timedelta(minutes=tf_minutes)
    return -1 if now >= last_close else -2

def fetch_m15_and_pick_last_closed(exchange, pair: str):
    df15 = fetch_ohlcv(exchange, pair, timeframe=M15_TIMEFRAME, limit=M15_LIMIT)
    if df15 is None or df15.empty: return None, None
    idx = get_last_closed_index(df15, M15_MIN)
    return df15, idx

def fetch_m15_and_pick_running(exchange, pair: str):
    df15 = fetch_ohlcv(exchange, pair, timeframe=M15_TIMEFRAME, limit=M15_LIMIT)
    if df15 is None or df15.empty: return None, None
    return df15, -1

def compute_m15_fields(df15: pd.DataFrame, idx: int, pk_tz: str):
    ema21_15 = ema(df15["close"], 21)
    rsi14_15 = rsi(df15["close"], 14)
    hi = float(df15["high"].iloc[idx]); lo = float(df15["low"].iloc[idx])
    o  = float(df15["open"].iloc[idx]); c  = float(df15["close"].iloc[idx])
    e  = float(ema21_15.iloc[idx]); r = float(rsi14_15.iloc[idx])
    low_vs_ema = "at"
    if lo > e: low_vs_ema = "above"
    elif lo < e: low_vs_ema = "below"
    high_vs_ema = "at"
    if hi > e: high_vs_ema = "above"
    elif hi < e: high_vs_ema = "below"
    t_pk = df15["time"].iloc[idx].tz_convert(pk_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    return {"o": o, "c": c, "hi": hi, "lo": lo, "ema": e, "rsi": r,
            "low_vs_ema": low_vs_ema, "high_vs_ema": high_vs_ema, "time_pk": t_pk}

# ---- 15m RUNNING candle strong reversal at 0.618/0.786 ----
def m15_running_strong_reversal(df15: pd.DataFrame, idx_run: int, gz_mid: float, gz_786: float,
                                leg_up: bool, atr_val_1h: float, tol_frac_base: float):
    if df15 is None or df15.empty: return False, None, None
    n = len(df15)
    if n < 3: return False, None, None

    close_now = float(df15["close"].iloc[idx_run]) if n else None
    base_price = max(close_now or 0.0, 1e-12)
    atr_frac = (atr_val_1h or 0.0) / base_price
    tol_frac = max(tol_frac_base, atr_frac * 0.75)  # adaptive
    tol_mid_abs = gz_mid * tol_frac
    tol_786_abs = gz_786 * tol_frac

    cur_o = float(df15["open"].iloc[idx_run]); cur_c = float(df15["close"].iloc[idx_run])
    cur_h = float(df15["high"].iloc[idx_run]); cur_l = float(df15["low"].iloc[idx_run])

    prev_o = float(df15["open"].iloc[idx_run-1]); prev_c = float(df15["close"].iloc[idx_run-1])

    touch_mid = (cur_l <= (gz_mid + tol_mid_abs)) and (cur_h >= (gz_mid - tol_mid_abs))
    touch_786 = (cur_l <= (gz_786 + tol_786_abs)) and (cur_h >= (gz_786 - tol_786_abs))
    if not (touch_mid or touch_786):
        return False, None, None

    want_bull = bool(leg_up)
    want_bear = not want_bull

    strong_bull = (
        detect_bullish_engulfing(prev_o, prev_c, cur_o, cur_c, atr_val_1h)
        or detect_hammer(cur_o, cur_c, cur_h, cur_l, atr_val_1h)
        or detect_pinbar_lower(cur_o, cur_c, cur_h, cur_l, atr_val_1h)
        or (detect_marubozu(cur_o, cur_c, cur_h, cur_l) == "bull")
    )
    strong_bear = (
        detect_bearish_engulfing(prev_o, prev_c, cur_o, cur_c, atr_val_1h)
        or detect_shooting_star(cur_o, cur_c, cur_h, cur_l, atr_val_1h)
        or detect_pinbar_upper(cur_o, cur_c, cur_h, cur_l, atr_val_1h)
        or (detect_marubozu(cur_o, cur_c, cur_h, cur_l) == "bear")
    )

    sig_name = None
    if want_bull and strong_bull and cur_c > cur_o:
        sig_name = "Bullish Reversal (15m)"
    elif want_bear and strong_bear and cur_c < cur_o:
        sig_name = "Bearish Reversal (15m)"
    else:
        return False, None, None

    level_str = "0.618" if touch_mid else ("0.786" if touch_786 else None)
    return True, level_str, sig_name

# ---- Most recent prior 1H CLOSED touch (0.618/0.786) — FIXED tie-break ----
def find_last_h1_touch_level(df: pd.DataFrame, stable_idx: int, gz_mid: float, gz_786: float, tol_frac: float, leg_up: bool) -> str:
    if df is None or df.empty: return None
    n = len(df)
    abs_idx = stable_idx if stable_idx >= 0 else (n + stable_idx)
    if abs_idx is None or abs_idx <= 0 or abs_idx >= n:
        return None

    tol_mid_abs = gz_mid * tol_frac
    tol_786_abs = gz_786 * tol_frac

    for j in range(abs_idx - 1, -1, -1):
        hi = float(df["high"].iloc[j]); lo = float(df["low"].iloc[j])
        touch_mid = (lo <= (gz_mid + tol_mid_abs)) and (hi >= (gz_mid - tol_mid_abs))
        touch_786 = (lo <= (gz_786 + tol_786_abs)) and (hi >= (gz_786 - tol_786_abs))
        if not (touch_mid or touch_786):
            continue
        if touch_mid and touch_786:
            if leg_up:
                d_mid = abs(lo - gz_mid); d_786 = abs(lo - gz_786)
            else:
                d_mid = abs(hi - gz_mid); d_786 = abs(hi - gz_786)
            return "0.618" if d_mid <= d_786 else "0.786"
        return "0.618" if touch_mid else "0.786"
    return None

# ---------------- Reclaim + Reversal + 15m confirmation helpers ----------------
def crossed_and_reclaimed(c_o, c_h, c_l, c_c, level, tol_frac, want_above: bool):
    tol = level * tol_frac
    if want_above:
        crossed = (c_l < level - (tol if RECLAIM_REQUIRE_CROSS else -tol))
        return crossed and (c_c >= level - tol)
    else:
        crossed = (c_h > level + (tol if RECLAIM_REQUIRE_CROSS else -tol))
        return crossed and (c_c <= level + tol)

def count_15m_closes(df15, t_start, t_end, level, above: bool):
    seg = df15[(df15["time"] >= t_start) & (df15["time"] < t_end)]
    if seg.empty: return 0, None
    cond = (seg["close"] >= level) if above else (seg["close"] <= level)
    return int(cond.sum()), seg

def m15_ok(df15, t_start, t_end, level, want_above: bool, use_ema=True):
    cnt, seg = count_15m_closes(df15, t_start, t_end, level, want_above)
    if cnt < M15_CLOSINGS_REQUIRED: return False
    if not use_ema: return True
    e21 = ema(seg["close"], 21)
    if want_above:
        return bool((seg["close"] >= e21).sum() >= max(1, len(seg)//3))
    else:
        return bool((seg["close"] <= e21).sum() >= max(1, len(seg)//3))

def is_bull_reversal(prev_o, prev_c, cur_o, cur_c, cur_h, cur_l):
    prev_body = abs(prev_c - prev_o)
    retr = (cur_c - cur_o) / max(prev_body, 1e-12)
    wick_lower = cur_o - cur_l if cur_c>=cur_o else cur_c - cur_l
    rng = cur_h - cur_l
    return (cur_c > cur_o) and (retr >= REVERSAL_MIN_RETRACE or (rng>0 and wick_lower/rng >= 0.4))

def is_bear_reversal(prev_o, prev_c, cur_o, cur_c, cur_h, cur_l):
    prev_body = abs(prev_c - prev_o)
    retr_now = (cur_o - cur_c) / max(prev_body, 1e-12)
    wick_upper = cur_h - (cur_o if cur_c>=cur_o else cur_c)
    rng = cur_h - cur_l
    return (cur_c < cur_o) and (retr_now >= REVERSAL_MIN_RETRACE or (rng>0 and wick_upper/rng >= 0.4))

# ---------------- Trend-only helper (safe) ----------------
def calc_trend_only(exchange, pair, timeframe, limit=300):
    try:
        data = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=min(limit, 1000))
        if not data: return "Skipped"
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
        ph, pl = find_pivots(df, PIVOT_LEFT, PIVOT_RIGHT)
        if len(ph) < 2 or len(pl) < 2: return "Sideways"
        _, _, _, trend = classify_structure(df, ph, pl)
        return trend
    except Exception:
        return "Skipped"

# ----------------------------- Main -----------------------------
def analyze_and_save():
    print(f"Timeframe: {TIMEFRAME} | 1H uses last CLOSED candle | 15m uses RUNNING candle")
    print("Fetching top coins (CoinGecko)...")
    try:
        coins = get_top_coins_by_mcap(TOP_N)
    except Exception as e:
        print(f"Failed to fetch top coins: {e}")
        coins = []
    print(f"Got {len(coins)} coins (stables excluded).")
    if not coins:
        print("No data fetched — aborting before exchange calls.")
        return pd.DataFrame()

    exchange = load_exchange()
    rows = []
    hard_flags = []  # final include flag for both conditions

    for c in coins:
        sym = c.get("symbol","").upper()
        name = c.get("name","").strip()
        pair = map_symbol_to_usdt(exchange, sym)
        if not pair:
            if LOG_ERRORS: print(f"[map] skipping {sym}: no {PAIR_QUOTE} spot pair")
            continue

        try:
            df = fetch_ohlcv(exchange, pair, timeframe=TIMEFRAME, limit=CANDLES_LIMIT)
            if df is None or df.empty:
                if LOG_ERRORS: print(f"[{pair}] no OHLCV")
                continue

            idx_closed_1h = get_last_closed_index(df, TF_MIN)
            stable_idx = (-1 if USE_RUNNING_CANDLE else idx_closed_1h)
            choose_idx = stable_idx

            run_remaining_hms = "00:00:00"  # using closed 1H

            stable_time_min = (df["time"].iloc[stable_idx].value // 10**9) / 60.0

            current_price = float(df["close"].iloc[choose_idx])
            close_stab = float(df["close"].iloc[stable_idx])

            ph_idx, pl_idx = find_pivots(df)
            if len(ph_idx) < 2 or len(pl_idx) < 2:
                if LOG_ERRORS: print(f"[{pair}] insufficient pivots")
                continue

            _, _, structure, trend = classify_structure(df, ph_idx, pl_idx)

            # SMART LEG (pivot-strict)
            leg_info = active_leg_smart(df, ph_idx, pl_idx, stable_idx, PIVOT_LEFT, PIVOT_RIGHT)
            if leg_info[0] is None:
                if LOG_ERRORS: print(f"[{pair}] no active leg (smart)")
                continue
            leg_up, L, H, _ = leg_info

            # Fib levels (mid=0.618, _786=0.786)
            gz_mid = level_from_leg(L, H, GOLDEN_RATIO, leg_up)
            gz_786 = level_from_leg(L, H, RATIO_786, leg_up)

            # ATR & band
            atr_series = atr(df["high"], df["low"], df["close"], period=14)
            atr14_val = float(atr_series.iloc[stable_idx]) if not math.isnan(atr_series.iloc[stable_idx]) else None
            atr_pct = (atr14_val / close_stab) if (atr14_val and close_stab > 0) else 0.0

            band_pct_effective = max(GZ_BAND_PCT_BASE, ATR_MULT_GZ * atr_pct)
            gz_low = gz_mid * (1.0 - band_pct_effective)
            gz_high = gz_mid * (1.0 + band_pct_effective)

            if MIN_GZ_WIDTH_ABS > 0:
                width_tmp = gz_high - gz_low
                if width_tmp < MIN_GZ_WIDTH_ABS:
                    pad = (MIN_GZ_WIDTH_ABS - width_tmp) / 2.0
                    gz_low -= pad; gz_high += pad

            fibStart, fibEnd = ((L, H) if leg_up else (H, L))

            eff_sr_tol = max(SR_TOL_PCT_BASE, ATR_MULT_SR * atr_pct)
            eff_ema_tol = max(EMA_TOL_PCT_BASE, ATR_MULT_EMA * atr_pct)
            eff_tl_tol = max(TL_TOL_PCT_BASE, ATR_MULT_TL * atr_pct)
            eff_vp_tol = max(VP_TOL_PCT_BASE, ATR_MULT_VP * atr_pct)

            # WHERE price vs band
            _, dist_pct_price, where, nearest = distance_to_zone(current_price, gz_low, gz_high)

            gz_prices = [gz_low, gz_mid, gz_high]
            strong_lvls = sr_strong_levels(df, ph_idx, pl_idx, SR_MIN_TOUCHES, eff_sr_tol)
            conf_sr = conf_sr_match_bool(gz_prices, strong_lvls, eff_sr_tol)
            conf_ema = conf_ema_21_50_bool(df, stable_idx, gz_prices, eff_ema_tol)
            conf_tl = conf_trendline_channel_bool(df, ph_idx, pl_idx, stable_time_min, gz_prices, eff_tl_tol)
            conf_vp = conf_vp_hvn_lvn_bool(df, lookback=350, bins=48, gz_prices=gz_prices, tol_pct=eff_vp_tol)

            # Candle/indicators
            ema21_series = ema(df["close"], 21)
            ema21_now = float(ema21_series.iloc[stable_idx])
            ema21_prev = float(ema21_series.iloc[stable_idx-1]) if (stable_idx-1) >= -len(df) else ema21_now
            ema21_slope = ema21_now - ema21_prev

            rsi_series = rsi(df["close"], 14)
            rsi_now = float(rsi_series.iloc[stable_idx])
            rsi_bias = "Neutral" if 45.0 < rsi_now < 55.0 else ("Long" if rsi_now >= 55.0 else "Short")

            ema21_trend_ok = ema21_confirm_for_trend(close_stab, ema21_now, ema21_prev, trend)
            rsi_trend_ok = ((trend == "Uptrend" and rsi_bias == "Long") or (trend == "Downtrend" and rsi_bias == "Short"))
            momentum_score = 1.0 if (ema21_trend_ok and rsi_trend_ok) else (0.5 if (ema21_trend_ok or rsi_trend_ok) else 0.0)
            momentum_bias = ("Aligned" if momentum_score >= 1.0 else ("Partial" if momentum_score >= 0.5 else "Against"))

            ob_value, ob_confirm = detect_order_block(df)
            rsi_div = rsi_divergence(df, rsi_series)

            base = df["volume"].iloc[max(0, stable_idx-20):stable_idx].astype(float)
            mean_v = float(base.mean()) if len(base)>0 else 0.0
            cur_v = float(df["volume"].iloc[stable_idx])
            ratio = (cur_v/mean_v) if mean_v>0 else 1.0
            if ratio >= 1.3:   vol_flow = f"rising x{ratio:.2f} (confirm)"
            elif ratio <= 0.7: vol_flow = f"dry-up x{ratio:.2f}"
            else:              vol_flow = f"normal x{ratio:.2f}"

            micro_trig = "none"
            mom_div = f"momentum:{momentum_bias.lower()} | div:{rsi_div}"

            # ---------- 1H last CLOSED candle touch checks (FIXED tie-break) ----------
            h1h = float(df["high"].iloc[stable_idx]); l1h = float(df["low"].iloc[stable_idx])
            tol_h1_frac = max(SR_TOL_PCT_BASE, atr_pct)  # ATR adaptive
            tol_mid_abs_h1 = gz_mid * tol_h1_frac
            tol_786_abs_h1 = gz_786 * tol_h1_frac

            h1_touch_mid  = (l1h <= (gz_mid + tol_mid_abs_h1)) and (h1h >= (gz_mid - tol_mid_abs_h1))
            h1_touch_786  = (l1h <= (gz_786 + tol_786_abs_h1)) and (h1h >= (gz_786 - tol_786_abs_h1))
            if h1_touch_mid and h1_touch_786:
                d_mid = abs((l1h if leg_up else h1h) - gz_mid)
                d_786 = abs((l1h if leg_up else h1h) - gz_786)
                h1_touched_level = "0.618" if d_mid <= d_786 else "0.786"
            else:
                h1_touched_level = "0.618" if h1_touch_mid else ("0.786" if h1_touch_786 else None)

            # ---------- last prior 1H touch ----------
            h1_last_touch_level = find_last_h1_touch_level(
                df=df, stable_idx=stable_idx, gz_mid=gz_mid, gz_786=gz_786,
                tol_frac=tol_h1_frac, leg_up=bool(leg_up)
            )

            # ---------- 15m RUNNING candle STRONG REVERSAL ----------
            df15_run, idx15_run = fetch_m15_and_pick_running(exchange, pair)
            m15_running_ok = False
            m15_rev_level = None
            m15_rev_signal = None
            if df15_run is not None and idx15_run is not None and len(df15_run) >= 3:
                m15_running_ok, m15_rev_level, m15_rev_signal = m15_running_strong_reversal(
                    df15=df15_run, idx_run=idx15_run, gz_mid=gz_mid, gz_786=gz_786,
                    leg_up=leg_up, atr_val_1h=atr14_val, tol_frac_base=SR_TOL_PCT_BASE
                )

            # ---------- 15m last CLOSED info ----------
            df15, idx15 = fetch_m15_and_pick_last_closed(exchange, pair)
            m15_touch_time = m15_touch_high = m15_touch_low = m15_low_vs_ema21 = m15_high_vs_ema21 = m15_rsi14 = None
            hit_minutes = None
            if df15 is not None and idx15 is not None:
                f = compute_m15_fields(df15, idx15, PK_TZ)
                m15_touch_time = f["time_pk"]
                m15_touch_high = round(f["hi"], 6)
                m15_touch_low  = round(f["lo"], 6)
                m15_rsi14      = round(f["rsi"], 3)
                m15_low_vs_ema21  = f["low_vs_ema"]
                m15_high_vs_ema21 = f["high_vs_ema"]
                hit_minutes = int(max(0, (pd.Timestamp.now(tz="UTC") - df15["time"].iloc[idx15]).total_seconds() // 60))

            # ---------- Candle confirmation / signal ----------
            candle_signal, candle_score, candle_side = derive_candle_signal(
                trend_tf=trend, where=where, dist_pct_price=dist_pct_price,
                df=df, stable_idx=stable_idx, atr_val=atr14_val
            )
            pa_confirm = "none: normal"
            if ("strong bullish" in candle_signal.lower()) or ("strong bearish" in candle_signal.lower()):
                pa_confirm = "confirm: " + candle_signal

            # ---------- Strategy: Reclaim (i-1) + Reversal (i) + 15m holds ----------
            strategy_ok = False
            strat_dir = None
            strat_reclaim_tag = None
            strat_reclaim_level = None
            strat_entry = strat_sl = strat_tp = None

            if len(df) >= 3 and (stable_idx - 1) >= -len(df):
                # reclaim on previous closed 1H
                i_prev = stable_idx - 1
                o_prev = float(df["open"].iloc[i_prev]); h_prev = float(df["high"].iloc[i_prev])
                l_prev = float(df["low"].iloc[i_prev]);  c_prev = float(df["close"].iloc[i_prev])
                # reversal on current closed 1H
                o_cur = float(df["open"].iloc[stable_idx]); h_cur = float(df["high"].iloc[stable_idx])
                l_cur = float(df["low"].iloc[stable_idx]);  c_cur = float(df["close"].iloc[stable_idx])

                want_above = True if leg_up else False
                candidate_lvls = [("0.618", gz_mid), ("0.786", gz_786)]

                picked = None
                for tag, lvl in candidate_lvls:
                    if crossed_and_reclaimed(o_prev, h_prev, l_prev, c_prev, lvl, TOUCH_TOL_FRAC, want_above):
                        picked = (tag, lvl)
                        break

                if picked is not None:
                    # reversal check on current candle
                    if leg_up:
                        rev_ok = is_bull_reversal(o_prev, c_prev, o_cur, c_cur, h_cur, l_cur)
                        dir_after = "LONG"
                    else:
                        rev_ok = is_bear_reversal(o_prev, c_prev, o_cur, c_cur, h_cur, l_cur)
                        dir_after = "SHORT"

                    if rev_ok and df15 is not None:
                        # 15m window = [open(i), open(i)+1h)
                        t_start = df["time"].iloc[stable_idx]
                        t_end = t_start + pd.Timedelta(minutes=TF_MIN)
                        lvl_tag, lvl_val = picked
                        if m15_ok(df15, t_start, t_end, lvl_val, want_above, use_ema=USE_M15_EMA21_FILTER):
                            # ENTRY/TP/SL
                            atr_buf = (ATR_BUFFER_MULT * (atr14_val or 0.0))
                            if dir_after == "LONG":
                                # TP rule
                                tp = level_from_leg(L, H, 0.618, True) if lvl_tag == "0.786" else level_from_leg(L, H, TARGET_FOR_618, True)
                                # SL rule (leg extreme minus buffer)
                                sl = (L - atr_buf) if SL_MODE == "leg_100" else (lvl_val - atr_buf)
                            else:
                                tp = level_from_leg(L, H, 0.618, False) if lvl_tag == "0.786" else level_from_leg(L, H, 1.0 - TARGET_FOR_618, False)
                                # SL rule (leg extreme plus buffer)
                                sl = (H + atr_buf) if SL_MODE == "leg_100" else (lvl_val + atr_buf)

                            entry = c_cur

                            strat_dir = dir_after
                            strat_reclaim_tag = lvl_tag
                            strat_reclaim_level = float(lvl_val)
                            strat_entry = float(entry)
                            strat_sl = float(sl)
                            strat_tp = float(tp)
                            strategy_ok = True

            # ---------- Recommendation (legacy generic) ----------
            in_gz = (where == "inside") or (dist_pct_price is not None and dist_pct_price <= NEAR_THRESHOLD_PCT)
            if trend == "Uptrend" and in_gz and rsi_bias in ("Long","Neutral") and ema21_trend_ok and (close_stab >= gz_low):
                recommendation = build_recommendation("LONG", close_stab, gz_low, gz_high, atr14_val)
            elif trend == "Downtrend" and in_gz and rsi_bias in ("Short","Neutral") and ema21_trend_ok and (close_stab <= gz_high):
                recommendation = build_recommendation("SHORT", close_stab, gz_low, gz_high, atr14_val)
            else:
                recommendation = "NO_TRADE"

            confidence_level = compute_confidence_level(
                close_price=current_price, gz_level=gz_mid,
                conf_sr=bool(conf_sr), conf_ema=bool(conf_ema), conf_tl=bool(conf_tl), conf_vp=bool(conf_vp),
                candle_score=float(candle_score), trend_tf=trend, leg_up=bool(leg_up), momentum_score=float(momentum_score)
            )
            ck_score, ck_flags = confluence_king_score_builder(conf_sr, conf_ema, conf_tl, conf_vp, ob_confirm, rsi_div)

            confirm_close_ts = df["time"].iloc[stable_idx]
            mins_since = max(0, int((pd.Timestamp.now(tz="UTC") - confirm_close_ts).total_seconds() // 60))

            s_sup, s_sup_t, s_res, s_res_t = strong_sr_details(df, ph_idx, pl_idx, SR_MIN_TOUCHES, eff_sr_tol, close_stab)

            # --------- Row payload ---------
            row_payload = {
                "symbol": sym,
                "name": name,
                "pair": pair,
                "timeframe": TIMEFRAME,
                "run_remaining_hms": run_remaining_hms,
                "currentPrice": round(current_price, 6),
                "swing_low": round(L, 6),
                "swing_high": round(H, 6),
                "fibStart": round(fibStart, 6),
                "fibEnd": round(fibEnd, 6),
                "goldenZone": f"{round(gz_low,6)} - {round(gz_high,6)}",
                "where_vs_gz": where,
                "nearest_gz_bound": nearest,
                "Fibonacci zone": "inside" if where=="inside" else ("near" if (dist_pct_price is not None and dist_pct_price <= NEAR_THRESHOLD_PCT) else "far"),
                "Price Action candle confirmation": pa_confirm,
                "Volume/Orderflow": vol_flow,
                "Momentum/Divergence": mom_div,
                "Micro-structure trigger": micro_trig,
                "structu": structure,
                "trend": trend,
                "trend_1D": "Skipped",
                "trend_4H": "Skipped",
                "trend_1H": "Skipped",
                "candle_signal": candle_signal,
                "candle_score": round(float(candle_score),2),
                "ema21": round(ema21_now, 6),
                "ema21_slope": round(ema21_slope, 6),
                "rsi14": round(rsi_now, 3),
                "rsi_bias": rsi_bias,
                "conf_ema21_trend": bool(ema21_trend_ok),
                "conf_rsi_trend": bool(rsi_trend_ok),
                "momentum_bias": momentum_bias,
                "momentum_score": round(momentum_score, 2),
                "confluence_king_score": int(ck_score),
                "confluence_king_flags": ck_flags,
                "rsi_divergence": rsi_div,
                "conf_sr_match": bool(conf_sr),
                "conf_ema_21_50": bool(conf_ema),
                "conf_trendline_channel": bool(conf_tl),
                "conf_vp_hvn_lvn": bool(conf_vp),
                "order_block_value": None,
                "order_block_confirm": None,
                "recommendation": recommendation,
                "confidence_level": int(confidence_level),
                "atr14": (round(atr14_val, 6) if atr14_val is not None else None),
                "strongSupport": s_sup,
                "strongSupportTouches": s_sup_t,
                "strongSupportLevel": s_sup,
                "strongResistance": s_res,
                "strongResistanceTouches": s_res_t,
                "strongResistanceLevel": s_res,
                "triggerTime_min": int(mins_since),
                # 15m display (closed)
                "m15_touch_time": m15_touch_time,
                "m15_touch_high": m15_touch_high,
                "m15_touch_low": m15_touch_low,
                "m15_low_vs_ema21": m15_low_vs_ema21,
                "m15_high_vs_ema21": m15_high_vs_ema21,
                "m15_rsi14": m15_rsi14,
                # TP/SL placeholders (filled by either generic or strategy)
                "cc_entry": None, "cc_sl": None, "cc_tp1": None, "cc_tp2": None,
                # ---- NEW summary fields ----
                "NEW_TP": None, "NEW_SL": None,
                "HitMinuts": hit_minutes,
                # Hard filter fields
                "h1_touched_level": h1_touched_level,
                "h1_last_touch_level": h1_last_touch_level,
                "m15_running_reversal": bool(m15_running_ok),
                "m15_reversal_level": m15_rev_level,
                "m15_reversal_signal": m15_rev_signal
            }

            # --------- customConfirmation (generic) + TP/SL attach (generic) ---------
            cc = build_custom_confirmation(row_payload)
            if cc in ("STRONG_LONG","STRONG_SHORT") and not strategy_ok:
                direction = "LONG" if cc == "STRONG_LONG" else "SHORT"
                cc_entry, cc_sl, cc_tp1, cc_tp2 = build_tp_sl_from_zone(
                    direction=direction, close_stab=close_stab, gz_low=gz_low, gz_high=gz_high, atr_val=atr14_val
                )
                row_payload["cc_entry"] = cc_entry
                row_payload["cc_sl"] = cc_sl
                row_payload["cc_tp1"] = cc_tp1
                row_payload["cc_tp2"] = cc_tp2
                cc = f"{cc} | Entry: {cc_entry} | SL: {cc_sl} | TP1: {cc_tp1} | TP2: {cc_tp2}"

            # --------- overwrite with Reclaim+Reversal strategy if it fired ---------
            if strategy_ok:
                r6 = lambda x: round(float(x), 6)
                row_payload["cc_entry"] = r6(strat_entry)
                row_payload["cc_sl"]    = r6(strat_sl)
                row_payload["cc_tp1"]   = r6(strat_tp)
                row_payload["cc_tp2"]   = r6(strat_tp)

                # -------- NEW_TP / NEW_SL (Summary display) --------
                atr_buf = (ATR_BUFFER_MULT * (atr14_val or 0.0))
                if strat_dir == "LONG":
                    new_sl = (L - atr_buf)
                    if strat_reclaim_tag == "0.786":
                        new_tp = level_from_leg(L, H, 0.618, True)
                    else:  # "0.618"
                        new_tp = level_from_leg(L, H, TARGET_FOR_618, True)
                else:  # SHORT
                    new_sl = (H + atr_buf)
                    if strat_reclaim_tag == "0.786":
                        new_tp = level_from_leg(L, H, 0.618, False)
                    else:  # "0.618"
                        new_tp = level_from_leg(L, H, 1.0 - TARGET_FOR_618, False)

                row_payload["NEW_TP"] = r6(new_tp)
                row_payload["NEW_SL"] = r6(new_sl)

                cc_msg = (f"RECLAIM+REVERSAL+15m({M15_CLOSINGS_REQUIRED}) OK | {strat_dir} | "
                          f"reclaim={strat_reclaim_tag} @ {r6(strat_reclaim_level)} | "
                          f"Entry={r6(strat_entry)} | SL={r6(strat_sl)} | TP={r6(strat_tp)}")
                row_payload["customConfirmation"] = cc_msg
            else:
                row_payload["customConfirmation"] = cc

            rows.append(row_payload)

            # HARD FILTER FLAG (1H touched & 15m running strong reversal)
            h1_touched_any = (h1_touched_level is not None)
            hard_flags.append(bool(h1_touched_any and m15_running_ok))

            # optional HTF trends if near/inside
            need_htf = (where == "inside") or (dist_pct_price is not None and dist_pct_price <= NEAR_THRESHOLD_PCT)
            if need_htf:
                rl = (exchange.rateLimit/1000.0 if hasattr(exchange,"rateLimit") else 0.25)
                rows[-1]["trend_1D"] = calc_trend_only(exchange, pair, TREND_TIMEFRAMES["1D"], TREND_CANDLES_LIMIT); time.sleep(rl)
                rows[-1]["trend_4H"] = calc_trend_only(exchange, pair, TREND_TIMEFRAMES["4H"], TREND_CANDLES_LIMIT); time.sleep(rl)
                rows[-1]["trend_1H"] = calc_trend_only(exchange, pair, TREND_TIMEFRAMES["1H"], TREND_CANDLES_LIMIT); time.sleep(rl)

            time.sleep(exchange.rateLimit / 1000.0 if hasattr(exchange, "rateLimit") else 0.25)

        except Exception as e:
            if LOG_ERRORS:
                print(f"[{pair}] error: {e}")
            continue

    # ----------------- build frames -----------------
    if not rows:
        print("No data rows produced — possibly rate limits or symbol mapping.")
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)

    # FINAL FILTER (strategy hard gate)
    if len(hard_flags) == len(df_all):
        df_all["__ok__"] = hard_flags
        df_filtered = df_all[df_all["__ok__"] == True].copy()
        df_filtered.drop(columns=["__ok__"], inplace=True, errors="ignore")
    else:
        df_filtered = df_all.copy()

    if df_filtered.empty:
        print("No rows after '1H touch (0.618/0.786) AND 15m RUNNING strong reversal' filter.")
        return df_filtered

    # DETAIL = filtered frame with ALL columns
    df_detail = df_filtered.copy()

    # Ensure console ordering columns exist
    for col in COLUMN_ORDER:
        if col not in df_detail.columns:
            df_detail[col] = None

    # SUMMARY = only requested columns (ensure existence first)
    for col in SUMMARY_COLUMNS:
        if col not in df_detail.columns:
            df_detail[col] = None
    df_summary = df_detail[SUMMARY_COLUMNS].copy()

    # ----------------- save ONE Excel with 2 sheets -----------------
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        xlsx_path = os.path.join(SAVE_DIR, f"{FILE_PREFIX}_{ts}.xlsx")

        # engine='openpyxl' works for .xlsx (install: pip install openpyxl)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            df_detail.to_excel(writer, sheet_name="Detail", index=False)

        print("\nExcel saved (2 sheets):")
        print(f" Sheet1 (Summary): {xlsx_path}")
        print(f" Sheet2 (Detail) : {xlsx_path}")
    except Exception as e:
        print(f"Excel save failed: {e}")

    # Optional pretty print
    try:
        df_pretty = df_detail.reindex(columns=[c for c in COLUMN_ORDER if c in df_detail.columns])
        print(f"\n=== H1 TOUCH (0.618/0.786) + 15m RUNNING STRONG REVERSAL — {TIMEFRAME} ===")
        print(df_pretty.to_string(index=False))
    except Exception:
        print(df_summary.head())

    return df_summary, df_detail

if __name__ == "__main__":
    analyze_and_save()
