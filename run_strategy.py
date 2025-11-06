#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMA21 Filter — send single-column summary to Telegram
Style: code1-like console output + same content to Telegram when matches exist
  • Only send Telegram message when there are matches (silent if none)
  • STRICT EMA: last CLOSED 1H candle's OPEN & CLOSE both above EMA21
  • Current 24h % in [12, 15], Peak ≤ 20%
  • Strong Support & Strong Resistance via pivot clustering (1H)

Requirements:
  pip install requests pandas python-dateutil
"""

import os, sys, time, math, html, requests, pandas as pd
from datetime import datetime, timezone
from dateutil import tz
from typing import List, Dict, Optional, Tuple

# ---------- Telegram ----------
try:
    from notifier import notify
except Exception:
    def notify(msg: str):
        print("[notify:NOOP]", msg)

def notify_chunks(html_text: str, chunk_size: int = 3800):
    """Telegram limit ~4096 chars; send in safe chunks."""
    for i in range(0, len(html_text), chunk_size):
        notify(html_text[i:i+chunk_size])

# ---------- Config ----------
EMA_LENGTH = 21

# 24h filter
MIN_24H_PCT = 12.0
MAX_24H_PCT = 15.0
MAX_24H_PEAK_PCT = 20.0
EMA_CHECK_INTERVAL = "1h"

# Symbols universe
MAX_COINS = 100
QUOTE_WHITELIST = {"USDT","FDUSD","TUSD","USDC","USD"}
QUOTE_PRIORITY  = ["USDT","FDUSD","TUSD","USDC","USD"]

# Timezone for logs
PK_TZ = tz.gettz("Asia/Karachi")

# Binance endpoints — prefer GitHub Actions env name first
BASE = (
    os.environ.get("BINANCE_PUBLIC_BASE") or
    os.environ.get("BINANCE_BASE_URL") or
    "https://data-api.binance.vision"
)
EP_TICKER_24H = f"{BASE}/api/v3/ticker/24hr"
EP_KLINES     = f"{BASE}/api/v3/klines"

# Pivot/SR settings
PIVOT_L = 2
PIVOT_R = 2

# ---------------- Utils ----------------
def now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp()*1000)

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def atr_df(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c1 = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h-l).abs(), (h-c1).abs(), (l-c1).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def fetch_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    r = requests.get(EP_KLINES, params={"symbol":symbol,"interval":interval,"limit":limit}, timeout=15)
    r.raise_for_status()
    kl = r.json()
    if not kl: return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time","qv","n","tb_base","tb_quote","ignore"]
    df = pd.DataFrame(kl, columns=cols)
    for c in ["open","high","low","close","volume","qv","tb_base","tb_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["open_time","close_time","n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
    # drop running candle (if any)
    if len(df) and int(df.iloc[-1]["close_time"]) > now_utc_ms():
        df = df.iloc[:-1].copy()
    return df.reset_index(drop=True)

def _split_base_quote(sym: str) -> Optional[Tuple[str,str]]:
    for q in QUOTE_PRIORITY:
        if sym.endswith(q):
            return sym[:-len(q)], q
    return None

def fetch_top_symbols() -> List[str]:
    r = requests.get(EP_TICKER_24H, timeout=15); r.raise_for_status()
    data = r.json()
    best_for_base: Dict[str, Dict] = {}
    q_rank = {q:i for i,q in enumerate(QUOTE_PRIORITY)}
    for row in data:
        sym = row.get("symbol","")
        split = _split_base_quote(sym)
        if not split: continue
        base, quote = split
        if base in {"USDT","BUSD","USDC","TUSD","FDUSD","DAI","UST","USTC"}: continue
        if any(base.endswith(suf) for suf in ("UP","DOWN","BULL","BEAR")): continue
        if quote not in QUOTE_WHITELIST: continue
        try:
            qv = float(row.get("quoteVolume","0") or 0.0)
        except Exception:
            qv = 0.0
        if base not in best_for_base:
            best_for_base[base] = {"sym": sym, "qv": qv, "quote": quote}
        else:
            cur = best_for_base[base]
            if q_rank[quote] < q_rank[cur["quote"]] or (quote == cur["quote"] and qv > cur["qv"]):
                best_for_base[base] = {"sym": sym, "qv": qv, "quote": quote}
    ranked = sorted(best_for_base.values(), key=lambda d: d["qv"], reverse=True)
    return [d["sym"] for d in ranked[:MAX_COINS]]

# ---- 24h stats map (current & peak) ----
def fetch_24h_stats_map() -> Dict[str, Dict[str, float]]:
    """
    Returns: { "SYMBOL": {"cur_pct": float, "peak_pct": float, "peak_price": float} }
      cur_pct   = (lastPrice - openPrice)/openPrice * 100
      peak_pct  = (highPrice - openPrice)/openPrice * 100
      peak_price= highPrice
    """
    r = requests.get(EP_TICKER_24H, timeout=15)
    r.raise_for_status()
    out: Dict[str, Dict[str, float]] = {}
    for row in r.json():
        sym = row.get("symbol","")
        try:
            op = float(row.get("openPrice","0") or 0.0)
            last = float(row.get("lastPrice","0") or 0.0)
            hi = float(row.get("highPrice","0") or 0.0)
        except Exception:
            continue
        if op <= 0:
            continue
        cur_pct = (last - op) / op * 100.0
        peak_pct = (hi - op) / op * 100.0
        out[sym] = {"cur_pct": cur_pct, "peak_pct": peak_pct, "peak_price": hi}
    return out

# ---- check latest CLOSED 1H candle: OPEN & CLOSE above EMA21 ----
def is_above_ema21_last_closed(symbol: str, interval: str = EMA_CHECK_INTERVAL) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    df = fetch_klines(symbol, interval, limit=max(EMA_LENGTH+50, 120))
    if df is None or df.empty or len(df) < EMA_LENGTH+1:
        return False, None, None, None
    e = ema(df["close"], EMA_LENGTH)

    last_open  = float(df["open"].iloc[-1])
    last_close = float(df["close"].iloc[-1])
    ema_now = float(e.iloc[-1]) if not math.isnan(e.iloc[-1]) else None
    if ema_now is None:
        return False, last_close, None, None

    # Require both open & close above EMA21
    cond = (last_open > ema_now) and (last_close > ema_now)
    return cond, last_close, ema_now, float(df["close"].iloc[-2])

# ---------- Strong Support & Resistance via pivot clustering ----------
def _is_pivot_high(df: pd.DataFrame, i: int, L: int, R: int) -> bool:
    lo = max(0, i - L); hi = min(len(df), i + R + 1)
    return df["high"].iloc[i] == df["high"].iloc[lo:hi].max()

def _is_pivot_low(df: pd.DataFrame, i: int, L: int, R: int) -> bool:
    lo = max(0, i - L); hi = min(len(df), i + R + 1)
    return df["low"].iloc[i] == df["low"].iloc[lo:hi].min()

def _gather_pivots(df: pd.DataFrame, L: int, R: int, lookback: int = 300):
    lows, highs = [], []
    start = max(0, len(df) - lookback)
    for i in range(start, len(df)):
        if _is_pivot_low(df, i, L, R):
            lows.append((i, float(df["low"].iloc[i])))
        if _is_pivot_high(df, i, L, R):
            highs.append((i, float(df["high"].iloc[i])))
    return lows, highs

def _cluster_levels(points, tol_abs: float):
    if not points: return []
    pts = sorted(points, key=lambda x: x[1])
    clusters = []
    cur_vals, cur_idx = [], []
    def push():
        if not cur_vals: return
        clusters.append({"level": sum(cur_vals)/len(cur_vals), "touches": len(cur_vals), "last_idx": max(cur_idx)})
    for idx, price in pts:
        if not cur_vals:
            cur_vals, cur_idx = [price], [idx]; continue
        if abs(price - (sum(cur_vals)/len(cur_vals))) <= tol_abs:
            cur_vals.append(price); cur_idx.append(idx)
        else:
            push(); cur_vals, cur_idx = [price],[idx]
    push()
    return clusters

def find_strong_sr(df: pd.DataFrame, L: int = PIVOT_L, R: int = PIVOT_R, lookback: int = 300) -> Dict[str, Optional[float]]:
    """
    Returns dict with both support and resistance levels based on clustered pivots.
    Picks strongest by touches; for resistance ties, prefers higher level; for support, prefers higher level (closest below) last.
    """
    out = {"support": None, "resistance": None}
    if df is None or df.empty: return out

    last_close = float(df["close"].iloc[-1])
    atr = atr_df(df, 14)
    atr_last = float(atr.iloc[-1]) if not math.isnan(atr.iloc[-1]) else None
    tol_abs = max(0.0015 * last_close, (0.25 * atr_last) if atr_last else 0.0)

    lows, highs = _gather_pivots(df, L, R, lookback=lookback)
    low_clusters  = _cluster_levels(lows,  tol_abs)
    high_clusters = _cluster_levels(highs, tol_abs)

    if low_clusters:
        cands = [c for c in low_clusters if c["level"] <= last_close]
        if cands:
            cands.sort(key=lambda c: (c["touches"], c["level"], c["last_idx"]), reverse=True)
            out["support"] = cands[0]["level"]

    if high_clusters:
        cands = [c for c in high_clusters if c["level"] >= last_close]
        if cands:
            # Prefer more touches, then higher level, then recency
            cands.sort(key=lambda c: (c["touches"], c["level"], c["last_idx"]), reverse=True)
            out["resistance"] = cands[0]["level"]

    return out

# ---------- Main ----------
def main():
    print(f"[{datetime.now(PK_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}] Selecting top {MAX_COINS} symbols (USDⓈ quotes)…")
    top_symbols = fetch_top_symbols()
    print(f"Top Symbols ({len(top_symbols)}):", ", ".join(top_symbols))

    print(f"Fetching 24h stats and filtering for {MIN_24H_PCT}%–{MAX_24H_PCT}% AND peak ≤ {MAX_24H_PEAK_PCT}% …")
    stats_map = fetch_24h_stats_map()

    # Filter 1: current 24h % in range & peak ≤ cap
    candidates = []
    for s in top_symbols:
        st = stats_map.get(s)
        if not st:
            continue
        cur_pct = st["cur_pct"]; peak_pct = st["peak_pct"]
        if (MIN_24H_PCT <= cur_pct <= MAX_24H_PCT) and (peak_pct <= MAX_24H_PEAK_PCT):
            candidates.append(s)

    # Filter 2: last 1H CLOSED candle open & close above EMA21 + compute Strong Support/Resistance
    summaries = []
    for sym in candidates:
        try:
            ok, last_close, ema_now, prev_close = is_above_ema21_last_closed(sym, EMA_CHECK_INTERVAL)
            if not ok:
                continue
            # 1H data for SR
            df1h = fetch_klines(sym, EMA_CHECK_INTERVAL, limit=500)
            sr = find_strong_sr(df1h, L=PIVOT_L, R=PIVOT_R, lookback=300)
            support = sr.get("support")
            resistance = sr.get("resistance")

            cur_pct   = stats_map[sym]["cur_pct"]
            peak_pct  = stats_map[sym]["peak_pct"]
            peak_price= stats_map[sym]["peak_price"]

            if support is not None and resistance is not None:
                line = (
                    f"{sym} — Peak: {peak_pct:.2f}% | Peak Price: {peak_price:.6f} | "
                    f"Current: {cur_pct:.2f}% | Strong Support (1H): {support:.6f} | "
                    f"Strong Resistance (1H): {resistance:.6f}"
                )
            elif support is not None:
                line = (
                    f"{sym} — Peak: {peak_pct:.2f}% | Peak Price: {peak_price:.6f} | "
                    f"Current: {cur_pct:.2f}% | Strong Support (1H): {support:.6f} | "
                    f"Strong Resistance (1H): N/A"
                )
            elif resistance is not None:
                line = (
                    f"{sym} — Peak: {peak_pct:.2f}% | Peak Price: {peak_price:.6f} | "
                    f"Current: {cur_pct:.2f}% | Strong Support (1H): N/A | "
                    f"Strong Resistance (1H): {resistance:.6f}"
                )
            else:
                line = (
                    f"{sym} — Peak: {peak_pct:.2f}% | Peak Price: {peak_price:.6f} | "
                    f"Current: {cur_pct:.2f}% | Strong Support (1H): N/A | "
                    f"Strong Resistance (1H): N/A"
                )

            summaries.append(line)
            time.sleep(0.05)
        except Exception as e:
            print(f"[warn] {sym} failed: {e}", file=sys.stderr)

    # ---- Console + Telegram output (single-column like code1) ----
    if summaries:
        print("\n### Single-column results (Summary) ###")
        for s in summaries:
            print(s)
        print(f"\nTotal Matches: {len(summaries)}")

        # Build Telegram message (HTML-safe)
        header = "✅ EMA21 Filtered Coins (1H, O&C > EMA21)\n"
        body   = "\n".join(summaries)
        footer = f"\n\nTotal Matches: {len(summaries)}"
        msg = f"<b>{header}</b><pre>{html.escape(body)}</pre>{html.escape(footer)}"
        notify_chunks(msg)
    else:
        print("\nNo coins matched the filters.")
        # silent on Telegram

if __name__ == "__main__":
    main()
