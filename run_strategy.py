#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMA21 Filter — send single-column summary to Telegram
Improved readability:
  • Bold coin symbol
  • Separator lines between records
  • Sends only when matches exist (silent if none)
  • Includes Strong Support & Resistance (1H)
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
MIN_24H_PCT = 12.0
MAX_24H_PCT = 15.0
MAX_24H_PEAK_PCT = 20.0
EMA_CHECK_INTERVAL = "1h"
MAX_COINS = 100

QUOTE_WHITELIST = {"USDT","FDUSD","TUSD","USDC","USD"}
QUOTE_PRIORITY  = ["USDT","FDUSD","TUSD","USDC","USD"]

PK_TZ = tz.gettz("Asia/Karachi")

BASE = (
    os.environ.get("BINANCE_PUBLIC_BASE") or
    os.environ.get("BINANCE_BASE_URL") or
    "https://data-api.binance.vision"
)
EP_TICKER_24H = f"{BASE}/api/v3/ticker/24hr"
EP_KLINES     = f"{BASE}/api/v3/klines"

PIVOT_L = 2
PIVOT_R = 2

# ---------- Utils ----------
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
    best_for_base = {}
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

def fetch_24h_stats_map() -> Dict[str, Dict[str, float]]:
    r = requests.get(EP_TICKER_24H, timeout=15)
    r.raise_for_status()
    out = {}
    for row in r.json():
        sym = row.get("symbol","")
        try:
            op = float(row.get("openPrice","0") or 0.0)
            last = float(row.get("lastPrice","0") or 0.0)
            hi = float(row.get("highPrice","0") or 0.0)
        except Exception:
            continue
        if op <= 0: continue
        cur_pct = (last - op) / op * 100.0
        peak_pct = (hi - op) / op * 100.0
        out[sym] = {"cur_pct": cur_pct, "peak_pct": peak_pct, "peak_price": hi}
    return out

def is_above_ema21_last_closed(symbol: str, interval: str = EMA_CHECK_INTERVAL):
    df = fetch_klines(symbol, interval, limit=max(EMA_LENGTH+50, 120))
    if df is None or df.empty or len(df) < EMA_LENGTH+1:
        return False, None, None, None
    e = ema(df["close"], EMA_LENGTH)
    last_open = float(df["open"].iloc[-1])
    last_close = float(df["close"].iloc[-1])
    ema_now = float(e.iloc[-1]) if not math.isnan(e.iloc[-1]) else None
    if ema_now is None:
        return False, last_close, None, None
    cond = (last_open > ema_now) and (last_close > ema_now)
    return cond, last_close, ema_now, float(df["close"].iloc[-2])

# ---- SR Detection ----
def _is_pivot_high(df, i, L, R): return df["high"].iloc[i] == df["high"].iloc[max(0,i-L):min(len(df),i+R+1)].max()
def _is_pivot_low(df, i, L, R):  return df["low"].iloc[i]  == df["low"].iloc[max(0,i-L):min(len(df),i+R+1)].min()

def _gather_pivots(df, L, R, lookback=300):
    lows, highs = [], []
    start = max(0, len(df)-lookback)
    for i in range(start, len(df)):
        if _is_pivot_low(df,i,L,R): lows.append((i,float(df["low"].iloc[i])))
        if _is_pivot_high(df,i,L,R): highs.append((i,float(df["high"].iloc[i])))
    return lows, highs

def _cluster_levels(points, tol_abs):
    if not points: return []
    pts = sorted(points,key=lambda x:x[1])
    clusters, curv, curi = [], [], []
    def push():
        if curv: clusters.append({"level":sum(curv)/len(curv),"touches":len(curv),"last_idx":max(curi)})
    for idx,price in pts:
        if not curv: curv,curi=[price],[idx]; continue
        if abs(price-(sum(curv)/len(curv)))<=tol_abs: curv.append(price);curi.append(idx)
        else: push();curv,curi=[price],[idx]
    push(); return clusters

def find_strong_sr(df,L=PIVOT_L,R=PIVOT_R,lookback=300):
    out={"support":None,"resistance":None}
    if df.empty: return out
    last=float(df["close"].iloc[-1])
    atr=atr_df(df,14)
    atr_last=float(atr.iloc[-1]) if not math.isnan(atr.iloc[-1]) else 0
    tol_abs=max(0.0015*last,0.25*atr_last)
    lows,highs=_gather_pivots(df,L,R,lookback)
    low_c=_cluster_levels(lows,tol_abs)
    high_c=_cluster_levels(highs,tol_abs)
    if low_c:
        c=[x for x in low_c if x["level"]<=last]
        if c: c.sort(key=lambda x:(x["touches"],x["level"],x["last_idx"]),reverse=True); out["support"]=c[0]["level"]
    if high_c:
        c=[x for x in high_c if x["level"]>=last]
        if c: c.sort(key=lambda x:(x["touches"],x["level"],x["last_idx"]),reverse=True); out["resistance"]=c[0]["level"]
    return out

# ---------- Main ----------
def main():
    print(f"[{datetime.now(PK_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}] Selecting top {MAX_COINS} symbols…")
    syms=fetch_top_symbols()
    print(f"Top Symbols ({len(syms)}):", ", ".join(syms))
    stats=fetch_24h_stats_map()
    candidates=[s for s in syms if s in stats and MIN_24H_PCT<=stats[s]["cur_pct"]<=MAX_24H_PCT and stats[s]["peak_pct"]<=MAX_24H_PEAK_PCT]

    summaries=[]
    for sym in candidates:
        try:
            ok,last,ema_now,prev=is_above_ema21_last_closed(sym)
            if not ok: continue
            df=fetch_klines(sym,"1h",500)
            sr=find_strong_sr(df)
            cur,peak,price=stats[sym]["cur_pct"],stats[sym]["peak_pct"],stats[sym]["peak_price"]
            line=(f"{sym} — Peak: {peak:.2f}% | Peak Price: {price:.6f} | Current: {cur:.2f}% | "
                  f"Strong Support (1H): {sr['support']:.6f}" if sr["support"] else "N/A")
            if sr["resistance"]: line += f" | Strong Resistance (1H): {sr['resistance']:.6f}"
            summaries.append(line)
            time.sleep(0.05)
        except Exception as e:
            print("[warn]",sym,e,file=sys.stderr)

    if summaries:
        print("\n### Single-column results (Summary) ###")
        for i,s in enumerate(summaries,1):
            print(f"{i}. {s}")
            print("─"*90)
        print(f"\nTotal Matches: {len(summaries)}")

        # Build Telegram message
        header="✅ <b>EMA21 Filtered Coins (1H, O&C > EMA21)</b>\n"
        body="\n\n".join([f"<b>{s.split(' — ')[0]}</b> — {html.escape(' — '.join(s.split(' — ')[1:]))}" for s in summaries])
        footer=f"\n\nTotal Matches: {len(summaries)}"
        msg=f"{header}<pre>{body}</pre>{html.escape(footer)}"
        notify_chunks(msg)
    else:
        print("\nNo coins matched the filters.")

if __name__=="__main__":
    main()
