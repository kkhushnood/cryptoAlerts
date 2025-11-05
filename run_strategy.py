#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMA21 Filter — sends final table directly to Telegram (no CSV)
Adds: Peak Price (24h High) column in output
Fixes included:
  • Uses Binance Vision mirror to avoid 451 region block
  • Handles missing 'tabulate' automatically
  • Sends results as formatted table to Telegram

Requirements:
  pip install requests pandas python-dateutil
"""

import os, time, math, html, requests, pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

# ---------- Telegram ----------
try:
    from notifier import notify     # uses your working notifier.py
except Exception:
    def notify(msg: str):
        print("[notify:NOOP]", msg)

def send_table(headers, rows, title="Filtered Results", max_rows=25):
    """Send formatted table to Telegram (auto-fallback if tabulate not installed)"""
    try:
        from tabulate import tabulate
        table = tabulate(rows[:max_rows], headers=headers, tablefmt="plain")
    except ModuleNotFoundError:
        # fallback simple table
        lines = ["\t".join(headers)]
        for row in rows[:max_rows]:
            lines.append("\t".join(str(x) for x in row))
        table = "\n".join(lines)
    safe = html.escape(table)
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    notify(f"<b>{title}</b>\n<pre>{safe}{extra}</pre>")

# ---------- Config ----------
EMA_LENGTH = 21
MIN_24H_PCT, MAX_24H_PCT, MAX_24H_PEAK_PCT = 12.0, 15.0, 20.0
EMA_CHECK_INTERVAL = "1h"
MAX_COINS = 100
QUOTE_WHITELIST = {"USDT","FDUSD","TUSD","USDC","USD"}
QUOTE_PRIORITY  = ["USDT","FDUSD","TUSD","USDC","USD"]

# ✅ FIXED BASE (Binance Vision mirror avoids 451)
BASE = os.environ.get("BINANCE_BASE_URL", "https://data-api.binance.vision")
EP_TICKER_24H = f"{BASE}/api/v3/ticker/24hr"
EP_KLINES     = f"{BASE}/api/v3/klines"

# ---------- Helpers ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def to_float(x, d=None):
    try: return float(x)
    except: return d

def _split_base_quote(sym: str) -> Optional[Tuple[str,str]]:
    for q in QUOTE_PRIORITY:
        if sym.endswith(q): return sym[:-len(q)], q
    return None

def fetch_top_symbols() -> List[str]:
    """Top symbols by quote volume"""
    r = requests.get(EP_TICKER_24H, timeout=15); r.raise_for_status()
    data = r.json()
    best, qrank = {}, {q:i for i,q in enumerate(QUOTE_PRIORITY)}
    for row in data:
        sym=row["symbol"]; sp=_split_base_quote(sym)
        if not sp: continue
        b,q=sp
        if b in {"USDT","BUSD","USDC","TUSD","FDUSD","DAI","UST","USTC"}: continue
        if any(b.endswith(suf) for suf in ("UP","DOWN","BULL","BEAR")): continue
        if q not in QUOTE_WHITELIST: continue
        qv=to_float(row.get("quoteVolume",0),0)
        cur=best.get(b)
        if not cur or qrank[q]<qrank[cur["quote"]] or (q==cur["quote"] and qv>cur["qv"]):
            best[b]={"sym":sym,"qv":qv,"quote":q}
    return [v["sym"] for v in sorted(best.values(),key=lambda x:x["qv"],reverse=True)[:MAX_COINS]]

def fetch_24h_stats() -> Dict[str,Dict[str,float]]:
    """Return cur_pct, peak_pct, and peak_price (24h high) for all symbols."""
    r = requests.get(EP_TICKER_24H, timeout=15); r.raise_for_status()
    out = {}
    for row in r.json():
        op  = to_float(row.get("openPrice"))
        la  = to_float(row.get("lastPrice"))
        hi  = to_float(row.get("highPrice"))
        if not op:
            continue
        cur_pct  = (la - op) / op * 100.0
        peak_pct = (hi - op) / op * 100.0
        out[row["symbol"]] = {
            "cur_pct": cur_pct,
            "peak_pct": peak_pct,
            "peak_price": hi  # 24h high = price at max% up
        }
    return out

def fetch_klines(sym,interval,limit=300):
    r=requests.get(EP_KLINES,params={"symbol":sym,"interval":interval,"limit":limit},timeout=15)
    r.raise_for_status()
    cols=["open_time","open","high","low","close","volume","close_time","qv","n","tb_base","tb_quote","ignore"]
    df=pd.DataFrame(r.json(),columns=cols)
    for c in ["open","high","low","close"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    if len(df) and int(df.iloc[-1]["close_time"])>int(datetime.now(timezone.utc).timestamp()*1000): df=df.iloc[:-1]
    return df.reset_index(drop=True)

def atr(df: pd.DataFrame, n:int=14)->pd.Series:
    h,l,c1=df["high"],df["low"],df["close"].shift(1)
    tr=pd.concat([(h-l).abs(),(h-c1).abs(),(l-c1).abs()],axis=1).max(axis=1)
    return tr.rolling(n,min_periods=n).mean()

def pivots(df,L=2,R=2):
    lows,highs=[],[]
    for i in range(max(0,len(df)-300),len(df)):
        if df["low"].iloc[i]==df["low"].iloc[max(0,i-L):min(len(df),i+R+1)].min():
            lows.append((i,float(df["low"].iloc[i])))
        if df["high"].iloc[i]==df["high"].iloc[max(0,i-L):min(len(df),i+R+1)].max():
            highs.append((i,float(df["high"].iloc[i])))
    return lows,highs

def cluster(points,tol):
    if not points: return []
    pts=sorted(points,key=lambda x:x[1]);res=[];curv=[];curi=[]
    def push(): 
        if curv: res.append({"lvl":sum(curv)/len(curv),"touch":len(curv),"last":max(curi)})
    for i,p in pts:
        if not curv: curv=[p];curi=[i];continue
        if abs(p-sum(curv)/len(curv))<=tol: curv.append(p);curi.append(i)
        else: push();curv=[p];curi=[i]
    push();return res

def strong_sr(df):
    out={"support":None,"support_touches":0,"support_dist_pct":None,
         "resistance":None,"resistance_touches":0,"resistance_dist_pct":None}
    if df.empty: return out
    last=float(df["close"].iloc[-1])
    atrv=float(atr(df).iloc[-1])
    tol=max(0.0015*last,0.25*atrv)
    lows,highs=pivots(df)
    lows,highs=cluster(lows,tol),cluster(highs,tol)
    if lows:
        c=[x for x in lows if x["lvl"]<=last]
        if c:
            s=sorted(c,key=lambda x:(x["touch"],x["lvl"],x["last"]),reverse=True)[0]
            out.update(support=s["lvl"],support_touches=s["touch"],support_dist_pct=(s["lvl"]-last)/last*100)
    if highs:
        c=[x for x in highs if x["lvl"]>=last]
        if c:
            s=sorted(c,key=lambda x:(x["touch"],-x["lvl"],x["last"]),reverse=True)[0]
            out.update(resistance=s["lvl"],resistance_touches=s["touch"],resistance_dist_pct=(s["lvl"]-last)/last*100)
    return out

def above_ema21(sym):
    df=fetch_klines(sym,"1h",150)
    if df.empty or len(df)<EMA_LENGTH+1: return False,None,None,None
    e=ema(df["close"],EMA_LENGTH)
    last,ema_now,prev=float(df["close"].iloc[-1]),float(e.iloc[-1]),float(df["close"].iloc[-2])
    return last>ema_now,last,ema_now,prev

# ---------- Main ----------
def main():
    headers=[
        "Symbol","24h % Change","24h % Peak","Peak Price (24h High)","EMA TF",
        "Prev Close","Last Close","EMA21 (Last)","Above EMA21?",
        "Strong Support (1H)","Support Touches","Support Distance %",
        "Strong Resistance (1H)","Resistance Touches","Resistance Distance %"
    ]

    print(f"Fetching top {MAX_COINS} symbols from Binance Vision...")
    syms=fetch_top_symbols()
    stats=fetch_24h_stats()
    rows=[]
    for s in syms:
        st=stats.get(s)
        if not st: continue
        if not (MIN_24H_PCT<=st["cur_pct"]<=MAX_24H_PCT and st["peak_pct"]<=MAX_24H_PEAK_PCT):
            continue
        try:
            ok,last,ema_now,prev=above_ema21(s)
            if ok:
                df=fetch_klines(s,"1h",500)
                sr=strong_sr(df)
                rows.append([
                    s,
                    f"{st['cur_pct']:.2f}",
                    f"{st['peak_pct']:.2f}",
                    f"{st['peak_price']:.6f}",   # NEW: show 24h max% up price
                    "1H",
                    f"{prev:.5f}",
                    f"{last:.5f}",
                    f"{ema_now:.6f}",
                    "YES",
                    f"{sr['support']:.6f}" if sr['support'] else "",
                    sr['support_touches'],
                    f"{sr['support_dist_pct']:.3f}" if sr['support_dist_pct'] else "",
                    f"{sr['resistance']:.6f}" if sr['resistance'] else "",
                    sr['resistance_touches'],
                    f"{sr['resistance_dist_pct']:.3f}" if sr['resistance_dist_pct'] else ""
                ])
            time.sleep(0.05)
        except Exception as e:
            print("warn",s,e)

    if rows:
        send_table(headers,rows,"✅ EMA21 Filtered Coins (1H)")
        notify(f"Total Matches: <b>{len(rows)}</b>")
    else:
        notify("❌ No coins matched filters (12–15 % 24h, Peak ≤ 20 %, Above EMA21 1H)")

if __name__=="__main__":
    main()
