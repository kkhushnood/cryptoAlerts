#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMA21 Filter — sends final table directly to Telegram (no CSV)
Filters:
  - 24h change between 12%–15%
  - Peak ≤ 20%
  - Above EMA21 (latest CLOSED 1H)
Output:
  Table with columns:
  Symbol | 24h % Change | 24h % Peak | EMA TF | Prev Close | Last Close |
  EMA21 (Last) | Above EMA21? | Strong Support (1H) | Support Touches | Support Distance % |
  Strong Resistance (1H) | Resistance Touches | Resistance Distance %

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os, time, math, html, requests, pandas as pd
from datetime import datetime, timezone
from dateutil import tz
from typing import List, Dict, Optional, Tuple

# ---------- Telegram ----------
try:
    from notifier import notify   # must be your working notifier.py
except Exception as e:
    def notify(msg: str):
        print("[notify:NOOP]", msg)

def send_table_to_telegram(headers, rows, title="Filtered Results", max_rows=25):
    """Send table in chunks to Telegram"""
    from tabulate import tabulate
    total = len(rows)
    show_rows = rows[:max_rows]
    table = tabulate(show_rows, headers=headers, tablefmt="plain")
    safe = html.escape(table)
    extra = f"\n(+{total - max_rows} more rows)" if total > max_rows else ""
    notify(f"<b>{title}</b>\n<pre>{safe}{extra}</pre>")

# ---------- Config ----------
EMA_LENGTH = 21
MIN_24H_PCT, MAX_24H_PCT, MAX_24H_PEAK_PCT = 12.0, 15.0, 20.0
EMA_CHECK_INTERVAL = "1h"
MAX_COINS = 100
QUOTE_WHITELIST  = {"USDT","FDUSD","TUSD","USDC","USD"}
QUOTE_PRIORITY   = ["USDT","FDUSD","TUSD","USDC","USD"]
BASE = "https://api.binance.com"
EP_TICKER_24H = f"{BASE}/api/v3/ticker/24hr"
EP_KLINES = f"{BASE}/api/v3/klines"

# ---------- Helpers ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _split_base_quote(sym: str) -> Optional[Tuple[str,str]]:
    for q in QUOTE_PRIORITY:
        if sym.endswith(q):
            return sym[:-len(q)], q
    return None

def fetch_top_symbols() -> List[str]:
    """Fetch top coins by quote volume (USD pairs)"""
    r = requests.get(EP_TICKER_24H, timeout=15); r.raise_for_status()
    data = r.json()
    best: Dict[str, Dict] = {}
    q_rank = {q:i for i,q in enumerate(QUOTE_PRIORITY)}
    for row in data:
        sym = row.get("symbol",""); sp = _split_base_quote(sym)
        if not sp: continue
        b,q = sp
        if b in {"USDT","BUSD","USDC","TUSD","FDUSD","DAI","UST","USTC"}: continue
        if any(b.endswith(suf) for suf in ("UP","DOWN","BULL","BEAR")): continue
        if q not in QUOTE_WHITELIST: continue
        qv = to_float(row.get("quoteVolume","0"),0)
        cur = best.get(b)
        if not cur or q_rank[q]<q_rank[cur["quote"]] or (q==cur["quote"] and qv>cur["qv"]):
            best[b] = {"sym":sym,"qv":qv,"quote":q}
    return [v["sym"] for v in sorted(best.values(), key=lambda x:x["qv"], reverse=True)[:MAX_COINS]]

def fetch_24h_stats() -> Dict[str,Dict[str,float]]:
    r = requests.get(EP_TICKER_24H,timeout=15); r.raise_for_status()
    out={}
    for row in r.json():
        op=to_float(row.get("openPrice",0)); la=to_float(row.get("lastPrice",0)); hi=to_float(row.get("highPrice",0))
        if not op: continue
        out[row["symbol"]]={"cur_pct":(la-op)/op*100,"peak_pct":(hi-op)/op*100}
    return out

def fetch_klines(sym,interval,limit=500):
    r=requests.get(EP_KLINES,params={"symbol":sym,"interval":interval,"limit":limit},timeout=15)
    r.raise_for_status(); kl=r.json()
    if not kl: return pd.DataFrame()
    cols=["open_time","open","high","low","close","volume","close_time","qv","n","tb_base","tb_quote","ignore"]
    df=pd.DataFrame(kl,columns=cols)
    for c in ["open","high","low","close"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    if len(df) and int(df.iloc[-1]["close_time"])>int(datetime.now(timezone.utc).timestamp()*1000):
        df=df.iloc[:-1]
    return df.reset_index(drop=True)

def atr_df(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c1 = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h-l).abs(), (h-c1).abs(), (l-c1).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _is_pivot_high(df, i, L=2, R=2):
    lo=max(0,i-L); hi=min(len(df),i+R+1)
    return df["high"].iloc[i]==df["high"].iloc[lo:hi].max()

def _is_pivot_low(df, i, L=2, R=2):
    lo=max(0,i-L); hi=min(len(df),i+R+1)
    return df["low"].iloc[i]==df["low"].iloc[lo:hi].min()

def find_strong_sr(df: pd.DataFrame, lookback: int = 300) -> Dict[str, Optional[float]]:
    out={"support":None,"support_touches":0,"support_dist_pct":None,
         "resistance":None,"resistance_touches":0,"resistance_dist_pct":None}
    if df is None or df.empty: return out
    last_close=float(df["close"].iloc[-1])
    atr_last=float(atr_df(df,14).iloc[-1])
    tol_abs=max(0.0015*last_close,0.25*atr_last)
    lows,highs=[],[]
    start=max(0,len(df)-lookback)
    for i in range(start,len(df)):
        if _is_pivot_low(df,i): lows.append((i,float(df["low"].iloc[i])))
        if _is_pivot_high(df,i): highs.append((i,float(df["high"].iloc[i])))
    def cluster(points):
        if not points: return []
        pts=sorted(points,key=lambda x:x[1])
        clust=[];curv=[];curi=[]
        def push():
            if curv: clust.append({"lvl":sum(curv)/len(curv),"touch":len(curv),"last":max(curi)})
        for i,p in pts:
            if not curv: curv=[p];curi=[i];continue
            if abs(p-sum(curv)/len(curv))<=tol_abs: curv.append(p);curi.append(i)
            else: push();curv=[p];curi=[i]
        push();return clust
    lows=cluster(lows); highs=cluster(highs)
    if lows:
        cands=[c for c in lows if c["lvl"]<=last_close]
        if cands:
            cands.sort(key=lambda c:(c["touch"],c["lvl"],c["last"]),reverse=True)
            s=cands[0];out["support"]=s["lvl"];out["support_touches"]=s["touch"]
            out["support_dist_pct"]=(s["lvl"]-last_close)/last_close*100
    if highs:
        cands=[c for c in highs if c["lvl"]>=last_close]
        if cands:
            cands.sort(key=lambda c:(c["touch"],-c["lvl"],c["last"]),reverse=True)
            r=cands[0];out["resistance"]=r["lvl"];out["resistance_touches"]=r["touch"]
            out["resistance_dist_pct"]=(r["lvl"]-last_close)/last_close*100
    return out

def is_above_ema21(sym,interval="1h"):
    df=fetch_klines(sym,interval,limit=150)
    if df.empty or len(df)<EMA_LENGTH+1: return False,None,None,None
    e=ema(df["close"],EMA_LENGTH)
    last=float(df["close"].iloc[-1]); ema_now=float(e.iloc[-1])
    prev=float(df["close"].iloc[-2])
    return last>ema_now,last,ema_now,prev

# ---------- main ----------
def main():
    headers=[
        "Symbol","24h % Change","24h % Peak","EMA TF",
        "Prev Close","Last Close","EMA21 (Last)","Above EMA21?",
        "Strong Support (1H)","Support Touches","Support Distance %",
        "Strong Resistance (1H)","Resistance Touches","Resistance Distance %"
    ]
    print("Fetching top symbols...")
    syms=fetch_top_symbols()
    stats=fetch_24h_stats()
    rows=[]
    for s in syms:
        st=stats.get(s)
        if not st: continue
        if not (MIN_24H_PCT<=st["cur_pct"]<=MAX_24H_PCT and st["peak_pct"]<=MAX_24H_PEAK_PCT):
            continue
        try:
            ok,last,ema_now,prev=is_above_ema21(s)
            if ok:
                df=fetch_klines(s,"1h",500)
                sr=find_strong_sr(df)
                rows.append([
                    s,f"{st['cur_pct']:.2f}",f"{st['peak_pct']:.2f}","1H",
                    f"{prev:.5f}",f"{last:.5f}",f"{ema_now:.6f}","YES",
                    f"{sr['support']:.6f}" if sr['support'] else "",
                    sr['support_touches'],
                    f"{sr['support_dist_pct']:.3f}" if sr['support_dist_pct'] else "",
                    f"{sr['resistance']:.6f}" if sr['resistance'] else "",
                    sr['resistance_touches'],
                    f"{sr['resistance_dist_pct']:.3f}" if sr['resistance_dist_pct'] else "",
                ])
            time.sleep(0.05)
        except Exception as e:
            print("warn",s,e)
    if rows:
        title="✅ EMA21 Filtered Coins (1H)"
        send_table_to_telegram(headers,rows,title)
        notify(f"Total Matches: <b>{len(rows)}</b>")
    else:
        notify("❌ No coins matched filters (12–15% 24h, Peak ≤20%, Above EMA21 1H)")

if __name__=="__main__":
    main()
