#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMA-21 filter — sends results directly to Telegram
Filters:
  • 24h change 12–15 %
  • 24h peak ≤ 20 %
  • Above EMA-21 on latest CLOSED 1H candle
Output:
  Telegram message (no CSV)

Requirements:
  pip install requests pandas python-dateutil tabulate
Env vars:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os, time, math, html, requests, pandas as pd
from datetime import datetime, timezone
from dateutil import tz
from typing import List, Dict, Optional, Tuple

# ---------- Telegram ----------
try:
    from notifier import notify  # uses your working notifier.py
except Exception as e:
    def notify(msg: str):
        print("[notify:NOOP]", msg)

def send_table_to_telegram(headers, rows, title, max_rows=25):
    """Pretty print table and send to Telegram in chunks"""
    from tabulate import tabulate
    total = len(rows)
    shown = rows[:max_rows]
    table = tabulate(shown, headers=headers, tablefmt="plain")
    safe = html.escape(table)
    extra = f"\n(+{total - max_rows} more)" if total > max_rows else ""
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
    try: return float(x)
    except Exception: return default

def _split_base_quote(sym: str) -> Optional[Tuple[str,str]]:
    for q in QUOTE_PRIORITY:
        if sym.endswith(q): return sym[:-len(q)], q
    return None

def fetch_top_symbols() -> List[str]:
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

def fetch_klines(sym,interval,limit=200):
    r=requests.get(EP_KLINES,params={"symbol":sym,"interval":interval,"limit":limit},timeout=15)
    r.raise_for_status(); kl=r.json()
    if not kl: return pd.DataFrame()
    cols=["open_time","open","high","low","close","volume","close_time","qv","n","tb_base","tb_quote","ignore"]
    df=pd.DataFrame(kl,columns=cols)
    for c in ["open","high","low","close"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    if len(df) and int(df.iloc[-1]["close_time"])>int(datetime.now(timezone.utc).timestamp()*1000):
        df=df.iloc[:-1]
    return df.reset_index(drop=True)

def is_above_ema21(sym,interval="1h"):
    df=fetch_klines(sym,interval,limit=150)
    if df.empty or len(df)<EMA_LENGTH+1: return False,None,None,None
    e=ema(df["close"],EMA_LENGTH)
    last=float(df["close"].iloc[-1]); ema_now=float(e.iloc[-1])
    prev=float(df["close"].iloc[-2])
    return last>ema_now,last,ema_now,prev

# ---------- main ----------
def main():
    headers=["Symbol","24h %","Peak %","Prev Close","Last Close","EMA21","Above EMA21?"]
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
                rows.append([s,f"{st['cur_pct']:.2f}",f"{st['peak_pct']:.2f}",
                             f"{prev:.4f}",f"{last:.4f}",f"{ema_now:.4f}","YES"])
            time.sleep(0.05)
        except Exception as e:
            print("warn",s,e)
    if rows:
        title="✅ Coins 12–15 % 24h & Peak ≤ 20 & Above EMA21 (1H)"
        send_table_to_telegram(headers,rows,title)
        notify(f"Total matches: <b>{len(rows)}</b>")
    else:
        notify(f"❌ No coins matched (24h {MIN_24H_PCT}–{MAX_24H_PCT} %, Peak ≤ {MAX_24H_PEAK_PCT} %, Above EMA21 1H)")

if __name__=="__main__":
    main()
