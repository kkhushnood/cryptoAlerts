# -*- coding: utf-8 -*-
"""
run_strategy.py
Fetch BTC/USDT (or any symbol) price from Binance public API with mirror fallback,
then send it to Telegram via notifier.notify().

Usage:
  python run_strategy.py
  python run_strategy.py --symbol ETHUSDT
  python run_strategy.py --base-url https://api.binance.us
"""

import argparse
from decimal import Decimal, getcontext
import time
from typing import Iterable, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- try to import notifier.notify; if missing or broken, use a safe no-op ---
try:
    from notifier import notify as _notify_real  # type: ignore
    def notify(text: str) -> None:
        _notify_real(text)
except Exception as _e:
    def notify(text: str) -> None:
        # No Telegram env / missing notifier.py — keep pipeline green
        print(f"[notify:NOOP] {text}")

# High precision for crypto prices
getcontext().prec = 28

# Primary + mirrors (order matters)
BINANCE_BASE_URLS: tuple[str, ...] = (
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api-gcp.binance.com",
    # Public data mirror (no auth needed)
    "https://data-api.binance.vision",
)

DEFAULT_SYMBOL = "BTCUSDT"

class PriceFetchError(RuntimeError):
    pass

def _session_with_retries(total: int = 3, backoff: float = 0.3) -> requests.Session:
    """Requests session with retry for transient errors."""
    s = requests.Session()
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; price-fetcher/1.0)"
    })
    return s

def _try_fetch_price(session: requests.Session, base_url: str, symbol: str, timeout: float) -> Optional[Decimal]:
    """Try one base_url; return Decimal price or None on handled failure."""
    url = f"{base_url.rstrip('/')}/api/v3/ticker/price"
    try:
        r = session.get(url, params={"symbol": symbol}, timeout=timeout)
    except requests.RequestException:
        return None

    if r.status_code == 451:
        return None
    if r.status_code != 200:
        return None

    try:
        data = r.json()
        price_str = data.get("price")
        if not price_str:
            return None
        return Decimal(price_str)
    except Exception:
        return None

def get_symbol_price(
    symbol: str = DEFAULT_SYMBOL,
    base_urls: Iterable[str] = BINANCE_BASE_URLS,
    timeout: float = 5.0,
    per_host_attempts: int = 2,
) -> Decimal:
    """
    Binance public API se <symbol> ki current price Decimal me return karta hai.
    Multiple mirrors try karta hai; 451/temporary errors par rotate karta hai.
    """
    symbol = symbol.upper().strip()
    session = _session_with_retries()

    tried = []
    for base in base_urls:
        for _ in range(per_host_attempts):
            price = _try_fetch_price(session, base, symbol, timeout)
            if price is not None:
                return price
            time.sleep(0.2)
        tried.append(base)

    hint = (
        "Server ne 451 ya network error diye ho sakte hain (region/legal restrictions). "
        "Agar aap Binance.US use karte hain to `--base-url https://api.binance.us` try karein; "
        "warna allowed region/Data mirror ensure karein."
    )
    raise PriceFetchError(
        f"Failed to fetch {symbol} price from Binance public mirrors: {', '.join(tried)}. {hint}"
    )

def main():
    parser = argparse.ArgumentParser(description="Fetch current price from Binance public API (with mirror fallback).")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom Binance base URL (e.g., https://api.binance.us). If set, only this URL is used."
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout seconds")
    parser.add_argument("--attempts", type=int, default=2, help="Attempts per host before moving on")
    parser.add_argument("--no-notify", action="store_true", help="Only print price; do not send Telegram notification")
    args = parser.parse_args()

    try:
        if args.base_url:
            price = get_symbol_price(args.symbol, base_urls=(args.base_url,), timeout=args.timeout, per_host_attempts=args.attempts)
        else:
            price = get_symbol_price(args.symbol, timeout=args.timeout, per_host_attempts=args.attempts)

        # print to logs
        print(f"{args.symbol}: {price}")

        # send Telegram notification unless disabled
        if not args.no_notify:
            notify(f"<b>{args.symbol}</b>: <code>{price}</code>")

    except PriceFetchError as e:
        print(str(e))
        # also alert failure (optional)
        try:
            if not args.no_notify:
                notify(f"⚠️ Price fetch failed: {e}")
        except Exception:
            pass
        raise SystemExit(1)

if __name__ == "__main__":
    main()
