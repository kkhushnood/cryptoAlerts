# -*- coding: utf-8 -*-
"""
run_strategy.py
Simple Binance public API price fetcher for BTCUSDT.

Usage (CLI):
    python run_strategy.py
    python run_strategy.py --symbol ETHUSDT
    python run_strategy.py --base-url https://api.binance.us

Returns (in code): Decimal price via get_symbol_price()

Notes:
- No API key required (public endpoint).
- Uses Decimal for precision-safe arithmetic.
"""

import argparse
from decimal import Decimal, getcontext
import requests

# Set high precision for crypto prices
getcontext().prec = 28

DEFAULT_BASE_URL = "https://api.binance.com"
DEFAULT_SYMBOL = "BTCUSDT"

class PriceFetchError(RuntimeError):
    pass

def get_symbol_price(symbol: str = DEFAULT_SYMBOL,
                     base_url: str = DEFAULT_BASE_URL,
                     timeout: float = 5.0) -> Decimal:
    """
    Binance public API se <symbol> ki current price Decimal me return karta hai.
    Raises PriceFetchError on failure.
    """
    symbol = symbol.upper().strip()
    url = f"{base_url.rstrip('/')}/api/v3/ticker/price"
    params = {"symbol": symbol}

    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # Expected: {"symbol":"BTCUSDT","price":"12345.67000000"}
        price_str = data.get("price")
        if price_str is None:
            raise PriceFetchError(f"Unexpected response payload: {data!r}")
        return Decimal(price_str)
    except (requests.RequestException, ValueError) as e:
        raise PriceFetchError(f"Failed to fetch {symbol} price from {base_url}: {e}") from e

def main():
    parser = argparse.ArgumentParser(description="Fetch current price from Binance public API.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading pair symbol, e.g., BTCUSDT")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Binance API base URL (use https://api.binance.us for Binance.US)")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    try:
        price = get_symbol_price(args.symbol, args.base_url, args.timeout)
        print(f"{args.symbol}: {price}")
    except PriceFetchError as e:
        # Clean, user-friendly error on CLI
        print(str(e))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
