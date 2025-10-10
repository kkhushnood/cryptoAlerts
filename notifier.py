# notifier.py
# Sends Telegram message using env vars:
#   TELEGRAM_BOT_TOKEN  -> BotFather token (e.g. 123456789:AAH-...)
#   TELEGRAM_CHAT_ID    -> numeric ID (DM: positive, group/channel: usually -100...), or @channel_username
#   TELEGRAM_DEBUG      -> "1" to print success JSON (masked/limited)

import os
import re
import json
import requests

RAW_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
RAW_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")
DEBUG     = os.environ.get("TELEGRAM_DEBUG", "0") == "1"

BASE = "https://api.telegram.org"

# ---------- helpers ----------
def _mask(s: str) -> str:
    if not s:
        return "<EMPTY>"
    return (s[:6] + "..." + s[-6:]) if len(s) > 15 else "***"

def _extract_token(s: str) -> str:
    """
    Accept messy inputs and extract a Telegram token.
    Handles:
      - full URL: https://api.telegram.org/bot<token>/sendMessage
      - token with 'bot' prefix
      - full BotFather message (find first token-shaped substring)
    """
    t = s.strip().strip('"').strip("'")

    # If a full URL was pasted
    m = re.search(r"/bot([^/\s]+)/?", t)
    if m:
        t = m.group(1)

    # If someone left 'bot' prefix
    if t.lower().startswith("bot"):
        t = t[3:]

    # First token-looking substring
    m2 = re.search(r"(\d+:[A-Za-z0-9_-]{30,})", t)
    return m2.group(1) if m2 else ""

def _clean_token(raw: str) -> str:
    if not raw:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty/missing in environment.")
    token = _extract_token(raw)
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN looks malformed. Save ONLY the token from @BotFather, "
            "e.g. 123456789:AAH-... (no quotes/newlines/URL). "
            f"Received (masked): { _mask(raw) }"
        )
    return token

def _clean_chat_id(raw: str) -> str:
    """
    Accepts numeric id (e.g. 6039..., -100...) or @username.
    Returns trimmed string; raises if empty.
    """
    cid = raw.strip()
    if not cid:
        raise RuntimeError("TELEGRAM_CHAT_ID is empty/missing in environment.")
    return cid

BOT_TOKEN = _clean_token(RAW_TOKEN)
CHAT_ID   = _clean_chat_id(RAW_CHAT)

def _post(method: str, payload: dict) -> dict:
    url = f"{BASE}/bot{BOT_TOKEN}/{method}"
    r = requests.post(url, data=payload, timeout=20)
    # Telegram sometimes returns 200 with ok=false -> must check JSON
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Telegram HTTP {r.status_code}, non-JSON body: {r.text[:300]}")
    if not data.get("ok", False):
        # Common pitfalls:
        #  - 400 chat not found -> wrong CHAT_ID or bot not started/added/admin
        #  - 401 unauthorized   -> token wrong/expired
        #  - 404 not found      -> path broken (malformed token/extra 'bot')
        raise RuntimeError(f"Telegram error {data.get('error_code')}: {data.get('description')}")
    return data

# ---------- public API ----------
def notify(text: str) -> None:
    """
    Send a Telegram message to CHAT_ID. Uses HTML parse_mode by default.
    """
    payload = {
        "chat_id": CHAT_ID,                 # e.g. "6039..." or "-100123..." or "@mychannel"
        "text": text,
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }
    resp = _post("sendMessage", payload)
    if DEBUG:
        # compact debug output (no secrets)
        print("Telegram sendMessage ok ->", json.dumps(resp, ensure_ascii=False)[:600])

# ---------- optional diagnostics (manual use) ----------
def tg_get_me() -> dict:
    return _post("getMe", {})

def tg_get_chat() -> dict:
    return _post("getChat", {"chat_id": CHAT_ID})

if __name__ == "__main__":
    # quick manual test (only runs if you execute notifier.py directly)
    notify("✅ Telegram notifier test — it works!")
