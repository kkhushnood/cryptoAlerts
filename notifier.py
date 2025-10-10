# notifier.py
import os, re, requests, json

RAW_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

def _extract_token(s: str) -> str:
    """
    Extract a BotFather token from messy input: full messages, URLs, or prefixed with 'bot'.
    Returns cleaned token or empty string if not found.
    """
    t = s.strip().strip('"').strip("'")
    # If a full URL was pasted: https://api.telegram.org/bot<token>/sendMessage
    m = re.search(r"/bot([^/\s]+)/?", t)
    if m:
        t = m.group(1)

    # remove accidental 'bot' prefix
    if t.lower().startswith("bot"):
        t = t[3:]

    # Best-effort: find first token-looking pattern anywhere in the string
    m2 = re.search(r"(\d+:[A-Za-z0-9_-]{30,})", t)
    if m2:
        return m2.group(1)
    return ""

def _clean_token(raw: str) -> str:
    token = _extract_token(raw)
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN looks malformed. Save ONLY the token from @BotFather, "
            "e.g. 123456789:AAH-... (no quotes/newlines/URL)."
        )
    return token

BOT_TOKEN = _clean_token(RAW_TOKEN)

BASE = "https://api.telegram.org"

def _post(method: str, payload: dict) -> dict:
    url = f"{BASE}/bot{BOT_TOKEN}/{method}"
    r = requests.post(url, data=payload, timeout=20)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Telegram HTTP {r.status_code}, non-JSON body: {r.text[:300]}")
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram error {data.get('error_code')}: {data.get('description')}")
    return data

def notify(text: str) -> None:
    payload = {
        "chat_id": CHAT_ID,  # numeric ID for DM/group/channel (channel/group usually starts with -100)
        "text": text,
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }
    _post("sendMessage", payload)
