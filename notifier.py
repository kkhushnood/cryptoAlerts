# notifier.py
import os, re, requests, json

def _clean_token(raw: str) -> str:
    t = raw.strip()
    if t.lower().startswith("bot"):
        t = t[3:]
    if not re.match(r"^\d+:[A-Za-z0-9_-]{30,}$", t):
        raise RuntimeError("TELEGRAM_BOT_TOKEN looks malformed.")
    return t

BOT_TOKEN = _clean_token(os.environ["TELEGRAM_BOT_TOKEN"])
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]          # string is fine
DEBUG     = os.environ.get("TELEGRAM_DEBUG", "0") == "1"

BASE = "https://api.telegram.org"

def _post(method: str, payload: dict) -> dict:
    url = f"{BASE}/bot{BOT_TOKEN}/{method}"
    r = requests.post(url, data=payload, timeout=20)
    # Telegram sometimes returns 200 even for errors -> must check JSON.ok
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Telegram HTTP {r.status_code}, non-JSON body: {r.text[:300]}")
    if not data.get("ok", False):
        # Raise with the Telegram error details
        raise RuntimeError(f"Telegram error {data.get('error_code')}: {data.get('description')}")
    return data

def notify(text: str) -> None:
    payload = {
        "chat_id": CHAT_ID,      # e.g. "6039469607" (DM) or "-1001234567890" (group/channel)
        "text": text,
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }
    resp = _post("sendMessage", payload)
    if DEBUG:
        print("Telegram sendMessage ok ->", json.dumps(resp, ensure_ascii=False)[:500])

# Optional diagnostics you can call once while debugging:
def get_me() -> dict:
    return _post("getMe", {})

def get_chat() -> dict:
    return _post("getChat", {"chat_id": CHAT_ID})
