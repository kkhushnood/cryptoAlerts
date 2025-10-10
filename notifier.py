# notifier.py
import os, re, requests

def _clean_token(raw: str) -> str:
    t = raw.strip()
    if t.lower().startswith("bot"):
        t = t[3:]
    if not re.match(r"^\d+:[A-Za-z0-9_-]{30,}$", t):
        raise RuntimeError("TELEGRAM_BOT_TOKEN looks malformed.")
    return t

BOT_TOKEN = "8447796045:AAFcJgKfBh1aEBmjEiqK-2IxkUaDQ7BQHws"
CHAT_ID   = "6039469607"

def notify(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "test hello",
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }
    r = requests.post(url, data=payload, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")
