# notifier.py
import os
import requests

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

def notify(text: str) -> None:
    """
    Send a simple Telegram message to CHAT_ID.
    Raises RuntimeError if Telegram returns non-200 status.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, data=payload, timeout=20)
        # if Telegram returns an error, surface it so CI fails visibly
        if r.status_code != 200:
            raise RuntimeError(f"Telegram API error {r.status_code}: {r.text}")
    except Exception as e:
        # don't crash silently in CI
        print("Notify failed:", e)
        raise
