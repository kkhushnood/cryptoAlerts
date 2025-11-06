#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Telegram Notifier
------------------------
Used by run_strategy.py to send messages.

Environment variables required:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import requests

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

def notify(msg: str):
    """Send a message to Telegram chat"""
    if not BOT_TOKEN or not CHAT_ID:
        print("[notify:missing TOKEN or CHAT_ID]", msg)
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    }

    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            print("[notify:OK]", msg[:60])
        else:
            print(f"[notify:FAIL] {r.status_code}: {r.text}")
    except Exception as e:
        print("[notify:EXC]", e)
