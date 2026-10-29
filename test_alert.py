"""
test_alert.py - Smartphone Push-Alert Test (Telegram & Discord)
Sendet einen sofortigen Test-Alarm aufs Smartphone, um zu prüfen, ob die Benachrichtigungen funktionieren.
"""

import os
import sys
import json
import ssl
import urllib.request
import urllib.error
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


def _get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl._create_unverified_context()
        except Exception:
            return ssl.create_default_context()


def send_test_telegram() -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Nicht konfiguriert (TELEGRAM_BOT_TOKEN oder TELEGRAM_CHAT_ID fehlt in .env).")
        return False

    ctx = _get_ssl_context()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": int(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID.isdigit() else TELEGRAM_CHAT_ID,
        "text": (
            "🤖 <b>Autonomous Finance Agent · Test-Alarm</b>\n\n"
            "✅ <b>Verbindung erfolgreich hergestellt!</b>\n"
            "Dein KI-Trading-Schwarm (14 Spezial-Unteragenten) ist mit deinem Smartphone verbunden.\n\n"
            "Du wirst ab sofort in Echtzeit benachrichtigt bei:\n"
            "• 🚀 Penny-Stock Ausbrüchen & SEC 8-K Katalysatoren\n"
            "• 📈 Trade-Eröffnungen & 2-Stufen Gewinn-Mitnahmen\n"
            "• 🧬 Geburt neuer Unteragenten & Meta-Evolution"
        ),
        "parse_mode": "HTML",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=ctx, timeout=6.0) as resp:
            if resp.status == 200:
                print("✅ [Telegram] Test-Nachricht erfolgreich aufs Smartphone gesendet!")
                return True
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        if "chat not found" in err_body.lower():
            print("⚠️ [Telegram] Chat noch nicht initialisiert: Bitte öffne in Telegram @Henry_trading_agent_bot und drücke 'START'!")
        else:
            print(f"❌ [Telegram] HTTP Fehler: {err_body}")
    except Exception as e:
        print(f"❌ [Telegram] Fehler beim Senden: {e}")
    return False


def send_test_discord() -> bool:
    if not DISCORD_WEBHOOK_URL:
        print("[Discord] Nicht konfiguriert (DISCORD_WEBHOOK_URL fehlt in .env).")
        return False

    payload = json.dumps({
        "content": (
            "**🤖 Autonomous Finance Agent · Test-Alarm**\n"
            "✅ **Verbindung erfolgreich hergestellt!**\n"
            "Dein KI-Trading-Schwarm (14 Spezial-Unteragenten) ist mit deinem Discord-Kanal verbunden.\n"
            "Echtzeit-Meldungen zu Trades, Penny Stocks und Agenten-Evolution sind ab jetzt aktiv!"
        )
    }).encode("utf-8")

    try:
        ctx = _get_ssl_context()
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "AutonomousFinanceAgent/2.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=5.0) as resp:
            if resp.status in (200, 204):
                print("✅ [Discord] Test-Nachricht erfolgreich im Discord-Kanal gesendet!")
                return True
    except Exception as e:
        print(f"❌ [Discord] Fehler beim Senden: {e}")
    return False


if __name__ == "__main__":
    print("=== 📲 Teste Smartphone Push-Benachrichtigungen ===")
    t_ok = send_test_telegram()
    d_ok = send_test_discord()
