"""
test_alert.py - Smartphone Push-Alert Test (Telegram & Discord)
Sendet einen sofortigen Test-Alarm aufs Smartphone, um zu prüfen, ob die Benachrichtigungen funktionieren.
"""

import os
import sys
import json
import urllib.request
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
WHATSAPP_PHONE_NUMBER = os.getenv("WHATSAPP_PHONE_NUMBER", "").strip()
WHATSAPP_API_KEY = os.getenv("WHATSAPP_API_KEY", "").strip()


def send_test_telegram() -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Nicht konfiguriert (TELEGRAM_BOT_TOKEN oder TELEGRAM_CHAT_ID fehlt in .env).")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": (
            "🤖 *Autonomous Finance Agent · Test-Alarm*\n\n"
            "✅ *Verbindung erfolgreich hergestellt!*\n"
            "Dein KI-Trading-Schwarm (14 Spezial-Unteragenten) ist mit deinem Smartphone verbunden.\n\n"
            "Du wirst ab sofort benachrichtigt bei:\n"
            "• 🚀 Penny-Stock Ausbrüchen & 8-K Katalysatoren\n"
            "• 📈 Trade-Eröffnungen & Gewinn-Mitnahmen\n"
            "• 🧬 Geburt neuer Spezial-Hilfsagenten"
        ),
        "parse_mode": "Markdown",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                print("✅ [Telegram] Test-Nachricht erfolgreich aufs Smartphone gesendet!")
                return True
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
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "AutonomousFinanceAgent/2.0"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status in (200, 204):
                print("✅ [Discord] Test-Nachricht erfolgreich im Discord-Kanal gesendet!")
                return True
    except Exception as e:
        print(f"❌ [Discord] Fehler beim Senden: {e}")
    return False


def send_test_whatsapp() -> bool:
    if not WHATSAPP_PHONE_NUMBER or not WHATSAPP_API_KEY:
        print("[WhatsApp] Nicht konfiguriert (WHATSAPP_PHONE_NUMBER oder WHATSAPP_API_KEY fehlt in .env).")
        return False

    import urllib.parse
    clean_phone = WHATSAPP_PHONE_NUMBER.replace("+", "").replace(" ", "").replace("-", "")
    text_content = (
        "🤖 Autonomous Finance Agent · Test-Alarm\n\n"
        "✅ WhatsApp-Verbindung erfolgreich hergestellt!\n"
        "Dein KI-Trading-Schwarm ist jetzt direkt mit deinem WhatsApp verbunden.\n\n"
        "Alerts für Trades, Penny Stocks & Agenten-Evolution werden ab sofort hierher gesendet."
    )
    encoded_text = urllib.parse.quote(text_content)
    wa_url = f"https://api.callmebot.com/whatsapp.php?phone={clean_phone}&text={encoded_text}&apikey={WHATSAPP_API_KEY}"

    try:
        req = urllib.request.Request(wa_url, headers={"User-Agent": "AutonomousFinanceAgent/2.0"})
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            if resp.status == 200:
                print("✅ [WhatsApp] Test-Nachricht erfolgreich auf dein WhatsApp gesendet!")
                return True
    except Exception as e:
        print(f"❌ [WhatsApp] Fehler beim Senden: {e}")
    return False


if __name__ == "__main__":
    print("=== 📲 Teste Smartphone Push-Benachrichtigungen ===")
    t_ok = send_test_telegram()
    d_ok = send_test_discord()
    w_ok = send_test_whatsapp()
    if not t_ok and not d_ok and not w_ok:
        print("\n👉 Bitte trage deinen Telegram-Bot, Discord-Webhook oder WhatsApp-Key in .env ein.")

