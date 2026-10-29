"""
watchdog_daemon.py
==================
Unattended Watchdog & Process Supervisor für den Autonomous Trading Agent.
Stellt sicher, dass der Agent während Henrys Abwesenheit 24/7 unterbrechungsfrei läuft.
Sollte der Prozess jemals unerwartet beenden (z.B. durch temporäre Netzwerkfehler,
Windows-Treiber-Updates oder Speicherdruck), startet der Watchdog den Agenten
binnen 5 Sekunden vollautomatisch neu und informiert Henry per Telegram.
"""

import datetime
import os
import subprocess
import sys
import time
from pathlib import Path


def send_telegram_alert(message: str) -> None:
    try:
        from config import AgentConfig
        from telegram_command_listener import TelegramCommandListener
        cfg = AgentConfig()
        if cfg.telegram_bot_token and cfg.telegram_chat_id:
            tg = TelegramCommandListener(cfg.telegram_bot_token, cfg.telegram_chat_id)
            tg.send_message(message)
    except Exception as e:
        print(f"[Watchdog] Konnte Telegram-Warnung nicht senden: {e}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    cmd = [sys.executable, str(project_dir / "main.py"), "--loop", "300"]
    restart_count = 0

    print("=" * 70)
    print("🛡️  AUTONOMOUS FINANCE AGENT - RESILIENT WATCHDOG SUPERVISOR")
    print(f"📁 Projekt-Verzeichnis: {project_dir}")
    print(f"⚙️  Kommando: {' '.join(cmd)}")
    print(f"🕒 Gestartet am: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    send_telegram_alert(
        "🛡️ <b>Watchdog Supervisor aktiviert:</b>\n\n"
        "Der autonome Trading-Agent wird während deiner 3-tägigen Reise rund um die Uhr "
        "überwacht. Bei Ausfällen erfolgt ein sofortiger, automatischer Neustart."
    )

    while True:
        try:
            print(f"\n[Watchdog] Starte Trading-Agent (Lauf #{restart_count + 1})...")
            proc = subprocess.Popen(cmd, cwd=str(project_dir))
            proc.wait()
            rc = proc.returncode

            # Wenn der Prozess manuell mit Exit 0 beendet wurde, respektieren wir das
            if rc == 0:
                print(f"[Watchdog] Prozess sauber mit Exit-Code 0 beendet.")
                break

            restart_count += 1
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            warn_msg = (
                f"⚠️ <b>Watchdog-Meldung ({now_str}):</b>\n\n"
                f"Der Trading-Agent wurde unerwartet beendet (Exit-Code {rc}).\n"
                f"🔄 <b>Automatischer Neustart #{restart_count} in 5 Sekunden...</b>\n\n"
                f"<i>Alle Sicherheitsfilter & Stop-Losses bleiben voll aktiv.</i>"
            )
            print(f"[Watchdog] WARNUNG: Prozess mit Code {rc} beendet. Neustart in 5 Sekunden...")
            send_telegram_alert(warn_msg)
            time.sleep(5.0)

        except KeyboardInterrupt:
            print("\n[Watchdog] Manuell durch Benutzer (Strg+C) beendet.")
            try:
                proc.terminate()
            except Exception:
                pass
            break
        except Exception as e:
            print(f"[Watchdog] Kritischer Fehler im Supervisor: {e}")
            time.sleep(10.0)


if __name__ == "__main__":
    main()
