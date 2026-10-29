"""
background_evolution_daemon.py - 24/7 Parallele Hintergrund-Strategie-Zucht & Evolution
Ermöglicht ununterbrochene Weiterentwicklung des Agenten-Schwarms:
- Läuft in einem unabhängigen Hintergrund-Thread (ohne den Haupt-Handelszyklus zu bremsen).
- Züchtet alle 30-45 Minuten neue Generationen von Handelsstrategien (Sakana AI / AlphaZero Breeding).
- Führt Walk-Forward-Krisentests durch.
- Befördert Elite-Strategien automatisch zu neuen Spezial-Unteragenten im ADAS-Schwarm.
- Informiert Henry via Telegram über neu geborene Elite-Agenten.
"""

import datetime
import json
import logging
import os
import threading
import time
from typing import Dict, Any, Optional, List

from genetic_strategy_breeder import run_strategy_breeding_cycle

logger = logging.getLogger("BackgroundEvolution")


class BackgroundEvolutionDaemon:
    _instance: Optional["BackgroundEvolutionDaemon"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, interval_seconds: int = 1800):
        if getattr(self, "_initialized", False):
            return
        self.interval_seconds = interval_seconds
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.total_tournaments_run = 0
        self.total_elites_bred = 0
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file = os.path.join(base_dir, "data", "evolution_tournaments.jsonl")
        self._initialized = True

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="BackgroundEvolutionWorker")
        self._thread.start()
        logger.info(f"24/7 Parallele Hintergrund-Zucht gestartet (Intervall: {self.interval_seconds}s)")

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("24/7 Hintergrund-Zucht beendet.")

    def run_single_tournament(self, pop_size: int = 4) -> List[Dict[str, Any]]:
        """Führt ein einzelnes Zucht- und Selektions-Turnier aus."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
        try:
            elites = run_strategy_breeding_cycle(num_generations=1, pop_per_gen=pop_size)
            self.total_tournaments_run += 1
            n_elites = len(elites)
            self.total_elites_bred += n_elites

            record = {
                "timestamp": now_iso,
                "tournament_id": self.total_tournaments_run,
                "candidates_evaluated": pop_size,
                "elites_certified": n_elites,
                "elite_names": [e.get("class_name") for e in elites] if elites else [],
            }
            try:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass

            if n_elites > 0:
                try:
                    from config import AgentConfig
                    from telegram_command_listener import TelegramCommandListener
                    cfg = AgentConfig()
                    if cfg.telegram_bot_token and cfg.telegram_chat_id:
                        tg = TelegramCommandListener(cfg.telegram_bot_token, cfg.telegram_chat_id)
                        e_names = ", ".join(record["elite_names"])
                        tg.send_message(
                            f"🧬 <b>Autonome Evolution & Zucht:</b>\n\n"
                            f"✨ <b>{n_elites} neue Elite-Strategie(n) gezüchtet:</b>\n"
                            f"<code>{e_names}</code>\n\n"
                            f"<i>Vollautomatisch als neue Spezial-Unteragenten in den Schwarm integriert!</i>"
                        )
                except Exception:
                    pass

            return elites
        except Exception as e:
            logger.debug(f"Fehler im Evolutions-Turnier: {e}")
            return []

    def _run_loop(self) -> None:
        # Erste Zucht nach 60 Sekunden Vorlauf
        if not self._stop_event.wait(60.0):
            self.run_single_tournament(pop_size=4)

        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval_seconds):
                break
            self.run_single_tournament(pop_size=4)


_daemon_instance: Optional[BackgroundEvolutionDaemon] = None


def get_background_evolution_daemon(interval_seconds: int = 1800) -> BackgroundEvolutionDaemon:
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = BackgroundEvolutionDaemon(interval_seconds=interval_seconds)
    return _daemon_instance
