import os
import shutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from config import AgentConfig

console = Console(width=120)


def run_diagnostics() -> None:
    # Erstelle .env automatisch aus .env.example, falls noch nicht vorhanden
    base_dir = os.path.dirname(__file__)
    env_path = os.path.join(base_dir, ".env")
    env_example = os.path.join(base_dir, ".env.example")
    if not os.path.exists(env_path) and os.path.exists(env_example):
        shutil.copyfile(env_example, env_path)
        console.print(f"[green]✔ Datei .env wurde aus .env.example angelegt: {env_path}[/green]")

    config = AgentConfig()
    table = Table(title="🔌 API- & Broker-Diagnose (Multi-Asset Agent)", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Komponente", style="bold")
    table.add_column("Status")
    table.add_column("Details & Empfehlung")

    # 1. Gemini API Check
    if config.gemini_api_key and config.gemini_api_key != "your_gemini_api_key_here":
        table.add_row(
            "Google Gemini KI-Analyst",
            "[bold green]● Konfiguriert[/bold green]",
            f"Nutzt Modell '{config.gemini_model}' für tiefes News- & Katalysator-Reasoning.",
        )
    else:
        table.add_row(
            "Google Gemini KI-Analyst",
            "[yellow]○ Quant-Fallback aktiv[/yellow]",
            "Trage GEMINI_API_KEY in .env ein, um zusätzlich LLM-Katalysator-Analysen zu aktivieren.",
        )

    # 2. Execution Mode Check
    table.add_row(
        "Ausführungs-Modus",
        f"[bold cyan]{config.execution_mode}[/bold cyan]",
        "LOCAL_PAPER speichert lokal in data/portfolio_state.json. Änderbar in .env auf ALPACA_PAPER oder CCXT_SANDBOX.",
    )

    # 3. Alpaca API Check
    if config.alpaca_api_key and config.alpaca_secret_key:
        table.add_row(
            "Alpaca Markets (Aktien & Krypto)",
            "[bold green]● API-Keys vorhanden[/bold green]",
            f"Paper-Modus: {config.alpaca_paper}. Bereit für Order-Routing.",
        )
    else:
        table.add_row(
            "Alpaca Markets (Aktien & Krypto)",
            "[dim]○ Nicht verbunden[/dim]",
            "Optional: Kostenlosen Paper-Trading Key auf https://alpaca.markets erstellen und in .env eintragen.",
        )

    # 4. CCXT Exchange Check
    if config.ccxt_api_key and config.ccxt_secret_key:
        table.add_row(
            f"CCXT Krypto-Börse ({config.ccxt_exchange_id})",
            "[bold green]● API-Keys vorhanden[/bold green]",
            f"Sandbox-Modus: {config.ccxt_sandbox}. Bereit für Krypto-Order-Routing.",
        )
    else:
        table.add_row(
            f"CCXT Krypto-Börse ({config.ccxt_exchange_id})",
            "[dim]○ Nicht verbunden[/dim]",
            "Optional: Testnet/API-Keys deiner Börse (Binance, Coinbase, Kraken) in .env eintragen.",
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold]Deine Konfigurationsdatei liegt hier:[/bold] {env_path}\n"
            f"Öffne sie im Editor, um jederzeit deine API-Keys oder Risikogrenzen (aktuell max. {config.max_risk_per_trade_pct*100:.1f}% Risiko pro Trade) anzupassen.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    run_diagnostics()
