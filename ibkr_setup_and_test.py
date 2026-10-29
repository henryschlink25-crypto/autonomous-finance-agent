import socket
from config import AgentConfig
from ibkr_bridge import IBKRBridge


def run_ibkr_connection_check() -> None:
    cfg = AgentConfig()
    bridge = IBKRBridge(cfg)
    print("================================================================================")
    print(" 🔌 INTERACTIVE BROKERS (IBKR) AUTONOMOUS API-BRIDGE DIAGNOSTICS")
    print("================================================================================")

    open_ports = []
    for port, label in IBKRBridge.CANDIDATE_PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.8)
        res = sock.connect_ex((cfg.ibkr_host, port))
        sock.close()
        status = "🟢 OFFEN (EREICHBAR)" if res == 0 else "⚪ Geschlossen"
        print(f" • Port {port} ({label:<25}): {status}")
        if res == 0:
            open_ports.append((port, label))

    print("--------------------------------------------------------------------------------")
    if not open_ports:
        print(" ⏳ STATUS: Die Python API-Brücke (ibkr_bridge.py) ist komplett installiert und bereit,")
        print("    wartet aber aktuell darauf, dass du auf deinem Windows-PC die")
        print("    IBKR Trader Workstation (TWS) oder das IB Gateway startest.")
        print("\n 👉 Sobald TWS läuft (API-Haken bei Port 7497 Paper oder 7496 Live gesetzt),")
        print("    erkennt der Agent den offenen Port im 5-Minuten-Loop vollautomatisch!")
        return

    ib = bridge.connect_session(timeout=4.0)
    if ib is not None:
        print(f" ✅ VERBUNDEN: {bridge.last_status_message}")
        accts = ib.managedAccounts()
        print(f" • Gefundene IBKR-Konten: {accts}")
        res = bridge.sync_account_and_positions({}, {})
        if res is not None:
            net_liq, cash_usd, positions, _ = res
            print(f" • NetLiquidation (Gesamt-Equity): ${net_liq:,.2f} (Basiswährung: {bridge.base_currency})")
            print(f" • Verfügbares Cash:               ${cash_usd:,.2f}")
            print(f" • Offene IBKR-Positionen:         {len(positions)} ({list(positions.keys())})")
        ib.disconnect()
    else:
        print(f" ⚠️ Port ist offen, aber API-Handshake wurde noch nicht bestätigt: {bridge.last_status_message}")
        print("    Prüfe in TWS unter: Datei -> Globale Konfiguration -> API -> Einstellungen ->")
        print("    'ActiveX und Socket-Clients aktivieren' anhaken & '127.0.0.1' als Trusted IP eintragen.")


if __name__ == "__main__":
    run_ibkr_connection_check()
