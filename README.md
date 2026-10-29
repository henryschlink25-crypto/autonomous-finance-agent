# 🤖 Autonomous Multi-Asset Finance & Execution Agent

Ein modularer, autonomer KI-Finanzagent für **Marktanalyse, News-Research und regelbasierte Order-Ausführung** über **Aktien/ETFs** und **Kryptowährungen**.

## Architektur-Prinzip: "AI for Research, Math for Risk"
Der größte Fehler bei autonomen Trading-Agenten ist es, einem Sprachmodell (LLM) direkten, ungefilterten Zugriff auf die Broker-API zu geben. Dieses System trennt deshalb strikt in **4 Schichten**:

1. **Multi-Asset Data & News Collector (`data_collector.py`)**
   - Lädt Kursdaten & berechnet institutionelle Indikatoren: `RSI(14)`, `SMA(20/50)`, `MACD-Histogramm`, `ATR(14)` (Average True Range für Volatilitäts-Stopps), annualisierte Volatilität und Volumen-Anomalien.
   - Zieht aktuelle Finanz-Schlagzeilen (RSS) für jedes Asset und berechnet einen ersten Lexikon-Sentiment-Score.
2. **Hybrid AI & Quant Analyst (`ai_analyst.py`)**
   - Kombiniert ein deterministisches Multi-Faktor-Modell (Trend + Momentum + Volumen + News-Sentiment) mit **Google Gemini** (`google-genai` Structured JSON Output), sofern ein API-Key in `.env` hinterlegt ist.
   - Liefert strukturierte `TradingSignal`-Objekte inkl. Konfidenz, Einstiegspreis, ATR-basiertem Stop-Loss, Take-Profit und nachvollziehbarer Research-These.
3. **Deterministic Risk Manager (`risk_manager.py`)**
   - **Harte mathematische Guardrails**, die keine KI überstimmen kann:
     - **Fixed-Fractional Position Sizing**: Berechnet die exakte Stückzahl anhand des Abstands zum Stop-Loss, sodass pro Trade maximal `MAX_RISK_PER_TRADE_PCT` (Standard: `1.5%`) des Gesamtkapitals riskiert werden.
     - **Klumpenrisiko-Schutz**: Maximal `MAX_POSITION_WEIGHT_PCT` (Standard: `20%`) pro Einzelposition.
     - **Daily Drawdown Circuit Breaker**: Stoppt neue Käufe automatisch, falls das Portfolio an einem Tag mehr als `MAX_DAILY_DRAWDOWN_PCT` (Standard: `4%`) verliert.
     - **Mindest-Konfidenz & Chance-Risiko-Verhältnis (R:R)**: Filtert schwache Signale konsequent heraus.
4. **Multi-Asset Execution Engine (`execution_engine.py`)**
   - **`LOCAL_PAPER` (Standard)**: Sofort lauffähige lokale Paper-Trading-Simulation inkl. 5 bps Gebühren/Slippage, automatischer Stop-Loss/Take-Profit-Auslösung, persistentem Portfolio-State (`data/portfolio_state.json`) und lückenlosem Audit-Log (`data/audit_journal.jsonl`).
   - **`ALPACA_PAPER`**: Anbindung an die Alpaca Markets API für Aktien, ETFs & Krypto.
   - **`CCXT_SANDBOX`**: Anbindung an Krypto-Börsen (Binance, Coinbase, Kraken etc.) über CCXT.

## Schnellstart

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. Optional: .env.example nach .env kopieren und API-Keys eintragen
cp .env.example .env

# 3. Einen vollständigen autonomen Analyse- & Trading-Zyklus ausführen
python main.py --once

# 4. Dauerhafter autonomer Modus (z.B. alle 15 Minuten = 900 Sekunden)
python main.py --loop 900

# 5. Nur aktuellen Portfolio-Status anzeigen
python main.py --status
```
