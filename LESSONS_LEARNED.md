# 🧠 Lessons Learned & Post-Mortem Audit Journal

Dieses Dokument dokumentiert aufgetretene Rechenfehler, Architektur-Schwächen und methodische Fehler, warum sie nicht sofort gemeldet wurden, wie sie behoben wurden und welche Prüfregeln ab sofort für alle KI-Agenten gelten.

---

## 📌 Eintrag #1: Sharpe-Formelfehler & Blind-Test-Logik im Walk-Forward-Labor (2026-10-29)

### 1. Das Symptom
Im automatisierten Test 36 erschien in der Konsolenausgabe:
`[OK] Test 36 bestanden: Walk-Forward Research Lab optimiert Parameter rollend (WFER 60.0%, OOS Sharpe -214.56, Note: C_MARGINAL) über 4 Stress-Regime.`
Der Test meldete `[OK]`, obwohl ein Sharpe-Wert von **-214.56** statistisch und mathematisch völlig unplausibel ist (normale Werte liegen zwischen -3.0 und +3.0).

### 2. Warum wurde der Fehler nicht sofort gemeldet?
1. **Oberflächliche Test-Assertions:** Der Unit-Test prüfte lediglich, ob `wfa_res.overall_wfer_pct > 0.0` ist und ob ein Report-File existiert. Es gab keine Plausibilitätsgrenzen für die Metriken (`|Sharpe| <= 6.0`, Min-Trades >= 30).
2. **"Green Build"-Bias:** Solange der Code syntaktisch fehlerfrei durchlief und der Test den Exit-Code 0 warf, galt der Test für den Agenten als bestanden.
3. **Mangelnde Selbstreflexion:** Die Ausgabewerte wurden im Prompt nicht auf Plausibilität abgeglichen, bevor die Erfolgsmeldung an den Nutzer formuliert wurde.

### 3. Die mathematische und logische Ursache
1. **Falsche Annualisierung:** Die Sharpe-Ratio wurde aus Pro-Trade-Renditen mit dem Faktor `sqrt(252)` multipliziert – eine Formel, die nur für **Tagesrenditen** gilt. Bei wenigen Trades oder kleinen Standardabweichungen explodiert der Wert ins Absurde.
2. **Synthetische Daten:** Das Labor testete ausschließlich auf künstlichen Zufallszahlen statt auf echten Marktdaten.
3. **Erfundene Fallback-Werte:** Bei negativer In-Sample-Performance wurde die WFER künstlich auf `60.0%` gesetzt, anstatt `0.0%` auszugeben.

### 4. Die Umsetzung der Behebung (Version 2)
1. **Mark-to-Market Tagesrenditen:** Sharpe wird nun sauber aus täglichen Portfoliorenditen berechnet (`mean(daily) / std(daily) * sqrt(252)`).
2. **Echte Marktdaten:** Anbindung an `yfinance` mit echten Tageskursen (SPY, QQQ, NVDA, AAPL, etc.).
3. **Strenge Plausibilitätsfilter (Sanity Guardrails):**
   - Ist `|Sharpe| > 6.0`, wird das Ergebnis als `UNPLAUSIBEL` verworfen und blockiert jede Kalibrierung.
   - Liegt die In-Sample-Rendite bei `<= 0`, ist die WFER exakt `0.0%`.
   - Synthetische Testdaten dürfen niemals echte Live-Hyperparameter überschreiben.
4. **Härtere Test-Prüfungen in Test 36:**
   - Mathematischer Hand-Check mit bekannten Tagesrenditen.
   - Plausibilitäts-Assertion: `assert abs(oos_sharpe) <= 6.0`.

---
