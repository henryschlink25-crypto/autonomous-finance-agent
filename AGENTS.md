# Autonomous Trading Agent - System Rules & Agent Architecture

## Project Scope
Autonomous multi-asset quantitative trading agent operating with Alpaca Paper API and Interactive Brokers (IBKR Ireland - IBIE).
Follows the Sakana AI ADAS (Automated Design of Agentic Systems) architecture with evolutionary specialist sub-agents, step-by-step reasoning self-reflection, and Darwinian pruning.

## Mandatory Risk & Safety Principles
1. **Never Trade Without Capital Protection**:
   - Every single position MUST carry an ATR-based Trailing Stop (Chandelier exit) or Stop-Loss bracket.
   - Max risk per trade: 1.5% - 2.5% of total portfolio equity.
   - Maximum daily drawdown circuit breaker: 4.0%.
2. **Deterministic Pre-Trade Filters**:
   - 15-minute Opening-Bell Cooldown (09:30 - 09:45 ET).
   - Earnings blackout (no new entries within 3 days of quarterly earnings announcements).
   - Macro VIX traffic light: RED (>30) restricts entries strictly to safe-havens (GLD, TLT).
3. **Multi-Exchange Routing**:
   - US Equities: Smart routing (`SMART` in USD).
   - German Equities (`.DE`): Xetra routing (`IBIS` in EUR).
   - Crypto: Cash/Spot or Alpaca Crypto (`USD`).
4. **Code Quality & Testing**:
   - Python 3.10 standard with type hints and Pydantic v2 data models.
   - All code changes must pass `python test_audit.py` before deployment.
   - Sensitive credentials (`.env`, private keys) must NEVER be committed to Git.
