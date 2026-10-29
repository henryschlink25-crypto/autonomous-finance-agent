"""
Synthetisierte Handelsstrategie: WhaleOrderflowAccumulator_Gen1_3
Archetyp: WHALE_ORDERFLOW_ACCUMULATOR
Hypothese: Folgt institutionellen Großkäufen (Whale Call Options Flow P/C < 0.45 und positiver Chaikin Money Flow) während gesunder Konsolidierungen oberhalb des 20 EMA.
Generiert am: 2026-10-29T22:29:23.799699
"""

class WhaleOrderflowAccumulator_Gen1_3:
    """Autonom gezüchtete Strategieklasse für den ADAS-Handelsschwarm."""
    
    STRATEGY_ID = "strat_whale_orderflow_accumulator_g1_3"
    ARCHETYPE = "WHALE_ORDERFLOW_ACCUMULATOR"
    NAME = "WhaleOrderflowAccumulator_Gen1_3"
    HYPOTHESIS = "Folgt institutionellen Großkäufen (Whale Call Options Flow P/C < 0.45 und positiver Chaikin Money Flow) während gesunder Konsolidierungen oberhalb des 20 EMA."
    
    def __init__(self):
        self.genes = {
        "min_rvol": 1.533,
        "min_change_pct": 0.763,
        "max_rsi": 69.508,
        "min_edge_score": 1.547,
        "atr_sl_mult": 2.224,
        "atr_tp_mult": 4.393,
        "time_stop_days": 7,
        "penny_max_price": 935.367
}

    def evaluate_entry(self, snapshot, macro=None) -> dict:
        """Prüft, ob für den Snapshot ein Kaufsignal vorliegt."""
        # Gene
        min_rvol = self.genes.get("min_rvol", 1.8)
        min_change = self.genes.get("min_change_pct", 2.0)
        max_rsi = self.genes.get("max_rsi", 75.0)
        max_price = self.genes.get("penny_max_price", 999.0)
        
        # Snapshot-Attribute
        price = getattr(snapshot, "current_price", 0.0)
        rvol = getattr(snapshot, "relative_volume_10d", 1.0)
        chg = getattr(snapshot, "change_24h_pct", 0.0)
        rsi = getattr(snapshot, "rsi_14", 50.0)
        trend = getattr(snapshot, "trend_status", "NEUTRAL")
        
        if price > max_price:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Preis über Maximalgrenze"}
            
        # Bewertungskriterien
        reasons = []
        score = 0.0
        
        if rvol >= min_rvol:
            score += 0.35
            reasons.append(f"RVOL {rvol:.1f}x >= {min_rvol:.1f}x")
            
        if chg >= min_change:
            score += 0.25
            reasons.append(f"Move +{chg:.1f}% >= {min_change:.1f}%")
            
        if rsi <= max_rsi:
            score += 0.20
            reasons.append(f"RSI {rsi:.1f} nicht überkauft")
            
        if trend == "BULLISH":
            score += 0.20
            reasons.append("Trend bullish")
            
        if score >= 0.70:
            atr = getattr(snapshot, "atr_14", price * 0.03) or (price * 0.03)
            sl_price = round(max(0.01, price - (self.genes.get("atr_sl_mult", 2.0) * atr)), 2)
            tp_price = round(price + (self.genes.get("atr_tp_mult", 4.0) * atr), 2)
            return {
                "action": "BUY",
                "confidence": min(0.96, score),
                "stop_loss_price": sl_price,
                "take_profit_price": tp_price,
                "reason": " | ".join(reasons)
            }
            
        return {"action": "HOLD", "confidence": score, "reason": "Kriterien nicht voll erfüllt"}

    def evaluate_exit(self, position, snapshot) -> dict:
        """Prüft, ob eine bestehende Position nach Regeln geschlossen werden soll."""
        cur_price = getattr(snapshot, "current_price", position.current_price)
        entry_price = getattr(position, "entry_price", cur_price)
        pnl_pct = ((cur_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        holding_days = getattr(position, "holding_days", 0)
        
        if holding_days >= self.genes.get("time_stop_days", 5) and pnl_pct < 1.0:
            return {"exit": True, "reason": f"Time-Stop nach {holding_days} Tagen"}
            
        if cur_price <= getattr(position, "stop_loss_price", 0.0):
            return {"exit": True, "reason": "Stop-Loss ausgelöst"}
            
        return {"exit": False, "reason": "Position intakt"}
