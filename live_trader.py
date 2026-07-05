"""
live_trader.py — Live-Trading-Bot für Echtausführung.

Führt kontinuierlich aus:
  1. Marktdaten abrufen (Alpaca real-time bars)
  2. Signale generieren (strategy.generate_signals)
  3. Orders ausführen (execution.py + broker_alpaca.py)
  4. Governance-Journal führen
  5. Reconciliation-Drift prüfen

Aufruf:
    python live_trader.py --config config.yaml --paper
    
    oder ohne Config-File (env vars): 
    ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python live_trader.py --symbol SPY --paper
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional
import json

import pandas as pd
import numpy as np

from strategy import generate_signals
from backtest import run_backtest, D
from execution import MockExchange, ReconciliationLoop, should_retry, RejectionReason, Order, OrderStatus
from governance import TradeJournal, IncidentLog, ThesisRequiredError

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("live_trader.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class LiveTradingConfig:
    """Konfiguration für den Live-Trading-Bot."""
    
    def __init__(
        self,
        symbol: str = "SPY",
        paper: bool = True,
        lookback_bars: int = 250,  # ~1 Jahr an Tagesbars
        poll_interval_seconds: int = 60,  # Alle 60 Sekunden prüfen
        risk_pct: float = 0.01,
        max_hold_days: int = 250,
        cost_bps: float = 5.0,
        slippage_bps: float = 5.0,
        max_position_pct: float = 0.05,  # Max 5% der Capital pro Position
        data_freshness_threshold_sec: int = 180,  # Warnung wenn Daten älter als 3 Min
        journal_path: str = "/tmp/live_journal.json",
        incident_log_path: str = "/tmp/incidents.json",
        state_path: str = "/tmp/trader_state.json",
    ):
        self.symbol = symbol
        self.paper = paper
        self.lookback_bars = lookback_bars
        self.poll_interval_seconds = poll_interval_seconds
        self.risk_pct = risk_pct
        self.max_hold_days = max_hold_days
        self.cost_bps = cost_bps
        self.slippage_bps = slippage_bps
        self.max_position_pct = max_position_pct
        self.data_freshness_threshold_sec = data_freshness_threshold_sec
        self.journal_path = journal_path
        self.incident_log_path = incident_log_path
        self.state_path = state_path


class TradingMetrics:
    """Tracking von Trade-Metriken."""
    
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = D(0)
        self.max_drawdown = D(0)
        self.peak_capital = D(100_000)
    
    def record_trade(self, pnl: Decimal):
        """Trade-Ergebnis aufzeichnen."""
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        self.total_pnl += pnl
    
    def update_drawdown(self, current_capital: Decimal):
        """Drawdown tracking."""
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        drawdown = self.peak_capital - current_capital
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_summary(self) -> dict:
        """Metriken-Zusammenfassung."""
        win_rate = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0.0
        )
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate_pct": round(win_rate, 2),
            "total_pnl": str(self.total_pnl),
            "max_drawdown": str(self.max_drawdown),
        }
    
    def log_summary(self):
        """Metriken in Logs ausgeben."""
        summary = self.get_summary()
        logger.info(
            f"Trading Metrics: Trades={summary['total_trades']}, "
            f"WinRate={summary['win_rate_pct']}%, "
            f"PnL={summary['total_pnl']}, "
            f"MaxDD={summary['max_drawdown']}"
        )


class LiveTrader:
    """Live-Trading-Bot — Echtausführung mit Governance."""
    
    def __init__(self, config: LiveTradingConfig, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock
        
        # Exchange
        if use_mock:
            self.exchange = MockExchange(seed=42)
            logger.info("MockExchange initialisiert (Demo-Modus)")
        else:
            try:
                from broker_alpaca import AlpacaExchange
                self.exchange = AlpacaExchange(paper=config.paper)
                mode = "Paper" if config.paper else "LIVE"
                logger.info(f"AlpacaExchange initialisiert ({mode})")
            except (ImportError, RuntimeError) as e:
                logger.error(f"AlpacaExchange konnte nicht geladen werden: {e}")
                logger.warning("Fallback auf MockExchange")
                self.exchange = MockExchange(seed=42)
                self.use_mock = True
        
        # Reconciliation
        self.recon = ReconciliationLoop(self.exchange)
        
        # Governance
        self.journal = TradeJournal(path=config.journal_path)
        self.incident_log = IncidentLog(path=config.incident_log_path)
        
        # Trading-State
        self.in_position = False
        self.entry_order: Optional[Order] = None
        self.bars_held = 0
        self.entry_date = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.size = None
        
        # Capital management
        self.capital = D(100_000)
        
        # Historische Daten
        self.price_history: list[dict] = []
        self.last_bar_time: Optional[datetime] = None
        
        # Metriken
        self.metrics = TradingMetrics()
        
        # Thread safety
        self._trade_lock = threading.Lock()
        
        # State persistence
        self.load_state()
        
        logger.info(
            f"LiveTrader initialisiert: {config.symbol}, "
            f"Risk={config.risk_pct*100}%, MaxPos={config.max_position_pct*100}%, "
            f"Capital={self.capital}"
        )
    
    def load_state(self):
        """Persistenten State laden (falls vorhanden)."""
        if not os.path.exists(self.config.state_path):
            logger.info("Kein persistenter State gefunden, starte mit Defaults")
            return
        
        try:
            with open(self.config.state_path, "r") as f:
                state = json.load(f)
            
            self.capital = D(state.get("capital", "100000"))
            self.in_position = state.get("in_position", False)
            self.entry_price = D(state["entry_price"]) if state.get("entry_price") else None
            self.stop_price = D(state["stop_price"]) if state.get("stop_price") else None
            self.target_price = D(state["target_price"]) if state.get("target_price") else None
            self.size = D(state["size"]) if state.get("size") else None
            self.bars_held = state.get("bars_held", 0)
            
            if state.get("entry_date"):
                self.entry_date = datetime.fromisoformat(state["entry_date"])
            
            logger.info(
                f"State geladen: capital={self.capital}, "
                f"in_position={self.in_position}, bars_held={self.bars_held}"
            )
        except Exception as e:
            logger.exception(f"Fehler beim Laden des States: {e}")
    
    def save_state(self):
        """Persistenten State speichern."""
        try:
            state = {
                "capital": str(self.capital),
                "in_position": self.in_position,
                "entry_price": str(self.entry_price) if self.entry_price else None,
                "stop_price": str(self.stop_price) if self.stop_price else None,
                "target_price": str(self.target_price) if self.target_price else None,
                "size": str(self.size) if self.size else None,
                "bars_held": self.bars_held,
                "entry_date": self.entry_date.isoformat() if self.entry_date else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            with open(self.config.state_path, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"State gespeichert: {self.config.state_path}")
        except Exception as e:
            logger.exception(f"Fehler beim Speichern des States: {e}")
    
    def load_historical_data(self) -> pd.DataFrame:
        """
        Historische OHLCV-Daten laden.
        In Produktion: von Alpaca abrufen via get_bars().
        Für Demo: CSV laden oder synthetische Daten generieren.
        """
        try:
            from broker_alpaca import AlpacaExchange
            if not self.use_mock:
                logger.info(f"Laden historischer Daten für {self.config.symbol} von Alpaca...")
                raise NotImplementedError("Alpaca live bar loading not yet implemented")
        except (ImportError, NotImplementedError, RuntimeError):
            pass
        
        # Fallback: CSV oder synthetische Daten
        logger.info("Lade Demo-Daten aus CSV oder generiere synthetisch...")
        try:
            from data import load_ohlcv_csv
            return load_ohlcv_csv(f"{self.config.symbol.lower()}_data.csv")
        except FileNotFoundError:
            logger.warning(f"CSV {self.config.symbol.lower()}_data.csv nicht gefunden.")
            from data import generate_synthetic_ohlcv
            df = generate_synthetic_ohlcv(
                start_date="2024-01-01",
                n_days=self.config.lookback_bars,
                seed=42
            )
            logger.info(f"Nutze synthetische Daten: {len(df)} Bars")
            return df[-self.config.lookback_bars:].copy()
    
    def get_latest_bar(self) -> Optional[dict]:
        """Neuesten Bar abrufen."""
        if not self.price_history:
            logger.error("Kein price_history verfügbar")
            return None
        
        return self.price_history[-1]
    
    def check_data_freshness(self) -> bool:
        """Prüfe, ob Daten aktuell sind. Rückgabe: True wenn frisch."""
        if self.last_bar_time is None:
            return True
        
        age = (datetime.now(timezone.utc) - self.last_bar_time).total_seconds()
        if age > self.config.data_freshness_threshold_sec:
            logger.warning(
                f"Daten veraltet: {age:.0f}s alt "
                f"(Threshold: {self.config.data_freshness_threshold_sec}s)"
            )
            return False
        
        return True
    
    def update_historical_data(self, df: pd.DataFrame):
        """
        Konvertiere DataFrame zu Bar-Liste.
        Wird nur inkrementell aktualisiert (nicht vollständig neu generiert).
        """
        new_bars = []
        existing_dates = {bar["date"] for bar in self.price_history}
        
        for date, row in df.iterrows():
            if date in existing_dates:
                continue
            
            bar = {
                "date": date,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
                "regime": row.get("regime", ""),
                "atr": float(row.get("atr", 0)) if "atr" in row else None,
                "signal": int(row.get("signal", 0)) if "signal" in row else 0,
                "stop_price": float(row.get("stop_price", 0)) if "stop_price" in row else None,
                "target_price": float(row.get("target_price", 0)) if "target_price" in row else None,
                "exit_signal": int(row.get("exit_signal", 0)) if "exit_signal" in row else 0,
            }
            new_bars.append(bar)
        
        self.price_history.extend(new_bars)
        
        # Halte nur die letzten lookback_bars
        if len(self.price_history) > self.config.lookback_bars:
            self.price_history = self.price_history[-self.config.lookback_bars:]
        
        if new_bars:
            self.last_bar_time = datetime.now(timezone.utc)
            logger.debug(f"Updated {len(new_bars)} neue Bars, total: {len(self.price_history)}")
    
    def generate_signals_on_latest_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signale nur auf den letzten Reihen regenerieren (performance)."""
        # Nur die letzten max(lookback_bars, 50) Zeilen verwenden für Signale
        lookback = max(self.config.lookback_bars, 50)
        df_recent = df.tail(lookback).copy()
        
        signals = generate_signals(
            df_recent,
            trend_sma_period=150,
            momentum_lookback=126,
            atr_period=14,
            stop_atr_mult=3.0,
        )
        return signals
    
    def check_entry_signal(self, latest_bar: dict) -> bool:
        """Prüfe, ob aktueller Bar ein Einstiegssignal hat."""
        if self.in_position:
            return False
        
        if latest_bar.get("signal", 0) != 1:
            return False
        
        if pd.isna(latest_bar.get("atr")):
            logger.warning("ATR ist NaN — Einstieg übersprungen")
            return False
        
        return True
    
    def check_exit_conditions(self, latest_bar: dict) -> Optional[str]:
        """Prüfe Ausstiegsbedingungen. Rückgabe: exit_reason oder None."""
        if not self.in_position:
            return None
        
        self.bars_held += 1
        
        # Stop getroffen
        if latest_bar["low"] <= float(self.stop_price):
            return "stop"
        
        # Target getroffen
        if latest_bar["high"] >= float(self.target_price):
            return "target"
        
        # Trend-Exit
        if latest_bar.get("exit_signal", 0) == 1:
            return "trend_reversal"
        
        # Zeit-Stop
        if self.bars_held >= self.config.max_hold_days:
            return "time_stop"
        
        return None
    
    def execute_entry(self, latest_bar: dict):
        """Einstiegs-Order ausführen (with thread safety)."""
        with self._trade_lock:
            if self.in_position:
                logger.warning("Einstieg übersprungen: bereits in Position")
                return
            
            try:
                risk_amount = self.capital * D(self.config.risk_pct)
                stop_distance = D(latest_bar["close"]) - D(latest_bar["stop_price"])
                
                if stop_distance <= 0:
                    logger.warning(f"Stop-Distanz ungültig: {stop_distance}")
                    return
                
                # Berechne Position-Größe
                raw_size = (risk_amount / stop_distance).quantize(D("0.0001"))
                
                # Wende Position-Limit an
                max_position_capital = self.capital * D(self.config.max_position_pct)
                position_value = raw_size * D(latest_bar["close"])
                
                if position_value > max_position_capital:
                    capped_size = (max_position_capital / D(latest_bar["close"])).quantize(D("0.0001"))
                    logger.info(
                        f"Position gecappt: {raw_size} → {capped_size} "
                        f"(Limit: {self.config.max_position_pct*100}% = {max_position_capital})"
                    )
                    self.size = capped_size
                else:
                    self.size = raw_size
                
                # Order abschicken
                self.entry_order = self.exchange.submit_order(
                    symbol=self.config.symbol,
                    side="buy",
                    quantity=self.size
                )
                
                self.recon.track(self.entry_order)
                
                if self.entry_order.status == OrderStatus.REJECTED:
                    reason = self.entry_order.reject_reason
                    logger.error(f"Einstiegs-Order abgelehnt: {reason.value if reason else '?'}")
                    
                    # Retry-Entscheidung
                    if should_retry(reason):
                        logger.info("Retry nach kurzer Wartezeit...")
                        time.sleep(2)
                        self.entry_order = self.exchange.submit_order(
                            symbol=self.config.symbol,
                            side="buy",
                            quantity=self.size
                        )
                        self.recon.track(self.entry_order)
                    else:
                        logger.info(f"Keine Wiederholung für {reason.value}")
                        self.incident_log.record(
                            title="Entry-Order rejected (no retry)",
                            description=f"Reason: {reason.value if reason else 'unknown'}",
                            severity="warning",
                            fix="Check market conditions and order parameters"
                        )
                        return
                
                if self.entry_order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    self.in_position = True
                    self.entry_date = latest_bar["date"]
                    self.entry_price = D(latest_bar["close"])
                    self.stop_price = D(latest_bar["stop_price"])
                    self.target_price = D(latest_bar["target_price"])
                    self.bars_held = 0
                    
                    logger.info(
                        f"✓ EINSTIEG: {self.config.symbol} @ {self.entry_price}, "
                        f"Qty={self.size}, Stop={self.stop_price}, Target={self.target_price}"
                    )
                    
                    # Journal-Eintrag
                    try:
                        self.journal.log(
                            symbol=self.config.symbol,
                            action="taken",
                            thesis="Kurs über 150-Tage-SMA UND positive 126-Tage-Rendite",
                            signal_score=1.0,
                            regime=latest_bar.get("regime", "unknown")
                        )
                    except ThesisRequiredError as e:
                        logger.error(f"Journal-Fehler: {e}")
                    
                    self.save_state()
                else:
                    logger.warning(f"Order nicht gefüllt: {self.entry_order.status.value}")
            
            except Exception as e:
                logger.exception(f"Fehler beim Einstieg: {e}")
                self.in_position = False
                self.incident_log.record(
                    title="Entry execution error",
                    description=str(e),
                    severity="error",
                    fix="Review logs and market conditions"
                )
    
    def execute_exit(self, latest_bar: dict, exit_reason: str) -> Decimal:
        """Ausstiegs-Order ausführen (with thread safety). Rückgabe: neue capital."""
        with self._trade_lock:
            if not self.in_position:
                logger.warning("Exit übersprungen: nicht in Position")
                return self.capital
            
            try:
                exit_price = D(latest_bar["close"])
                if exit_reason == "stop":
                    exit_price = self.stop_price
                elif exit_reason == "target":
                    exit_price = self.target_price
                
                # Slippage/Kosten anwenden
                friction = D(self.config.cost_bps + self.config.slippage_bps) / D(10_000)
                fill_price = exit_price * (D(1) - friction) if exit_reason != "stop" else exit_price
                
                pnl = (fill_price - self.entry_price) * self.size
                risk_amount = (self.entry_price - self.stop_price) * self.size
                r_multiple = float(pnl / risk_amount) if risk_amount != 0 else 0.0
                new_capital = self.capital + pnl
                
                logger.info(
                    f"✗ AUSSTIEG ({exit_reason}): @ {exit_price}, "
                    f"PnL={pnl} ({r_multiple:.2f}R), Capital={new_capital}"
                )
                
                # Metriken
                self.metrics.record_trade(pnl)
                self.metrics.update_drawdown(new_capital)
                
                # Journal-Update
                self.journal.update_outcome(
                    symbol=self.config.symbol,
                    timestamp=self.entry_date.isoformat(),
                    outcome=f"{exit_reason}: {r_multiple:.2f}R"
                )
                
                self.capital = new_capital
                self.in_position = False
                self.entry_order = None
                
                self.save_state()
                
                return new_capital
            
            except Exception as e:
                logger.exception(f"Fehler beim Ausstieg: {e}")
                self.incident_log.record(
                    title="Exit execution error",
                    description=str(e),
                    severity="error",
                    fix="Review logs and manually close position"
                )
                return self.capital
    
    def reconcile_state(self):
        """Reconciliation-Drift prüfen."""
        drift = self.recon.reconcile()
        if drift:
            logger.warning("Reconciliation-Drift gefunden:")
            for msg in drift:
                logger.warning(f"  - {msg}")
                self.incident_log.record(
                    title="Reconciliation drift detected",
                    description=msg,
                    severity="warning",
                    fix="Review and sync local state with exchange"
                )
    
    def run_loop(self, max_iterations: int = None):
        """Hauptschleife des Trading-Bots."""
        logger.info(
            f"Starte Trading-Loop (Interval: {self.config.poll_interval_seconds}s, "
            f"DataFreshness: {self.config.data_freshness_threshold_sec}s)"
        )
        
        # Historische Daten laden
        df_hist = self.load_historical_data()
        self.update_historical_data(df_hist)
        
        iteration = 0
        last_metrics_log = time.time()
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                logger.debug(f"--- Iteration {iteration} ---")
                
                # Signale auf den neuesten Daten generieren
                signals_df = self.generate_signals_on_latest_data(df_hist)
                self.update_historical_data(signals_df)
                
                # Prüfe Daten-Frische
                if not self.check_data_freshness():
                    logger.warning("Überspringe Iteration wegen veralteter Daten")
                    time.sleep(self.config.poll_interval_seconds)
                    continue
                
                latest_bar = self.get_latest_bar()
                if latest_bar is None:
                    logger.error("Kein aktueller Bar verfügbar")
                    time.sleep(self.config.poll_interval_seconds)
                    continue
                
                logger.debug(f"Bar: {latest_bar['date']} Close={latest_bar['close']:.2f}")
                
                # Reconciliation
                self.reconcile_state()
                
                if not self.in_position:
                    # Einstieg-Check
                    if self.check_entry_signal(latest_bar):
                        logger.info("Einstiegs-Signal gefunden!")
                        self.execute_entry(latest_bar)
                
                else:
                    # Ausstieg-Check
                    exit_reason = self.check_exit_conditions(latest_bar)
                    if exit_reason:
                        logger.info(f"Ausstiegs-Signal: {exit_reason}")
                        self.execute_exit(latest_bar, exit_reason)
                
                logger.info(f"Status: in_position={self.in_position}, capital={self.capital}")
                
                # Periodisch Metriken ausgeben
                now = time.time()
                if now - last_metrics_log > 3600:  # Jede Stunde
                    self.metrics.log_summary()
                    last_metrics_log = now
                
                time.sleep(self.config.poll_interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Beende Bot auf Tastendruck...")
        except Exception as e:
            logger.exception(f"Kritischer Fehler in der Hauptschleife: {e}")
            self.incident_log.record(
                title="Critical error in main loop",
                description=str(e),
                severity="critical",
                fix="Restart bot and review logs"
            )
        
        finally:
            logger.info(f"Trading-Loop beendet nach {iteration} Iterationen")
            self.metrics.log_summary()
            self.save_state()


def main():
    parser = argparse.ArgumentParser(description="Live-Trading-Bot")
    parser.add_argument("--symbol", default="SPY", help="Trading-Symbol (default: SPY)")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper-Trading (default)")
    parser.add_argument("--live", action="store_true", help="Live-Trading (WARNUNG: echtes Kapital!)")
    parser.add_argument("--mock", action="store_true", help="MockExchange (Demo-Modus)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling-Intervall (Sekunden)")
    parser.add_argument("--lookback", type=int, default=250, help="Historische Bars (Tage)")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="Risiko pro Trade (%)")
    parser.add_argument("--max-pos-pct", type=float, default=0.05, help="Max Position % (default: 5%)")
    parser.add_argument("--iterations", type=int, help="Max. Iterationen (default: unbegrenzt)")
    
    args = parser.parse_args()
    
    if args.live and not args.mock:
        confirm = input(
            "⚠️  WARNUNG: Live-Trading mit echtem Kapital! "
            "Fortfahren? (ja/nein): "
        )
        if confirm.lower() != "ja":
            logger.info("Abgebrochen.")
            return
    
    config = LiveTradingConfig(
        symbol=args.symbol,
        paper=not args.live,
        lookback_bars=args.lookback,
        poll_interval_seconds=args.poll_interval,
        risk_pct=args.risk_pct,
        max_position_pct=args.max_pos_pct,
    )
    
    trader = LiveTrader(config, use_mock=args.mock)
    trader.run_loop(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
