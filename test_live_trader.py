"""
test_live_trader.py — Tests für refaktoriertes live_trader.py

Tests für:
  1. State Persistence (save/load)
  2. Position Sizing Limits
  3. Thread Safety
  4. Data Freshness Checks
  5. Trading Metrics
  6. Incremental Bar Updates
"""

import pytest
import os
import json
import tempfile
import threading
import time
from decimal import Decimal as D
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from live_trader import (
    LiveTradingConfig,
    LiveTrader,
    TradingMetrics,
)


class TestLiveTradingConfig:
    """Tests für Konfiguration."""
    
    def test_config_defaults(self):
        """Teste default-Konfiguration."""
        config = LiveTradingConfig()
        assert config.symbol == "SPY"
        assert config.paper == True
        assert config.risk_pct == 0.01
        assert config.max_position_pct == 0.05
        assert config.data_freshness_threshold_sec == 180
    
    def test_config_custom(self):
        """Teste custom-Konfiguration."""
        config = LiveTradingConfig(
            symbol="AAPL",
            risk_pct=0.02,
            max_position_pct=0.10
        )
        assert config.symbol == "AAPL"
        assert config.risk_pct == 0.02
        assert config.max_position_pct == 0.10


class TestTradingMetrics:
    """Tests für Trading-Metriken."""
    
    def test_metrics_init(self):
        """Teste Metriken-Initialisierung."""
        metrics = TradingMetrics()
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.total_pnl == D(0)
        assert metrics.max_drawdown == D(0)
    
    def test_record_winning_trade(self):
        """Teste Aufzeichnung eines Gewinners."""
        metrics = TradingMetrics()
        metrics.record_trade(D(100))
        
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.total_pnl == D(100)
    
    def test_record_losing_trade(self):
        """Teste Aufzeichnung eines Verlierers."""
        metrics = TradingMetrics()
        metrics.record_trade(D(-50))
        
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 0
        assert metrics.total_pnl == D(-50)
    
    def test_multiple_trades(self):
        """Teste mehrere Trades."""
        metrics = TradingMetrics()
        metrics.record_trade(D(100))
        metrics.record_trade(D(-50))
        metrics.record_trade(D(200))
        
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.total_pnl == D(250)
    
    def test_win_rate_calculation(self):
        """Teste Win-Rate-Berechnung."""
        metrics = TradingMetrics()
        metrics.record_trade(D(100))
        metrics.record_trade(D(100))
        metrics.record_trade(D(-50))
        
        summary = metrics.get_summary()
        assert summary["total_trades"] == 3
        assert summary["winning_trades"] == 2
        assert abs(summary["win_rate_pct"] - 66.67) < 0.1
    
    def test_drawdown_tracking(self):
        """Teste Drawdown-Tracking."""
        metrics = TradingMetrics()
        metrics.update_drawdown(D(100_000))
        assert metrics.peak_capital == D(100_000)
        assert metrics.max_drawdown == D(0)
        
        metrics.update_drawdown(D(90_000))
        assert metrics.peak_capital == D(100_000)
        assert metrics.max_drawdown == D(10_000)
        
        metrics.update_drawdown(D(95_000))
        assert metrics.peak_capital == D(100_000)
        assert metrics.max_drawdown == D(10_000)
    
    def test_summary_format(self):
        """Teste Summary-Format."""
        metrics = TradingMetrics()
        metrics.record_trade(D(100))
        summary = metrics.get_summary()
        
        assert "total_trades" in summary
        assert "winning_trades" in summary
        assert "win_rate_pct" in summary
        assert "total_pnl" in summary
        assert "max_drawdown" in summary


class TestStateManagement:
    """Tests für State-Persistierung."""
    
    @pytest.fixture
    def temp_state_file(self):
        """Temporäre State-Datei."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_save_state(self, temp_state_file):
        """Teste State-Speicherung."""
        config = LiveTradingConfig(state_path=temp_state_file)
        
        with patch('live_trader.MockExchange'):
            with patch('live_trader.ReconciliationLoop'):
                with patch('live_trader.TradeJournal'):
                    with patch('live_trader.IncidentLog'):
                        trader = LiveTrader(config, use_mock=True)
                        trader.capital = D(150_000)
                        trader.in_position = True
                        trader.entry_price = D(100.5)
                        trader.stop_price = D(99.0)
                        trader.target_price = D(105.0)
                        trader.size = D(100)
                        trader.bars_held = 5
                        
                        trader.save_state()
        
        # Verifiziere gespeicherte Datei
        assert os.path.exists(temp_state_file)
        with open(temp_state_file, 'r') as f:
            state = json.load(f)
        
        assert state["capital"] == "150000"
        assert state["in_position"] == True
        assert state["entry_price"] == "100.5"
        assert state["bars_held"] == 5
    
    def test_load_state(self, temp_state_file):
        """Teste State-Laden."""
        # Erstelle State-Datei
        state_data = {
            "capital": "150000",
            "in_position": True,
            "entry_price": "100.5",
            "stop_price": "99.0",
            "target_price": "105.0",
            "size": "100",
            "bars_held": 5,
            "entry_date": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(temp_state_file, 'w') as f:
            json.dump(state_data, f)
        
        config = LiveTradingConfig(state_path=temp_state_file)
        
        with patch('live_trader.MockExchange'):
            with patch('live_trader.ReconciliationLoop'):
                with patch('live_trader.TradeJournal'):
                    with patch('live_trader.IncidentLog'):
                        trader = LiveTrader(config, use_mock=True)
        
        # Verifiziere geladene Werte
        assert trader.capital == D(150_000)
        assert trader.in_position == True
        assert trader.entry_price == D(100.5)
        assert trader.bars_held == 5


class TestPositionSizing:
    """Tests für Position-Sizing mit Limits."""
    
    @pytest.fixture
    def trader(self):
        """Trader-Instanz für Tests."""
        config = LiveTradingConfig(
            risk_pct=0.01,
            max_position_pct=0.05,
            state_path=tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        )
        with patch('live_trader.MockExchange'):
            with patch('live_trader.ReconciliationLoop'):
                with patch('live_trader.TradeJournal'):
                    with patch('live_trader.IncidentLog'):
                        trader = LiveTrader(config, use_mock=True)
        return trader
    
    def test_position_sizing_no_cap(self, trader):
        """Teste Position-Sizing ohne Cap."""
        trader.capital = D(100_000)
        
        latest_bar = {
            "close": D(100),
            "stop_price": D(95),
            "target_price": D(110),
        }
        
        with patch.object(trader.exchange, 'submit_order') as mock_order:
            mock_order.return_value = Mock(
                status=Mock(name='FILLED'),
            )
            from execution import OrderStatus
            mock_order.return_value.status = OrderStatus.FILLED
            
            with patch.object(trader, 'save_state'):
                trader.execute_entry(latest_bar)
        
        # Bei 1% Risk und $5 Stop: Size = 100_000 * 0.01 / 5 = 200
        # Position value = 200 * 100 = $20_000 (20% > 5% limit)
        # Also sollte gecappt werden zu 5% * 100_000 / 100 = 50 shares
        
        # Verifiziere dass Order mit gecappter Size aufgerufen wurde
        assert trader.size <= D(100_000) * D(0.05) / D(100)
    
    def test_max_position_pct_respected(self, trader):
        """Teste dass max_position_pct respektiert wird."""
        trader.capital = D(100_000)
        max_pos_value = D(100_000) * D(trader.config.max_position_pct)
        
        latest_bar = {
            "close": D(100),
            "stop_price": D(99),
            "target_price": D(110),
        }
        
        with patch.object(trader.exchange, 'submit_order') as mock_order:
            from execution import OrderStatus
            mock_order.return_value = Mock(status=OrderStatus.FILLED)
            
            with patch.object(trader, 'save_state'):
                trader.execute_entry(latest_bar)
        
        # Position value sollte <= max_position_pct * capital
        position_value = trader.size * D(latest_bar["close"])
        assert position_value <= max_pos_value


class TestDataFreshness:
    """Tests für Daten-Frische-Prüfung."""
    
    @pytest.fixture
    def trader(self):
        """Trader-Instanz für Tests."""
        config = LiveTradingConfig(
            data_freshness_threshold_sec=60,
            state_path=tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        )
        with patch('live_trader.MockExchange'):
            with patch('live_trader.ReconciliationLoop'):
                with patch('live_trader.TradeJournal'):
                    with patch('live_trader.IncidentLog'):
                        trader = LiveTrader(config, use_mock=True)
        return trader
    
    def test_fresh_data(self, trader):
        """Teste dass frische Daten erkannt werden."""
        trader.last_bar_time = datetime.now(timezone.utc)
        assert trader.check_data_freshness() == True
    
    def test_stale_data(self, trader):
        """Teste dass veraltete Daten erkannt werden."""
        # Setze last_bar_time auf 2 Minuten in der Vergangenheit
        trader.last_bar_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        assert trader.check_data_freshness() == False
    
    def test_no_bar_time_set(self, trader):
        """Teste Fall wenn last_bar_time nicht gesetzt."""
        trader.last_bar_time = None
        assert trader.check_data_freshness() == True


class TestIncrementalBarUpdates:
    """Tests für inkrementelle Bar-Updates."""
    
    @pytest.fixture
    def trader(self):
        """Trader-Instanz für Tests."""
        config = LiveTradingConfig(
            lookback_bars=10,
            state_path=tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        )
        with patch('live_trader.MockExchange'):
            with patch('live_trader.ReconciliationLoop'):
                with patch('live_trader.TradeJournal'):
                    with patch('live_trader.IncidentLog'):
                        trader = LiveTrader(config, use_mock=True)
        return trader
    
    def test_incremental_update_no_duplicates(self, trader):
        """Teste dass keine doppelten Bars hinzugefügt werden."""
        import pandas as pd
        
        # Erstelle DataFrame mit 5 Bars
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df1 = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000] * 5,
        }, index=dates)
        
        trader.update_historical_data(df1)
        initial_count = len(trader.price_history)
        assert initial_count == 5
        
        # Update mit denselben Bars
        trader.update_historical_data(df1)
        assert len(trader.price_history) == 5, "Duplikate sollten nicht hinzugefügt werden"
        
        # Update mit neuen Bars
        dates2 = pd.date_range('2024-01-06', periods=3, freq='D')
        df2 = pd.DataFrame({
            'open': [105, 106, 107],
            'high': [106, 107, 108],
            'low': [104, 105, 106],
            'close': [105.5, 106.5, 107.5],
            'volume': [1000] * 3,
        }, index=dates2)
        
        combined_df = pd.concat([df1, df2])
        trader.update_historical_data(combined_df)
        assert len(trader.price_history) == 8, "Neue Bars sollten hinzugefügt werden"
    
    def test_lookback_window_maintained(self, trader):
        """Teste dass Lookback-Fenster beibehalten wird."""
        import pandas as pd
        
        # Erstelle 20 Bars, aber Lookback ist nur 10
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'open': range(100, 120),
            'high': range(101, 121),
            'low': range(99, 119),
            'close': range(100, 120),
            'volume': [1000] * 20,
        }, index=dates)
        
        trader.update_historical_data(df)
        
        # Sollte nur die letzten 10 Bars halten
        assert len(trader.price_history) == 10
        # Und die neuesten sollten die höchsten Close-Werte haben
        assert trader.price_history[-1]["close"] == 119


class TestThreadSafety:
    """Tests für Thread-Sicherheit."""
    
    @pytest.fixture
    def trader(self):
        """Trader-Instanz für Tests."""
        config = LiveTradingConfig(
            state_path=tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        )
        with patch('live_trader.MockExchange'):
            with patch('live_trader.ReconciliationLoop'):
                with patch('live_trader.TradeJournal'):
                    with patch('live_trader.IncidentLog'):
                        trader = LiveTrader(config, use_mock=True)
        return trader
    
    def test_concurrent_entry_attempts(self, trader):
        """Teste dass konkurrierende Entry-Versuche sicher sind."""
        trader.capital = D(100_000)
        
        latest_bar = {
            "close": D(100),
            "stop_price": D(95),
            "target_price": D(110),
            "date": datetime.now(),
            "regime": "uptrend",
        }
        
        entry_count = [0]
        lock = threading.Lock()
        
        def try_entry():
            with patch.object(trader.exchange, 'submit_order') as mock_order:
                from execution import OrderStatus
                mock_order.return_value = Mock(status=OrderStatus.FILLED)
                
                with patch.object(trader, 'save_state'):
                    trader.execute_entry(latest_bar)
                    
                    if trader.in_position:
                        with lock:
                            entry_count[0] += 1
        
        # Starte 3 Threads gleichzeitig
        threads = [threading.Thread(target=try_entry) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Nur EINE sollte erfolgreich in_position setzen
        assert entry_count[0] <= 1, "Nur ein Entry sollte erfolgreich sein"
        assert trader.in_position in [True, False]


# Run Tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
