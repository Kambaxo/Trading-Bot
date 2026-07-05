"""
governance.py — Trade-Journal und Incident-Logging
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional


class ThesisRequiredError(Exception):
    """Fehler wenn Thesis fehlerhaft."""
    pass


class TradeJournal:
    """Trade-Journal für Dokumentation."""
    
    def __init__(self, path: str = "/tmp/live_journal.json"):
        self.path = path
        self.trades = []
        self.load()
    
    def load(self):
        """Lade existierendes Journal."""
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self.trades = json.load(f)
            except Exception as e:
                print(f"Fehler beim Laden des Journals: {e}")
                self.trades = []
    
    def log(
        self,
        symbol: str,
        action: str,
        thesis: str,
        signal_score: float,
        regime: str,
    ):
        """Protokolliere Trade-Eintrag."""
        if not thesis or len(thesis.strip()) == 0:
            raise ThesisRequiredError("Thesis ist erforderlich")
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": action,
            "thesis": thesis,
            "signal_score": signal_score,
            "regime": regime,
        }
        
        self.trades.append(entry)
        self.save()
    
    def update_outcome(
        self,
        symbol: str,
        timestamp: str,
        outcome: str,
    ):
        """Aktualisiere Outcome eines Trades."""
        # Finde den neuesten Trade für dieses Symbol
        for trade in reversed(self.trades):
            if trade["symbol"] == symbol and "outcome" not in trade:
                trade["outcome"] = outcome
                trade["exit_timestamp"] = datetime.now(timezone.utc).isoformat()
                break
        
        self.save()
    
    def save(self):
        """Speichere Journal."""
        try:
            with open(self.path, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            print(f"Fehler beim Speichern des Journals: {e}")


class IncidentLog:
    """Incident-Logging für Fehler und Warnungen."""
    
    def __init__(self, path: str = "/tmp/incidents.json"):
        self.path = path
        self.incidents = []
        self.load()
    
    def load(self):
        """Lade existierendes Incident-Log."""
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self.incidents = json.load(f)
            except Exception as e:
                print(f"Fehler beim Laden des Incident-Logs: {e}")
                self.incidents = []
    
    def record(
        self,
        title: str,
        description: str,
        severity: str,  # "info", "warning", "error", "critical"
        fix: Optional[str] = None,
    ):
        """Protokolliere Incident."""
        incident = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "description": description,
            "severity": severity,
            "fix": fix,
        }
        
        self.incidents.append(incident)
        self.save()
    
    def save(self):
        """Speichere Incident-Log."""
        try:
            with open(self.path, 'w') as f:
                json.dump(self.incidents, f, indent=2)
        except Exception as e:
            print(f"Fehler beim Speichern des Incident-Logs: {e}")
