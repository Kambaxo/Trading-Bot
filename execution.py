"""
execution.py — Order-Ausführung und Reconciliation
"""

from enum import Enum
from decimal import Decimal as D
from dataclasses import dataclass
from typing import Optional, List


class OrderStatus(Enum):
    """Order-Status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RejectionReason(Enum):
    """Gründe für Order-Ablehnung."""
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INVALID_QUANTITY = "invalid_quantity"
    MARKET_CLOSED = "market_closed"
    TECHNICAL_ERROR = "technical_error"
    RATE_LIMIT = "rate_limit"


@dataclass
class Order:
    """Order-Objekt."""
    order_id: str
    symbol: str
    side: str  # "buy" oder "sell"
    quantity: D
    status: OrderStatus
    reject_reason: Optional[RejectionReason] = None
    filled_quantity: D = D(0)


def should_retry(reason: Optional[RejectionReason]) -> bool:
    """Entscheide ob eine Order wiederholt werden sollte."""
    if reason is None:
        return False
    
    # Retry bei technischen Fehlern und Rate Limits
    return reason in [RejectionReason.TECHNICAL_ERROR, RejectionReason.RATE_LIMIT]


class MockExchange:
    """Mock-Exchange für Demo und Testing."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.orders = {}
        self.order_counter = 0
        self.position = {}
        import random
        random.seed(seed)
    
    def submit_order(self, symbol: str, side: str, quantity: D) -> Order:
        """Simuliere Order-Ausführung."""
        self.order_counter += 1
        order_id = f"mock-{self.order_counter}"
        
        # Mock: 95% erfolgreich, 5% technischer Fehler
        import random
        if random.random() < 0.95:
            status = OrderStatus.FILLED
            reject_reason = None
        else:
            status = OrderStatus.REJECTED
            reject_reason = RejectionReason.TECHNICAL_ERROR
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=status,
            reject_reason=reject_reason,
            filled_quantity=quantity if status == OrderStatus.FILLED else D(0)
        )
        
        self.orders[order_id] = order
        
        # Update Position
        if status == OrderStatus.FILLED:
            if symbol not in self.position:
                self.position[symbol] = D(0)
            
            if side == "buy":
                self.position[symbol] += quantity
            elif side == "sell":
                self.position[symbol] -= quantity
        
        return order
    
    def get_position(self, symbol: str) -> D:
        """Gebe aktuelle Position zurück."""
        return self.position.get(symbol, D(0))


class ReconciliationLoop:
    """Reconciliation zwischen lokalem State und Exchange."""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.tracked_orders = []
    
    def track(self, order: Order):
        """Tracke eine Order."""
        self.tracked_orders.append(order)
    
    def reconcile(self) -> List[str]:
        """
        Prüfe auf Diskrepanzen.
        Rückgabe: Liste von Fehlermeldungen (leer wenn alles ok)
        """
        drift_messages = []
        
        # Vereinfachte Reconciliation: prüfe ob Tracked Orders konsistent sind
        for order in self.tracked_orders[-10:]:  # Nur letzte 10 prüfen
            if order.status == OrderStatus.FILLED:
                position = self.exchange.get_position(order.symbol)
                if position != order.quantity:
                    drift_messages.append(
                        f"Position mismatch for {order.symbol}: "
                        f"local={order.quantity}, exchange={position}"
                    )
        
        return drift_messages
