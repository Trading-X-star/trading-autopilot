"""
Decimal utilities for financial calculations.
All monetary values MUST use Decimal, not float.
"""
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Union, Optional
import functools

# Precision for different types
PRICE_PRECISION = Decimal("0.01")      # 2 decimal places for prices
QUANTITY_PRECISION = Decimal("1")       # Whole numbers for shares
PERCENT_PRECISION = Decimal("0.0001")   # 4 decimal places for percentages
PNL_PRECISION = Decimal("0.01")         # 2 decimal places for P&L

def to_decimal(value: Union[str, int, float, Decimal, None], default: Decimal = Decimal("0")) -> Decimal:
    """Safely convert any value to Decimal."""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        if isinstance(value, float):
            # Avoid float precision issues
            return Decimal(str(value))
        return Decimal(value)
    except (InvalidOperation, ValueError, TypeError):
        return default

def round_price(value: Union[Decimal, float, str]) -> Decimal:
    """Round to price precision (2 decimal places)."""
    return to_decimal(value).quantize(PRICE_PRECISION, rounding=ROUND_HALF_UP)

def round_percent(value: Union[Decimal, float, str]) -> Decimal:
    """Round to percentage precision (4 decimal places)."""
    return to_decimal(value).quantize(PERCENT_PRECISION, rounding=ROUND_HALF_UP)

def round_quantity(value: Union[Decimal, float, str]) -> int:
    """Round to whole shares."""
    return int(to_decimal(value).quantize(QUANTITY_PRECISION, rounding=ROUND_HALF_UP))

def calculate_pnl(entry_price: Decimal, exit_price: Decimal, quantity: int, side: str = "buy") -> Decimal:
    """Calculate P&L with proper precision."""
    entry = to_decimal(entry_price)
    exit = to_decimal(exit_price)
    qty = Decimal(quantity)

    if side.lower() == "buy":
        pnl = (exit - entry) * qty
    else:
        pnl = (entry - exit) * qty

    return pnl.quantize(PNL_PRECISION, rounding=ROUND_HALF_UP)

def calculate_percent_change(old_value: Decimal, new_value: Decimal) -> Decimal:
    """Calculate percentage change with proper precision."""
    old = to_decimal(old_value)
    new = to_decimal(new_value)

    if old == 0:
        return Decimal("0")

    change = ((new - old) / old) * Decimal("100")
    return change.quantize(PERCENT_PRECISION, rounding=ROUND_HALF_UP)

def decimal_safe(func):
    """Decorator to ensure function returns Decimal-safe values."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, float):
            return to_decimal(result)
        if isinstance(result, dict):
            return {k: to_decimal(v) if isinstance(v, float) else v for k, v in result.items()}
        return result
    return wrapper

# Export commonly used
__all__ = [
    'Decimal', 'to_decimal', 'round_price', 'round_percent', 
    'round_quantity', 'calculate_pnl', 'calculate_percent_change',
    'decimal_safe', 'PRICE_PRECISION', 'PERCENT_PRECISION'
]
