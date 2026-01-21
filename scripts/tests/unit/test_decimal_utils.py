"""Tests for decimal utilities."""
import pytest
from decimal import Decimal
import sys
sys.path.insert(0, 'services/shared/utils')
from decimal_utils import (
    to_decimal, round_price, round_percent, round_quantity,
    calculate_pnl, calculate_percent_change
)

class TestToDecimal:
    def test_from_float(self):
        assert to_decimal(10.5) == Decimal("10.5")

    def test_from_string(self):
        assert to_decimal("100.25") == Decimal("100.25")

    def test_from_int(self):
        assert to_decimal(100) == Decimal("100")

    def test_from_none(self):
        assert to_decimal(None) == Decimal("0")

    def test_from_invalid(self):
        assert to_decimal("invalid") == Decimal("0")

class TestRounding:
    def test_round_price(self):
        assert round_price(100.125) == Decimal("100.13")
        assert round_price(100.124) == Decimal("100.12")

    def test_round_percent(self):
        assert round_percent(0.12345) == Decimal("0.1235")

    def test_round_quantity(self):
        assert round_quantity(10.6) == 11
        assert round_quantity(10.4) == 10

class TestPnLCalculation:
    def test_long_profit(self):
        pnl = calculate_pnl(Decimal("100"), Decimal("110"), 10, "buy")
        assert pnl == Decimal("100.00")

    def test_long_loss(self):
        pnl = calculate_pnl(Decimal("100"), Decimal("90"), 10, "buy")
        assert pnl == Decimal("-100.00")

    def test_short_profit(self):
        pnl = calculate_pnl(Decimal("100"), Decimal("90"), 10, "sell")
        assert pnl == Decimal("100.00")

class TestPercentChange:
    def test_positive_change(self):
        change = calculate_percent_change(Decimal("100"), Decimal("110"))
        assert change == Decimal("10.0000")

    def test_negative_change(self):
        change = calculate_percent_change(Decimal("100"), Decimal("90"))
        assert change == Decimal("-10.0000")

    def test_zero_base(self):
        change = calculate_percent_change(Decimal("0"), Decimal("100"))
        assert change == Decimal("0")
