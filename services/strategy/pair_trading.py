"""Pair Trading Strategy"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("pair-trading")

PAIRS = [
    ("SBER", "VTBR"), ("GAZP", "NVTK"), ("LKOH", "ROSN"),
    ("NLMK", "CHMF"), ("MTSS", "MGNT"), ("TATN", "SNGS"),
]

@dataclass
class PairSignal:
    pair: tuple
    long_ticker: str
    short_ticker: str
    zscore: float
    spread: float
    mean_spread: float
    confidence: float
    action: str

class PairTradingStrategy:
    def __init__(self, lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.positions = {}
    
    def calc_spread(self, pa: list, pb: list):
        if len(pa) < self.lookback or len(pb) < self.lookback: return None, None, None
        spreads = [a/b for a, b in zip(pa[-self.lookback:], pb[-self.lookback:])]
        mean = sum(spreads) / len(spreads)
        std = (sum((s - mean)**2 for s in spreads) / len(spreads)) ** 0.5
        if std == 0: return spreads[-1], mean, 0
        curr = pa[-1] / pb[-1]
        return curr, mean, (curr - mean) / std
    
    def calc_corr(self, pa: list, pb: list) -> float:
        if len(pa) < 20 or len(pb) < 20: return 0
        n = min(len(pa), len(pb), self.lookback)
        a, b = pa[-n:], pb[-n:]
        ma, mb = sum(a)/n, sum(b)/n
        cov = sum((a[i]-ma)*(b[i]-mb) for i in range(n)) / n
        sa = (sum((x-ma)**2 for x in a)/n)**0.5
        sb = (sum((x-mb)**2 for x in b)/n)**0.5
        return cov / (sa * sb) if sa and sb else 0
    
    def analyze(self, ta: str, tb: str, pa: list, pb: list) -> Optional[PairSignal]:
        pair = (ta, tb)
        corr = self.calc_corr(pa, pb)
        if abs(corr) < 0.5: return PairSignal(pair, "", "", 0, 0, 0, 0, "hold")
        spread, mean, z = self.calc_spread(pa, pb)
        if z is None: return None
        conf = min(abs(z)/3, 1.0) * abs(corr)
        in_pos = pair in self.positions
        if not in_pos:
            if z > self.entry_z: return PairSignal(pair, tb, ta, z, spread, mean, conf, "open")
            if z < -self.entry_z: return PairSignal(pair, ta, tb, z, spread, mean, conf, "open")
        else:
            if abs(z) < self.exit_z: return PairSignal(pair, "", "", z, spread, mean, conf, "close")
        return PairSignal(pair, "", "", z, spread, mean, 0, "hold")
    
    def get_signals(self, prices: dict) -> list:
        signals = []
        for ta, tb in PAIRS:
            if ta in prices and tb in prices:
                sig = self.analyze(ta, tb, prices[ta], prices[tb])
                if sig and sig.action != "hold": signals.append(sig)
        return signals

pair_strategy = PairTradingStrategy()
