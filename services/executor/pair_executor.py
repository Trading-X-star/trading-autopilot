"""Pair Trading Executor - автоисполнение парных сделок"""
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional
import httpx

logger = logging.getLogger("pair-executor")

@dataclass
class PairPosition:
    pair: tuple
    long_ticker: str
    short_ticker: str
    long_qty: int
    short_qty: int
    long_entry: float
    short_entry: float
    entry_zscore: float
    entry_time: datetime = field(default_factory=datetime.now)
    
    @property
    def notional(self) -> float:
        return self.long_qty * self.long_entry + self.short_qty * self.short_entry

class PairExecutor:
    def __init__(self, strategy_url: str, datafeed_url: str, capital: float = 100_000):
        self.strategy_url = strategy_url
        self.datafeed_url = datafeed_url
        self.capital_per_pair = capital
        self.positions: Dict[tuple, PairPosition] = {}
        self.history = []
        self.enabled = False
        self.min_confidence = 0.6
        self.check_interval = 60
        self._client = None
    
    async def get_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10)
        return self._client
    
    async def get_signals(self) -> list:
        client = await self.get_client()
        try:
            r = await client.get(f"{self.strategy_url}/pairs/scan")
            return r.json()
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return {"signals": [], "all_pairs": []}
    
    async def get_price(self, ticker: str) -> Optional[float]:
        client = await self.get_client()
        try:
            r = await client.get(f"{self.datafeed_url}/price/{ticker}")
            data = r.json()
            return data.get("price") or data.get("last_price")
        except Exception as e:
            logger.error(f"Price error {ticker}: {e}")
            return None
    
    async def execute_order(self, ticker: str, side: str, qty: int) -> dict:
        """Симуляция ордера (логирование)"""
        price = await self.get_price(ticker)
        logger.info(f"📝 ORDER: {side.upper()} {qty}x {ticker} @ {price}")
        return {"status": "executed", "ticker": ticker, "side": side, "qty": qty, "price": price}
    
    async def open_pair(self, signal: dict) -> Optional[PairPosition]:
        pair = tuple(signal["pair"])
        
        if pair in self.positions:
            logger.info(f"Pair {pair} already open")
            return None
        
        conf = signal.get("confidence", 0)
        if conf < self.min_confidence:
            logger.info(f"Low confidence: {conf}")
            return None
        
        long_ticker = signal["long"]
        short_ticker = signal["short"]
        
        long_price = await self.get_price(long_ticker)
        short_price = await self.get_price(short_ticker)
        
        if not long_price or not short_price:
            logger.error(f"Cannot get prices: {long_ticker}={long_price}, {short_ticker}={short_price}")
            return None
        
        half_capital = self.capital_per_pair / 2
        long_qty = int(half_capital / long_price)
        short_qty = int(half_capital / short_price)
        
        if long_qty < 1 or short_qty < 1:
            logger.error("Insufficient capital")
            return None
        
        logger.info(f"🔓 Opening pair: LONG {long_qty}x{long_ticker}@{long_price} / SHORT {short_qty}x{short_ticker}@{short_price}")
        
        await self.execute_order(long_ticker, "buy", long_qty)
        await self.execute_order(short_ticker, "sell", short_qty)
        
        position = PairPosition(
            pair=pair, long_ticker=long_ticker, short_ticker=short_ticker,
            long_qty=long_qty, short_qty=short_qty,
            long_entry=long_price, short_entry=short_price,
            entry_zscore=signal.get("zscore", 0)
        )
        
        self.positions[pair] = position
        logger.info(f"✅ Pair opened: {pair}, notional: {position.notional:.0f}₽")
        return position
    
    async def close_pair(self, pair: tuple, reason: str = "signal") -> Optional[dict]:
        if pair not in self.positions:
            return None
        
        pos = self.positions[pair]
        long_price = await self.get_price(pos.long_ticker) or pos.long_entry
        short_price = await self.get_price(pos.short_ticker) or pos.short_entry
        
        logger.info(f"🔒 Closing pair: {pair}")
        await self.execute_order(pos.long_ticker, "sell", pos.long_qty)
        await self.execute_order(pos.short_ticker, "buy", pos.short_qty)
        
        long_pnl = (long_price - pos.long_entry) * pos.long_qty
        short_pnl = (pos.short_entry - short_price) * pos.short_qty
        total_pnl = long_pnl + short_pnl
        pnl_pct = total_pnl / pos.notional * 100
        
        result = {
            "pair": list(pair), "long_pnl": round(long_pnl, 2), "short_pnl": round(short_pnl, 2),
            "total_pnl": round(total_pnl, 2), "pnl_pct": round(pnl_pct, 2), "reason": reason,
            "duration_min": (datetime.now() - pos.entry_time).seconds // 60
        }
        
        self.history.append(result)
        del self.positions[pair]
        logger.info(f"✅ Pair closed: {pair}, P&L: {total_pnl:.0f}₽ ({pnl_pct:.2f}%)")
        return result
    
    async def check_exits(self, all_pairs: list):
        for pd in all_pairs:
            pair_str = pd.get("pair", "")
            if "/" not in pair_str:
                continue
            ta, tb = pair_str.split("/")
            pair = (ta, tb)
            if pair not in self.positions:
                continue
            
            z = pd.get("zscore", 0)
            pos = self.positions[pair]
            
            if abs(z) < 0.5:
                await self.close_pair(pair, "convergence")
            elif abs(z) > abs(pos.entry_zscore) + 1.5:
                await self.close_pair(pair, "stop_loss")
    
    async def run_cycle(self):
        if not self.enabled:
            return
        
        data = await self.get_signals()
        signals = data.get("signals", [])
        all_pairs = data.get("all_pairs", [])
        
        await self.check_exits(all_pairs)
        
        for sig in signals:
            if sig.get("action") == "open":
                await self.open_pair(sig)
    
    async def run_loop(self):
        logger.info("🚀 Pair Executor started")
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            await asyncio.sleep(self.check_interval)
    
    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "positions": len(self.positions),
            "open_pairs": [
                {"pair": list(p.pair), "long": f"{p.long_qty}x{p.long_ticker}@{p.long_entry}",
                 "short": f"{p.short_qty}x{p.short_ticker}@{p.short_entry}",
                 "entry_z": p.entry_zscore, "notional": p.notional}
                for p in self.positions.values()
            ],
            "history": self.history[-10:],
            "total_pnl": sum(h["total_pnl"] for h in self.history)
        }

pair_executor = PairExecutor(
    strategy_url="http://strategy:8005",
    datafeed_url="http://datafeed:8006"
)
