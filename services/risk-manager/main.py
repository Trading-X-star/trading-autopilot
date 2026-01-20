#!/usr/bin/env python3
"""Risk Manager v2.0 - Enhanced with drawdown protection & confidence scaling"""
import asyncio, os, json, logging
from datetime import datetime, date
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, Dict, List
import httpx, asyncpg, redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("risk-manager")

# Metrics
CHECKS_TOTAL = Counter("risk_checks_total", "Checks", ["result"])
STOPS_TRIGGERED = Counter("stops_triggered_total", "Stops", ["ticker", "type"])
POSITION_RISK = Gauge("position_risk_pct", "Risk %", ["ticker"])
DRAWDOWN_GAUGE = Gauge("portfolio_drawdown_pct", "Current drawdown")
DAILY_PNL_GAUGE = Gauge("daily_pnl", "Daily PnL")
CONSECUTIVE_LOSSES = Gauge("consecutive_losses", "Consecutive losses")

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

RISK_LIMITS = {
    RiskProfile.CONSERVATIVE: {
        "max_position_pct": 3.0,
        "max_daily_loss_pct": 1.5,
        "max_portfolio_exposure_pct": 30.0,
        "max_drawdown_pct": 5.0,          # NEW
        "atr_mult_sl": 2.0,
        "atr_mult_tp": 3.0,                # R:R = 1.5
        "max_positions": 5,
        "max_consecutive_losses": 3,       # NEW
        "pause_after_losses": 5,           # NEW
        "min_confidence": 0.55,            # NEW
        "max_daily_trades": 10,
    },
    RiskProfile.BALANCED: {
        "max_position_pct": 5.0,
        "max_daily_loss_pct": 2.0,
        "max_portfolio_exposure_pct": 50.0,
        "max_drawdown_pct": 10.0,
        "atr_mult_sl": 2.0,
        "atr_mult_tp": 3.0,
        "max_positions": 10,
        "max_consecutive_losses": 4,
        "pause_after_losses": 6,
        "min_confidence": 0.50,
        "max_daily_trades": 15,
    },
    RiskProfile.AGGRESSIVE: {
        "max_position_pct": 10.0,
        "max_daily_loss_pct": 3.0,
        "max_portfolio_exposure_pct": 70.0,
        "max_drawdown_pct": 15.0,
        "atr_mult_sl": 1.5,
        "atr_mult_tp": 2.5,
        "max_positions": 15,
        "max_consecutive_losses": 5,
        "pause_after_losses": 8,
        "min_confidence": 0.45,
        "max_daily_trades": 25,
    },
}

class OrderRequest(BaseModel):
    account_id: str = "default"
    ticker: str
    side: str
    quantity: int
    price: float
    risk_profile: str = "balanced"
    confidence: float = 0.5              # NEW: from ML model

@dataclass
class Position:
    ticker: str
    side: str
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float = 0
    take_profit: float = 0
    highest_price: float = 0
    lowest_price: float = 0              # NEW: for short trailing
    confidence: float = 0.5              # NEW
    opened_at: datetime = field(default_factory=datetime.now)
    
    @property
    def pnl(self): 
        return (self.current_price - self.entry_price) * self.quantity * (1 if self.side == "long" else -1)
    
    @property
    def pnl_pct(self): 
        if not self.entry_price:
            return 0
        return (self.current_price / self.entry_price - 1) * 100 * (1 if self.side == "long" else -1)
    
    @property
    def risk_pct(self):
        """Current risk as % of entry"""
        if self.side == "long":
            return (self.entry_price - self.stop_loss) / self.entry_price * 100
        else:
            return (self.stop_loss - self.entry_price) / self.entry_price * 100


class RiskManager:
    def __init__(self):
        self.pool = None
        self.redis = None
        self.http = None
        self.positions: Dict[str, Position] = {}
        self.running = False
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.today = date.today()
        
        # Drawdown tracking
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.initial_balance = 0.0
        
        # Loss tracking
        self.consecutive_losses = 0
        self.is_paused = False
        self.pause_reason = ""
        
        # Trailing config
        self.trailing_activation_pct = 1.5
        self.trailing_distance_pct = 1.0
        
        self.executor_url = os.getenv("EXECUTOR_URL", "http://executor:8007")
    
    async def start(self):
        self.pool = await asyncpg.create_pool(
            os.getenv("DB_DSN", os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading")),
            min_size=2, max_size=10
        )
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"), 
            decode_responses=True
        )
        self.http = httpx.AsyncClient(timeout=10.0)
        await self._init_db()
        await self._load_balance()
        self.running = True
        asyncio.create_task(self._monitor_loop())
        asyncio.create_task(self._trade_listener())
        asyncio.create_task(self._daily_reset_loop())
        logger.info("✅ Risk Manager v2.0 started")
        logger.info(f"   Balance: {self.current_balance:,.0f}")
        logger.info(f"   Peak: {self.peak_balance:,.0f}")
    
    async def stop(self):
        self.running = False
        if self.http: await self.http.aclose()
        if self.redis: await self.redis.close()
        if self.pool: await self.pool.close()
    
    async def _init_db(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS position_history (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20),
                    side VARCHAR(10),
                    entry_price DECIMAL(20,4),
                    exit_price DECIMAL(20,4),
                    quantity INT,
                    pnl DECIMAL(20,2),
                    pnl_pct DECIMAL(10,2),
                    exit_reason VARCHAR(50),
                    confidence DECIMAL(5,3),
                    exit_time TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id SERIAL PRIMARY KEY,
                    date DATE UNIQUE,
                    pnl DECIMAL(20,2),
                    trades INT,
                    wins INT,
                    losses INT,
                    max_drawdown DECIMAL(10,4)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS account_balance (
                    id SERIAL PRIMARY KEY,
                    balance DECIMAL(20,2),
                    peak_balance DECIMAL(20,2),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
    
    async def _load_balance(self):
        """Load balance from DB or use default"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT balance, peak_balance FROM account_balance ORDER BY id DESC LIMIT 1")
            if row:
                self.current_balance = float(row['balance'])
                self.peak_balance = float(row['peak_balance'])
            else:
                self.current_balance = 1_000_000
                self.peak_balance = 1_000_000
                await conn.execute(
                    "INSERT INTO account_balance (balance, peak_balance) VALUES ($1, $2)",
                    self.current_balance, self.peak_balance
                )
            self.initial_balance = self.current_balance
    
    async def _save_balance(self):
        """Save current balance to DB"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO account_balance (balance, peak_balance) VALUES ($1, $2)",
                self.current_balance, self.peak_balance
            )
    
    async def _get_price(self, ticker: str) -> float:
        try:
            data = await self.redis.get(f"price:{ticker}")
            return json.loads(data).get("price", 0) if data else 0
        except:
            return 0
    
    async def _get_atr(self, ticker: str) -> float:
        try:
            data = await self.redis.zrevrange(f"history:{ticker}", 0, 14)
            if len(data) < 10:
                return 0
            prices = [json.loads(d)["price"] for d in data]
            return sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices))) / (len(prices)-1)
        except:
            return 0
    
    def _get_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_balance <= 0:
            return 0
        return (self.peak_balance - self.current_balance) / self.peak_balance
    
    def _get_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        total_exposure = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return total_exposure / self.current_balance if self.current_balance > 0 else 0
    
    async def check_order(self, order: OrderRequest) -> dict:
        """Enhanced order validation with all risk checks"""
        
        try:
            profile = RiskProfile(order.risk_profile)
        except:
            profile = RiskProfile.BALANCED
        
        limits = RISK_LIMITS[profile]
        
        # 1. Check if trading is paused
        if self.is_paused:
            CHECKS_TOTAL.labels(result="rejected").inc()
            return {
                "approved": False,
                "reason": f"Trading paused: {self.pause_reason}"
            }
        
        # 2. Check confidence threshold
        if order.confidence < limits["min_confidence"]:
            CHECKS_TOTAL.labels(result="rejected").inc()
            return {
                "approved": False,
                "reason": f"Confidence {order.confidence:.1%} below threshold {limits['min_confidence']:.0%}"
            }
        
        # 3. Check daily loss limit
        if self.daily_pnl / self.initial_balance < -limits["max_daily_loss_pct"] / 100:
            self._pause_trading("Daily loss limit reached")
            CHECKS_TOTAL.labels(result="rejected").inc()
            return {
                "approved": False,
                "reason": f"Daily loss limit {limits['max_daily_loss_pct']}% reached"
            }
        
        # 4. Check max drawdown
        drawdown = self._get_drawdown()
        if drawdown > limits["max_drawdown_pct"] / 100:
            self._pause_trading(f"Max drawdown {drawdown:.1%} reached")
            CHECKS_TOTAL.labels(result="rejected").inc()
            return {
                "approved": False,
                "reason": f"Drawdown {drawdown:.1%} exceeds {limits['max_drawdown_pct']}%"
            }
        
        # 5. Check max positions
        if len(self.positions) >= limits["max_positions"]:
            CHECKS_TOTAL.labels(result="rejected").inc()
            return {
                "approved": False,
                "reason": f"Max positions ({limits['max_positions']}) reached"
            }
        
        # 5.5. Check max daily trades
        if self.daily_trades >= limits["max_daily_trades"]:
            CHECKS_TOTAL.labels(result="rejected").inc()
            return {
                "approved": False,
                "reason": f"Max daily trades ({limits['max_daily_trades']}) reached"
            }

        # 6. Check if already have position in this ticker
        if order.ticker.upper() in self.positions:
            CHECKS_TOTAL.labels(result="rejected").inc()
            return {
                "approved": False,
                "reason": f"Already have position in {order.ticker}"
            }
        
        # 7. Check portfolio exposure
        current_exposure = self._get_exposure()
        new_exposure = order.quantity * order.price / self.current_balance
        if (current_exposure + new_exposure) > limits["max_portfolio_exposure_pct"] / 100:
            CHECKS_TOTAL.labels(result="rejected").inc()
            return {
                "approved": False,
                "reason": f"Portfolio exposure would exceed {limits['max_portfolio_exposure_pct']}%"
            }
        
        # 8. Calculate position size (with confidence scaling)
        order_value = order.quantity * order.price
        position_pct = (order_value / self.current_balance) * 100
        
        # Scale by confidence (0.5 conf = 50% size, 0.8 conf = 100% size)
        conf_scale = min(1.0, (order.confidence - 0.3) * 2)
        adjusted_max_pct = limits["max_position_pct"] * conf_scale
        
        # Scale down after consecutive losses
        if self.consecutive_losses >= limits["max_consecutive_losses"]:
            loss_scale = 0.5 ** (self.consecutive_losses - limits["max_consecutive_losses"] + 1)
            adjusted_max_pct *= max(0.25, loss_scale)
            logger.warning(f"⚠️ Reduced size due to {self.consecutive_losses} consecutive losses")
        
        if position_pct > adjusted_max_pct:
            adjusted_qty = int((self.current_balance * adjusted_max_pct / 100) / order.price)
            if adjusted_qty < 1:
                CHECKS_TOTAL.labels(result="rejected").inc()
                return {
                    "approved": False,
                    "reason": f"Position too small after adjustments"
                }
            order.quantity = adjusted_qty
            position_pct = (adjusted_qty * order.price / self.current_balance) * 100
        
        # 9. Calculate stops
        atr = await self._get_atr(order.ticker)
        if atr <= 0:
            atr = order.price * 0.02  # Default 2%
        
        sl_distance = atr * limits["atr_mult_sl"]
        tp_distance = atr * limits["atr_mult_tp"]
        
        if order.side == "buy":
            stop_loss = order.price - sl_distance
            take_profit = order.price + tp_distance
        else:
            stop_loss = order.price + sl_distance
            take_profit = order.price - tp_distance
        
        CHECKS_TOTAL.labels(result="approved").inc()
        
        return {
            "approved": True,
            "order_id": f"ORD-{datetime.now().strftime('%H%M%S')}-{order.ticker}",
            "adjusted_quantity": order.quantity,
            "position_size_pct": round(position_pct, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_reward": round(tp_distance / sl_distance, 2),
            "confidence": order.confidence,
            "consecutive_losses": self.consecutive_losses,
            "current_drawdown": round(self._get_drawdown() * 100, 2)
        }
    
    def _pause_trading(self, reason: str):
        """Pause trading with reason"""
        self.is_paused = True
        self.pause_reason = reason
        logger.warning(f"🛑 TRADING PAUSED: {reason}")
    
    def resume_trading(self):
        """Resume trading"""
        self.is_paused = False
        self.pause_reason = ""
        self.consecutive_losses = 0
        logger.info("🟢 Trading RESUMED")
    
    async def add_position(self, ticker: str, side: str, qty: int, price: float, confidence: float = 0.5) -> Position:
        """Add position with enhanced tracking"""
        
        atr = await self._get_atr(ticker)
        if atr <= 0:
            atr = price * 0.02
        
        # Use balanced profile for stops
        limits = RISK_LIMITS[RiskProfile.BALANCED]
        sl_dist = atr * limits["atr_mult_sl"]
        tp_dist = atr * limits["atr_mult_tp"]
        
        if side == "long":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
        
        pos = Position(
            ticker=ticker,
            side=side,
            quantity=qty,
            entry_price=price,
            current_price=price,
            stop_loss=sl,
            take_profit=tp,
            highest_price=price,
            lowest_price=price,
            confidence=confidence
        )
        
        self.positions[ticker] = pos
        logger.info(f"📊 {side.upper()} {ticker} @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Conf: {confidence:.1%}")
        
        return pos
    
    async def close_position(self, ticker: str, reason: str) -> Optional[dict]:
        """Close position with PnL tracking"""
        
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        
        # Send close order to executor
        try:
            await self.http.post(
                f"{self.executor_url}/order",
                json={
                    "ticker": ticker,
                    "side": "sell" if pos.side == "long" else "buy",
                    "quantity": pos.quantity
                }
            )
        except Exception as e:
            logger.error(f"Failed to send close order: {e}")
        
        # Update tracking
        pnl = pos.pnl
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.current_balance += pnl
        
        # Update peak
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Track consecutive losses
        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            
            # Check if should pause
            limits = RISK_LIMITS[RiskProfile.BALANCED]
            if self.consecutive_losses >= limits["pause_after_losses"]:
                self._pause_trading(f"{self.consecutive_losses} consecutive losses")
        
        # Update metrics
        STOPS_TRIGGERED.labels(ticker=ticker, type=reason).inc()
        DRAWDOWN_GAUGE.set(self._get_drawdown() * 100)
        DAILY_PNL_GAUGE.set(self.daily_pnl)
        CONSECUTIVE_LOSSES.set(self.consecutive_losses)
        
        # Save to DB
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO position_history 
                (ticker, side, entry_price, exit_price, quantity, pnl, pnl_pct, exit_reason, confidence)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, ticker, pos.side, pos.entry_price, pos.current_price, 
                pos.quantity, pnl, pos.pnl_pct, reason, pos.confidence)
        
        await self._save_balance()
        
        result_emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{result_emoji} Closed {ticker} | {reason} | PnL: {pnl:+,.0f} ({pos.pnl_pct:+.2f}%)")
        
        del self.positions[ticker]
        
        return {
            "ticker": ticker,
            "reason": reason,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pos.pnl_pct, 2),
            "consecutive_losses": self.consecutive_losses
        }
    
    async def _monitor_loop(self):
        """Monitor positions and manage stops"""
        logger.info("📡 Position monitor started")
        
        while self.running:
            for ticker, pos in list(self.positions.items()):
                price = await self._get_price(ticker)
                if price <= 0:
                    continue
                
                pos.current_price = price
                
                if pos.side == "long":
                    # Check stop loss
                    if price <= pos.stop_loss:
                        await self.close_position(ticker, "stop_loss")
                        continue
                    
                    # Check take profit
                    if price >= pos.take_profit:
                        await self.close_position(ticker, "take_profit")
                        continue
                    
                    # Update trailing stop
                    if price > pos.highest_price:
                        pos.highest_price = price
                        if pos.pnl_pct >= self.trailing_activation_pct:
                            new_sl = price * (1 - self.trailing_distance_pct / 100)
                            if new_sl > pos.stop_loss:
                                pos.stop_loss = new_sl
                                logger.info(f"📈 Trailing {ticker}: SL → {new_sl:.2f}")
                
                else:  # Short position
                    # Check stop loss
                    if price >= pos.stop_loss:
                        await self.close_position(ticker, "stop_loss")
                        continue
                    
                    # Check take profit
                    if price <= pos.take_profit:
                        await self.close_position(ticker, "take_profit")
                        continue
                    
                    # Update trailing stop for shorts
                    if price < pos.lowest_price:
                        pos.lowest_price = price
                        if pos.pnl_pct >= self.trailing_activation_pct:
                            new_sl = price * (1 + self.trailing_distance_pct / 100)
                            if new_sl < pos.stop_loss:
                                pos.stop_loss = new_sl
                                logger.info(f"📉 Trailing {ticker}: SL → {new_sl:.2f}")
                
                POSITION_RISK.labels(ticker=ticker).set(pos.risk_pct)
            
            # Update global metrics
            DRAWDOWN_GAUGE.set(self._get_drawdown() * 100)
            
            await asyncio.sleep(1)
    
    async def _trade_listener(self):
        """Listen for trade events from Redis stream"""
        logger.info("📡 Trade listener started")
        last_id = "$"
        
        while self.running:
            try:
                result = await self.redis.xread(
                    {"stream:trades": last_id}, 
                    count=10, 
                    block=5000
                )
                
                for stream, msgs in (result or []):
                    for msg_id, data in msgs:
                        last_id = msg_id
                        ticker = data.get("ticker", "").upper()
                        side = data.get("side", "")
                        qty = int(data.get("quantity", 1))
                        price = float(data.get("price", 0))
                        conf = float(data.get("confidence", 0.5))
                        
                        if side == "buy" and ticker not in self.positions:
                            await self.add_position(ticker, "long", qty, price, conf)
                        elif side == "sell" and ticker in self.positions:
                            await self.close_position(ticker, "manual")
                            
            except Exception as e:
                logger.error(f"Trade listener error: {e}")
                await asyncio.sleep(5)
    
    async def _daily_reset_loop(self):
        """Reset daily counters at midnight"""
        while self.running:
            now = datetime.now()
            if now.date() != self.today:
                # Save daily stats
                async with self.pool.acquire() as conn:
                    # Count wins/losses for today
                    rows = await conn.fetch("""
                        SELECT pnl FROM position_history 
                        WHERE DATE(exit_time) = $1
                    """, self.today)
                    
                    wins = sum(1 for r in rows if r['pnl'] > 0)
                    losses = sum(1 for r in rows if r['pnl'] <= 0)
                    
                    await conn.execute("""
                        INSERT INTO daily_stats (date, pnl, trades, wins, losses, max_drawdown)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (date) DO UPDATE SET 
                            pnl = $2, trades = $3, wins = $4, losses = $5, max_drawdown = $6
                    """, self.today, self.daily_pnl, self.daily_trades, wins, losses, self._get_drawdown())
                
                logger.info(f"📅 Daily reset | PnL: {self.daily_pnl:+,.0f} | W/L: {wins}/{losses}")
                
                # Reset counters
                self.today = now.date()
                self.daily_pnl = 0
                self.daily_trades = 0
                self.initial_balance = self.current_balance
                
                # Resume if paused for daily loss
                if self.is_paused and "Daily loss" in self.pause_reason:
                    self.resume_trading()
            
            await asyncio.sleep(60)
    
    def get_status(self) -> dict:
        """Get comprehensive status"""
        return {
            "balance": round(self.current_balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "drawdown_pct": round(self._get_drawdown() * 100, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "positions": len(self.positions),
            "exposure_pct": round(self._get_exposure() * 100, 2),
            "consecutive_losses": self.consecutive_losses,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason
        }


svc = RiskManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()

app = FastAPI(title="Risk Manager v2.0", lifespan=lifespan)
# ============================================================
# METRICS ENDPOINT (fixed - no 307 redirects)
# ============================================================
@app.get("/metrics")
@app.get("/metrics/")
async def prometheus_metrics():
    from fastapi import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

# OLD: metrics mount removed

@app.get("/health")
async def health():
    return svc.get_status()

@app.post("/check")
async def check(order: OrderRequest):
    return await svc.check_order(order)

@app.get("/profiles")
async def profiles():
    return [{"name": p.value, **RISK_LIMITS[p]} for p in RiskProfile]

@app.get("/positions")
async def positions():
    return {
        t: {**asdict(p), "pnl": round(p.pnl, 2), "pnl_pct": round(p.pnl_pct, 2)}
        for t, p in svc.positions.items()
    }

@app.post("/position")
async def add_pos(ticker: str, side: str, quantity: int, price: float, confidence: float = 0.5):
    pos = await svc.add_position(ticker.upper(), side, quantity, price, confidence)
    return asdict(pos)

@app.delete("/position/{ticker}")
async def close_pos(ticker: str, reason: str = "manual"):
    result = await svc.close_position(ticker.upper(), reason)
    if not result:
        raise HTTPException(404, "Position not found")
    return result

@app.put("/position/{ticker}/stop")
async def upd_stop(ticker: str, stop_loss: float):
    ticker = ticker.upper()
    if ticker not in svc.positions:
        raise HTTPException(404, "Position not found")
    svc.positions[ticker].stop_loss = stop_loss
    return {"ok": True, "stop_loss": stop_loss}

@app.post("/resume")
async def resume():
    svc.resume_trading()
    return {"ok": True, "status": svc.get_status()}

@app.get("/history")
async def history(limit: int = 20):
    async with svc.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM position_history ORDER BY exit_time DESC LIMIT $1", 
            limit
        )
        return [dict(r) for r in rows]

@app.get("/daily-stats")
async def daily_stats(days: int = 7):
    async with svc.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM daily_stats ORDER BY date DESC LIMIT $1",
            days
        )
        return [dict(r) for r in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
