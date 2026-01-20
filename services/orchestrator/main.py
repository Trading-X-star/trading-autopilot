#!/usr/bin/env python3
"""Orchestrator v2 - Trading Coordinator"""
import asyncio, os, json, logging
from datetime import datetime, time
from contextlib import asynccontextmanager
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import asyncpg, httpx
import redis.asyncio as aioredis
from fastapi import Response, FastAPI
from prometheus_client import Gauge, Counter, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("orchestrator")

TRADING_STATE = Gauge("trading_state", "Trading enabled")
TRADING_MODE = Gauge("trading_mode_active", "Trading mode", ["mode"])
MARKET_REGIME = Gauge("market_regime_active", "Market regime", ["regime"])
UPTIME = Gauge("system_uptime_seconds", "Uptime")
COMMANDS = Counter("orchestrator_commands_total", "Commands", ["command"])

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"

class TradingMode(Enum):
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    DEFENSIVE = "defensive"
    STOPPED = "stopped"

@dataclass
class TradingConfig:
    mode: TradingMode
    min_confidence: float
    max_position_pct: float
    stop_loss_pct: float
    take_profit_pct: float

CONFIGS = {
    TradingMode.AGGRESSIVE: TradingConfig(TradingMode.AGGRESSIVE, 0.45, 0.15, 0.07, 0.15),
    TradingMode.NORMAL: TradingConfig(TradingMode.NORMAL, 0.55, 0.10, 0.05, 0.10),
    TradingMode.CONSERVATIVE: TradingConfig(TradingMode.CONSERVATIVE, 0.65, 0.07, 0.04, 0.08),
    TradingMode.DEFENSIVE: TradingConfig(TradingMode.DEFENSIVE, 0.75, 0.05, 0.03, 0.06),
    TradingMode.STOPPED: TradingConfig(TradingMode.STOPPED, 1.0, 0.0, 0.02, 0.0),
}

class Orchestrator:
    def __init__(self):
        self.db: Optional[asyncpg.Pool] = None
        self.redis: Optional[aioredis.Redis] = None
        self.http: Optional[httpx.AsyncClient] = None
        self.running = False
        self.start_time: Optional[datetime] = None
        self.trading_enabled = False
        self.config = CONFIGS[TradingMode.NORMAL]
        self.market_regime = MarketRegime.SIDEWAYS
        self.position_highs: Dict[str, float] = {}
    
    async def start(self):
        try:
            self.db = await asyncpg.create_pool(
                os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading"),
                min_size=2, max_size=10
            )
            logger.info("✅ PostgreSQL connected")
        except Exception as e:
            logger.error(f"PostgreSQL error: {e}")
            raise
        
        self.redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
        self.http = httpx.AsyncClient(timeout=30)
        
        # Restore state
        state = await self.redis.get("orchestrator:trading_enabled")
        self.trading_enabled = state == "1"
        
        mode = await self.redis.get("orchestrator:trading_mode")
        if mode:
            try:
                self.config = CONFIGS[TradingMode(mode)]
            except:
                pass
        
        self.start_time = datetime.now()
        self.running = True
        
        asyncio.create_task(self._main_loop())
        asyncio.create_task(self._position_loop())
        asyncio.create_task(self._metrics_loop())
        
        logger.info(f"✅ Orchestrator started | mode={self.config.mode.value} trading={self.trading_enabled}")
    
    async def stop(self):
        self.running = False
        await self.redis.set("orchestrator:trading_enabled", "1" if self.trading_enabled else "0")
        await self.redis.set("orchestrator:trading_mode", self.config.mode.value)
        if self.http: await self.http.aclose()
        if self.redis: await self.redis.close()
        if self.db: await self.db.close()
        logger.info("🛑 Orchestrator stopped")
    
    def _is_trading_hours(self) -> bool:
        now = datetime.now()
        return now.weekday() < 5 and time(7, 0) <= now.time() <= time(21, 0)
    
    async def _analyze_market(self) -> MarketRegime:
        try:
            async with self.db.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT AVG(volatility_20) as vol, 
                           AVG(CASE WHEN close > sma_20 THEN 1.0 ELSE -1.0 END) as trend
                    FROM features WHERE date = (SELECT MAX(date) FROM features)
                """)
                vol = float(row['vol'] or 0.02) if row else 0.02
                trend = float(row['trend'] or 0) if row else 0
                
                sent_row = await conn.fetchrow("""
                    SELECT AVG(sentiment_avg) as s FROM news_sentiment_daily 
                    WHERE date >= CURRENT_DATE - INTERVAL '2 days'
                """)
                sent = float(sent_row['s'] or 0) if sent_row else 0
            
            if sent < -0.5 and trend < -0.4: return MarketRegime.CRISIS
            if vol > 0.04: return MarketRegime.CRISIS
            if vol > 0.025: return MarketRegime.HIGH_VOL
            if trend > 0.4: return MarketRegime.BULL
            if trend < -0.4: return MarketRegime.BEAR
            return MarketRegime.SIDEWAYS
        except Exception as e:
            logger.warning(f"Market analysis error: {e}")
            return MarketRegime.SIDEWAYS
    
    async def _make_decisions(self):
        regime = await self._analyze_market()
        self.market_regime = regime
        
        old_mode = self.config.mode
        new_mode = TradingMode.NORMAL
        
        if regime == MarketRegime.CRISIS:
            new_mode = TradingMode.STOPPED
        elif regime == MarketRegime.BEAR:
            new_mode = TradingMode.DEFENSIVE
        elif regime == MarketRegime.HIGH_VOL:
            new_mode = TradingMode.CONSERVATIVE
        elif regime == MarketRegime.BULL:
            new_mode = TradingMode.AGGRESSIVE
        
        if new_mode != old_mode:
            self.config = CONFIGS[new_mode]
            await self.redis.set("orchestrator:trading_mode", new_mode.value)
            logger.info(f"🔄 Mode: {old_mode.value} → {new_mode.value} | regime={regime.value}")
        
        # Update metrics
        for m in TradingMode:
            TRADING_MODE.labels(mode=m.value).set(1 if m == new_mode else 0)
        for r in MarketRegime:
            MARKET_REGIME.labels(regime=r.value).set(1 if r == regime else 0)
    
    async def _manage_positions(self):
        if self.config.mode == TradingMode.STOPPED or not self.trading_enabled:
            return
        
        try:
            resp = await self.http.get("http://executor:8007/positions")
            positions = resp.json().get('positions', [])
        except:
            return
        
        for pos in positions:
            ticker = pos.get('ticker')
            if not ticker or ticker == 'RUB':
                continue
            
            entry = float(pos.get('avg_price', 0))
            current = float(pos.get('current_price', 0))
            qty = int(pos.get('quantity', 0))
            
            if entry <= 0 or current <= 0 or qty <= 0:
                continue
            
            pnl_pct = (current - entry) / entry
            
            # Track highs for trailing stop
            if ticker not in self.position_highs:
                self.position_highs[ticker] = current
            elif current > self.position_highs[ticker]:
                self.position_highs[ticker] = current
            
            action = None
            reason = None
            
            # Stop loss
            if pnl_pct < -self.config.stop_loss_pct:
                action, reason = 'STOP_LOSS', f"PnL {pnl_pct:.1%}"
            # Take profit
            elif pnl_pct > self.config.take_profit_pct:
                action, reason = 'TAKE_PROFIT', f"PnL {pnl_pct:.1%}"
            # Trailing stop
            elif pnl_pct > 0.05:
                high = self.position_highs.get(ticker, current)
                drop = (high - current) / high
                if drop > 0.03:
                    action, reason = 'TRAILING_STOP', f"Drop {drop:.1%} from high"
            
            if action:
                try:
                    await self.http.post("http://executor:8007/order", json={
                        'ticker': ticker, 'side': 'sell', 'quantity': qty,
                        'reason': f"{action}: {reason}"
                    })
                    self.position_highs.pop(ticker, None)
                    logger.info(f"📤 {action} {ticker}: {reason}")
                except Exception as e:
                    logger.error(f"Exit {ticker} failed: {e}")
    
    async def _main_loop(self):
        while self.running:
            try:
                if self._is_trading_hours():
                    await self._make_decisions()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Main loop: {e}")
                await asyncio.sleep(30)
    
    async def _position_loop(self):
        while self.running:
            try:
                if self._is_trading_hours():
                    await self._manage_positions()
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Position loop: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_loop(self):
        while self.running:
            if self.start_time:
                UPTIME.set((datetime.now() - self.start_time).total_seconds())
            TRADING_STATE.set(1 if self.trading_enabled else 0)
            await asyncio.sleep(10)
    
    async def set_trading(self, enabled: bool) -> dict:
        self.trading_enabled = enabled
        await self.redis.set("orchestrator:trading_enabled", "1" if enabled else "0")
        TRADING_STATE.set(1 if enabled else 0)
        COMMANDS.labels(command="trading_toggle").inc()
        logger.info(f"🔄 Trading {'STARTED' if enabled else 'STOPPED'}")
        return {"trading": enabled, "mode": self.config.mode.value}
    
    async def set_mode(self, mode: str) -> dict:
        try:
            new_mode = TradingMode(mode)
            old_mode = self.config.mode
            self.config = CONFIGS[new_mode]
            await self.redis.set("orchestrator:trading_mode", new_mode.value)
            logger.info(f"🔄 Mode: {old_mode.value} → {new_mode.value} (manual)")
            return {"old": old_mode.value, "new": new_mode.value}
        except ValueError:
            return {"error": f"Invalid mode: {mode}. Valid: {[m.value for m in TradingMode]}"}
    
    async def get_state(self) -> dict:
        return {
            "trading": self.trading_enabled,
            "mode": self.config.mode.value,
            "regime": self.market_regime.value,
            "config": {
                "min_confidence": self.config.min_confidence,
                "max_position_pct": self.config.max_position_pct,
                "stop_loss_pct": self.config.stop_loss_pct,
                "take_profit_pct": self.config.take_profit_pct,
            },
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }

svc = Orchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()

app = FastAPI(title="Orchestrator v2", lifespan=lifespan)
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
    # Extended health check
    checks = {"redis": False, "status": "healthy"}
    try:
        if hasattr(globals().get(list(globals().keys())[0]), 'redis'):
            await globals()[list(globals().keys())[0]].redis.ping()
            checks["redis"] = True
    except: pass
    return {"status": "healthy", "trading": svc.trading_enabled, "mode": svc.config.mode.value}

@app.get("/state")
async def state():
    return await svc.get_state()

@app.post("/trading/start")
async def start_trading():
    return await svc.set_trading(True)

@app.post("/trading/stop")
async def stop_trading():
    return await svc.set_trading(False)

@app.post("/mode/{mode}")
async def set_mode(mode: str):
    return await svc.set_mode(mode)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
