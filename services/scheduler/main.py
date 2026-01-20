#!/usr/bin/env python3
"""
Trading Brain v5.3 - FULL TRADING SYSTEM
- Pattern Recognition: panic/capitulation detection
- ML Signal Engine: buy/sell signal generation
- Risk Management: Kelly criterion, stop-loss, drawdown control
- APScheduler, Circuit Breaker, Rate Limiting
- Prometheus metrics for Grafana
"""

import asyncio
import os
import logging
import json
import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import random

import asyncpg
import redis.asyncio as aioredis
import httpx
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import uvicorn
import yaml
from backtest_routes import router as backtest_router, get_engines, BacktestEngine, AutoExecuteEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trading_brain")


def to_float(val, default=0.0):
    if val is None: return default
    try: return float(val)
    except: return default


# ============================================================
# PATTERN RECOGNITION - Panic & Capitulation Detection
# ============================================================
class PatternType(Enum):
    NORMAL = "normal"
    PANIC_SELLING = "panic_selling"
    CAPITULATION = "capitulation"
    EUPHORIA = "euphoria"
    RECOVERY = "recovery"


@dataclass
class PatternSignal:
    pattern: PatternType
    confidence: float
    affected_tickers: List[str]
    avg_drop: float
    timestamp: datetime = field(default_factory=datetime.now)


class PatternRecognition:
    """Detects market panic and capitulation patterns"""
    
    PANIC_THRESHOLD = -0.05      # -5% drop
    CAPITULATION_THRESHOLD = -0.10  # -10% drop
    EUPHORIA_THRESHOLD = 0.08    # +8% gain
    MIN_TICKERS_PCT = 0.70       # 70% of tickers must be affected
    
    def __init__(self):
        self.last_pattern: Optional[PatternSignal] = None
        self.pattern_history: deque = deque(maxlen=100)
    
    async def detect(self, price_changes: Dict[str, float]) -> PatternSignal:
        """
        Detect patterns based on price changes
        price_changes: {ticker: pct_change} e.g. {'SBER': -0.12, 'GAZP': -0.08}
        """
        if not price_changes:
            return PatternSignal(PatternType.NORMAL, 0.0, [], 0.0)
        
        total = len(price_changes)
        
        # Count tickers by movement
        panic_tickers = [t for t, ch in price_changes.items() if ch <= self.PANIC_THRESHOLD]
        capitulation_tickers = [t for t, ch in price_changes.items() if ch <= self.CAPITULATION_THRESHOLD]
        euphoria_tickers = [t for t, ch in price_changes.items() if ch >= self.EUPHORIA_THRESHOLD]
        
        panic_ratio = len(panic_tickers) / total
        capitulation_ratio = len(capitulation_tickers) / total
        euphoria_ratio = len(euphoria_tickers) / total
        
        avg_change = sum(price_changes.values()) / total
        
        # Determine pattern
        if capitulation_ratio >= self.MIN_TICKERS_PCT:
            pattern = PatternSignal(
                PatternType.CAPITULATION,
                confidence=min(capitulation_ratio, 0.99),
                affected_tickers=capitulation_tickers,
                avg_drop=avg_change
            )
        elif panic_ratio >= self.MIN_TICKERS_PCT:
            pattern = PatternSignal(
                PatternType.PANIC_SELLING,
                confidence=min(panic_ratio, 0.95),
                affected_tickers=panic_tickers,
                avg_drop=avg_change
            )
        elif euphoria_ratio >= self.MIN_TICKERS_PCT:
            pattern = PatternSignal(
                PatternType.EUPHORIA,
                confidence=min(euphoria_ratio, 0.90),
                affected_tickers=euphoria_tickers,
                avg_drop=avg_change
            )
        elif avg_change > 0.02 and self.last_pattern and self.last_pattern.pattern in [PatternType.PANIC_SELLING, PatternType.CAPITULATION]:
            pattern = PatternSignal(PatternType.RECOVERY, confidence=0.7, affected_tickers=[], avg_drop=avg_change)
        else:
            pattern = PatternSignal(PatternType.NORMAL, confidence=0.5, affected_tickers=[], avg_drop=avg_change)
        
        self.last_pattern = pattern
        self.pattern_history.append(pattern)
        return pattern


# ============================================================
# ML SIGNAL ENGINE - Buy/Sell Signal Generation
# ============================================================
class SignalType(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingSignal:
    ticker: str
    signal: SignalType
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


class MLSignalEngine:
    """
    ML-based signal generation using features + sentiment
    Simple rule-based model (can be replaced with real ML)
    """
    
    def __init__(self):
        self.signals_generated = 0
        self.model_version = "rule_based_v1"
    
    async def generate_signal(
        self,
        ticker: str,
        ohlcv: Dict[str, float],
        sentiment: float,
        pattern: PatternType,
        fear_greed: float
    ) -> TradingSignal:
        """
        Generate trading signal based on multiple factors
        
        ohlcv: {'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}
        sentiment: -1.0 to 1.0
        pattern: current market pattern
        fear_greed: 0-100
        """
        close = ohlcv.get('close', 0)
        open_price = ohlcv.get('open', close)
        high = ohlcv.get('high', close)
        low = ohlcv.get('low', close)
        
        if close == 0:
            return TradingSignal(ticker, SignalType.HOLD, 0.0, 0, 0, 0, "No data")
        
        # Calculate indicators
        daily_return = (close - open_price) / open_price if open_price else 0
        volatility = (high - low) / close if close else 0
        
        # Score calculation (-100 to +100)
        score = 0
        reasons = []
        
        # 1. Contrarian: Buy on panic, sell on euphoria
        if pattern == PatternType.CAPITULATION:
            score += 40
            reasons.append("Capitulation detected - contrarian buy")
        elif pattern == PatternType.PANIC_SELLING:
            score += 25
            reasons.append("Panic selling - potential reversal")
        elif pattern == PatternType.EUPHORIA:
            score -= 30
            reasons.append("Euphoria - potential top")
        elif pattern == PatternType.RECOVERY:
            score += 15
            reasons.append("Recovery in progress")
        
        # 2. Fear & Greed (contrarian)
        if fear_greed < 20:
            score += 25
            reasons.append(f"Extreme fear ({fear_greed}) - buy signal")
        elif fear_greed < 35:
            score += 10
            reasons.append(f"Fear zone ({fear_greed})")
        elif fear_greed > 80:
            score -= 25
            reasons.append(f"Extreme greed ({fear_greed}) - sell signal")
        elif fear_greed > 65:
            score -= 10
            reasons.append(f"Greed zone ({fear_greed})")
        
        # 3. Sentiment
        score += int(sentiment * 20)
        if abs(sentiment) > 0.3:
            reasons.append(f"Sentiment: {sentiment:.2f}")
        
        # 4. Technical: oversold/overbought based on daily move
        if daily_return < -0.05:
            score += 15
            reasons.append(f"Oversold today ({daily_return:.1%})")
        elif daily_return > 0.05:
            score -= 10
            reasons.append(f"Overbought today ({daily_return:.1%})")
        
        # Determine signal
        confidence = min(abs(score) / 100, 0.95)
        
        if score >= 50:
            signal = SignalType.STRONG_BUY
        elif score >= 20:
            signal = SignalType.BUY
        elif score <= -50:
            signal = SignalType.STRONG_SELL
        elif score <= -20:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        
        # Calculate targets
        if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            target = close * (1 + 0.05 + volatility)  # 5% + volatility
            stop = close * (1 - 0.03 - volatility/2)  # 3% + half volatility
        elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
            target = close * (1 - 0.05 - volatility)
            stop = close * (1 + 0.03 + volatility/2)
        else:
            target = close
            stop = close * 0.95
        
        self.signals_generated += 1
        
        return TradingSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            entry_price=close,
            target_price=round(target, 2),
            stop_loss=round(stop, 2),
            reasoning=" | ".join(reasons) if reasons else "Neutral conditions"
        )


# ============================================================
# RISK MANAGEMENT - Kelly Criterion, Stop-Loss, Drawdown
# ============================================================
@dataclass
class Position:
    ticker: str
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=datetime.now)


@dataclass 
class RiskMetrics:
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    total_exposure: float
    positions_count: int


class RiskManager:
    """
    Risk management with Kelly criterion and drawdown control
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_position_pct: float = 0.10,
        max_daily_loss_pct: float = 0.02,
        max_drawdown_pct: float = 0.15
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.trades_today = 0
        self.max_trades_per_day = 20
    
    def kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        f* = (p * b - q) / b
        where p = win_rate, q = 1-p, b = win/loss ratio
        """
        if win_loss_ratio <= 0:
            return 0
        
        q = 1 - win_rate
        kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Use fractional Kelly (half) for safety
        kelly = max(0, min(kelly * 0.5, self.max_position_pct))
        return kelly
    
    def calculate_position_size(
        self,
        ticker: str,
        price: float,
        signal_confidence: float,
        win_rate: float = 0.55,
        win_loss_ratio: float = 1.5
    ) -> int:
        """Calculate number of shares to buy"""
        
        # Check daily limits
        if self.trades_today >= self.max_trades_per_day:
            logger.warning(f"Max daily trades reached ({self.max_trades_per_day})")
            return 0
        
        # Check drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= self.max_drawdown_pct:
            logger.warning(f"Max drawdown reached ({current_drawdown:.1%})")
            return 0
        
        # Check daily loss
        if self.daily_pnl <= -self.initial_capital * self.max_daily_loss_pct:
            logger.warning(f"Max daily loss reached ({self.daily_pnl:,.0f})")
            return 0
        
        # Kelly sizing adjusted by confidence
        kelly_pct = self.kelly_criterion(win_rate, win_loss_ratio)
        adjusted_pct = kelly_pct * signal_confidence
        
        # Cap at max position
        position_pct = min(adjusted_pct, self.max_position_pct)
        position_value = self.current_capital * position_pct
        
        if price <= 0:
            return 0
        
        shares = int(position_value / price)
        return shares
    
    def open_position(self, ticker: str, price: float, quantity: int, stop_loss: float, take_profit: float) -> bool:
        """Open a new position"""
        if ticker in self.positions:
            logger.warning(f"Position already exists for {ticker}")
            return False
        
        self.positions[ticker] = Position(
            ticker=ticker,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        self.trades_today += 1
        logger.info(f"Opened {ticker}: {quantity} @ {price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        return True
    
    def check_stops(self, current_prices: Dict[str, float]) -> List[str]:
        """Check and trigger stop-losses and take-profits"""
        closed = []
        
        for ticker, pos in list(self.positions.items()):
            price = current_prices.get(ticker)
            if not price:
                continue
            
            pnl = (price - pos.entry_price) * pos.quantity
            
            # Stop-loss hit
            if price <= pos.stop_loss:
                logger.warning(f"STOP-LOSS {ticker}: {price:.2f} <= {pos.stop_loss:.2f} | PnL: {pnl:,.0f}")
                self.close_position(ticker, price, "stop_loss")
                closed.append(ticker)
            
            # Take-profit hit
            elif price >= pos.take_profit:
                logger.info(f"TAKE-PROFIT {ticker}: {price:.2f} >= {pos.take_profit:.2f} | PnL: {pnl:,.0f}")
                self.close_position(ticker, price, "take_profit")
                closed.append(ticker)
        
        return closed
    
    def close_position(self, ticker: str, price: float, reason: str = "manual") -> float:
        """Close position and return PnL"""
        if ticker not in self.positions:
            return 0
        
        pos = self.positions.pop(ticker)
        pnl = (price - pos.entry_price) * pos.quantity
        
        self.current_capital += pnl
        self.daily_pnl += pnl
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        logger.info(f"Closed {ticker} ({reason}): PnL {pnl:,.0f} | Capital: {self.current_capital:,.0f}")
        return pnl
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        exposure = sum(p.entry_price * p.quantity for p in self.positions.values())
        
        return RiskMetrics(
            current_drawdown=drawdown,
            max_drawdown=self.max_drawdown_pct,
            daily_pnl=self.daily_pnl,
            total_exposure=exposure,
            positions_count=len(self.positions)
        )
    
    def reset_daily(self):
        """Reset daily counters (call at market open)"""
        self.daily_pnl = 0.0
        self.trades_today = 0


# ============================================================
# EXISTING COMPONENTS (Circuit Breaker, Rate Limiter, etc.)
# ============================================================
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit Breaker OPEN (failures: {self.failures})")
    
    def can_execute(self) -> bool:
        if self.state == "CLOSED": return True
        if self.state == "OPEN":
            if (datetime.now() - self.last_failure_time).total_seconds() > self.timeout_seconds:
                self.state = "HALF_OPEN"
                return True
            return False
        return True


class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
    
    def allow_request(self) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False


class ConfigLoader:
    @staticmethod
    def load(config_path: str = "/app/config.yaml") -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return ConfigLoader.defaults()
    
    @staticmethod
    def defaults() -> Dict[str, Any]:
        return {
            'scheduler': {'enabled': True, 'collect_features_cron': '0 */1 * * *', 'calculate_sentiment_cron': '0 * * * *', 'pattern_detection_cron': '*/5 * * * *'},
            'circuit_breaker': {'failure_threshold': 5, 'timeout_seconds': 60},
            'rate_limiter': {'max_requests': 100, 'window_seconds': 60},
            'risk': {'initial_capital': 1000000, 'max_position_pct': 0.10, 'max_daily_loss_pct': 0.02, 'max_drawdown_pct': 0.15}
        }


class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache: return None
        value, ts = self._cache[key]
        if datetime.now() - ts > timedelta(seconds=self._ttl):
            del self._cache[key]
            return None
        return value
    
    def set(self, key: str, value: Any):
        if len(self._cache) >= self._max_size:
            expired = [k for k, (_, ts) in self._cache.items() if datetime.now() - ts > timedelta(seconds=self._ttl)]
            for k in expired: del self._cache[k]
        self._cache[key] = (value, datetime.now())
    
    def clear(self): self._cache.clear()
    def size(self) -> int: return len(self._cache)


async def fetch_with_retry(client: httpx.AsyncClient, url: str, max_retries: int = 3, backoff: float = 1.0) -> Optional[httpx.Response]:
    for attempt in range(max_retries):
        try:
            resp = await client.get(url, timeout=30)
            if resp.status_code < 500: return resp
        except (httpx.TimeoutException, httpx.NetworkError): pass
        if attempt < max_retries - 1:
            await asyncio.sleep(backoff * (2 ** attempt))
    return None


# ============================================================
# PROMETHEUS METRICS
# ============================================================
JOBS_TOTAL = Counter('scheduler_jobs_total', 'Jobs executed', ['job', 'status'])
FEAR_GREED_INDEX = Gauge('fear_greed_index', 'Current Fear & Greed Index')
CIRCUIT_BREAKER_STATE = Gauge('circuit_breaker_state', 'Circuit breaker state', ['service'])
RATE_LIMIT_USAGE = Gauge('rate_limit_usage', 'Rate limit usage %')
SCHEDULER_RUNS = Counter('scheduler_runs_total', 'Scheduler runs', ['job', 'status'])

# New v5.3 metrics
PATTERN_DETECTED = Counter('pattern_detected_total', 'Patterns detected', ['pattern'])
SIGNALS_GENERATED = Counter('signals_generated_total', 'Trading signals', ['signal'])
CURRENT_DRAWDOWN = Gauge('current_drawdown', 'Current portfolio drawdown')
DAILY_PNL = Gauge('daily_pnl', 'Daily profit/loss')
POSITIONS_COUNT = Gauge('positions_count', 'Open positions count')
CAPITAL = Gauge('portfolio_capital', 'Current portfolio capital')


# ============================================================
# KNOWN TICKERS
# ============================================================
KNOWN_TICKERS = {
    'SBER', 'GAZP', 'LKOH', 'GMKN', 'NVTK', 'ROSN', 'VTBR', 'MTSS',
    'MGNT', 'TATN', 'YNDX', 'TCSG', 'NLMK', 'CHMF', 'PLZL', 'ALRS',
    'POLY', 'FIVE', 'IRAO', 'HYDR', 'PHOR', 'RUAL', 'MAGN', 'AFLT',
    'FEES', 'CBOM', 'RTKM', 'MOEX', 'AFKS', 'SNGS', 'OZON'
}


# ============================================================
# ENUMS & DATACLASSES
# ============================================================
class MarketRegime(Enum):
    CRISIS = "crisis"
    SIDEWAYS = "sideways"
    TRENDING_UP = "trending_up"


@dataclass
class TradingConfig:
    min_confidence: float = 0.45
    max_position_pct: float = 0.10
    max_daily_trades: int = 20


@dataclass
class GlobalConfig:
    trading: TradingConfig = field(default_factory=TradingConfig)
    regime: MarketRegime = MarketRegime.SIDEWAYS
    fear_greed: float = 50.0


# ============================================================
# TRADING BRAIN v5.3 - FULL SYSTEM
# ============================================================
class TradingBrain:
    """Main class - v5.3 FULL TRADING SYSTEM"""
    
    def __init__(self):
        self.pg = None
        self.redis = None
        self.http_client = None
        self.config = GlobalConfig()
        self.yaml_config = ConfigLoader.load()
        
        # Caches
        self.pattern_cache = TTLCache(ttl_seconds=300, max_size=100)
        self.feature_cache = TTLCache(ttl_seconds=600, max_size=100)
        
        # Semaphores
        self.fetch_semaphore = asyncio.Semaphore(10)
        self.db_semaphore = asyncio.Semaphore(5)
        
        # v5.2 components
        cb_config = self.yaml_config.get('circuit_breaker', {})
        rl_config = self.yaml_config.get('rate_limiter', {})
        self.circuit_breaker = CircuitBreaker(cb_config.get('failure_threshold', 5), cb_config.get('timeout_seconds', 60))
        self.rate_limiter = RateLimiter(rl_config.get('max_requests', 100), rl_config.get('window_seconds', 60))
        
        # v5.3 NEW components
        risk_config = self.yaml_config.get('risk', {})
        self.pattern_recognition = PatternRecognition()
        self.signal_engine = MLSignalEngine()
        self.risk_manager = RiskManager(
            initial_capital=risk_config.get('initial_capital', 1_000_000),
            max_position_pct=risk_config.get('max_position_pct', 0.10),
            max_daily_loss_pct=risk_config.get('max_daily_loss_pct', 0.02),
            max_drawdown_pct=risk_config.get('max_drawdown_pct', 0.15)
        )
        
        self.scheduler = AsyncIOScheduler()
        self.current_pattern: Optional[PatternSignal] = None
        self.latest_signals: Dict[str, TradingSignal] = {}
    
    async def start(self):
        self.pg = await asyncpg.create_pool(
            os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading"),
            min_size=5, max_size=20, command_timeout=60
        )
        self.redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
        self.http_client = httpx.AsyncClient(timeout=30)
        await self._ensure_tables()
        await self._load_config()
        if self.yaml_config.get('scheduler', {}).get('enabled', True):
            await self._setup_scheduler()
        logger.info("Trading Brain v5.3 FULL SYSTEM started")
    
    async def stop(self):
        if self.scheduler.running: self.scheduler.shutdown()
        if self.pg: await self.pg.close()
        if self.redis: await self.redis.close()
        if self.http_client: await self.http_client.aclose()
        self.pattern_cache.clear()
        self.feature_cache.clear()
        logger.info("Trading Brain stopped")
    
    async def _ensure_tables(self):
        async with self.pg.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    date DATE, ticker VARCHAR(20), open FLOAT, high FLOAT, low FLOAT, close FLOAT,
                    volume FLOAT, updated_at TIMESTAMPTZ DEFAULT NOW(), PRIMARY KEY (date, ticker)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, ticker VARCHAR(20), signal VARCHAR(20),
                    confidence FLOAT, entry_price FLOAT, target_price FLOAT, stop_loss FLOAT,
                    reasoning TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id SERIAL PRIMARY KEY, pattern VARCHAR(30), confidence FLOAT,
                    affected_tickers TEXT, avg_change FLOAT, created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_features_ticker_date ON features(ticker, date DESC)")
        logger.info("Database tables ready")
    
    async def _load_config(self):
        try:
            data = await self.redis.get("global_config")
            if data:
                d = json.loads(data)
                self.config.regime = MarketRegime(d.get('regime', 'sideways'))
                self.config.fear_greed = d.get('fear_greed', 50)
        except Exception as e:
            logger.warning(f"Config load failed: {e}")
    
    async def _setup_scheduler(self):
        sched = self.yaml_config.get('scheduler', {})
        self.scheduler.add_job(self.collect_features_parallel, CronTrigger.from_crontab(sched.get('collect_features_cron', '0 */1 * * *')), id='collect_features', misfire_grace_time=10)
        self.scheduler.add_job(self.calculate_fear_greed, CronTrigger.from_crontab(sched.get('calculate_sentiment_cron', '0 * * * *')), id='calculate_sentiment', misfire_grace_time=10)
        self.scheduler.add_job(self.detect_patterns, CronTrigger.from_crontab(sched.get('pattern_detection_cron', '*/5 * * * *')), id='detect_patterns', misfire_grace_time=10)
        self.scheduler.add_job(self.generate_signals, CronTrigger.from_crontab('*/10 * * * *'), id='generate_signals', misfire_grace_time=10)
        self.scheduler.add_job(self.check_risk, CronTrigger.from_crontab('* * * * *'), id='check_risk', misfire_grace_time=10)
        self.scheduler.start()
        logger.info("APScheduler started with 5 jobs")
    
    async def collect_features_parallel(self):
        """Collect market data"""
        if not self.circuit_breaker.can_execute():
            SCHEDULER_RUNS.labels(job='collect_features', status='skipped').inc()
            return
        
        async def fetch_ticker(ticker: str):
            async with self.fetch_semaphore:
                if not self.rate_limiter.allow_request(): return ticker, None
                try:
                    resp = await fetch_with_retry(self.http_client, f"http://datafeed:8006/history/{ticker}?days=5")
                    self.circuit_breaker.record_success()
                    return ticker, resp.json() if resp else None
                except:
                    self.circuit_breaker.record_failure()
                    return ticker, None
        
        try:
            results = await asyncio.gather(*[fetch_ticker(t) for t in KNOWN_TICKERS])
            async with self.pg.acquire() as conn:
                batch = []
                for ticker, data in results:
                    if not data: continue
                    for c in data:
                        batch.append((c.get('date'), ticker, to_float(c.get('open')), to_float(c.get('high')), to_float(c.get('low')), to_float(c.get('close')), to_float(c.get('volume'))))
                if batch:
                    await conn.executemany("INSERT INTO features (date,ticker,open,high,low,close,volume,updated_at) VALUES ($1,$2,$3,$4,$5,$6,$7,NOW()) ON CONFLICT (date,ticker) DO UPDATE SET close=EXCLUDED.close,volume=EXCLUDED.volume", batch)
            SCHEDULER_RUNS.labels(job='collect_features', status='success').inc()
        except Exception as e:
            SCHEDULER_RUNS.labels(job='collect_features', status='failed').inc()
            logger.error(f"Collect features: {e}")
    
    async def calculate_fear_greed(self) -> Dict[str, Any]:
        """Calculate Fear & Greed Index"""
        value = 50.0
        try:
            async with self.pg.acquire() as conn:
                row = await conn.fetchrow("SELECT AVG(CASE WHEN close > 0 THEN 1 ELSE 0 END) as ratio FROM features WHERE date >= CURRENT_DATE - 5")
                if row and row['ratio']: value = to_float(row['ratio'], 0.5) * 100
            FEAR_GREED_INDEX.set(value)
            self.config.fear_greed = value
            SCHEDULER_RUNS.labels(job='calculate_sentiment', status='success').inc()
        except Exception as e:
            logger.warning(f"F&G error: {e}")
            SCHEDULER_RUNS.labels(job='calculate_sentiment', status='failed').inc()
        return {'value': round(value, 1), 'emotion': 'fear' if value < 40 else 'greed' if value > 60 else 'neutral'}
    
    async def detect_patterns(self) -> Dict[str, Any]:
        """Detect market patterns (panic, capitulation, etc.)"""
        try:
            async with self.pg.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT ticker, 
                           (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(LAG(close) OVER (PARTITION BY ticker ORDER BY date), 0) as pct_change
                    FROM features WHERE date >= CURRENT_DATE - 1
                """)
                price_changes = {r['ticker']: float(r['pct_change'] or 0) for r in rows if r['pct_change'] is not None}
            
            pattern = await self.pattern_recognition.detect(price_changes)
            self.current_pattern = pattern
            
            PATTERN_DETECTED.labels(pattern=pattern.pattern.value).inc()
            
            # Save to DB
            async with self.pg.acquire() as conn:
                await conn.execute(
                    "INSERT INTO patterns (pattern, confidence, affected_tickers, avg_change) VALUES ($1, $2, $3, $4)",
                    pattern.pattern.value, pattern.confidence, ','.join(pattern.affected_tickers[:10]), pattern.avg_drop
                )
            
            # Update regime based on pattern
            if pattern.pattern == PatternType.CAPITULATION:
                self.config.regime = MarketRegime.CRISIS
            elif pattern.pattern == PatternType.RECOVERY:
                self.config.regime = MarketRegime.TRENDING_UP
            
            SCHEDULER_RUNS.labels(job='detect_patterns', status='success').inc()
            logger.info(f"Pattern: {pattern.pattern.value} (confidence: {pattern.confidence:.2f})")
            
            return {'pattern': pattern.pattern.value, 'confidence': pattern.confidence, 'affected': len(pattern.affected_tickers), 'avg_change': f"{pattern.avg_drop:.2%}"}
        except Exception as e:
            SCHEDULER_RUNS.labels(job='detect_patterns', status='failed').inc()
            logger.error(f"Pattern detection: {e}")
            return {'error': str(e)}
    
    async def generate_signals(self) -> Dict[str, Any]:
        """Generate trading signals for all tickers"""
        signals = []
        try:
            # Reload config from Redis
            await self._load_config()
            pattern = self.current_pattern.pattern if self.current_pattern else PatternType.NORMAL
            
            async with self.pg.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (ticker) ticker, open, high, low, close, volume
                    FROM features ORDER BY ticker, date DESC
                """)
            
            for row in rows:
                ticker = row['ticker']
                ohlcv = {'open': float(row['open'] or 0), 'high': float(row['high'] or 0), 'low': float(row['low'] or 0), 'close': float(row['close'] or 0), 'volume': float(row['volume'] or 0)}
                
                signal = await self.signal_engine.generate_signal(
                    ticker=ticker,
                    ohlcv=ohlcv,
                    sentiment=random.uniform(-0.3, 0.3),  # Replace with real sentiment
                    pattern=pattern,
                    fear_greed=self.config.fear_greed
                )
                
                self.latest_signals[ticker] = signal
                
                if signal.signal != SignalType.HOLD:
                    signals.append(signal)
                    SIGNALS_GENERATED.labels(signal=signal.signal.value).inc()
                    
                    # Save to DB
                    async with self.pg.acquire() as conn:
                        await conn.execute(
                            "INSERT INTO signals (ticker,signal,confidence,entry_price,target_price,stop_loss,reasoning) VALUES ($1,$2,$3,$4,$5,$6,$7)",
                            ticker, signal.signal.value, signal.confidence, signal.entry_price, signal.target_price, signal.stop_loss, signal.reasoning
                        )
            
            SCHEDULER_RUNS.labels(job='generate_signals', status='success').inc()
            logger.info(f"Generated {len(signals)} actionable signals")
            return {'total': len(rows), 'actionable': len(signals), 'signals': [{'ticker': s.ticker, 'signal': s.signal.value, 'confidence': s.confidence} for s in signals[:5]]}
        except Exception as e:
            SCHEDULER_RUNS.labels(job='generate_signals', status='failed').inc()
            logger.error(f"Signal generation: {e}")
            return {'error': str(e)}
    
    async def check_risk(self) -> Dict[str, Any]:
        """Check risk metrics and stop-losses"""
        try:
            # Get current prices
            async with self.pg.acquire() as conn:
                rows = await conn.fetch("SELECT DISTINCT ON (ticker) ticker, close FROM features ORDER BY ticker, date DESC")
            prices = {r['ticker']: r['close'] for r in rows}
            
            # Check stops
            closed = self.risk_manager.check_stops(prices)
            
            # Update metrics
            metrics = self.risk_manager.get_metrics()
            CURRENT_DRAWDOWN.set(metrics.current_drawdown)
            DAILY_PNL.set(metrics.daily_pnl)
            POSITIONS_COUNT.set(metrics.positions_count)
            CAPITAL.set(self.risk_manager.current_capital)
            
            SCHEDULER_RUNS.labels(job='check_risk', status='success').inc()
            return asdict(metrics)
        except Exception as e:
            SCHEDULER_RUNS.labels(job='check_risk', status='failed').inc()
            return {'error': str(e)}
    
    async def execute_signal(self, ticker: str) -> Dict[str, Any]:
        """Execute a trading signal"""
        if ticker not in self.latest_signals:
            return {'error': 'No signal for ticker'}
        
        signal = self.latest_signals[ticker]
        if signal.signal == SignalType.HOLD:
            return {'error': 'Signal is HOLD'}
        
        # Calculate position size
        shares = self.risk_manager.calculate_position_size(
            ticker, signal.entry_price, signal.confidence
        )
        
        if shares == 0:
            return {'error': 'Risk limits prevent trade'}
        
        # Open position (for BUY signals)
        if signal.signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            self.risk_manager.open_position(ticker, signal.entry_price, shares, signal.stop_loss, signal.target_price)
            return {'status': 'executed', 'ticker': ticker, 'shares': shares, 'entry': signal.entry_price}
        
        return {'status': 'signal_logged', 'ticker': ticker, 'signal': signal.signal.value}
    
    async def health_check(self) -> Dict[str, Any]:
        db_ok = redis_ok = False
        try:
            async with self.pg.acquire() as conn: await conn.fetchval("SELECT 1")
            db_ok = True
        except: pass
        try:
            await self.redis.ping()
            redis_ok = True
        except: pass
        
        metrics = self.risk_manager.get_metrics()
        CIRCUIT_BREAKER_STATE.labels(service='datafeed').set({"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}.get(self.circuit_breaker.state, -1))
        RATE_LIMIT_USAGE.set((len(self.rate_limiter.requests) / self.rate_limiter.max_requests) * 100)
        
        return {
            'status': 'healthy' if db_ok and redis_ok else 'degraded',
            'version': '5.3.0',
            'db': 'ok' if db_ok else 'error',
            'redis': 'ok' if redis_ok else 'error',
            'scheduler': 'running' if self.scheduler.running else 'stopped',
            'circuit_breaker': self.circuit_breaker.state,
            'current_pattern': self.current_pattern.pattern.value if self.current_pattern else 'none',
            'risk': asdict(metrics),
            'signals_count': len([s for s in self.latest_signals.values() if s.signal != SignalType.HOLD])
        }


# ============================================================
# FASTAPI APP
# ============================================================
brain = TradingBrain()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await brain.start()
    yield
    await brain.stop()


app = FastAPI(title="Trading Brain", version="5.3.0", lifespan=lifespan)

# Include backtest routes
backtest_engine, auto_execute_engine = get_engines()

@app.post("/backtest")
async def run_backtest(start_date: str = "2025-01-01", end_date: str = "2025-12-31", tickers: str = None):
    ticker_list = tickers.split(',') if tickers else None
    result = await backtest_engine.run(brain.pg, start_date, end_date, ticker_list)
    from dataclasses import asdict
    return asdict(result)

@app.get("/backtest/trades")
async def get_backtest_trades():
    return {'trades': backtest_engine.trades[-50:], 'total': len(backtest_engine.trades)}

@app.get("/backtest/equity")
async def get_equity_curve():
    return {'equity': backtest_engine.equity_curve[-100:], 'total_points': len(backtest_engine.equity_curve)}

@app.get("/auto-execute")
async def get_auto_execute_status():
    return auto_execute_engine.get_status()

@app.post("/auto-execute/enable")
async def enable_auto_execute(mode: str = "paper"):
    auto_execute_engine.enabled = True
    auto_execute_engine.mode = mode
    return {'status': 'enabled', 'mode': mode}

@app.post("/auto-execute/disable")
async def disable_auto_execute():
    auto_execute_engine.enabled = False
    return {'status': 'disabled'}

@app.post("/auto-execute/run")
async def run_auto_execute():
    async with brain.pg.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT ON (ticker) ticker, close FROM features ORDER BY ticker, date DESC")
    prices = {r['ticker']: float(r['close']) for r in rows}
    results = await auto_execute_engine.execute(brain.latest_signals, brain.risk_manager, prices)
    return {'executed': len(results), 'orders': results}

@app.get("/auto-execute/orders")
async def get_executed_orders():
    return {'orders': auto_execute_engine.executed_orders[-50:], 'total': len(auto_execute_engine.executed_orders)}


@app.get("/health")
async def health(): return await brain.health_check()

@app.get("/config")
async def get_config(): return {'regime': brain.config.regime.value, 'fear_greed': brain.config.fear_greed, 'trading': asdict(brain.config.trading)}

@app.get("/sentiment")
async def get_sentiment(): return await brain.calculate_fear_greed()

@app.get("/metrics")
async def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/patterns")
async def get_patterns():
    if brain.current_pattern:
        return {'pattern': brain.current_pattern.pattern.value, 'confidence': brain.current_pattern.confidence, 'affected_tickers': brain.current_pattern.affected_tickers[:10], 'avg_change': f"{brain.current_pattern.avg_drop:.2%}"}
    return {'pattern': 'none'}

@app.post("/patterns/detect")
async def trigger_patterns(): return await brain.detect_patterns()

@app.get("/signals")
async def get_signals():
    actionable = [s for s in brain.latest_signals.values() if s.signal != SignalType.HOLD]
    return {'total': len(brain.latest_signals), 'actionable': len(actionable), 'signals': [{'ticker': s.ticker, 'signal': s.signal.value, 'confidence': round(s.confidence, 2), 'entry': s.entry_price, 'target': s.target_price, 'stop': s.stop_loss, 'reasoning': s.reasoning} for s in sorted(actionable, key=lambda x: x.confidence, reverse=True)[:10]]}

@app.post("/signals/generate")
async def trigger_signals(): return await brain.generate_signals()

@app.get("/signals/{ticker}")
async def get_signal(ticker: str):
    ticker = ticker.upper()
    if ticker in brain.latest_signals:
        s = brain.latest_signals[ticker]
        return {'ticker': s.ticker, 'signal': s.signal.value, 'confidence': s.confidence, 'entry': s.entry_price, 'target': s.target_price, 'stop': s.stop_loss, 'reasoning': s.reasoning}
    return {'error': 'No signal'}

@app.post("/signals/{ticker}/execute")
async def execute_signal(ticker: str): return await brain.execute_signal(ticker.upper())

@app.get("/risk")
async def get_risk(): return asdict(brain.risk_manager.get_metrics())

@app.get("/positions")
async def get_positions():
    return {'count': len(brain.risk_manager.positions), 'positions': [{'ticker': p.ticker, 'entry': p.entry_price, 'qty': p.quantity, 'stop': p.stop_loss, 'target': p.take_profit} for p in brain.risk_manager.positions.values()]}

@app.post("/collect-features")
async def trigger_features():
    asyncio.create_task(brain.collect_features_parallel())
    return {'status': 'collecting'}

@app.get("/scheduler/status")
async def scheduler_status():
    jobs = [{'id': j.id, 'next': str(j.next_run_time)} for j in brain.scheduler.get_jobs()] if brain.scheduler.running else []
    return {'running': brain.scheduler.running, 'jobs': jobs}

@app.get("/circuit-breaker")
async def cb_status(): return {'state': brain.circuit_breaker.state, 'failures': brain.circuit_breaker.failures}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
