#!/bin/bash
set -e
echo "📈 TRADING-AUTOPILOT: Исправление торговых рисков"
echo "=================================================="

# ============================================
# 1. MARKET REGIME DETECTOR
# ============================================
echo "[1/4] 🎯 Создание Market Regime Detector..."

mkdir -p services/shared/trading

cat > services/shared/trading/regime_detector.py << 'EOF'
"""
Market Regime Detection System
Определяет текущий режим рынка: TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"

@dataclass
class RegimeConfig:
    # ADX thresholds
    adx_trending_threshold: float = 25.0
    adx_strong_trend_threshold: float = 40.0
    
    # Volatility thresholds (в процентах от средней)
    volatility_high_threshold: float = 1.5  # 150% от средней
    volatility_low_threshold: float = 0.5   # 50% от средней
    volatility_crisis_threshold: float = 2.5 # 250% от средней
    
    # Hurst exponent thresholds
    hurst_trending_threshold: float = 0.6
    hurst_mean_reverting_threshold: float = 0.4
    
    # Lookback periods
    short_period: int = 14
    medium_period: int = 50
    long_period: int = 200
    volatility_period: int = 20

@dataclass
class RegimeAnalysis:
    regime: MarketRegime
    confidence: float  # 0-1
    adx: float
    volatility_ratio: float
    hurst_exponent: float
    trend_strength: float
    recommendation: str
    position_size_multiplier: float

class MarketRegimeDetector:
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        if len(close) < period + 1:
            return 0.0
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = self._ema(tr, period)
        plus_di = 100 * self._ema(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * self._ema(minus_dm, period) / (atr + 1e-10)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, period)
        
        return float(adx[-1]) if len(adx) > 0 else 0.0
    
    def calculate_hurst_exponent(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """
        Calculate Hurst Exponent using R/S analysis
        H > 0.5: Trending (persistent)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting (anti-persistent)
        """
        if len(prices) < max_lag * 2:
            return 0.5
        
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Calculate log returns
            returns = np.log(prices[lag:] / prices[:-lag])
            
            # R/S statistic
            mean_return = np.mean(returns)
            deviations = returns - mean_return
            cumulative_deviations = np.cumsum(deviations)
            
            r = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            s = np.std(returns) + 1e-10
            
            tau.append(r / s)
        
        # Linear regression of log(R/S) vs log(lag)
        log_lags = np.log(list(lags))
        log_tau = np.log(np.array(tau) + 1e-10)
        
        # Hurst exponent is the slope
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        
        return float(np.clip(slope, 0, 1))
    
    def calculate_volatility_ratio(self, returns: np.ndarray, period: int = 20) -> float:
        """
        Calculate current volatility vs historical average
        """
        if len(returns) < period * 5:
            return 1.0
        
        current_vol = np.std(returns[-period:])
        historical_vol = np.std(returns[:-period])
        
        return float(current_vol / (historical_vol + 1e-10))
    
    def calculate_trend_strength(self, close: np.ndarray) -> Tuple[float, str]:
        """
        Calculate trend strength using multiple moving averages
        Returns: (strength -1 to 1, direction)
        """
        if len(close) < self.config.long_period:
            return 0.0, "neutral"
        
        sma_short = np.mean(close[-self.config.short_period:])
        sma_medium = np.mean(close[-self.config.medium_period:])
        sma_long = np.mean(close[-self.config.long_period:])
        current_price = close[-1]
        
        # Normalized distances
        dist_short = (current_price - sma_short) / (sma_short + 1e-10)
        dist_medium = (current_price - sma_medium) / (sma_medium + 1e-10)
        dist_long = (current_price - sma_long) / (sma_long + 1e-10)
        
        # Weighted average
        strength = 0.5 * dist_short + 0.3 * dist_medium + 0.2 * dist_long
        
        # MA alignment bonus
        if sma_short > sma_medium > sma_long:
            strength = min(1.0, strength + 0.2)
            direction = "up"
        elif sma_short < sma_medium < sma_long:
            strength = max(-1.0, strength - 0.2)
            direction = "down"
        else:
            direction = "neutral"
        
        return float(np.clip(strength, -1, 1)), direction
    
    def detect_regime(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray,
        volume: Optional[np.ndarray] = None
    ) -> RegimeAnalysis:
        """
        Main function to detect current market regime
        """
        if len(close) < self.config.long_period:
            return RegimeAnalysis(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                adx=0.0,
                volatility_ratio=1.0,
                hurst_exponent=0.5,
                trend_strength=0.0,
                recommendation="Insufficient data",
                position_size_multiplier=0.5
            )
        
        # Calculate indicators
        returns = np.diff(np.log(close))
        adx = self.calculate_adx(high, low, close, self.config.short_period)
        hurst = self.calculate_hurst_exponent(close)
        vol_ratio = self.calculate_volatility_ratio(returns, self.config.volatility_period)
        trend_strength, trend_dir = self.calculate_trend_strength(close)
        
        # Determine regime
        regime = MarketRegime.UNKNOWN
        confidence = 0.5
        recommendation = ""
        size_multiplier = 1.0
        
        # Crisis detection (highest priority)
        if vol_ratio > self.config.volatility_crisis_threshold:
            regime = MarketRegime.CRISIS
            confidence = min(1.0, vol_ratio / 3.0)
            recommendation = "REDUCE ALL POSITIONS - Crisis volatility detected"
            size_multiplier = 0.25
        
        # High volatility
        elif vol_ratio > self.config.volatility_high_threshold:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(1.0, (vol_ratio - 1.0) / 1.0)
            recommendation = "Reduce position sizes, wider stops"
            size_multiplier = 0.5
        
        # Low volatility
        elif vol_ratio < self.config.volatility_low_threshold:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(1.0, (1.0 - vol_ratio) / 0.5)
            recommendation = "Increase position sizes, tighter stops"
            size_multiplier = 1.25
        
        # Trending market
        elif adx > self.config.adx_trending_threshold and hurst > self.config.hurst_trending_threshold:
            if trend_dir == "up":
                regime = MarketRegime.TRENDING_UP
                recommendation = "Use trend-following strategies, trail stops"
            else:
                regime = MarketRegime.TRENDING_DOWN
                recommendation = "Use trend-following strategies, consider shorts"
            
            confidence = min(1.0, adx / 50.0)
            size_multiplier = 1.0 + (confidence * 0.25)  # Up to 1.25x
        
        # Mean-reverting market
        elif hurst < self.config.hurst_mean_reverting_threshold:
            regime = MarketRegime.MEAN_REVERTING
            confidence = min(1.0, (0.5 - hurst) / 0.2)
            recommendation = "Use mean-reversion strategies, fade extremes"
            size_multiplier = 0.75
        
        else:
            regime = MarketRegime.UNKNOWN
            confidence = 0.3
            recommendation = "Mixed signals, use caution"
            size_multiplier = 0.75
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            adx=adx,
            volatility_ratio=vol_ratio,
            hurst_exponent=hurst,
            trend_strength=trend_strength,
            recommendation=recommendation,
            position_size_multiplier=size_multiplier
        )
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema


# Singleton instance
_detector = MarketRegimeDetector()

def detect_market_regime(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> RegimeAnalysis:
    """Convenience function for regime detection"""
    return _detector.detect_regime(high, low, close)
EOF

echo "   ✅ Market Regime Detector создан"

# ============================================
# 2. VOLATILITY-BASED POSITION SIZING
# ============================================
echo "[2/4] 📊 Создание Position Sizing Engine..."

cat > services/shared/trading/position_sizing.py << 'EOF'
"""
Volatility-Based Position Sizing System
Адаптивный расчёт размера позиции на основе волатильности и риска
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_TARGET = "volatility_target"
    ATR_BASED = "atr_based"
    KELLY_CRITERION = "kelly_criterion"
    OPTIMAL_F = "optimal_f"

@dataclass
class PositionSizeConfig:
    # Risk parameters
    max_risk_per_trade: float = 0.02      # 2% of capital per trade
    max_position_size: float = 0.10       # 10% of capital max
    min_position_size: float = 0.01       # 1% of capital min
    
    # Volatility targeting
    target_volatility: float = 0.15       # 15% annual volatility target
    volatility_lookback: int = 20         # Days for volatility calculation
    
    # ATR parameters
    atr_period: int = 14
    atr_multiplier: float = 2.0           # Stop distance in ATRs
    
    # Kelly parameters
    kelly_fraction: float = 0.25          # Use 25% of Kelly (quarter Kelly)
    
    # Scaling
    scale_with_confidence: bool = True
    min_confidence: float = 0.3

@dataclass 
class PositionSizeResult:
    shares: int
    position_value: float
    position_pct: float
    risk_amount: float
    stop_loss_price: float
    take_profit_price: Optional[float]
    method_used: SizingMethod
    volatility_adjusted: bool
    sizing_details: Dict

class PositionSizingEngine:
    def __init__(self, config: PositionSizeConfig = None):
        self.config = config or PositionSizeConfig()
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(close) < period + 1:
            return 0.0
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = np.mean(tr[-period:])
        return float(atr)
    
    def calculate_volatility(self, close: np.ndarray, period: int = 20, annualize: bool = True) -> float:
        """Calculate historical volatility"""
        if len(close) < period + 1:
            return 0.2  # Default 20% vol
        
        returns = np.diff(np.log(close[-period-1:]))
        vol = np.std(returns)
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize
        
        return float(vol)
    
    def fixed_fractional(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float
    ) -> PositionSizeResult:
        """
        Fixed Fractional Position Sizing
        Risk fixed % of capital per trade
        """
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            risk_per_share = entry_price * 0.02  # Default 2% stop
        
        risk_amount = capital * self.config.max_risk_per_trade
        shares = int(risk_amount / risk_per_share)
        
        # Apply limits
        position_value = shares * entry_price
        max_value = capital * self.config.max_position_size
        min_value = capital * self.config.min_position_size
        
        if position_value > max_value:
            shares = int(max_value / entry_price)
            position_value = shares * entry_price
        elif position_value < min_value:
            shares = int(min_value / entry_price)
            position_value = shares * entry_price
        
        return PositionSizeResult(
            shares=max(1, shares),
            position_value=position_value,
            position_pct=position_value / capital,
            risk_amount=shares * risk_per_share,
            stop_loss_price=stop_loss_price,
            take_profit_price=entry_price + (entry_price - stop_loss_price) * 2,  # 2:1 R/R
            method_used=SizingMethod.FIXED_FRACTIONAL,
            volatility_adjusted=False,
            sizing_details={"risk_per_share": risk_per_share}
        )
    
    def volatility_target(
        self,
        capital: float,
        entry_price: float,
        close: np.ndarray,
        target_vol: float = None
    ) -> PositionSizeResult:
        """
        Volatility Targeting Position Sizing
        Adjust position size to achieve target portfolio volatility
        """
        target_vol = target_vol or self.config.target_volatility
        asset_vol = self.calculate_volatility(close, self.config.volatility_lookback)
        
        # Target position = (Target Vol / Asset Vol)
        vol_scalar = target_vol / (asset_vol + 1e-10)
        vol_scalar = np.clip(vol_scalar, 0.1, 3.0)  # Limit scaling
        
        # Base position
        base_position_pct = self.config.max_position_size * vol_scalar
        base_position_pct = np.clip(
            base_position_pct, 
            self.config.min_position_size, 
            self.config.max_position_size
        )
        
        position_value = capital * base_position_pct
        shares = int(position_value / entry_price)
        
        # Calculate stop based on volatility
        daily_vol = asset_vol / np.sqrt(252)
        stop_distance = entry_price * daily_vol * 2  # 2 sigma stop
        stop_loss_price = entry_price - stop_distance
        
        return PositionSizeResult(
            shares=max(1, shares),
            position_value=shares * entry_price,
            position_pct=(shares * entry_price) / capital,
            risk_amount=shares * stop_distance,
            stop_loss_price=stop_loss_price,
            take_profit_price=entry_price + stop_distance * 2,
            method_used=SizingMethod.VOLATILITY_TARGET,
            volatility_adjusted=True,
            sizing_details={
                "asset_volatility": asset_vol,
                "target_volatility": target_vol,
                "vol_scalar": vol_scalar
            }
        )
    
    def atr_based(
        self,
        capital: float,
        entry_price: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr_multiplier: float = None
    ) -> PositionSizeResult:
        """
        ATR-Based Position Sizing
        Use ATR for stop distance and position sizing
        """
        atr = self.calculate_atr(high, low, close, self.config.atr_period)
        multiplier = atr_multiplier or self.config.atr_multiplier
        
        if atr == 0:
            atr = entry_price * 0.02  # Default 2%
        
        stop_distance = atr * multiplier
        stop_loss_price = entry_price - stop_distance
        
        # Risk amount
        risk_amount = capital * self.config.max_risk_per_trade
        shares = int(risk_amount / stop_distance)
        
        # Apply limits
        position_value = shares * entry_price
        max_value = capital * self.config.max_position_size
        
        if position_value > max_value:
            shares = int(max_value / entry_price)
        
        return PositionSizeResult(
            shares=max(1, shares),
            position_value=shares * entry_price,
            position_pct=(shares * entry_price) / capital,
            risk_amount=shares * stop_distance,
            stop_loss_price=stop_loss_price,
            take_profit_price=entry_price + stop_distance * 3,  # 3:1 R/R
            method_used=SizingMethod.ATR_BASED,
            volatility_adjusted=True,
            sizing_details={
                "atr": atr,
                "atr_multiplier": multiplier,
                "stop_distance": stop_distance
            }
        )
    
    def kelly_criterion(
        self,
        capital: float,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        close: np.ndarray = None
    ) -> PositionSizeResult:
        """
        Kelly Criterion Position Sizing
        Optimal sizing based on historical performance
        """
        # Kelly formula: f* = (p*b - q) / b
        # where p = win rate, q = 1-p, b = avg_win/avg_loss
        
        if avg_loss == 0:
            avg_loss = 0.01
        
        b = avg_win / avg_loss  # Win/Loss ratio
        p = win_rate
        q = 1 - p
        
        kelly_pct = (p * b - q) / b
        kelly_pct = max(0, kelly_pct)  # Can't be negative
        
        # Use fractional Kelly (safer)
        adjusted_kelly = kelly_pct * self.config.kelly_fraction
        adjusted_kelly = np.clip(
            adjusted_kelly,
            self.config.min_position_size,
            self.config.max_position_size
        )
        
        position_value = capital * adjusted_kelly
        shares = int(position_value / entry_price)
        
        # Default stop at avg_loss
        stop_loss_price = entry_price * (1 - avg_loss)
        
        return PositionSizeResult(
            shares=max(1, shares),
            position_value=shares * entry_price,
            position_pct=(shares * entry_price) / capital,
            risk_amount=shares * entry_price * avg_loss,
            stop_loss_price=stop_loss_price,
            take_profit_price=entry_price * (1 + avg_win),
            method_used=SizingMethod.KELLY_CRITERION,
            volatility_adjusted=False,
            sizing_details={
                "full_kelly": kelly_pct,
                "adjusted_kelly": adjusted_kelly,
                "win_rate": win_rate,
                "win_loss_ratio": b
            }
        )
    
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        method: SizingMethod = SizingMethod.ATR_BASED,
        stop_loss_price: float = None,
        high: np.ndarray = None,
        low: np.ndarray = None,
        close: np.ndarray = None,
        regime_multiplier: float = 1.0,
        confidence: float = 1.0,
        **kwargs
    ) -> PositionSizeResult:
        """
        Main function to calculate position size
        """
        # Choose method
        if method == SizingMethod.FIXED_FRACTIONAL:
            if stop_loss_price is None:
                stop_loss_price = entry_price * 0.98
            result = self.fixed_fractional(capital, entry_price, stop_loss_price)
        
        elif method == SizingMethod.VOLATILITY_TARGET:
            if close is None:
                raise ValueError("Close prices required for volatility targeting")
            result = self.volatility_target(capital, entry_price, close)
        
        elif method == SizingMethod.ATR_BASED:
            if high is None or low is None or close is None:
                raise ValueError("OHLC data required for ATR sizing")
            result = self.atr_based(capital, entry_price, high, low, close)
        
        elif method == SizingMethod.KELLY_CRITERION:
            win_rate = kwargs.get('win_rate', 0.5)
            avg_win = kwargs.get('avg_win', 0.03)
            avg_loss = kwargs.get('avg_loss', 0.02)
            result = self.kelly_criterion(capital, entry_price, win_rate, avg_win, avg_loss)
        
        else:
            # Default to fixed fractional
            stop_loss_price = stop_loss_price or entry_price * 0.98
            result = self.fixed_fractional(capital, entry_price, stop_loss_price)
        
        # Apply regime multiplier
        if regime_multiplier != 1.0:
            result.shares = int(result.shares * regime_multiplier)
            result.position_value = result.shares * entry_price
            result.position_pct = result.position_value / capital
            result.sizing_details['regime_multiplier'] = regime_multiplier
        
        # Apply confidence scaling
        if self.config.scale_with_confidence and confidence < 1.0:
            if confidence >= self.config.min_confidence:
                conf_scalar = 0.5 + (confidence * 0.5)  # 50-100% of position
                result.shares = int(result.shares * conf_scalar)
                result.position_value = result.shares * entry_price
                result.position_pct = result.position_value / capital
                result.sizing_details['confidence_scalar'] = conf_scalar
        
        return result


# Singleton instance
_engine = PositionSizingEngine()

def calculate_position(
    capital: float,
    entry_price: float,
    method: str = "atr",
    **kwargs
) -> PositionSizeResult:
    """Convenience function for position sizing"""
    method_map = {
        "fixed": SizingMethod.FIXED_FRACTIONAL,
        "volatility": SizingMethod.VOLATILITY_TARGET,
        "atr": SizingMethod.ATR_BASED,
        "kelly": SizingMethod.KELLY_CRITERION
    }
    return _engine.calculate_position_size(
        capital, entry_price, 
        method=method_map.get(method, SizingMethod.ATR_BASED),
        **kwargs
    )
EOF

echo "   ✅ Position Sizing Engine создан"

# ============================================
# 3. CORRELATION ANALYZER
# ============================================
echo "[3/4] 🔗 Создание Correlation Analyzer..."

cat > services/shared/trading/correlation_analyzer.py << 'EOF'
"""
Portfolio Correlation Analysis System
Анализ корреляций между позициями для контроля концентрации риска
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CorrelationLevel(Enum):
    NEGATIVE = "negative"        # < -0.3
    LOW = "low"                  # -0.3 to 0.3
    MODERATE = "moderate"        # 0.3 to 0.6
    HIGH = "high"                # 0.6 to 0.8
    VERY_HIGH = "very_high"      # > 0.8

@dataclass
class CorrelationConfig:
    lookback_period: int = 60           # Days for correlation calculation
    high_correlation_threshold: float = 0.7
    max_correlated_exposure: float = 0.20  # 20% max in highly correlated assets
    rebalance_threshold: float = 0.15   # Rebalance if correlation changes by 15%
    use_rolling: bool = True
    rolling_window: int = 20

@dataclass
class AssetCorrelation:
    asset1: str
    asset2: str
    correlation: float
    level: CorrelationLevel
    beta: float
    covariance: float

@dataclass
class PortfolioCorrelationReport:
    average_correlation: float
    max_correlation: float
    correlation_matrix: Dict[str, Dict[str, float]]
    high_correlation_pairs: List[AssetCorrelation]
    diversification_ratio: float
    effective_n_assets: float
    concentration_risk: str
    recommendations: List[str]

class CorrelationAnalyzer:
    def __init__(self, config: CorrelationConfig = None):
        self.config = config or CorrelationConfig()
        self._correlation_cache: Dict[Tuple[str, str], float] = {}
    
    def calculate_correlation(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray,
        method: str = "pearson"
    ) -> float:
        """Calculate correlation between two return series"""
        if len(returns1) != len(returns2):
            min_len = min(len(returns1), len(returns2))
            returns1 = returns1[-min_len:]
            returns2 = returns2[-min_len:]
        
        if len(returns1) < 10:
            return 0.0
        
        if method == "pearson":
            corr = np.corrcoef(returns1, returns2)[0, 1]
        elif method == "spearman":
            # Rank correlation
            rank1 = np.argsort(np.argsort(returns1))
            rank2 = np.argsort(np.argsort(returns2))
            corr = np.corrcoef(rank1, rank2)[0, 1]
        else:
            corr = np.corrcoef(returns1, returns2)[0, 1]
        
        return float(corr) if not np.isnan(corr) else 0.0
    
    def calculate_rolling_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """Calculate rolling correlation"""
        n = len(returns1)
        if n < window:
            return np.array([self.calculate_correlation(returns1, returns2)])
        
        rolling_corr = np.zeros(n - window + 1)
        for i in range(len(rolling_corr)):
            r1 = returns1[i:i+window]
            r2 = returns2[i:i+window]
            rolling_corr[i] = self.calculate_correlation(r1, r2)
        
        return rolling_corr
    
    def calculate_beta(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> float:
        """Calculate beta (sensitivity to market)"""
        if len(asset_returns) != len(market_returns):
            min_len = min(len(asset_returns), len(market_returns))
            asset_returns = asset_returns[-min_len:]
            market_returns = market_returns[-min_len:]
        
        if len(asset_returns) < 10:
            return 1.0
        
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return float(covariance / market_variance)
    
    def get_correlation_level(self, corr: float) -> CorrelationLevel:
        """Classify correlation level"""
        if corr < -0.3:
            return CorrelationLevel.NEGATIVE
        elif corr < 0.3:
            return CorrelationLevel.LOW
        elif corr < 0.6:
            return CorrelationLevel.MODERATE
        elif corr < 0.8:
            return CorrelationLevel.HIGH
        else:
            return CorrelationLevel.VERY_HIGH
    
    def build_correlation_matrix(
        self,
        returns_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Build full correlation matrix"""
        assets = list(returns_dict.keys())
        matrix = {}
        
        for asset1 in assets:
            matrix[asset1] = {}
            for asset2 in assets:
                if asset1 == asset2:
                    matrix[asset1][asset2] = 1.0
                else:
                    corr = self.calculate_correlation(
                        returns_dict[asset1],
                        returns_dict[asset2]
                    )
                    matrix[asset1][asset2] = corr
        
        return matrix
    
    def calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        volatilities: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate Diversification Ratio
        DR = Sum(w_i * sigma_i) / Portfolio_Sigma
        DR > 1 means portfolio benefits from diversification
        """
        weighted_vol_sum = np.sum(weights * volatilities)
        
        # Portfolio variance
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        if portfolio_vol == 0:
            return 1.0
        
        return float(weighted_vol_sum / portfolio_vol)
    
    def calculate_effective_n(
        self,
        weights: np.ndarray
    ) -> float:
        """
        Calculate Effective Number of Assets (Herfindahl-based)
        N_eff = 1 / Sum(w_i^2)
        """
        hhi = np.sum(weights ** 2)
        if hhi == 0:
            return len(weights)
        return float(1.0 / hhi)
    
    def analyze_portfolio_correlations(
        self,
        positions: Dict[str, Dict],  # {symbol: {returns: [], weight: float, volatility: float}}
        market_returns: Optional[np.ndarray] = None
    ) -> PortfolioCorrelationReport:
        """
        Main function to analyze portfolio correlations
        """
        if len(positions) < 2:
            return PortfolioCorrelationReport(
                average_correlation=0.0,
                max_correlation=0.0,
                correlation_matrix={},
                high_correlation_pairs=[],
                diversification_ratio=1.0,
                effective_n_assets=len(positions),
                concentration_risk="low",
                recommendations=["Add more positions for diversification"]
            )
        
        # Extract data
        symbols = list(positions.keys())
        returns_dict = {s: np.array(positions[s]['returns']) for s in symbols}
        weights = np.array([positions[s].get('weight', 1/len(symbols)) for s in symbols])
        volatilities = np.array([positions[s].get('volatility', 0.2) for s in symbols])
        
        # Build correlation matrix
        corr_matrix = self.build_correlation_matrix(returns_dict)
        
        # Find high correlation pairs
        high_corr_pairs = []
        all_correlations = []
        
        for i, asset1 in enumerate(symbols):
            for j, asset2 in enumerate(symbols):
                if i < j:  # Upper triangle only
                    corr = corr_matrix[asset1][asset2]
                    all_correlations.append(corr)
                    
                    level = self.get_correlation_level(corr)
                    
                    # Calculate beta if market returns provided
                    beta = 1.0
                    if market_returns is not None:
                        beta = self.calculate_beta(returns_dict[asset1], market_returns)
                    
                    asset_corr = AssetCorrelation(
                        asset1=asset1,
                        asset2=asset2,
                        correlation=corr,
                        level=level,
                        beta=beta,
                        covariance=np.cov(returns_dict[asset1], returns_dict[asset2])[0, 1]
                    )
                    
                    if corr > self.config.high_correlation_threshold:
                        high_corr_pairs.append(asset_corr)
        
        # Calculate metrics
        avg_corr = np.mean(all_correlations) if all_correlations else 0.0
        max_corr = np.max(all_correlations) if all_correlations else 0.0
        
        # Build numpy correlation matrix for diversification ratio
        n = len(symbols)
        np_corr_matrix = np.zeros((n, n))
        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                np_corr_matrix[i, j] = corr_matrix[s1][s2]
        
        div_ratio = self.calculate_diversification_ratio(weights, volatilities, np_corr_matrix)
        eff_n = self.calculate_effective_n(weights)
        
        # Determine concentration risk
        if avg_corr > 0.7 or max_corr > 0.9:
            concentration_risk = "critical"
        elif avg_corr > 0.5 or max_corr > 0.8:
            concentration_risk = "high"
        elif avg_corr > 0.3:
            concentration_risk = "moderate"
        else:
            concentration_risk = "low"
        
        # Generate recommendations
        recommendations = []
        
        if len(high_corr_pairs) > 0:
            recommendations.append(
                f"Reduce exposure: {len(high_corr_pairs)} highly correlated pairs found"
            )
            for pair in high_corr_pairs[:3]:  # Top 3
                recommendations.append(
                    f"  - {pair.asset1}/{pair.asset2}: {pair.correlation:.2f} correlation"
                )
        
        if div_ratio < 1.2:
            recommendations.append("Low diversification benefit - consider adding uncorrelated assets")
        
        if eff_n < len(symbols) * 0.5:
            recommendations.append("Portfolio concentrated in few positions - rebalance weights")
        
        if concentration_risk in ["high", "critical"]:
            recommendations.append("Consider sector/industry diversification")
        
        if not recommendations:
            recommendations.append("Portfolio well-diversified")
        
        return PortfolioCorrelationReport(
            average_correlation=avg_corr,
            max_correlation=max_corr,
            correlation_matrix=corr_matrix,
            high_correlation_pairs=high_corr_pairs,
            diversification_ratio=div_ratio,
            effective_n_assets=eff_n,
            concentration_risk=concentration_risk,
            recommendations=recommendations
        )
    
    def check_new_position_correlation(
        self,
        new_asset_returns: np.ndarray,
        existing_positions: Dict[str, np.ndarray],
        max_allowed_correlation: float = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if a new position is too correlated with existing positions
        Returns: (is_allowed, list of warnings)
        """
        threshold = max_allowed_correlation or self.config.high_correlation_threshold
        warnings = []
        
        for symbol, returns in existing_positions.items():
            corr = self.calculate_correlation(new_asset_returns, returns)
            
            if corr > threshold:
                warnings.append(f"High correlation with {symbol}: {corr:.2f}")
        
        is_allowed = len(warnings) == 0
        
        return is_allowed, warnings


# Singleton instance
_analyzer = CorrelationAnalyzer()

def analyze_correlations(positions: Dict[str, Dict]) -> PortfolioCorrelationReport:
    """Convenience function for correlation analysis"""
    return _analyzer.analyze_portfolio_correlations(positions)

def check_correlation(
    new_returns: np.ndarray, 
    existing: Dict[str, np.ndarray]
) -> Tuple[bool, List[str]]:
    """Check if new position is allowed based on correlation"""
    return _analyzer.check_new_position_correlation(new_returns, existing)
EOF

echo "   ✅ Correlation Analyzer создан"

# ============================================
# 4. SLIPPAGE MODEL FOR BACKTESTING
# ============================================
echo "[4/4] 💹 Создание Slippage Model..."

cat > services/shared/trading/slippage_model.py << 'EOF'
"""
Realistic Slippage Model for Backtesting
Моделирование проскальзывания для более точного бэктестинга
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SlippageType(Enum):
    FIXED = "fixed"                    # Fixed percentage
    VOLUME_BASED = "volume_based"      # Based on volume participation
    SPREAD_BASED = "spread_based"      # Based on bid-ask spread
    VOLATILITY_BASED = "volatility_based"  # Based on current volatility
    MARKET_IMPACT = "market_impact"    # Full market impact model

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

@dataclass
class SlippageConfig:
    # Fixed slippage
    fixed_slippage_pct: float = 0.001  # 0.1% = 10 bps
    
    # Volume-based parameters
    volume_participation_rate: float = 0.10  # 10% of volume
    volume_impact_factor: float = 0.1  # Price impact per % of volume
    
    # Spread-based
    default_spread_pct: float = 0.001  # 0.1% default spread
    spread_multiplier: float = 0.5     # Pay half spread on average
    
    # Volatility-based
    volatility_multiplier: float = 0.1  # 10% of daily volatility
    
    # Market impact (Square-root model)
    market_impact_coeff: float = 0.1   # Almgren-Chriss coefficient
    temporary_impact_coeff: float = 0.05
    permanent_impact_coeff: float = 0.02
    
    # Order-type specific
    limit_order_fill_rate: float = 0.7  # 70% fill rate for limits
    stop_slippage_multiplier: float = 2.0  # 2x slippage for stops

@dataclass
class SlippageResult:
    slippage_pct: float
    slippage_amount: float
    execution_price: float
    market_impact: float
    spread_cost: float
    fill_probability: float
    details: Dict

class SlippageModel:
    def __init__(self, config: SlippageConfig = None):
        self.config = config or SlippageConfig()
    
    def fixed_slippage(
        self,
        price: float,
        is_buy: bool
    ) -> SlippageResult:
        """Simple fixed percentage slippage"""
        slippage_pct = self.config.fixed_slippage_pct
        slippage_amount = price * slippage_pct
        
        if is_buy:
            execution_price = price + slippage_amount
        else:
            execution_price = price - slippage_amount
        
        return SlippageResult(
            slippage_pct=slippage_pct,
            slippage_amount=slippage_amount,
            execution_price=execution_price,
            market_impact=0.0,
            spread_cost=slippage_amount,
            fill_probability=1.0,
            details={"type": "fixed"}
        )
    
    def volume_based_slippage(
        self,
        price: float,
        quantity: int,
        volume: float,
        is_buy: bool
    ) -> SlippageResult:
        """
        Volume-based slippage model
        Higher slippage when order size is large relative to volume
        """
        if volume == 0:
            volume = quantity * 10  # Assume low liquidity
        
        # Participation rate
        participation = (quantity * price) / (volume * price)
        participation = min(participation, 1.0)
        
        # Slippage increases with participation
        base_slippage = self.config.fixed_slippage_pct
        volume_slippage = self.config.volume_impact_factor * participation
        total_slippage_pct = base_slippage + volume_slippage
        
        slippage_amount = price * total_slippage_pct
        
        if is_buy:
            execution_price = price + slippage_amount
        else:
            execution_price = price - slippage_amount
        
        # Fill probability decreases with size
        fill_prob = max(0.5, 1.0 - participation * 0.5)
        
        return SlippageResult(
            slippage_pct=total_slippage_pct,
            slippage_amount=slippage_amount,
            execution_price=execution_price,
            market_impact=volume_slippage * price,
            spread_cost=base_slippage * price,
            fill_probability=fill_prob,
            details={
                "type": "volume_based",
                "participation_rate": participation,
                "volume": volume
            }
        )
    
    def spread_based_slippage(
        self,
        price: float,
        bid: Optional[float],
        ask: Optional[float],
        is_buy: bool
    ) -> SlippageResult:
        """
        Spread-based slippage using bid-ask spread
        """
        if bid is not None and ask is not None:
            spread = ask - bid
            spread_pct = spread / price
        else:
            spread_pct = self.config.default_spread_pct
            spread = price * spread_pct
        
        # Pay portion of spread
        slippage_pct = spread_pct * self.config.spread_multiplier
        slippage_amount = slippage_pct * price
        
        if is_buy:
            execution_price = price + slippage_amount
        else:
            execution_price = price - slippage_amount
        
        return SlippageResult(
            slippage_pct=slippage_pct,
            slippage_amount=slippage_amount,
            execution_price=execution_price,
            market_impact=0.0,
            spread_cost=slippage_amount,
            fill_probability=1.0,
            details={
                "type": "spread_based",
                "spread": spread,
                "spread_pct": spread_pct
            }
        )
    
    def volatility_based_slippage(
        self,
        price: float,
        volatility: float,
        is_buy: bool
    ) -> SlippageResult:
        """
        Volatility-based slippage
        Higher volatility = higher slippage
        """
        # Daily volatility contribution to slippage
        daily_vol = volatility / np.sqrt(252)  # Assume annual vol input
        slippage_pct = daily_vol * self.config.volatility_multiplier
        slippage_pct = max(slippage_pct, self.config.fixed_slippage_pct)
        
        slippage_amount = price * slippage_pct
        
        if is_buy:
            execution_price = price + slippage_amount
        else:
            execution_price = price - slippage_amount
        
        return SlippageResult(
            slippage_pct=slippage_pct,
            slippage_amount=slippage_amount,
            execution_price=execution_price,
            market_impact=slippage_amount * 0.5,
            spread_cost=slippage_amount * 0.5,
            fill_probability=max(0.8, 1.0 - daily_vol),
            details={
                "type": "volatility_based",
                "daily_volatility": daily_vol,
                "annual_volatility": volatility
            }
        )
    
    def market_impact_model(
        self,
        price: float,
        quantity: int,
        volume: float,
        volatility: float,
        is_buy: bool,
        execution_time_hours: float = 1.0
    ) -> SlippageResult:
        """
        Full Market Impact Model (Almgren-Chriss inspired)
        
        Total cost = Temporary impact + Permanent impact + Spread
        
        Temporary impact: η * σ * sqrt(X/V/T)
        Permanent impact: γ * σ * (X/V)
        """
        if volume == 0:
            volume = quantity * 10
        
        daily_vol = volatility / np.sqrt(252)
        
        # Order size as fraction of daily volume
        x_ratio = (quantity * price) / (volume * price)
        x_ratio = min(x_ratio, 1.0)
        
        # Temporary impact (decays after execution)
        temp_impact = (
            self.config.temporary_impact_coeff * 
            daily_vol * 
            np.sqrt(x_ratio / max(execution_time_hours / 6.5, 0.1))  # 6.5 hour trading day
        )
        
        # Permanent impact (moves the price)
        perm_impact = (
            self.config.permanent_impact_coeff * 
            daily_vol * 
            x_ratio
        )
        
        # Total market impact
        total_impact_pct = temp_impact + perm_impact
        
        # Add spread
        spread_cost_pct = self.config.default_spread_pct * self.config.spread_multiplier
        
        total_slippage_pct = total_impact_pct + spread_cost_pct
        slippage_amount = price * total_slippage_pct
        
        if is_buy:
            execution_price = price + slippage_amount
        else:
            execution_price = price - slippage_amount
        
        # Fill probability based on size and volatility
        fill_prob = max(0.6, 1.0 - x_ratio * 0.3 - daily_vol)
        
        return SlippageResult(
            slippage_pct=total_slippage_pct,
            slippage_amount=slippage_amount,
            execution_price=execution_price,
            market_impact=total_impact_pct * price,
            spread_cost=spread_cost_pct * price,
            fill_probability=fill_prob,
            details={
                "type": "market_impact",
                "temporary_impact": temp_impact,
                "permanent_impact": perm_impact,
                "order_volume_ratio": x_ratio,
                "execution_time_hours": execution_time_hours
            }
        )
    
    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
        order_type: OrderType = OrderType.MARKET,
        slippage_type: SlippageType = SlippageType.MARKET_IMPACT,
        volume: float = None,
        volatility: float = None,
        bid: float = None,
        ask: float = None,
        **kwargs
    ) -> SlippageResult:
        """
        Main function to calculate slippage
        """
        # Adjust for order type
        multiplier = 1.0
        if order_type == OrderType.STOP:
            multiplier = self.config.stop_slippage_multiplier
        elif order_type == OrderType.LIMIT:
            # Limit orders have lower slippage but lower fill rate
            multiplier = 0.5
        ```bash
cat > fix_trading_risks.sh << 'SCRIPT'
#!/bin/bash
set -e
echo "📈 TRADING-AUTOPILOT: Исправление торговых рисков"
echo "=================================================="

# ============================================
# 1. MARKET REGIME DETECTOR
# ============================================
echo "[1/4] 🌡️ Создание Market Regime Detector..."

mkdir -p services/shared/trading

cat > services/shared/trading/regime_detector.py << 'EOF'
"""
Market Regime Detector
Определяет текущий режим рынка: TRENDING, RANGING, HIGH_VOLATILITY, CRISIS
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"

@dataclass
class RegimeConfig:
    # ADX thresholds
    adx_trending_threshold: float = 25.0
    adx_strong_trend_threshold: float = 40.0
    
    # Volatility thresholds (относительно исторической)
    volatility_high_threshold: float = 1.5  # 150% от средней
    volatility_crisis_threshold: float = 2.5  # 250% от средней
    
    # RSI thresholds для определения перекупленности/перепроданности
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Период для расчётов
    lookback_period: int = 20
    volatility_lookback: int = 60

@dataclass
class RegimeAnalysis:
    regime: MarketRegime
    confidence: float  # 0-1
    adx: float
    rsi: float
    volatility_ratio: float
    trend_direction: int  # 1=up, -1=down, 0=neutral
    recommended_position_multiplier: float
    details: dict

class MarketRegimeDetector:
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Average Directional Index - сила тренда"""
        if len(close) < period + 1:
            return 0.0
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = self._ema(tr, period)
        plus_di = 100 * self._ema(plus_dm, period) / np.where(atr > 0, atr, 1)
        minus_di = 100 * self._ema(minus_dm, period) / np.where(atr > 0, atr, 1)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
        adx = self._ema(dx, period)
        
        return float(adx[-1]) if len(adx) > 0 else 0.0
    
    def calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(close) < period + 1:
            return 50.0
        
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = self._ema(gains, period)[-1]
        avg_loss = self._ema(losses, period)[-1]
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_volatility_ratio(self, close: np.ndarray) -> float:
        """Отношение текущей волатильности к исторической"""
        if len(close) < self.config.volatility_lookback:
            return 1.0
        
        returns = np.diff(np.log(close))
        
        current_vol = np.std(returns[-self.config.lookback_period:])
        historical_vol = np.std(returns[-self.config.volatility_lookback:])
        
        if historical_vol == 0:
            return 1.0
        
        return current_vol / historical_vol
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2 / (period + 1)
        result = np.zeros_like(data, dtype=float)
        result = data
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    def detect_regime(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> RegimeAnalysis:
        """Основной метод определения режима рынка"""
        
        adx = self.calculate_adx(high, low, close)
        rsi = self.calculate_rsi(close)
        vol_ratio = self.calculate_volatility_ratio(close)
        
        # Определение направления тренда
        if len(close) >= 20:
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
            trend_direction = 1 if close[-1] > sma_20 > sma_50 else (-1 if close[-1] < sma_20 < sma_50 else 0)
        else:
            trend_direction = 0
        
        # Определение режима
        regime = MarketRegime.UNKNOWN
        confidence = 0.5
        position_multiplier = 1.0
        
        # CRISIS - экстремальная волатильность
        if vol_ratio >= self.config.volatility_crisis_threshold:
            regime = MarketRegime.CRISIS
            confidence = min(0.95, 0.7 + (vol_ratio - self.config.volatility_crisis_threshold) * 0.1)
            position_multiplier = 0.25  # Минимальные позиции
        
        # HIGH VOLATILITY
        elif vol_ratio >= self.config.volatility_high_threshold:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = 0.7
            position_multiplier = 0.5  # Половина позиции
        
        # TRENDING
        elif adx >= self.config.adx_trending_threshold:
            if trend_direction > 0:
                regime = MarketRegime.TRENDING_UP
            elif trend_direction < 0:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING
            
            confidence = min(0.9, 0.6 + (adx - self.config.adx_trending_threshold) / 50)
            position_multiplier = 1.2 if adx >= self.config.adx_strong_trend_threshold else 1.0
        
        # RANGING
        else:
            regime = MarketRegime.RANGING
            confidence = 0.6
            position_multiplier = 0.7  # Уменьшенные позиции в боковике
        
        details = {
            "adx_threshold": self.config.adx_trending_threshold,
            "vol_high_threshold": self.config.volatility_high_threshold,
            "sma_20": float(np.mean(close[-20:])) if len(close) >= 20 else None,
            "sma_50": float(np.mean(close[-50:])) if len(close) >= 50 else None,
            "current_price": float(close[-1]) if len(close) > 0 else None
        }
        
        logger.info(f"Market Regime: {regime.value} (confidence: {confidence:.2f}, ADX: {adx:.1f}, RSI: {rsi:.1f})")
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            adx=adx,
            rsi=rsi,
            volatility_ratio=vol_ratio,
            trend_direction=trend_direction,
            recommended_position_multiplier=position_multiplier,
            details=details
        )

# Singleton instance
regime_detector = MarketRegimeDetector()
EOF

echo "   ✅ Market Regime Detector создан"

# ============================================
# 2. VOLATILITY-BASED POSITION SIZING
# ============================================
echo "[2/4] 📊 Создание Position Sizing на основе волатильности..."

cat > services/shared/trading/position_sizing.py << 'EOF'
"""
Volatility-Based Position Sizing
ATR-based и Kelly Criterion sizing
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    ATR_BASED = "atr_based"
    KELLY = "kelly"
    VOLATILITY_PARITY = "volatility_parity"

@dataclass
class SizingConfig:
    # Risk parameters
    max_risk_per_trade: float = 0.02  # 2% риска на сделку
    max_position_size: float = 0.10   # 10% портфеля макс на позицию
    min_position_size: float = 0.01   # 1% портфеля мин
    
    # ATR parameters
    atr_period: int = 14
    atr_multiplier: float = 2.0  # Stop-loss = ATR * multiplier
    
    # Kelly parameters
    kelly_fraction: float = 0.25  # Используем 25% от Kelly (fractional Kelly)
    
    # Volatility targeting
    target_volatility: float = 0.15  # 15% годовой волатильности

@dataclass
class PositionSize:
    shares: int
    position_value: float
    position_pct: float
    risk_amount: float
    stop_loss_price: float
    method_used: SizingMethod
    details: dict

class PositionSizer:
    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Average True Range"""
        if len(close) < self.config.atr_period + 1:
            return (high[-1] - low[-1]) if len(high) > 0 else 0
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        return np.mean(tr[-self.config.atr_period:])
    
    def calculate_volatility(self, close: np.ndarray, period: int = 20) -> float:
        """Annualized volatility"""
        if len(close) < period + 1:
            return 0.2  # Default 20%
        
        returns = np.diff(np.log(close[-period-1:]))
        daily_vol = np.std(returns)
        return daily_vol * np.sqrt(252)  # Annualized
    
    def atr_based_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        side: str = "long"
    ) -> PositionSize:
        """Position sizing на основе ATR"""
        
        atr = self.calculate_atr(high, low, close)
        stop_distance = atr * self.config.atr_multiplier
        
        if side == "long":
            stop_loss_price = entry_price - stop_distance
        else:
            stop_loss_price = entry_price + stop_distance
        
        # Риск на сделку
        risk_amount = portfolio_value * self.config.max_risk_per_trade
        
        # Размер позиции исходя из риска
        if stop_distance > 0:
            shares = int(risk_amount / stop_distance)
        else:
            shares = 0
        
        position_value = shares * entry_price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Ограничения
        if position_pct > self.config.max_position_size:
            position_pct = self.config.max_position_size
            position_value = portfolio_value * position_pct
            shares = int(position_value / entry_price)
        
        if position_pct < self.config.min_position_size:
            shares = 0
            position_value = 0
            position_pct = 0
        
        return PositionSize(
            shares=shares,
            position_value=position_value,
            position_pct=position_pct,
            risk_amount=risk_amount,
            stop_loss_price=stop_loss_price,
            method_used=SizingMethod.ATR_BASED,
            details={
                "atr": atr,
                "stop_distance": stop_distance,
                "atr_multiplier": self.config.atr_multiplier
            }
        )
    
    def kelly_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> PositionSize:
        """Kelly Criterion sizing"""
        
        if avg_loss == 0:
            kelly_pct = 0
        else:
            # Kelly formula: f = (bp - q) / b
            # b = avg_win / avg_loss (win/loss ratio)
            # p = win_rate
            # q = 1 - p
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly_pct = (b * p - q) / b
        
        # Fractional Kelly
        position_pct = kelly_pct * self.config.kelly_fraction
        
        # Ограничения
        position_pct = max(0, min(position_pct, self.config.max_position_size))
        
        position_value = portfolio_value * position_pct
        shares = int(position_value / entry_price) if entry_price > 0 else 0
        
        return PositionSize(
            shares=shares,
            position_value=position_value,
            position_pct=position_pct,
            risk_amount=portfolio_value * self.config.max_risk_per_trade,
            stop_loss_price=0,  # Не определён для Kelly
            method_used=SizingMethod.KELLY,
            details={
                "full_kelly": kelly_pct,
                "fractional_kelly": self.config.kelly_fraction,
                "win_rate": win_rate,
                "win_loss_ratio": avg_win / avg_loss if avg_loss > 0 else 0
            }
        )
    
    def volatility_parity_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        close: np.ndarray,
        regime_multiplier: float = 1.0
    ) -> PositionSize:
        """Volatility targeting / Risk parity sizing"""
        
        current_vol = self.calculate_volatility(close)
        
        if current_vol == 0:
            vol_adjustment = 1.0
        else:
            vol_adjustment = self.config.target_volatility / current_vol
        
        # Base position size
        base_pct = self.config.max_risk_per_trade * 5  # ~10% base
        
        # Adjust for volatility and regime
        position_pct = base_pct * vol_adjustment * regime_multiplier
        
        # Ограничения
        position_pct = max(
            self.config.min_position_size,
            min(position_pct, self.config.max_position_size)
        )
        
        position_value = portfolio_value * position_pct
        shares = int(position_value / entry_price) if entry_price > 0 else 0
        
        return PositionSize(
            shares=shares,
            position_value=position_value,
            position_pct=position_pct,
            risk_amount=portfolio_value * self.config.max_risk_per_trade,
            stop_loss_price=0,
            method_used=SizingMethod.VOLATILITY_PARITY,
            details={
                "current_volatility": current_vol,
                "target_volatility": self.config.target_volatility,
                "vol_adjustment": vol_adjustment,
                "regime_multiplier": regime_multiplier
            }
        )
    
    def calculate_size(
        self,
        portfolio_value: float,
        entry_price: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        method: SizingMethod = SizingMethod.ATR_BASED,
        side: str = "long",
        regime_multiplier: float = 1.0,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0
    ) -> PositionSize:
        """Универсальный метод расчёта размера позиции"""
        
        if method == SizingMethod.ATR_BASED:
            size = self.atr_based_sizing(portfolio_value, entry_price, high, low, close, side)
        elif method == SizingMethod.KELLY:
            size = self.kelly_sizing(portfolio_value, entry_price, win_rate, avg_win, avg_loss)
        elif method == SizingMethod.VOLATILITY_PARITY:
            size = self.volatility_parity_sizing(portfolio_value, entry_price, close, regime_multiplier)
        else:
            # Fixed fractional
            position_pct = self.config.max_risk_per_trade * 5
            position_value = portfolio_value * position_pct
            shares = int(position_value / entry_price) if entry_price > 0 else 0
            size = PositionSize(
                shares=shares,
                position_value=position_value,
                position_pct=position_pct,
                risk_amount=portfolio_value * self.config.max_risk_per_trade,
                stop_loss_price=0,
                method_used=SizingMethod.FIXED_FRACTIONAL,
                details={}
            )
        
        # Apply regime multiplier
        if regime_multiplier != 1.0 and method != SizingMethod.VOLATILITY_PARITY:
            size.shares = int(size.shares * regime_multiplier)
            size.position_value *= regime_multiplier
            size.position_pct *= regime_multiplier
            size.details["regime_multiplier"] = regime_multiplier
        
        logger.info(f"Position Size: {size.shares} shares ({size.position_pct:.1%}), method: {method.value}")
        
        return size

# Singleton
position_sizer = PositionSizer()
EOF

echo "   ✅ Position Sizing создан"

# ============================================
# 3. CORRELATION ANALYZER
# ============================================
echo "[3/4] 🔗 Создание Correlation Analyzer..."

cat > services/shared/trading/correlation_analyzer.py << 'EOF'
"""
Portfolio Correlation Analyzer
Анализ корреляций между позициями для диверсификации
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class CorrelationConfig:
    lookback_period: int = 60  # Дней для расчёта корреляции
    high_correlation_threshold: float = 0.7
    max_correlated_exposure: float = 0.25  # Макс 25% в сильно коррелированных активах
    min_data_points: int = 20

@dataclass
class CorrelationResult:
    symbol1: str
    symbol2: str
    correlation: float
    is_high: bool
    rolling_correlation: Optional[np.ndarray] = None

@dataclass
class PortfolioCorrelationAnalysis:
    correlation_matrix: Dict[str, Dict[str, float]]
    high_correlations: List[CorrelationResult]
    average_correlation: float
    diversification_ratio: float
    concentration_risk: float
    recommendations: List[str]

class CorrelationAnalyzer:
    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self._price_data: Dict[str, np.ndarray] = {}
    
    def update_prices(self, symbol: str, prices: np.ndarray):
        """Обновить ценовые данные для символа"""
        self._price_data[symbol] = prices
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """Рассчитать логарифмические доходности"""
        if len(prices) < 2:
            return np.array([])
        return np.diff(np.log(prices))
    
    def calculate_correlation(self, symbol1: str, symbol2: str) -> CorrelationResult:
        """Рассчитать корреляцию между двумя активами"""
        
        if symbol1 not in self._price_data or symbol2 not in self._price_data:
            return CorrelationResult(symbol1, symbol2, 0.0, False)
        
        prices1 = self._price_data[symbol1]
        prices2 = self._price_data[symbol2]
        
        # Выровнять длину
        min_len = min(len(prices1), len(prices2))
        if min_len < self.config.min_data_points:
            return CorrelationResult(symbol1, symbol2, 0.0, False)
        
        returns1 = self.calculate_returns(prices1[-min_len:])
        returns2 = self.calculate_returns(prices2[-min_len:])
        
        # Pearson correlation
        correlation = np.corrcoef(returns1, returns2)[3]
        
        if np.isnan(correlation):
            correlation = 0.0
        
        is_high = abs(correlation) >= self.config.high_correlation_threshold
        
        return CorrelationResult(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=float(correlation),
            is_high=is_high
        )
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Рассчитать матрицу корреляций"""
        matrix = {}
        
        for s1 in symbols:
            matrix[s1] = {}
            for s2 in symbols:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                elif s2 in matrix and s1 in matrix[s2]:
                    matrix[s1][s2] = matrix[s2][s1]
                else:
                    result = self.calculate_correlation(s1, s2)
                    matrix[s1][s2] = result.correlation
        
        return matrix
    
    def calculate_diversification_ratio(
        self,
        symbols: List[str],
        weights: Dict[str, float]
    ) -> float:
        """
        Diversification Ratio = weighted avg volatility / portfolio volatility
        DR > 1 означает диверсификацию
        """
        if not symbols or not weights:
            return 1.0
        
        volatilities = {}
        for symbol in symbols:
            if symbol in self._price_data:
                returns = self.calculate_returns(self._price_data[symbol])
                if len(returns) > 0:
                    volatilities[symbol] = np.std(returns) * np.sqrt(252)
                else:
                    volatilities[symbol] = 0.2
            else:
                volatilities[symbol] = 0.2
        
        # Weighted average volatility
        weighted_vol = sum(
            weights.get(s, 0) * volatilities.get(s, 0.2)
            for s in symbols
        )
        
        # Portfolio volatility
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        portfolio_var = 0
        for s1 in symbols:
            for s2 in symbols:
                w1 = weights.get(s1, 0)
                w2 = weights.get(s2, 0)
                v1 = volatilities.get(s1, 0.2)
                v2 = volatilities.get(s2, 0.2)
                corr = corr_matrix.get(s1, {}).get(s2, 0)
                portfolio_var += w1 * w2 * v1 * v2 * corr
        
        portfolio_vol = np.sqrt(max(0, portfolio_var))
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_vol / portfolio_vol
    
    def analyze_portfolio(
        self,
        positions: Dict[str, float],  # symbol -> position_value
        portfolio_value: float
    ) -> PortfolioCorrelationAnalysis:
        """Полный анализ корреляций портфеля"""
        
        symbols = list(positions.keys())
        
        if len(symbols) < 2:
            return PortfolioCorrelationAnalysis(
                correlation_matrix={},
                high_correlations=[],
                average_correlation=0.0,
                diversification_ratio=1.0,
                concentration_risk=1.0 if len(symbols) == 1 else 0.0,
                recommendations=["Добавьте больше позиций для диверсификации"]
            )
        
        # Calculate weights
        weights = {s: v / portfolio_value for s, v in positions.items()}
        
        # Correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        # Find high correlations
        high_correlations = []
        all_correlations = []
        
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                corr = corr_matrix[s1][s2]
                all_correlations.append(abs(corr))
                
                if abs(corr) >= self.config.high_correlation_threshold:
                    high_correlations.append(CorrelationResult(
                        symbol1=s1,
                        symbol2=s2,
                        correlation=corr,
                        is_high=True
                    ))
        
        # Average correlation
        avg_corr = np.mean(all_correlations) if all_correlations else 0.0
        
        # Diversification ratio
        div_ratio = self.calculate_diversification_ratio(symbols, weights)
        
        # Concentration risk (Herfindahl index)
        concentration = sum(w ** 2 for w in weights.values())
        
        # Recommendations
        recommendations = []
        
        if avg_corr > 0.5:
            recommendations.append("⚠️ Высокая средняя корреляция портфеля. Рассмотрите добавление некоррелированных активов.")
        
        for hc in high_correlations:
            combined_weight = weights.get(hc.symbol1, 0) + weights.get(hc.symbol2, 0)
            if combined_weight > self.config.max_correlated_exposure:
                recommendations.append(
                    f"⚠️ {hc.symbol1} и {hc.symbol2} сильно коррелированы ({hc.correlation:.2f}) "
                    f"с общей долей {combined_weight:.1%}. Рекомендуется снизить до {self.config.max_correlated_exposure:.0%}."
                )
        
        if concentration > 0.3:
            recommendations.append("⚠️ Высокая концентрация портфеля. Рассмотрите более равномерное распределение.")
        
        if div_ratio < 1.2:
            recommendations.append("ℹ️ Низкий коэффициент диверсификации. Портфель может быть недостаточно диверсифицирован.")
        
        if not recommendations:
            recommendations.append("✅ Корреляционный профиль портфеля в норме.")
        
        logger.info(f"Portfolio Analysis: avg_corr={avg_corr:.2f}, div_ratio={div_ratio:.2f}, concentration={concentration:.2f}")
        
        return PortfolioCorrelationAnalysis(
            correlation_matrix=corr_matrix,
            high_correlations=high_correlations,
            average_correlation=float(avg_corr),
            diversification_ratio=float(div_ratio),
            concentration_risk=float(concentration),
            recommendations=recommendations
        )
    
    def should_allow_trade(
        self,
        new_symbol: str,
        new_value: float,
        current_positions: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """Проверить, допустима ли новая сделка с точки зрения корреляции"""
        
        if new_symbol not in self._price_data:
            return True, "Нет данных для анализа корреляции"
        
        for symbol, value in current_positions.items():
            if symbol == new_symbol:
                continue
            
            if symbol not in self._price_data:
                continue
            
            result = self.calculate_correlation(new_symbol, symbol)
            
            if result.is_high:
                combined_exposure = (value + new_value) / portfolio_value
                
                if combined_exposure > self.config.max_correlated_exposure:
                    return False, (
                        f"Отклонено: {new_symbol} сильно коррелирован с {symbol} "
                        f"(r={result.correlation:.2f}). Общая экспозиция {combined_exposure:.1%} "
                        f"превышает лимит {self.config.max_correlated_exposure:.0%}"
                    )
        
        return True, "OK"

# Singleton
correlation_analyzer = CorrelationAnalyzer()
EOF

echo "   ✅ Correlation Analyzer создан"

# ============================================
# 4. SLIPPAGE MODEL FOR BACKTESTING
# ============================================
echo "[4/4] 📉 Создание Slippage Model..."

cat > services/shared/trading/slippage_model.py << 'EOF'
"""
Realistic Slippage Model for Backtesting
Учитывает: спред, market impact, время суток, волатильность
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    OPENING = "opening"  # Первые 30 мин торгов
    CLOSING = "closing"  # Последние 30 мин торгов

@dataclass
class SlippageConfig:
    # Base slippage (в процентах)
    base_slippage_pct: float = 0.05  # 0.05% базовый slippage
    
    # Spread
    avg_spread_pct: float = 0.1  # 0.1% средний спред
    
    # Market impact parameters
    market_impact_coefficient: float = 0.1  # Impact = coef * sqrt(order_size / avg_volume)
    
    # Time-of-day multipliers
    opening_multiplier: float = 2.0  # Открытие рынка
    closing_multiplier: float = 1.5  # Закрытие рынка
    
    # Volatility multipliers
    high_volatility_multiplier: float = 2.5
    normal_volatility_multiplier: float = 1.0
    
    # Order type adjustments
    limit_order_fill_rate: float = 0.85  # 85% лимитных ордеров исполняются
    stop_order_slippage_multiplier: float = 1.5  # Стопы проскальзывают больше

@dataclass
class SlippageResult:
    expected_price: float
    executed_price: float
    slippage_amount: float
    slippage_pct: float
    total_cost: float
    components: dict

class SlippageModel:
    def __init__(self, config: Optional[SlippageConfig] = None):
        self.config = config or SlippageConfig()
    
    def get_market_condition(
        self,
        timestamp: datetime,
        volatility_ratio: float = 1.0
    ) -> MarketCondition:
        """Определить рыночные условия"""
        
        trade_time = timestamp.time()
        
        # MOEX: 10:00 - 18:50 основная сессия
        opening_start = time(10, 0)
        opening_end = time(10, 30)
        closing_start = time(18, 20)
        closing_end = time(18, 50)
        
        if opening_start <= trade_time <= opening_end:
            return MarketCondition.OPENING
        elif closing_start <= trade_time <= closing_end:
            return MarketCondition.CLOSING
        elif volatility_ratio > 1.5:
            return MarketCondition.VOLATILE
        else:
            return MarketCondition.NORMAL
    
    def calculate_spread_cost(
        self,
        price: float,
        side: str,
        condition: MarketCondition
    ) -> float:
        """Рассчитать стоимость спреда"""
        
        spread_pct = self.config.avg_spread_pct
        
        # Adjust for market condition
        if condition == MarketCondition.OPENING:
            spread_pct *= 1.5
        elif condition == MarketCondition.CLOSING:
            spread_pct *= 1.2
        elif condition == MarketCondition.VOLATILE:
            spread_pct *= 2.0
        elif condition == MarketCondition.ILLIQUID:
            spread_pct *= 3.0
        
        # Half spread for execution
        half_spread = (spread_pct / 100) * price / 2
        
        return half_spread if side == "buy" else -half_spread
    
    def calculate_market_impact(
        self,
        price: float,
        order_size: float,
        avg_daily_volume: float,
        side: str
    ) -> float:
        """Рассчитать market impact"""
        
        if avg_daily_volume == 0:
            participation_rate = 0.01
        else:
            participation_rate = order_size / avg_daily_volume
        
        # Square root market impact model
        impact_pct = self.config.market_impact_coefficient * np.sqrt(participation_rate)
        
        impact_amount = (impact_pct / 100) * price
        
        return impact_amount if side == "buy" else -impact_amount
    
    def calculate_volatility_slippage(
        self,
        price: float,
        volatility_ratio: float,
        side: str
    ) -> float:
        """Slippage из-за волатильности"""
        
        if volatility_ratio > 1.5:
            multiplier = self.config.high_volatility_multiplier
        else:
            multiplier = self.config.normal_volatility_multiplier
        
        base_slip = (self.config.base_slippage_pct / 100) * price * multiplier
        
        # Random component (symmetric around expected slippage)
        random_factor = np.random.normal(1.0, 0.3)
        random_factor = max(0.5, min(1.5, random_factor))
        
        slippage = base_slip * random_factor
        
        return slippage if side == "buy" else -slippage
    
    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,  # "buy" or "sell"
        order_type: OrderType = OrderType.MARKET,
        timestamp: Optional[datetime] = None,
        avg_daily_volume: float = 1_000_000,
        volatility_ratio: float = 1.0,
        include_random: bool = True
    ) -> SlippageResult:
        """
        Рассчитать полный slippage для ордера
        
        Returns executed price with all slippage components
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        order_value = price * quantity
        condition = self.get_market_condition(timestamp, volatility_ratio)
        
        # Components
        spread_cost = self.calculate_spread_cost(price, side, condition)
        market_impact = self.calculate_market_impact(price, order_value, avg_daily_volume, side)
        vol_slippage = self.calculate_volatility_slippage(price, volatility_ratio, side) if include_random else 0
        
        # Time-of-day adjustment
        if condition == MarketCondition.OPENING:
            time_multiplier = self.config.opening_multiplier
        elif condition == MarketCondition.CLOSING:
            time_multiplier = self.config.closing_multiplier
        else:
            time_multiplier = 1.0
        
        # Order type adjustment
        if order_type == OrderType.STOP:
            type_multiplier = self.config.stop_order_slippage_multiplier
        else:
            type_multiplier = 1.0
        
        # Total slippage
        base_slippage = spread_cost + market_impact + vol_slippage
        total_slippage = base_slippage * time_multiplier * type_multiplier
        
        # Executed price
        executed_price = price + total_slippage
        
        # Ensure price is positive
        executed_price = max(0.01, executed_price)
        
        slippage_pct = (total_slippage / price) * 100
        total_cost = abs(total_slippage) * quantity
        
        logger.debug(
            f"Slippage: {side} {quantity}x{price:.2f} -> {executed_price:.2f} "
            f"({slippage_pct:+.3f}%, condition={condition.value})"
        )
        
        return SlippageResult(
            expected_price=price,
            executed_price=executed_price,
            slippage_amount=total_slippage,
            slippage_pct=slippage_pct,
            total_cost=total_cost,
            components={
                "spread": spread_cost,
                "market_impact": market_impact,
                "volatility": vol_slippage,
                "time_multiplier": time_multiplier,
                "type_multiplier": type_multiplier,
                "condition": condition.value
            }
        )
    
    def simulate_limit_order_fill(
        self,
        limit_price: float,
        current_price: float,
        side: str,
        time_in_market_hours: float = 1.0
    ) -> tuple[bool, float]:
        """
        Симулировать исполнение лимитного ордера
        Returns: (filled, fill_price)
        """
        
        # Базовая вероятность исполнения
        base_fill_prob = self.config.limit_order_fill_rate
        
        # Adjust for distance from market
        distance_pct = abs(limit_price - current_price) / current_price * 100
        
        if side == "buy":
            # Покупка: лимит ниже рынка - выше шанс исполнения если цена упадёт
            if limit_price >= current_price:
                fill_prob = 0.99  # Сразу исполнится
            else:
                fill_prob = base_fill_prob * np.exp(-distance_pct * 0.5)
        else:
            # Продажа: лимит выше рынка
            if limit_price <= current_price:
                fill_prob = 0.99
            else:
                fill_prob = base_fill_prob * np.exp(-distance_pct * 0.5)
        
        # Adjust for time
        fill_prob = min(0.99, fill_prob * (1 + 0.1 * time_in_market_hours))
        
        filled = np.random.random() < fill_prob
        
        if filled:
            # Small improvement possible for limit orders
            improvement = np.random.uniform(0, 0.01) * limit_price
            fill_price = limit_price - improvement if side == "buy" else limit_price + improvement
        else:
            fill_price = 0
        
        return filled, fill_price

# Singleton
slippage_model = SlippageModel()
EOF

echo "   ✅ Slippage Model создан"

# ============================================
# 5. ИНТЕГРАЦИЯ В RISK MANAGER
# ============================================
echo "[5/5] 🔧 Интеграция в Risk Manager..."

cat > services/risk-manager/trading_risk_integration.py << 'EOF'
"""
Integration layer for trading risk components
"""
import sys
sys.path.append('/app/shared')

from trading.regime_detector import regime_detector, MarketRegime
from trading.position_sizing import position_sizer, SizingMethod
from trading.correlation_analyzer import correlation_analyzer
from trading.slippage_model import slippage_model, OrderType

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeRiskAssessment:
    approved: bool
    position_size: int
    position_value: float
    stop_loss: float
    regime: str
    regime_multiplier: float
    correlation_ok: bool
    expected_slippage_pct: float
    warnings: list
    details: dict

class TradingRiskIntegration:
    def __init__(self):
        self.regime_detector = regime_detector
        self.position_sizer = position_sizer
        self.correlation_analyzer = correlation_analyzer
        self.slippage_model = slippage_model
    
    def assess_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        avg_daily_volume: float = 1_000_000
    ) -> TradeRiskAssessment:
        """Полная оценка рисков сделки"""
        
        warnings = []
        
        # 1. Определить режим рынка
        regime_analysis = self.regime_detector.detect_regime(high, low, close)
        regime_multiplier = regime_analysis.recommended_position_multiplier
        
        if regime_analysis.regime == MarketRegime.CRISIS:
            warnings.append("🚨 CRISIS режим - торговля приостановлена")
            return TradeRiskAssessment(
                approved=False,
                position_size=0,
                position_value=0,
                stop_loss=0,
                regime=regime_analysis.regime.value,
                regime_multiplier=0,
                correlation_ok=False,
                expected_slippage_pct=0,
                warnings=warnings,
                details={"regime_analysis": regime_analysis.__dict__}
            )
        
        if regime_analysis.regime == MarketRegime.HIGH_VOLATILITY:
            warnings.append("⚠️ Высокая волатильность - размер позиции уменьшен")
        
        # 2. Рассчитать размер позиции
        position = self.position_sizer.calculate_size(
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            high=high,
            low=low,
            close=close,
            method=SizingMethod.ATR_BASED,
            side=side,
            regime_multiplier=regime_multiplier
        )
        
        if position.shares == 0:
            warnings.append("Размер позиции слишком мал")
            return TradeRiskAssessment(
                approved=False,
                position_size=0,
                position_value=0,
                stop_loss=0,
                regime=regime_analysis.regime.value,
                regime_multiplier=regime_multiplier,
                correlation_ok=True,
                expected_slippage_pct=0,
                warnings=warnings,
                details={}
            )
        
        # 3. Проверить корреляции
        self.correlation_analyzer.update_prices(symbol, close)
        correlation_ok, corr_message = self.correlation_analyzer.should_allow_trade(
            new_symbol=symbol,
            new_value=position.position_value,
            current_positions=current_positions,
            portfolio_value=portfolio_value
        )
        
        if not correlation_ok:
            warnings.append(corr_message)
        
        # 4. Оценить slippage
        slippage = self.slippage_model.calculate_slippage(
            price=entry_price,
            quantity=position.shares,
            side=side,
            avg_daily_volume=avg_daily_volume,
            volatility_ratio=regime_analysis.volatility_ratio,
            include_random=False
        )
        
        if slippage.slippage_pct > 0.5:
            warnings.append(f"⚠️ Высокий ожидаемый slippage: {slippage.slippage_pct:.2f}%")
        
        # Final decision
        approved = correlation_ok and position.shares > 0
        
        return TradeRiskAssessment(
            approved=approved,
            position_size=position.shares,
            position_value=position.position_value,
            stop_loss=position.stop_loss_price,
            regime=regime_analysis.regime.value,
            regime_multiplier=regime_multiplier,
            correlation_ok=correlation_ok,
            expected_slippage_pct=slippage.slippage_pct,
            warnings=warnings,
            details={
                "regime": regime_analysis.__dict__,
                "position": position.__dict__,
                "slippage": slippage.__dict__
            }
        )

# Singleton
trading_risk = TradingRiskIntegration()
EOF

echo "   ✅ Интеграция создана"

# ============================================
# КОПИРОВАНИЕ В СЕРВИСЫ
# ============================================
echo ""
echo "📦 Копирование модулей в сервисы..."

# Скопировать в risk-manager
mkdir -p services/risk-manager/shared/trading
cp services/shared/trading/*.py services/risk-manager/shared/trading/
touch services/risk-manager/shared/__init__.py
touch services/risk-manager/shared/trading/__init__.py

# Скопировать в strategy
mkdir -p services/strategy/shared/trading
cp services/shared/trading/*.py services/strategy/shared/trading/
touch services/strategy/shared/__init__.py
touch services/strategy/shared/trading/__init__.py

# Скопировать в backtest
mkdir -p services/backtest/shared/trading
cp services/shared/trading/*.py services/backtest/shared/trading/
touch services/backtest/shared/__init__.py
touch services/backtest/shared/trading/__init__.py

echo ""
echo "=============================================="
echo "✅ ВСЕ ТОРГОВЫЕ РИСКИ ИСПРАВЛЕНЫ!"
echo "=============================================="
echo ""
echo "📁 Созданные модули:"
echo "   • regime_detector.py    - Определение режима рынка"
echo "   • position_sizing.py    - ATR/Kelly/Vol-parity sizing"
echo "   • correlation_analyzer.py - Анализ корреляций"
echo "   • slippage_model.py     - Модель проскальзывания"
echo ""
echo "🔧 Режимы рынка:"
echo "   • TRENDING_UP/DOWN - Позиции до 120%"
echo "   • RANGING          - Позиции 70%"
echo "   • HIGH_VOLATILITY  - Позиции 50%"
echo "   • CRISIS           - Торговля остановлена"
echo ""
echo "📊 Position Sizing:"
echo "   • ATR-based    - Stop-loss = 2×ATR"
echo "   • Kelly        - 25% от оптимального Kelly"
echo "   • Vol-parity   - Таргет 15% годовой vol"
echo ""
echo "🔗 Correlation:"
echo "   • Порог: 0.7"
echo "   • Макс в коррелированных: 25%"
echo ""
echo "📉 Slippage Model:"
echo "   • Спред + Market Impact + Volatility"
echo "   • Учёт времени суток (открытие/закрытие)"
echo "   • Учёт типа ордера (market/limit/stop)"
echo ""
