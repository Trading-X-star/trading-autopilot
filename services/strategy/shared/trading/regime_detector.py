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
