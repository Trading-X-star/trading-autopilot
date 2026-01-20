"""Feature Generator для ML-модели v7"""
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger("feature_generator")

REQUIRED_FEATURES = [
    'return_5d', 'roc_5', 'close_to_sma_10', 'macd_hist', 'macd_signal',
    'close_to_sma_5', 'bb_position', 'return_10d', 'return_3d', 'return_2d',
    'close_to_sma_20', 'close_to_sma_50', 'return_1d', 'momentum_10', 'rsi_14',
    'bb_width', 'roc_10', 'volatility_20', 'volatility_10', 'volume_sma_20'
]

def calc_sma(values: list, period: int) -> float:
    if len(values) < period:
        return values[-1] if values else 0
    return np.mean(values[-period:])

def calc_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(closes: list) -> tuple:
    if len(closes) < 26:
        return 0, 0, 0
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    macd_line = ema12 - ema26
    signal = calc_ema(closes[-9:], 9) if len(closes) >= 9 else macd_line
    hist = macd_line - signal
    return macd_line, signal, hist

def calc_ema(values: list, period: int) -> float:
    if not values:
        return 0
    if len(values) < period:
        return np.mean(values)
    multiplier = 2 / (period + 1)
    ema = np.mean(values[:period])
    for price in values[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def generate_features(history: list) -> Optional[Dict[str, float]]:
    """Генерирует фичи из истории OHLCV"""
    if not history or len(history) < 50:
        return None
    
    try:
        closes = [h.get('close', h.get('price', h.get('c', 0))) for h in history]
        volumes = [h.get('volume', h.get('v', 1)) for h in history]
        highs = [h.get('high', h.get('h', h.get('price', c))) for h, c in zip(history, closes)]
        lows = [h.get('low', h.get('l', h.get('price', c))) for h, c in zip(history, closes)]
        
        c = closes[-1]
        
        # Returns
        return_1d = (c / closes[-2] - 1) if len(closes) >= 2 else 0
        return_2d = (c / closes[-3] - 1) if len(closes) >= 3 else 0
        return_3d = (c / closes[-4] - 1) if len(closes) >= 4 else 0
        return_5d = (c / closes[-6] - 1) if len(closes) >= 6 else 0
        return_10d = (c / closes[-11] - 1) if len(closes) >= 11 else 0
        
        # ROC
        roc_5 = ((c - closes[-6]) / closes[-6] * 100) if len(closes) >= 6 else 0
        roc_10 = ((c - closes[-11]) / closes[-11] * 100) if len(closes) >= 11 else 0
        
        # SMAs
        sma5 = calc_sma(closes, 5)
        sma10 = calc_sma(closes, 10)
        sma20 = calc_sma(closes, 20)
        sma50 = calc_sma(closes, 50)
        
        close_to_sma_5 = (c / sma5 - 1) if sma5 else 0
        close_to_sma_10 = (c / sma10 - 1) if sma10 else 0
        close_to_sma_20 = (c / sma20 - 1) if sma20 else 0
        close_to_sma_50 = (c / sma50 - 1) if sma50 else 0
        
        # MACD
        _, macd_signal, macd_hist = calc_macd(closes)
        
        # RSI
        rsi_14 = calc_rsi(closes, 14)
        
        # Bollinger Bands
        std20 = np.std(closes[-20:]) if len(closes) >= 20 else 0
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_width = (bb_upper - bb_lower) / sma20 if sma20 else 0
        bb_position = (c - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # Momentum
        momentum_10 = c - closes[-11] if len(closes) >= 11 else 0
        
        # Volatility
        volatility_10 = np.std(closes[-10:]) / np.mean(closes[-10:]) if len(closes) >= 10 else 0
        volatility_20 = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0
        
        # Volume
        volume_sma_20 = calc_sma(volumes, 20)
        
        return {
            'return_1d': return_1d,
            'return_2d': return_2d,
            'return_3d': return_3d,
            'return_5d': return_5d,
            'return_10d': return_10d,
            'roc_5': roc_5,
            'roc_10': roc_10,
            'close_to_sma_5': close_to_sma_5,
            'close_to_sma_10': close_to_sma_10,
            'close_to_sma_20': close_to_sma_20,
            'close_to_sma_50': close_to_sma_50,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'rsi_14': rsi_14,
            'bb_position': bb_position,
            'bb_width': bb_width,
            'momentum_10': momentum_10,
            'volatility_10': volatility_10,
            'volatility_20': volatility_20,
            'volume_sma_20': volume_sma_20
        }
    except Exception as e:
        logger.error(f"Feature generation error: {e}")
        return None
