"""Feature engineering for specialized models"""
import pandas as pd
import numpy as np

class FeatureBuilder:
    """Builds features for different market regimes"""
    
    @staticmethod
    def trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Features for trend detection"""
        f = pd.DataFrame(index=df.index)
        
        # Moving averages crossovers
        for short, long in [(5,20), (10,50), (20,100)]:
            f[f'sma_{short}_{long}_cross'] = (
                df['close'].rolling(short).mean() - 
                df['close'].rolling(long).mean()
            ) / df['close']
        
        # ADX - trend strength
        f['adx_14'] = df.get('adx_14', FeatureBuilder._calc_adx(df, 14))
        
        # Directional movement
        f['di_plus'] = df.get('di_plus', 0)
        f['di_minus'] = df.get('di_minus', 0)
        
        # Price momentum
        for p in [5, 10, 20]:
            f[f'momentum_{p}'] = df['close'].pct_change(p)
        
        # Higher highs / Lower lows
        f['hh_count'] = (df['high'] > df['high'].shift(1)).rolling(10).sum()
        f['ll_count'] = (df['low'] < df['low'].shift(1)).rolling(10).sum()
        
        # Trend consistency
        f['close_above_sma20'] = (df['close'] > df['close'].rolling(20).mean()).astype(int)
        f['close_above_sma50'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)
        
        return f.fillna(0)
    
    @staticmethod
    def flat_features(df: pd.DataFrame) -> pd.DataFrame:
        """Features for sideways/range detection"""
        f = pd.DataFrame(index=df.index)
        
        # Bollinger Band width (narrow = flat)
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        f['bb_width'] = (4 * std) / sma
        
        # ATR relative to price
        f['atr_pct'] = df.get('atr_14', df['high'] - df['low']) / df['close']
        
        # Range metrics
        for p in [5, 10, 20]:
            high_max = df['high'].rolling(p).max()
            low_min = df['low'].rolling(p).min()
            f[f'range_{p}'] = (high_max - low_min) / df['close']
        
        # ADX low = no trend = flat
        f['adx_14'] = df.get('adx_14', 20)
        
        # Mean reversion signals
        f['rsi_distance_50'] = abs(df.get('rsi_14', 50) - 50)
        
        # Volume decline in flat
        f['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Price oscillation around mean
        f['price_to_sma20'] = (df['close'] - sma) / sma
        
        return f.fillna(0)
    
    @staticmethod
    def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Features for volatility regime"""
        f = pd.DataFrame(index=df.index)
        
        # Historical volatility
        for p in [5, 10, 20, 60]:
            f[f'volatility_{p}'] = df['close'].pct_change().rolling(p).std() * np.sqrt(252)
        
        # Volatility of volatility
        f['vol_of_vol'] = f['volatility_20'].rolling(10).std()
        
        # ATR
        f['atr_14'] = df.get('atr_14', (df['high'] - df['low']).rolling(14).mean())
        f['atr_change'] = f['atr_14'].pct_change(5)
        
        # Intraday range
        f['daily_range'] = (df['high'] - df['low']) / df['close']
        f['range_expansion'] = f['daily_range'] / f['daily_range'].rolling(20).mean()
        
        # Gap analysis
        f['gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Volume-volatility correlation
        f['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Parkinson volatility
        f['parkinson_vol'] = np.sqrt(
            (1/(4*np.log(2))) * (np.log(df['high']/df['low'])**2).rolling(20).mean()
        )
        
        return f.fillna(0)
    
    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        high, low, close = df['high'], df['low'], df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(20)
