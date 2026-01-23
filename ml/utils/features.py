"""Advanced Feature Engineering v3 - Fixed"""
import pandas as pd
import numpy as np

class FeatureBuilder:
    @staticmethod
    def _safe_divide(a, b, fill=0.0):
        """Safe division returning pandas Series"""
        if isinstance(a, pd.Series):
            result = a / b.replace(0, np.nan)
            return result.fillna(fill).replace([np.inf, -np.inf], fill)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(b != 0, a / b, fill)
                return np.where(np.isfinite(result), result, fill)
    
    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        return df.replace([np.inf, -np.inf], 0).fillna(0)
    
    @staticmethod
    def trend_features(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # SMAs
        for p in [5, 10, 20, 50, 100, 200]:
            sma = close.rolling(p, min_periods=1).mean()
            f[f'close_to_sma_{p}'] = (close - sma) / sma
        
        # SMA Crossovers
        for short, long in [(5,20), (10,50), (20,100), (50,200)]:
            sma_s = close.rolling(short, min_periods=1).mean()
            sma_l = close.rolling(long, min_periods=1).mean()
            f[f'sma_{short}_{long}_cross'] = (sma_s - sma_l) / close
            f[f'sma_{short}_{long}_signal'] = (sma_s > sma_l).astype(int)
        
        # EMA
        for p in [8, 13, 21, 55]:
            ema = close.ewm(span=p, min_periods=1).mean()
            f[f'close_to_ema_{p}'] = (close - ema) / ema
        
        # MACD
        ema12 = close.ewm(span=12, min_periods=1).mean()
        ema26 = close.ewm(span=26, min_periods=1).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, min_periods=1).mean()
        f['macd'] = macd / close
        f['macd_signal'] = signal / close
        f['macd_hist'] = (macd - signal) / close
        
        # ADX
        f['adx_14'] = FeatureBuilder._calc_adx(df, 14)
        f['adx_28'] = FeatureBuilder._calc_adx(df, 28)
        
        # DI
        plus_di, minus_di = FeatureBuilder._calc_dm(df, 14)
        f['di_plus'] = plus_di
        f['di_minus'] = minus_di
        f['di_diff'] = plus_di - minus_di
        
        # Momentum/Returns
        for p in [1, 2, 3, 5, 10, 20]:
            f[f'return_{p}d'] = close.pct_change(p)
        
        # RSI
        f['rsi_14'] = FeatureBuilder._calc_rsi(close, 14)
        f['rsi_above_50'] = (f['rsi_14'] > 50).astype(int)
        
        # Higher Highs / Lower Lows
        f['hh_10'] = (high > high.shift(1)).rolling(10, min_periods=1).sum()
        f['ll_10'] = (low < low.shift(1)).rolling(10, min_periods=1).sum()
        f['trend_score'] = f['hh_10'] - f['ll_10']
        
        # SMA Alignment
        sma20 = close.rolling(20, min_periods=1).mean()
        sma50 = close.rolling(50, min_periods=1).mean()
        sma200 = close.rolling(200, min_periods=1).mean()
        f['sma_bullish'] = ((close > sma20) & (sma20 > sma50) & (sma50 > sma200)).astype(int)
        f['sma_bearish'] = ((close < sma20) & (sma20 < sma50) & (sma50 < sma200)).astype(int)
        
        # Volume
        vol_sma = volume.rolling(20, min_periods=1).mean()
        f['volume_ratio'] = volume / vol_sma
        
        # OBV trend
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_sma = obv.rolling(20, min_periods=1).mean()
        f['obv_trend'] = (obv - obv_sma) / (obv_sma.abs() + 1)
        
        return FeatureBuilder._clean(f)
    
    @staticmethod
    def flat_features(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # Bollinger Bands
        for p in [10, 20, 50]:
            sma = close.rolling(p, min_periods=1).mean()
            std = close.rolling(p, min_periods=1).std()
            f[f'bb_width_{p}'] = (4 * std) / sma
            f[f'bb_pct_{p}'] = (close - (sma - 2*std)) / (4 * std + 1e-10)
        
        # ATR
        atr14 = FeatureBuilder._calc_atr(df, 14)
        atr28 = FeatureBuilder._calc_atr(df, 28)
        f['atr_pct_14'] = atr14 / close
        f['atr_pct_28'] = atr28 / close
        f['atr_ratio'] = atr14 / (atr28 + 1e-10)
        
        # Range
        for p in [5, 10, 20]:
            hmax = high.rolling(p, min_periods=1).max()
            lmin = low.rolling(p, min_periods=1).min()
            rng = hmax - lmin
            f[f'range_{p}'] = rng / close
            f[f'range_pos_{p}'] = (close - lmin) / (rng + 1e-10)
        
        f['range_contracting'] = (f['range_5'] < f['range_20']).astype(int)
        
        # ADX
        f['adx_14'] = FeatureBuilder._calc_adx(df, 14)
        f['adx_below_25'] = (f['adx_14'] < 25).astype(int)
        
        # RSI
        rsi = FeatureBuilder._calc_rsi(close, 14)
        f['rsi_14'] = rsi
        f['rsi_distance_50'] = abs(rsi - 50) / 50
        f['rsi_neutral'] = ((rsi > 40) & (rsi < 60)).astype(int)
        
        # Volume
        vol_sma = volume.rolling(20, min_periods=1).mean()
        f['volume_ratio'] = volume / vol_sma
        f['volume_declining'] = (f['volume_ratio'] < 0.8).astype(int)
        
        # Consolidation score
        f['consolidation'] = (f['adx_below_25'] + f['rsi_neutral'] + f['range_contracting']) / 3
        
        return FeatureBuilder._clean(f)
    
    @staticmethod
    def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        open_p = df['open'].astype(float)
        volume = df['volume'].astype(float)
        returns = close.pct_change()
        
        # Historical Volatility
        for p in [5, 10, 20, 40, 60]:
            f[f'hv_{p}'] = returns.rolling(p, min_periods=1).std() * np.sqrt(252)
        
        # Volatility ratios
        f['hv_5_20_ratio'] = f['hv_5'] / (f['hv_20'] + 1e-10)
        f['hv_10_60_ratio'] = f['hv_10'] / (f['hv_60'] + 1e-10)
        f['vol_expanding'] = (f['hv_5_20_ratio'] > 1.2).astype(int)
        
        # Vol of Vol
        f['vov'] = f['hv_20'].rolling(20, min_periods=1).std()
        
        # ATR
        for p in [7, 14, 28]:
            atr = FeatureBuilder._calc_atr(df, p)
            f[f'atr_{p}'] = atr / close
        
        # Parkinson Volatility (using pandas)
        hl_log = np.log(high / low)
        f['parkinson'] = np.sqrt((1/(4*np.log(2))) * (hl_log**2).rolling(20, min_periods=1).mean())
        
        # Daily range
        f['daily_range'] = (high - low) / close
        f['range_ma'] = f['daily_range'].rolling(20, min_periods=1).mean()
        f['range_expansion'] = f['daily_range'] / (f['range_ma'] + 1e-10)
        
        # Gap
        gap = (open_p - close.shift(1)).abs()
        f['gap_pct'] = gap / (close.shift(1) + 1e-10)
        f['large_gap'] = (f['gap_pct'] > f['gap_pct'].rolling(50, min_periods=1).quantile(0.9)).astype(int)
        
        # Extreme moves
        f['extreme_up'] = (returns > returns.rolling(50, min_periods=1).quantile(0.95)).astype(int)
        f['extreme_down'] = (returns < returns.rolling(50, min_periods=1).quantile(0.05)).astype(int)
        f['extreme_count'] = (f['extreme_up'] + f['extreme_down']).rolling(10, min_periods=1).sum()
        
        # Volume
        f['volume_norm'] = volume / volume.rolling(20, min_periods=1).mean()
        
        # Z-score
        vol_mean = f['hv_20'].rolling(60, min_periods=1).mean()
        vol_std = f['hv_20'].rolling(60, min_periods=1).std()
        f['vol_zscore'] = (f['hv_20'] - vol_mean) / (vol_std + 1e-10)
        
        return FeatureBuilder._clean(f)
    
    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
    
    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean()
        
        plus_di = 100 * plus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        return dx.rolling(period, min_periods=1).mean().fillna(20)
    
    @staticmethod
    def _calc_dm(df: pd.DataFrame, period: int = 14) -> tuple:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean()
        
        plus_di = 100 * plus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10)
        
        return plus_di.fillna(0), minus_di.fillna(0)
