"""Enhanced Feature Engineering for Ensemble Models"""
import pandas as pd
import numpy as np
from typing import Optional

class FeatureBuilder:
    """Builds features for different market regimes"""
    
    @staticmethod
    def trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced features for trend detection"""
        f = pd.DataFrame(index=df.index)
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # === SMA Crossovers ===
        for short, long in [(5,20), (10,50), (20,100), (50,200)]:
            sma_short = close.rolling(short).mean()
            sma_long = close.rolling(long).mean()
            f[f'sma_{short}_{long}_cross'] = (sma_short - sma_long) / close
            f[f'sma_{short}_{long}_signal'] = (sma_short > sma_long).astype(int)
        
        # === EMA Crossovers ===
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        f['ema_12_26_cross'] = (ema_12 - ema_26) / close
        
        # === ADX - Trend Strength ===
        f['adx_14'] = FeatureBuilder._calc_adx(df, 14)
        f['adx_28'] = FeatureBuilder._calc_adx(df, 28)
        
        # === Directional Movement ===
        plus_dm, minus_dm = FeatureBuilder._calc_dm(df, 14)
        f['di_plus'] = plus_dm
        f['di_minus'] = minus_dm
        f['di_diff'] = plus_dm - minus_dm
        
        # === Momentum ===
        for p in [3, 5, 10, 20]:
            f[f'momentum_{p}'] = close.pct_change(p)
            f[f'roc_{p}'] = (close - close.shift(p)) / close.shift(p)
        
        # === Higher Highs / Lower Lows ===
        f['hh_count_5'] = (high > high.shift(1)).rolling(5).sum()
        f['ll_count_5'] = (low < low.shift(1)).rolling(5).sum()
        f['hh_count_10'] = (high > high.shift(1)).rolling(10).sum()
        f['ll_count_10'] = (low < low.shift(1)).rolling(10).sum()
        f['trend_consistency'] = f['hh_count_10'] - f['ll_count_10']
        
        # === Price Position ===
        for p in [10, 20, 50]:
            sma = close.rolling(p).mean()
            f[f'close_to_sma_{p}'] = (close - sma) / sma
        
        f['close_above_sma20'] = (close > close.rolling(20).mean()).astype(int)
        f['close_above_sma50'] = (close > close.rolling(50).mean()).astype(int)
        f['close_above_sma200'] = (close > close.rolling(200).mean()).astype(int)
        f['sma_alignment'] = f['close_above_sma20'] + f['close_above_sma50'] + f['close_above_sma200']
        
        # === OBV (On-Balance Volume) ===
        obv = (np.sign(close.diff()) * volume).cumsum()
        f['obv_slope_10'] = obv.diff(10) / obv.rolling(10).mean()
        f['obv_slope_20'] = obv.diff(20) / obv.rolling(20).mean()
        
        # === VWAP Deviation ===
        vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        f['vwap_deviation'] = (close - vwap) / vwap
        
        # === RSI Trend ===
        rsi = FeatureBuilder._calc_rsi(close, 14)
        f['rsi_14'] = rsi
        f['rsi_above_50'] = (rsi > 50).astype(int)
        f['rsi_trend'] = rsi.diff(5)
        
        # === MACD ===
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        f['macd'] = macd
        f['macd_signal'] = signal
        f['macd_hist'] = macd - signal
        f['macd_hist_slope'] = f['macd_hist'].diff(3)
        
        # === Ichimoku (simplified) ===
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        f['ichimoku_tk_cross'] = (tenkan - kijun) / close
        f['price_vs_kijun'] = (close - kijun) / kijun
        
        # === Seasonality ===
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            f['day_of_week'] = dates.dt.dayofweek
            f['month'] = dates.dt.month
            f['is_month_end'] = dates.dt.is_month_end.astype(int)
            f['is_quarter_end'] = dates.dt.is_quarter_end.astype(int)
        
        return f.fillna(0)
    
    @staticmethod
    def flat_features(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced features for sideways/range detection"""
        f = pd.DataFrame(index=df.index)
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # === Bollinger Band Width ===
        for p in [10, 20, 50]:
            sma = close.rolling(p).mean()
            std = close.rolling(p).std()
            f[f'bb_width_{p}'] = (4 * std) / sma
            f[f'bb_pct_{p}'] = (close - (sma - 2*std)) / (4 * std)
        
        # === ATR relative to price ===
        atr = FeatureBuilder._calc_atr(df, 14)
        f['atr_pct'] = atr / close
        f['atr_pct_20'] = FeatureBuilder._calc_atr(df, 20) / close
        f['atr_change'] = atr.pct_change(5)
        
        # === Range Metrics ===
        for p in [5, 10, 20]:
            high_max = high.rolling(p).max()
            low_min = low.rolling(p).min()
            f[f'range_{p}'] = (high_max - low_min) / close
            f[f'range_position_{p}'] = (close - low_min) / (high_max - low_min + 1e-10)
        
        # === Range Contraction/Expansion ===
        f['range_ratio_5_20'] = f['range_5'] / (f['range_20'] + 1e-10)
        f['range_contracting'] = (f['range_5'] < f['range_20']).astype(int)
        
        # === ADX (low = no trend = flat) ===
        f['adx_14'] = FeatureBuilder._calc_adx(df, 14)
        f['adx_below_25'] = (f['adx_14'] < 25).astype(int)
        f['adx_below_20'] = (f['adx_14'] < 20).astype(int)
        
        # === RSI Distance from 50 ===
        rsi = FeatureBuilder._calc_rsi(close, 14)
        f['rsi_distance_50'] = abs(rsi - 50)
        f['rsi_in_range'] = ((rsi > 40) & (rsi < 60)).astype(int)
        
        # === Volume Profile ===
        f['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        f['volume_declining'] = (f['volume_ma_ratio'] < 0.8).astype(int)
        f['volume_std_20'] = volume.rolling(20).std() / volume.rolling(20).mean()
        
        # === Price Oscillation ===
        sma_20 = close.rolling(20).mean()
        f['price_to_sma20'] = (close - sma_20) / sma_20
        f['oscillation_count'] = ((close - sma_20).diff().apply(np.sign).diff().abs() > 0).rolling(10).sum()
        
        # === Mean Reversion Score ===
        f['mean_reversion_score'] = (
            f['rsi_in_range'] + 
            f['adx_below_25'] + 
            f['volume_declining'] + 
            f['range_contracting']
        ) / 4
        
        return f.fillna(0)
    
    @staticmethod
    def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced features for volatility regime"""
        f = pd.DataFrame(index=df.index)
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        returns = close.pct_change()
        
        # === Historical Volatility ===
        for p in [5, 10, 20, 60]:
            f[f'volatility_{p}'] = returns.rolling(p).std() * np.sqrt(252)
        
        # === Volatility Ratios ===
        f['vol_ratio_5_20'] = f['volatility_5'] / (f['volatility_20'] + 1e-10)
        f['vol_ratio_10_60'] = f['volatility_10'] / (f['volatility_60'] + 1e-10)
        
        # === Volatility of Volatility ===
        f['vol_of_vol'] = f['volatility_20'].rolling(10).std()
        f['vol_regime_change'] = f['volatility_20'].diff(5).abs()
        
        # === ATR ===
        f['atr_14'] = FeatureBuilder._calc_atr(df, 14)
        f['atr_change_5'] = f['atr_14'].pct_change(5)
        f['atr_change_10'] = f['atr_14'].pct_change(10)
        
        # === Intraday Range ===
        f['daily_range'] = (high - low) / close
        f['daily_range_ma'] = f['daily_range'].rolling(20).mean()
        f['range_expansion'] = f['daily_range'] / (f['daily_range_ma'] + 1e-10)
        
        # === Gap Analysis ===
        f['gap'] = abs(df['open'].astype(float) - close.shift(1)) / close.shift(1)
        f['gap_ma'] = f['gap'].rolling(10).mean()
        f['large_gap'] = (f['gap'] > f['gap'].rolling(50).quantile(0.9)).astype(int)
        
        # === Parkinson Volatility ===
        f['parkinson_vol'] = np.sqrt(
            (1/(4*np.log(2))) * (np.log(high/low)**2).rolling(20).mean()
        )
        
        # === Garman-Klass Volatility ===
        f['gk_vol'] = np.sqrt(
            0.5 * (np.log(high/low)**2).rolling(20).mean() -
            (2*np.log(2)-1) * (np.log(close/df['open'].astype(float))**2).rolling(20).mean()
        )
        
        # === Volume-Volatility ===
        f['volume_norm'] = volume / volume.rolling(20).mean()
        f['vol_volume_corr'] = f['volatility_20'].rolling(20).corr(f['volume_norm'])
        
        # === Extreme Moves ===
        f['extreme_up'] = (returns > returns.rolling(50).quantile(0.95)).astype(int)
        f['extreme_down'] = (returns < returns.rolling(50).quantile(0.05)).astype(int)
        f['extreme_count_10'] = (f['extreme_up'] + f['extreme_down']).rolling(10).sum()
        
        # === Z-Score of Volatility ===
        vol_mean = f['volatility_20'].rolling(60).mean()
        vol_std = f['volatility_20'].rolling(60).std()
        f['vol_zscore'] = (f['volatility_20'] - vol_mean) / (vol_std + 1e-10)
        
        return f.fillna(0)
    
    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
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
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return adx.fillna(20)
    
    @staticmethod
    def _calc_dm(df: pd.DataFrame, period: int = 14) -> tuple:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
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
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        
        return plus_di.fillna(0), minus_di.fillna(0)
