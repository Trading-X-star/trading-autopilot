#!/usr/bin/env python3
"""
Train all ensemble models: Trend, Flat, Volatility
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import asyncio
import asyncpg

from trainers.trend_model import TrendModel
from trainers.flat_model import FlatModel
from trainers.volatility_model import VolatilityModel

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading")

async def load_data_from_db() -> pd.DataFrame:
    """Load OHLCV data from PostgreSQL"""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch("""
            SELECT ticker, date, open, high, low, close, volume,
                   rsi_14, macd, macd_signal, macd_hist, 
                   bb_upper, bb_lower, bb_pct, atr_14, adx_14
            FROM features
            WHERE date > NOW() - INTERVAL '3 years'
            ORDER BY ticker, date
        """)
        df = pd.DataFrame([dict(r) for r in rows])
        
        # Convert Decimal to float
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                except:
                    pass
        
        # Ensure numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                       'bb_upper', 'bb_lower', 'bb_pct', 'atr_14', 'adx_14']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"📊 Loaded {len(df)} rows from database")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        return df
    finally:
        await conn.close()

def train_all_models(df: pd.DataFrame, output_dir: Path) -> dict:
    """Train all three specialized models"""
    results = {}
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 1. Train Trend Model
    print("\n" + "="*50)
    print("🔄 Training TREND Model...")
    print("="*50)
    trend_model = TrendModel()
    trend_results = trend_model.train(df)
    trend_model.save(str(output_dir / "trend_model.joblib"))
    results['trend'] = trend_results
    print(f"✅ Trend: Accuracy={trend_results['accuracy']:.2%} (±{trend_results['accuracy_std']:.2%})")
    
    # 2. Train Flat Model
    print("\n" + "="*50)
    print("🔄 Training FLAT Model...")
    print("="*50)
    flat_model = FlatModel()
    flat_results = flat_model.train(df)
    flat_model.save(str(output_dir / "flat_model.joblib"))
    results['flat'] = flat_results
    print(f"✅ Flat: Accuracy={flat_results['accuracy']:.2%}, Flat ratio={flat_results['flat_ratio']:.2%}")
    
    # 3. Train Volatility Model
    print("\n" + "="*50)
    print("🔄 Training VOLATILITY Model...")
    print("="*50)
    vol_model = VolatilityModel()
    vol_results = vol_model.train(df)
    vol_model.save(str(output_dir / "volatility_model.joblib"))
    results['volatility'] = vol_results
    print(f"✅ Volatility: Accuracy={vol_results['accuracy']:.2%}")
    print(f"   LOW={vol_results['regime_distribution']['low']:.1%}, "
          f"MED={vol_results['regime_distribution']['medium']:.1%}, "
          f"HIGH={vol_results['regime_distribution']['high']:.1%}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=["db", "csv"], default="db")
    parser.add_argument("--csv-path", type=str, default="data/ohlcv.csv")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Ensemble Model Training")
    print(f"📁 Output: {output_dir}")
    
    # Load data
    df = asyncio.run(load_data_from_db())
    
    if len(df) < 1000:
        print(f"⚠️ Warning: Only {len(df)} samples")
    
    # Train
    start = datetime.now()
    results = train_all_models(df, output_dir)
    elapsed = (datetime.now() - start).total_seconds()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TRAINING COMPLETE")
    print("="*50)
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"📈 Trend:      {results['trend']['accuracy']:.2%}")
    print(f"📊 Flat:       {results['flat']['accuracy']:.2%}")
    print(f"📉 Volatility: {results['volatility']['accuracy']:.2%}")
    print(f"\n✅ Saved to {output_dir}/")

if __name__ == "__main__":
    main()
