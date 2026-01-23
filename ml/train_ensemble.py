#!/usr/bin/env python3
"""Train all ensemble models v3"""
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
        
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                except:
                    pass
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"📊 Loaded {len(df)} rows from database")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        return df
    finally:
        await conn.close()

def train_all(df: pd.DataFrame, output_dir: Path) -> dict:
    results = {}
    
    # Trend
    print("\n" + "="*50)
    print("🔄 Training TREND Model (v3 enhanced)...")
    print("="*50)
    trend_model = TrendModel(use_gpu=True)
    results['trend'] = trend_model.train(df)
    trend_model.save(str(output_dir / "trend_model.joblib"))
    print(f"✅ Trend: Accuracy={results['trend']['accuracy']:.2%}")
    
    # Flat
    print("\n" + "="*50)
    print("🔄 Training FLAT Model (v3 enhanced)...")
    print("="*50)
    flat_model = FlatModel(use_gpu=True)
    results['flat'] = flat_model.train(df)
    flat_model.save(str(output_dir / "flat_model.joblib"))
    print(f"✅ Flat: Accuracy={results['flat']['accuracy']:.2%}")
    
    # Volatility (hybrid)
    print("\n" + "="*50)
    print("🔄 Training VOLATILITY Model (v3 hybrid)...")
    print("="*50)
    vol_model = VolatilityModel(use_gpu=True)
    results['volatility'] = vol_model.train(df)
    vol_model.save(str(output_dir / "volatility_model.joblib"))
    print(f"✅ Volatility: Accuracy={results['volatility']['classification_accuracy']:.2%}")
    print(f"   Regression MAE: {results['volatility']['regression_mae']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=["db", "csv"], default="db")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Ensemble Model Training v3")
    print(f"📁 Output: {output_dir}")
    
    df = asyncio.run(load_data_from_db())
    
    start = datetime.now()
    results = train_all(df, output_dir)
    elapsed = (datetime.now() - start).total_seconds()
    
    print("\n" + "="*50)
    print("📊 TRAINING COMPLETE v3")
    print("="*50)
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"📈 Trend:      {results['trend']['accuracy']:.2%}")
    print(f"📊 Flat:       {results['flat']['accuracy']:.2%}")
    print(f"📉 Volatility: {results['volatility']['classification_accuracy']:.2%}")
    print(f"\n✅ Saved to {output_dir}/")

if __name__ == "__main__":
    main()
