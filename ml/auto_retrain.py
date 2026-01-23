#!/usr/bin/env python3
"""Auto-retrain system for ML ensemble"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import asyncio
import asyncpg

sys.path.insert(0, str(Path(__file__).parent))

from trainers.trend_model import TrendModel
from trainers.flat_model import FlatModel
from trainers.volatility_model import VolatilityModel
from ensemble_orchestrator import EnsembleOrchestrator

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading")
MODELS_DIR = Path(__file__).parent / "models"

class AutoRetrainer:
    """Automatic model retraining with drift detection"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.min_samples_for_retrain = 1000
        self.performance_threshold = 0.05  # Retrain if accuracy drops 5%
        
    async def check_drift(self) -> dict:
        """Check for data drift and model degradation"""
        orchestrator = EnsembleOrchestrator(str(self.models_dir))
        if not orchestrator.load_models():
            return {'status': 'models_not_found', 'should_retrain': True}
        
        drift_report = orchestrator.check_drift()
        
        should_retrain = (
            drift_report.get('status') == 'drift_detected' or
            drift_report.get('avg_confidence', 1.0) < 0.5
        )
        
        return {
            'status': drift_report.get('status'),
            'should_retrain': should_retrain,
            'drift_report': drift_report
        }
    
    async def load_recent_data(self, days: int = 90) -> 'pd.DataFrame':
        """Load recent data for retraining"""
        import pandas as pd
        from decimal import Decimal
        
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            rows = await conn.fetch(f"""
                SELECT ticker, date, open, high, low, close, volume,
                       rsi_14, macd, macd_signal, macd_hist, atr_14, adx_14
                FROM features
                WHERE date > NOW() - INTERVAL '{days} days'
                ORDER BY ticker, date
            """)
            
            df = pd.DataFrame([dict(r) for r in rows])
            
            # Convert Decimal
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                    except:
                        pass
            
            return df
        finally:
            await conn.close()
    
    async def retrain_all(self, df: 'pd.DataFrame') -> dict:
        """Retrain all models"""
        results = {}
        
        # Trend Model
        print("🔄 Retraining Trend Model...")
        trend_model = TrendModel(use_gpu=True)
        results['trend'] = trend_model.train(df)
        trend_model.save(str(self.models_dir / "trend_model.joblib"))
        print(f"✅ Trend: {results['trend']['accuracy']:.2%}")
        
        # Flat Model
        print("🔄 Retraining Flat Model...")
        flat_model = FlatModel(use_gpu=True)
        results['flat'] = flat_model.train(df)
        flat_model.save(str(self.models_dir / "flat_model.joblib"))
        print(f"✅ Flat: {results['flat']['accuracy']:.2%}")
        
        # Volatility Model
        print("🔄 Retraining Volatility Model...")
        vol_model = VolatilityModel(use_gpu=True)
        results['volatility'] = vol_model.train(df)
        vol_model.save(str(self.models_dir / "volatility_model.joblib"))
        print(f"✅ Volatility: {results['volatility']['classification_accuracy']:.2%}")
        
        # Save metadata
        metadata = {
            'retrained_at': datetime.now().isoformat(),
            'samples': len(df),
            'results': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} 
                       for k, v in results.items()}
        }
        
        with open(self.models_dir / "retrain_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return results
    
    async def run(self):
        """Main auto-retrain loop"""
        print("🔍 Checking for drift...")
        drift = await self.check_drift()
        
        if drift['should_retrain']:
            print(f"⚠️ Drift detected: {drift.get('status')}")
            print("📊 Loading recent data...")
            df = await self.load_recent_data(days=365)
            
            if len(df) >= self.min_samples_for_retrain:
                print(f"📈 {len(df)} samples loaded")
                results = await self.retrain_all(df)
                print("\n✅ Retraining complete!")
                return {'status': 'retrained', 'results': results}
            else:
                print(f"❌ Not enough data: {len(df)} < {self.min_samples_for_retrain}")
                return {'status': 'insufficient_data', 'samples': len(df)}
        else:
            print("✅ No drift detected, models are up to date")
            return {'status': 'no_retrain_needed'}

if __name__ == "__main__":
    retrainer = AutoRetrainer()
    result = asyncio.run(retrainer.run())
    print(f"\nResult: {result}")
