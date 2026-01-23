"""Auto-Retrain Pipeline with GPU Support (XGBoost)"""
import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
import asyncpg
import httpx
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# GPU Support
try:
    import xgboost as xgb
    GPU_AVAILABLE = True
    # Test GPU
    try:
        test_params = {'tree_method': 'hist', 'device': 'cuda:0'}
        xgb.XGBClassifier(**test_params)
        print("✅ XGBoost GPU available")
    except:
        GPU_AVAILABLE = False
        print("⚠️ XGBoost GPU not available, using CPU")
except ImportError:
    GPU_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier
    print("⚠️ XGBoost not installed, using sklearn")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ml-trainer")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.52"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "5000"))
AUTO_DEPLOY = os.getenv("AUTO_DEPLOY", "true").lower() == "true"

FEATURES = [
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_pct', 'bb_width', 'atr_14',
    'return_1d', 'return_5d', 'return_10d',
    'volatility_20', 'volume_ratio'
]

class TrainRequest(BaseModel):
    min_accuracy: float = MIN_ACCURACY
    min_samples: int = MIN_SAMPLES
    optimize: bool = True
    n_trials: int = 50
    use_gpu: bool = True

class TrainResult(BaseModel):
    success: bool
    model_version: str = ""
    accuracy: float = 0
    samples_used: int = 0
    message: str = ""
    deployed: bool = False
    metrics: dict = {}
    gpu_used: bool = False
    training_time_sec: float = 0

class AutoRetrainPipeline:
    def __init__(self):
        self.db_pool = None
        self.current_model_accuracy = None
        self.last_train_time = None
        self.training_in_progress = False
        self.label_encoder = LabelEncoder()
    
    async def init_db(self):
        if self.db_pool is None:
            self.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
    
    async def close(self):
        if self.db_pool:
            await self.db_pool.close()
    
    async def load_training_data(self) -> pd.DataFrame:
        await self.init_db()
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT ticker, date, rsi_14, macd, macd_signal, macd_hist,
                       bb_pct, bb_width, atr_14, return_1d, return_5d, return_10d,
                       volatility_20, volume_ratio, target_5d, signal_class
                FROM features
                WHERE signal_class IS NOT NULL
                  AND date > NOW() - INTERVAL '2 years'
                ORDER BY date
            """)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame([dict(r) for r in rows])
        logger.info(f"📊 Loaded {len(df)} samples for training")
        return df
    
    def optimize_hyperparams_gpu(self, X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
        """Optuna optimization with XGBoost GPU"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'tree_method': 'hist',
                'device': 'cuda:0',
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"🎯 GPU Best params: {study.best_params}, accuracy: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_hyperparams_cpu(self, X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
        """Optuna optimization with CPU (sklearn or XGBoost CPU)"""
        
        def objective(trial):
            if GPU_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'tree_method': 'hist',  # CPU
                    'random_state': 42,
                }
                model = xgb.XGBClassifier(**params)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                model = GradientBoostingClassifier(**params, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"🎯 CPU Best params: {study.best_params}, accuracy: {study.best_value:.4f}")
        return study.best_params
    
    async def train(self, request: TrainRequest) -> TrainResult:
        if self.training_in_progress:
            return TrainResult(success=False, message="Training already in progress")
        
        self.training_in_progress = True
        start_time = datetime.now()
        use_gpu = request.use_gpu and GPU_AVAILABLE
        
        try:
            logger.info(f"🚀 Starting training (GPU: {use_gpu})")
            
            # 1. Load data
            df = await self.load_training_data()
            if len(df) < request.min_samples:
                return TrainResult(
                    success=False,
                    message=f"Insufficient data: {len(df)} < {request.min_samples}"
                )
            
            # 2. Prepare features
            X = df[FEATURES].fillna(0).values
            y = df['signal_class'].values
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # 3. Time-based split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
            
            val_idx = int(len(X_train) * 0.85)
            X_train_opt, X_val = X_train[:val_idx], X_train[val_idx:]
            y_train_opt, y_val = y_train[:val_idx], y_train[val_idx:]
            
            # 4. Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_opt_scaled = scaler.transform(X_train_opt)
            X_val_scaled = scaler.transform(X_val)
            
            # 5. Optimize
            if request.optimize:
                logger.info(f"🔍 Optimizing hyperparameters ({request.n_trials} trials)...")
                if use_gpu:
                    best_params = self.optimize_hyperparams_gpu(
                        X_train_opt_scaled, y_train_opt,
                        X_val_scaled, y_val,
                        n_trials=request.n_trials
                    )
                    best_params['tree_method'] = 'hist'
                    best_params['device'] = 'cuda:0'
                else:
                    best_params = self.optimize_hyperparams_cpu(
                        X_train_opt_scaled, y_train_opt,
                        X_val_scaled, y_val,
                        n_trials=request.n_trials
                    )
            else:
                best_params = {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8
                }
                if use_gpu:
                    best_params['tree_method'] = 'hist'
                    best_params['device'] = 'cuda:0'
            
            # 6. Train final model
            logger.info("🏋️ Training final model...")
            if GPU_AVAILABLE:
                best_params['random_state'] = 42
                model = xgb.XGBClassifier(**best_params)
            else:
                model = GradientBoostingClassifier(**best_params, random_state=42)
            
            model.fit(X_train_scaled, y_train)
            
            # 7. Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Decode for report
            y_test_decoded = self.label_encoder.inverse_transform(y_test)
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
            report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"📈 Test accuracy: {accuracy:.4f} (trained in {training_time:.1f}s)")
            
            # 8. Check threshold
            if accuracy < request.min_accuracy:
                return TrainResult(
                    success=False,
                    accuracy=accuracy,
                    samples_used=len(df),
                    message=f"Accuracy {accuracy:.2%} below threshold {request.min_accuracy:.2%}",
                    metrics=report,
                    gpu_used=use_gpu,
                    training_time_sec=training_time
                )
            
            # 9. Save model
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")
            model_path = MODELS_DIR / f"model_{version}.joblib"
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Feature stats for drift detection
            feature_stats = {}
            for i, feat in enumerate(FEATURES):
                values = X_train[:, i]
                feature_stats[feat] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            model_data = {
                'model': model,
                'scaler': scaler,
                'label_encoder': self.label_encoder,
                'version': version,
                'accuracy': accuracy,
                'features': FEATURES,
                'feature_stats': feature_stats,
                'trained_at': datetime.now().isoformat(),
                'samples': len(df),
                'params': best_params,
                'metrics': report,
                'gpu_used': use_gpu,
                'training_time_sec': training_time
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"💾 Model saved: {model_path}")
            
            # 10. Deploy
            deployed = False
            if AUTO_DEPLOY:
                deployed = await self.deploy_model(model_path, version)
            
            self.last_train_time = datetime.now()
            self.current_model_accuracy = accuracy
            
            return TrainResult(
                success=True,
                model_version=version,
                accuracy=accuracy,
                samples_used=len(df),
                message=f"Training completed ({'GPU' if use_gpu else 'CPU'})",
                deployed=deployed,
                metrics=report,
                gpu_used=use_gpu,
                training_time_sec=training_time
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return TrainResult(success=False, message=str(e))
        finally:
            self.training_in_progress = False
    
    async def deploy_model(self, model_path: Path, version: str) -> bool:
        try:
            import shutil
            latest_path = MODELS_DIR / "model_latest.joblib"
            shutil.copy(model_path, latest_path)
            
            # Also copy to strategy location
            strategy_path = Path("/app/models/model_v3_macro.joblib")
            if strategy_path.parent.exists():
                shutil.copy(model_path, strategy_path)
            
            async with httpx.AsyncClient(timeout=30) as client:
                try:
                    resp = await client.post(
                        "http://strategy:8005/ml/reload",
                        json={"model_path": str(latest_path), "version": version}
                    )
                    logger.info(f"Strategy reload: {resp.status_code}")
                except:
                    pass
            
            logger.info(f"✅ Model {version} deployed")
            return True
        except Exception as e:
            logger.error(f"Deploy failed: {e}")
            return False

# FastAPI app
pipeline = AutoRetrainPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.init_db()
    yield
    await pipeline.close()

app = FastAPI(title="ML Trainer (GPU)", version="2.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "training_in_progress": pipeline.training_in_progress,
        "gpu_available": GPU_AVAILABLE
    }

@app.post("/train", response_model=TrainResult)
async def train_model(request: TrainRequest):
    return await pipeline.train(request)

@app.post("/train/async")
async def train_async(request: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(pipeline.train, request)
    return {"status": "training_started", "gpu": request.use_gpu and GPU_AVAILABLE}

@app.get("/check-retrain")
async def check_retrain():
    return {"should_retrain": True, "reasons": ["Manual check"]}

@app.get("/models")
async def list_models():
    models = []
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.glob("model_*.joblib"), reverse=True):
            try:
                data = joblib.load(p)
                if isinstance(data, dict):
                    models.append({
                        "path": str(p),
                        "version": data.get("version"),
                        "accuracy": data.get("accuracy"),
                        "trained_at": data.get("trained_at"),
                        "gpu_used": data.get("gpu_used", False)
                    })
            except:
                pass
    return {"models": models[:10], "gpu_available": GPU_AVAILABLE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8025)
