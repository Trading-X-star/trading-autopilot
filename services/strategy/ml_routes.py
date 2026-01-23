"""ML Routes for Strategy Service - интеграция с MLPredictorV4"""
from fastapi import APIRouter
from typing import Dict, List

router = APIRouter(prefix="/ml", tags=["ML"])

# Import will be done dynamically to avoid circular imports
predictor = None
ab_tester = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            from ml.predictor_v4 import MLPredictorV4
            predictor = MLPredictorV4()
        except Exception as e:
            print(f"Failed to load MLPredictorV4: {e}")
            # Fallback to v3
            from ml_predictor_v3 import MLPredictorV3
            predictor = MLPredictorV3()
    return predictor

def get_ab_tester():
    global ab_tester
    if ab_tester is None:
        try:
            from ml.predictor_v4 import ModelABTester, MLPredictorV4
            ab_tester = ModelABTester()
            # Add default model
            ab_tester.add_model("default", get_predictor(), 1.0)
        except:
            pass
    return ab_tester

@router.get("/info")
async def ml_info():
    """Информация о ML модели"""
    return get_predictor().info()

@router.get("/drift")
async def ml_drift():
    """Статус drift detection"""
    p = get_predictor()
    if hasattr(p, 'get_drift_status'):
        return p.get_drift_status()
    return {"drift_detected": False, "reason": "drift detection not available"}

@router.post("/predict")
async def ml_predict(features: Dict):
    """Single prediction"""
    prediction = get_predictor().predict(features)
    return {
        "signal": prediction.signal,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "is_valid": prediction.is_valid,
        "warnings": prediction.warnings,
        "latency_ms": prediction.latency_ms,
        "model_version": prediction.model_version
    }

@router.post("/predict/batch")
async def ml_predict_batch(features_list: List[Dict]):
    """Batch predictions"""
    p = get_predictor()
    if hasattr(p, 'predict_batch'):
        predictions = p.predict_batch(features_list)
    else:
        predictions = [p.predict(f) for f in features_list]
    
    return [
        {
            "signal": pred.signal,
            "confidence": pred.confidence,
            "is_valid": pred.is_valid,
            "latency_ms": pred.latency_ms
        }
        for pred in predictions
    ]

@router.post("/reload")
async def ml_reload(model_path: str = None, version: str = None):
    """Reload ML model"""
    global predictor
    try:
        from ml.predictor_v4 import MLPredictorV4
        if model_path:
            predictor = MLPredictorV4(model_path)
        else:
            predictor = MLPredictorV4()
        return {"status": "reloaded", "info": predictor.info()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/ab/stats")
async def ab_stats():
    """A/B testing statistics"""
    tester = get_ab_tester()
    if tester:
        return tester.get_stats()
    return {"error": "A/B tester not available"}
