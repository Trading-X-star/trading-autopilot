#!/usr/bin/env python3
from feature_generator import generate_features
"""Strategy Service - ML + Boxing Strategy"""
import asyncio
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import Response, FastAPI
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("strategy")

# ML imports
try:
    from ml_predictor_v3 import MLPredictorV3
    ml_predictor_v3 = MLPredictorV3("model_v7_ultimate.joblib")
    ML_AVAILABLE = ml_predictor_v3.ready
except Exception as e:
    logger.warning(f"ML v3 load failed: {e}")
    ML_AVAILABLE = False
    ml_predictor_v3 = None
    MLPredictorV3 = None

# # Legacy predictor
# try:
#     from ml_predictor import predictor as ml_predictor
# except:
#     ml_predictor = None

SIGNALS = Counter("signals_total", "Signals", ["ticker", "signal", "strategy"])
TICKERS = ["SBER", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "PLZL", "MGNT",
           "VTBR", "MTSS", "ALRS", "CHMF", "NLMK", "TATN", "SNGS", "MOEX", "AFKS"]


class BoxingStrategy:
    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold
        self.boxes = {}
    
    def detect_box(self, prices: list) -> dict:
        if len(prices) < self.lookback:
            return None
        
        recent = prices[-self.lookback:]
        high = max(recent)
        low = min(recent)
        box_height = (high - low) / low if low > 0 else 0
        
        if 0.02 <= box_height <= 0.10:
            touches_high = sum(1 for p in recent if p >= high * 0.99)
            touches_low = sum(1 for p in recent if p <= low * 1.01)
            
            if touches_high >= 2 and touches_low >= 2:
                return {"high": high, "low": low, "mid": (high + low) / 2,
                        "height_pct": box_height * 100, "confirmed": True}
        
        return {"high": high, "low": low, "mid": (high + low) / 2, 
                "height_pct": box_height * 100, "confirmed": False}
    
    def analyze(self, ticker: str, prices: list, current_price: float) -> dict:
        box = self.detect_box(prices)
        
        if not box:
            return {"signal": 0, "confidence": 0, "box": None, "reason": "Недостаточно данных"}
        
        self.boxes[ticker] = box
        
        if not box["confirmed"]:
            return {"signal": 0, "confidence": 0.3, "box": box, "reason": "Бокс не подтверждён"}
        
        high, low = box["high"], box["low"]
        box_range = high - low
        position = (current_price - low) / box_range if box_range > 0 else 0.5
        
        signal, confidence, reason = 0, 0, ""
        
        if position <= self.threshold / (box["height_pct"] / 100):
            signal, confidence = 1, 0.6 + (0.2 * (1 - position))
            reason = f"У нижней границы бокса ({position:.1%})"
        elif position >= 1 - self.threshold / (box["height_pct"] / 100):
            signal, confidence = -1, 0.6 + (0.2 * position)
            reason = f"У верхней границы бокса ({position:.1%})"
        elif current_price > high * 1.01:
            signal, confidence = 1, 0.5
            reason = f"Пробой верхней границы (+{((current_price/high)-1)*100:.1f}%)"
        elif current_price < low * 0.99:
            signal, confidence = -1, 0.5
            reason = f"Пробой нижней границы ({((current_price/low)-1)*100:.1f}%)"
        else:
            reason = f"В середине бокса ({position:.1%})"
        
        return {"signal": signal, "confidence": round(confidence, 3), "box": box,
                "position_in_box": round(position, 3), "reason": reason}


class Strategy:
    def __init__(self):
        self.redis = None
        self.boxing = BoxingStrategy(lookback=50, threshold=0.02)
        self.strategy_mode = os.getenv("STRATEGY_MODE", "combined")
    
    async def start(self):
        self.redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
        logger.info(f"✅ Strategy started (ML: {ML_AVAILABLE}, Mode: {self.strategy_mode})")
    
    async def stop(self):
        if self.redis: await self.redis.close()
    
    async def get_history(self, ticker: str, limit: int = 250) -> list:
        try:
            data = await self.redis.zrevrange(f"history:{ticker}", 0, limit-1)
            return [json.loads(d) for d in data][::-1]
        except:
            return []
    
    def calc_indicators(self, history: list) -> dict:
        if len(history) < 30:
            return {}
        
        closes = [h["price"] for h in history]
        highs = [h.get("high", h["price"]) for h in history]
        lows = [h.get("low", h["price"]) for h in history]
        volumes = [h.get("volume", 1000) for h in history]
        
        c = closes[-1]
        n = len(closes)
        if c <= 0: return {}
        
        def sma(data, p): return sum(data[-p:])/p if len(data)>=p else data[-1]
        def ema(data, p):
            if len(data)<p: return data[-1]
            m = 2/(p+1); e = sma(data[:p], p)
            for x in data[p:]: e = x*m + e*(1-m)
            return e
        def std(v): 
            if len(v) < 2: return 0
            m = sum(v)/len(v)
            return (sum((x-m)**2 for x in v)/len(v))**0.5
        
        ema12, ema26 = ema(closes, 12), ema(closes, 26)
        macd = ema12 - ema26
        macd_hist_vals = []
        for i in range(max(26, n-9), n):
            e12 = ema(closes[:i+1], 12)
            e26 = ema(closes[:i+1], 26)
            macd_hist_vals.append(e12 - e26)
        macd_sig = ema(macd_hist_vals, 9) if macd_hist_vals else macd
        
        sma20 = sma(closes, 20)
        bb_std = std(closes[-20:])
        
        gains, losses = [], []
        for i in range(1, min(15, n)):
            d = closes[-i] - closes[-(i+1)]
            gains.append(max(d, 0))
            losses.append(max(-d, 0))
        ag = sum(gains)/len(gains) if gains else 0
        al = sum(losses)/len(losses) if losses else 0
        rsi = 50 if al == 0 else 100 - 100/(1 + ag/al) if al > 0 else (100 if ag > 0 else 50)
        
        rets = [(closes[i]/closes[i-1]-1) for i in range(max(1, n-20), n) if closes[i-1] > 0]
        vol_sma = sum(volumes[-20:])/20 if len(volumes) >= 20 else sum(volumes)/len(volumes)
        
        return {
            "return_1d": c/closes[-2]-1 if n >= 2 and closes[-2] > 0 else 0,
            "return_5d": c/closes[-5]-1 if n >= 5 and closes[-5] > 0 else 0,
            "return_10d": c/closes[-10]-1 if n >= 10 and closes[-10] > 0 else 0,
            "return_20d": c/closes[-20]-1 if n >= 20 and closes[-20] > 0 else 0,
            "sma_5": sma(closes, 5), "sma_10": sma(closes, 10), "sma_20": sma20,
            "sma_50": sma(closes, 50), "sma_200": sma(closes, 200) if n >= 200 else sma(closes, 50),
            "ema_12": ema12, "ema_26": ema26,
            "macd": macd, "macd_signal": macd_sig, "macd_hist": macd - macd_sig,
            "rsi_14": rsi,
            "bb_upper": sma20 + 2*bb_std, "bb_middle": sma20, "bb_lower": sma20 - 2*bb_std,
            "bb_width": 4*bb_std/sma20 if sma20 > 0 else 0,
            "bb_pct": (c - (sma20 - 2*bb_std))/(4*bb_std) if bb_std > 0 else 0.5,
            "atr_14": sum(highs[i]-lows[i] for i in range(-min(14, n), 0))/min(14, n) if n > 0 else 0,
            "volatility_20": std(rets) if rets else 0,
            "volume_ratio": volumes[-1]/vol_sma if vol_sma > 0 else 1,
            "pct_from_high": c/max(highs)-1 if max(highs) > 0 else 0,
            "pct_from_low": c/min(x for x in lows if x > 0)-1 if any(x > 0 for x in lows) else 0,
            "close": c
        }
    
    async def analyze(self, ticker: str) -> dict:
        history = await self.get_history(ticker)
        
        if len(history) < 30:
            return {"ticker": ticker, "signal": "hold", "confidence": 0, 
                    "method": "none", "reason": f"Need 30 points, have {len(history)}"}
        
        closes = [h["price"] for h in history]
        c = closes[-1]
        ind = self.calc_indicators(history)
        
        ml_signal, ml_conf = 0, 0
        box_result = None
        
        # ML Strategy
        if self.strategy_mode in ["ml", "combined"] and ML_AVAILABLE and ml_predictor_v3 and ind:
            pred = ml_predictor_v3.predict(ind)
            ml_signal, ml_conf = pred.signal, pred.confidence
        
        # Boxing Strategy
        if self.strategy_mode in ["boxing", "combined"]:
            box_result = self.boxing.analyze(ticker, closes, c)
        
        # Combine strategies
        if self.strategy_mode == "combined" and box_result:
            box_signal = box_result["signal"]
            box_conf = box_result["confidence"]
            
            if ml_signal == box_signal and ml_signal != 0:
                final_signal = ml_signal
                final_conf = min((ml_conf + box_conf) / 2 + 0.1, 0.95)
                method = "ml+boxing"
            elif box_signal == 0 and ml_signal != 0:
                final_signal = ml_signal
                final_conf = ml_conf * 0.9
                method = "ml"
            elif ml_signal == 0 and box_signal != 0:
                final_signal = box_signal
                final_conf = box_conf * 0.9
                method = "boxing"
            elif ml_signal != box_signal:
                if ml_conf > box_conf:
                    final_signal = ml_signal
                    final_conf = ml_conf * 0.7
                    method = "ml (conflict)"
                else:
                    final_signal = box_signal
                    final_conf = box_conf * 0.7
                    method = "boxing (conflict)"
            else:
                final_signal = 0
                final_conf = 0.3
                method = "neutral"
        elif self.strategy_mode == "boxing" and box_result:
            final_signal = box_result["signal"]
            final_conf = box_result["confidence"]
            method = "boxing"
        else:
            final_signal = ml_signal
            final_conf = ml_conf
            method = "ml"
        
        signal = {1: "buy", -1: "sell", 0: "hold"}[final_signal]
        SIGNALS.labels(ticker=ticker, signal=signal, strategy=method).inc()
        
        if final_signal != 0 and final_conf > 0.4:
            await self.redis.xadd("stream:signals", {"ticker": ticker, "signal": signal,
                                  "confidence": str(final_conf), "price": str(c), "method": method}, maxlen=1000)
        
        result = {
            "ticker": ticker, "signal": signal, "confidence": round(final_conf, 3),
            "price": c, "method": method, "history_points": len(history),
            "indicators": {
                "rsi": round(ind.get("rsi_14", 0), 1) if ind else 0,
                "macd_hist": round(ind.get("macd_hist", 0), 4) if ind else 0,
                "bb_pct": round(ind.get("bb_pct", 0), 2) if ind else 0,
                "sma_20": round(ind.get("sma_20", 0), 2) if ind else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if box_result:
            result["boxing"] = {
                "box_high": round(box_result["box"]["high"], 2) if box_result["box"] else None,
                "box_low": round(box_result["box"]["low"], 2) if box_result["box"] else None,
                "box_confirmed": box_result["box"]["confirmed"] if box_result["box"] else False,
                "position_in_box": box_result.get("position_in_box"),
                "box_reason": box_result.get("reason")
            }
        
        return result
    
    async def scan(self) -> list:
        results = []
        for t in TICKERS:
            r = await self.analyze(t)
            results.append(r)
        results.sort(key=lambda x: (x["signal"] == "hold", -x["confidence"]))
        return results


svc = Strategy()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()

app = FastAPI(title="Strategy ML+Boxing v2", lifespan=lifespan)

# ============================================================
# METRICS
# ============================================================
@app.get("/metrics")
@app.get("/metrics/")
async def prometheus_metrics():
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

# ============================================================
# HEALTH & INFO
# ============================================================
@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "ml_available": ML_AVAILABLE, 
        "strategy_mode": svc.strategy_mode,
        "ml_info": ml_predictor_v3.info() if ml_predictor_v3 else None
    }

@app.get("/model/info")
async def model_info():
    """Информация о текущей модели"""
    if ml_predictor_v3:
        return ml_predictor_v3.info()
    return {"ready": False, "error": "Model not loaded"}

# ============================================================
# STRATEGY ENDPOINTS
# ============================================================
@app.get("/analyze/{ticker}")
async def analyze(ticker: str): 
    return await svc.analyze(ticker.upper())

@app.get("/scan")
async def scan(): 
    return await svc.scan()

@app.get("/boxes")
async def get_boxes():
    return svc.boxing.boxes

@app.post("/mode/{mode}")
async def set_mode(mode: str):
    if mode in ["ml", "boxing", "combined"]:
        svc.strategy_mode = mode
        return {"mode": mode}
    return {"error": "Invalid mode"}

# ============================================================
# PAIR TRADING
# ============================================================
try:
    from pair_trading import pair_strategy, PAIRS
    
    @app.get("/pairs")
    async def get_pairs():
        return {"pairs": PAIRS, "positions": dict(pair_strategy.positions)}
    
    @app.get("/pairs/scan")
    async def scan_pairs():
        prices = {}
        for ta, tb in PAIRS:
            for t in [ta, tb]:
                if t not in prices:
                    hist = await svc.get_history(t)
                    if hist:
                        prices[t] = [h["price"] for h in hist]
        
        signals = pair_strategy.get_signals(prices)
        results = []
        for s in signals:
            results.append({
                "pair": list(s.pair), "action": s.action,
                "long": s.long_ticker, "short": s.short_ticker,
                "zscore": round(s.zscore, 2), "confidence": round(s.confidence, 2)
            })
        
        all_pairs = []
        for ta, tb in PAIRS:
            if ta in prices and tb in prices:
                spread, mean, z = pair_strategy.calc_spread(prices[ta], prices[tb])
                corr = pair_strategy.calc_corr(prices[ta], prices[tb])
                if z is not None:
                    all_pairs.append({
                        "pair": f"{ta}/{tb}", "zscore": round(z, 2),
                        "correlation": round(corr, 2), "spread": round(spread, 4)
                    })
        
        return {"signals": results, "all_pairs": all_pairs}
    
    @app.post("/pairs/{action}")
    async def pair_action(action: str, pair_a: str, pair_b: str, long_t: str = "", short_t: str = ""):
        pair = (pair_a, pair_b)
        if action == "open" and long_t and short_t:
            pair_strategy.positions[pair] = {"long": long_t, "short": short_t}
            return {"status": "opened", "pair": pair}
        elif action == "close":
            pair_strategy.positions.pop(pair, None)
            return {"status": "closed", "pair": pair}
        return {"error": "Invalid action"}
except Exception as e:
    logger.warning(f"Pair trading not available: {e}")

# ============================================================
# ML V3 PREDICT
# ============================================================
@app.get("/predict/v3/{ticker}")
async def predict_v3(ticker: str):
    if not ml_predictor_v3:
        return {"error": "Model not loaded", "ticker": ticker}
    
    features = {}
    try:
        data = await svc.redis.hgetall(f"features:{ticker}")
        features = {k: float(v) for k, v in data.items()}
    except: pass
    
    if not features:
        return {"error": "No features available", "ticker": ticker}
    
    result = ml_predictor_v3.predict(features)
    return {
        "ticker": ticker,
        "signal": result.signal,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "is_valid": result.is_valid,
        "warnings": result.warnings,
        "latency_ms": result.latency_ms,
        "model_version": result.model_version
    }

# ============================================================
# RETRAIN ENDPOINT
# ============================================================
@app.post("/retrain")
async def retrain_model(reason: str = "manual"):
    """Запуск переобучения модели"""
    global ml_predictor_v3
    
    try:
        logger.info(f"🔄 Retrain triggered: {reason}")
        
        process = await asyncio.create_subprocess_exec(
            "python", "/app/trainer_v7_ultimate.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=7200)
        
        if process.returncode == 0:
            # Перезагружаем модель
            ml_predictor_v3 = MLPredictorV3("model_v7_ultimate.joblib")
            
            logger.info(f"✅ Retrain complete, version: {ml_predictor_v3.version}")
            
            return {
                "status": "success",
                "reason": reason,
                "version": ml_predictor_v3.version,
                "ready": ml_predictor_v3.ready,
                "output": stdout.decode()[-1000:] if stdout else ""
            }
        else:
            error_msg = stderr.decode()[-1000:] if stderr else "Unknown error"
            logger.error(f"❌ Retrain failed: {error_msg}")
            return {"status": "failed", "error": error_msg}
            
    except asyncio.TimeoutError:
        return {"status": "failed", "error": "Training timeout (10 min)"}
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return {"status": "failed", "error": str(e)}


@app.post("/features/update")
async def update_features():
    """Обновляет фичи для всех тикеров"""
    updated = 0
    errors = []
    
    for ticker in TICKERS:
        try:
            raw = await svc.redis.zrevrange(f"history:{ticker}", 0, 99)
            if not raw:
                continue
            history = [json.loads(x) for x in raw]
            features = generate_features(history)
            if features:
                await svc.redis.hset(f"features:{ticker}", mapping={k: str(v) for k, v in features.items()})
                updated += 1
        except Exception as e:
            errors.append(f"{ticker}: {e}")
    
    return {"updated": updated, "errors": errors[:5]}

@app.get("/features/{ticker}")
async def get_features(ticker: str):
    """Получает фичи для тикера"""
    data = await svc.redis.hgetall(f"features:{ticker}")
    if not data:
        return {"error": "No features", "ticker": ticker}
    return {"ticker": ticker, "features": {k: float(v) for k, v in data.items()}}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

# ============================================================
# FEATURE GENERATION
# ============================================================
import json

