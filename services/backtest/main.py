#!/usr/bin/env python3
"""Backtest Module v2.0 - Enhanced strategy testing"""
import os, json, logging, statistics
from datetime import datetime
from contextlib import asynccontextmanager
import httpx, asyncpg
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backtest")

class BacktestRequest(BaseModel):
    ticker: str = "SBER"
    strategy: str = "ml"
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"
    initial_capital: float = 1_000_000
    position_size_pct: float = 10.0
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    commission_pct: float = 0.05
    allow_short: bool = False

class MultiBacktestRequest(BaseModel):
    tickers: List[str] = ["SBER", "GAZP", "LKOH"]
    strategy: str = "ml"
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"
    initial_capital: float = 1_000_000
    position_size_pct: float = 10.0

def calc_rsi(prices, period=14):
    if len(prices) < period + 1: return 50
    gains, losses = [], []
    for i in range(1, min(period + 1, len(prices))):
        d = prices[-i] - prices[-(i+1)]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    ag, al = sum(gains)/len(gains), sum(losses)/len(losses)
    return 50 if al == 0 else 100 - 100/(1 + ag/al)

def calc_sma(prices, p): return sum(prices[-p:])/p if len(prices) >= p else prices[-1]

def calc_ema(prices, p):
    if len(prices) < p: return prices[-1]
    m = 2/(p+1); e = sum(prices[:p])/p
    for x in prices[p:]: e = x*m + e*(1-m)
    return e

def calc_bb(prices, p=20, k=2):
    if len(prices) < p: return prices[-1], prices[-1], prices[-1]
    sma = calc_sma(prices, p)
    std = statistics.stdev(prices[-p:]) if len(prices[-p:]) > 1 else 1
    return sma - k*std, sma, sma + k*std

def strategy_ml(prices, i):
    if i < 50: return 0
    s = prices[:i+1]
    rsi = calc_rsi(s)
    ema12, ema26 = calc_ema(s, 12), calc_ema(s, 26)
    macd = ema12 - ema26
    sma20, sma50 = calc_sma(s, 20), calc_sma(s, 50)
    score = 0
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    if macd > 0: score += 1
    else: score -= 1
    if sma20 > sma50: score += 1
    else: score -= 1
    if score >= 2: return 1
    if score <= -2: return -1
    return 0

def strategy_range(prices, i):
    if i < 50: return 0
    recent = prices[i-50:i+1]
    high, low = max(recent), min(recent)
    if (high - low) / low < 0.02: return 0
    pos = (prices[i] - low) / (high - low)
    if pos <= 0.1: return 1
    if pos >= 0.9: return -1
    return 0

def strategy_macd(prices, i):
    if i < 30: return 0
    s = prices[:i+1]
    rsi = calc_rsi(s)
    macd = calc_ema(s, 12) - calc_ema(s, 26)
    if rsi < 30 and macd > 0: return 1
    if rsi > 70 and macd < 0: return -1
    return 0

def strategy_bb(prices, i):
    if i < 25: return 0
    s = prices[:i+1]
    lower, mid, upper = calc_bb(s)
    price = prices[i]
    rsi = calc_rsi(s)
    if price < lower and rsi < 35: return 1
    if price > upper and rsi > 65: return -1
    return 0

def strategy_momentum(prices, i):
    if i < 50: return 0
    s = prices[:i+1]
    roc_10 = (prices[i] / prices[i-10] - 1) * 100
    roc_20 = (prices[i] / prices[i-20] - 1) * 100
    ema20, ema50 = calc_ema(s, 20), calc_ema(s, 50)
    if roc_10 > 3 and roc_20 > 5 and ema20 > ema50: return 1
    if roc_10 < -3 and roc_20 < -5 and ema20 < ema50: return -1
    return 0

STRATEGIES = {"ml": strategy_ml, "range": strategy_range, "macd_rsi": strategy_macd, "bollinger": strategy_bb, "momentum": strategy_momentum}

class BacktestEngine:
    def __init__(self):
        self.pool = None
        self.cache = {}
    
    async def start(self):
        try:
            self.pool = await asyncpg.create_pool(os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading"), min_size=2, max_size=5)
        except: pass
        logger.info("✅ Backtest Engine v2.0 started")
    
    async def stop(self):
        if self.pool: await self.pool.close()
    
    async def fetch_data(self, ticker: str, start: str, end: str) -> list:
        cache_key = f"{ticker}_{start}_{end}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        prices = []
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
        async with httpx.AsyncClient(timeout=30.0) as client:
            cursor = 0
            while True:
                resp = await client.get(url, params={"from": start, "till": end, "start": cursor})
                data = resp.json()
                rows = data.get("history", {}).get("data", [])
                cols = data.get("history", {}).get("columns", [])
                if not rows: break
                idx = {c: cols.index(c) for c in ["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"] if c in cols}
                for row in rows:
                    if idx.get("CLOSE") and row[idx["CLOSE"]]:
                        prices.append({"date": row[idx.get("TRADEDATE", 0)], "close": float(row[idx["CLOSE"]])})
                cursor += len(rows)
                if len(rows) < 100: break
        logger.info(f"Fetched {len(prices)} candles for {ticker}")
        self.cache[cache_key] = prices
        return prices
    
    def run_backtest(self, prices: list, req: BacktestRequest) -> dict:
        if len(prices) < 50:
            return {"error": "Not enough data", "candles": len(prices)}
        strategy_fn = STRATEGIES.get(req.strategy, strategy_ml)
        closes = [p["close"] for p in prices]
        capital = req.initial_capital
        position = None
        trades = []
        equity_curve = []
        peak = capital
        max_dd = 0
        daily_returns = []
        
        for i in range(50, len(closes)):
            price, date = closes[i], prices[i]["date"]
            signal = strategy_fn(closes, i)
            
            if position:
                pnl_pct = (price / position["entry"] - 1) * 100 if position["side"] == "long" else (position["entry"] / price - 1) * 100
                should_exit = pnl_pct <= -req.stop_loss_pct or pnl_pct >= req.take_profit_pct or (position["side"] == "long" and signal == -1) or (position["side"] == "short" and signal == 1)
                if should_exit:
                    commission = position["size"] * req.commission_pct / 100
                    exit_pnl = position["size"] * pnl_pct / 100 - commission
                    capital += exit_pnl
                    trades.append({"date_entry": position["date"], "date_exit": date, "side": position["side"], "entry": position["entry"], "exit": price, "pnl": round(exit_pnl, 2), "pnl_pct": round(pnl_pct, 2), "reason": "sl" if pnl_pct <= -req.stop_loss_pct else "tp" if pnl_pct >= req.take_profit_pct else "signal"})
                    daily_returns.append(pnl_pct)
                    position = None
            
            if not position:
                if signal == 1:
                    size = capital * req.position_size_pct / 100
                    commission = size * req.commission_pct / 100
                    capital -= commission
                    position = {"entry": price, "size": size, "date": date, "side": "long"}
                elif signal == -1 and req.allow_short:
                    size = capital * req.position_size_pct / 100
                    commission = size * req.commission_pct / 100
                    capital -= commission
                    position = {"entry": price, "size": size, "date": date, "side": "short"}
            
            equity = capital + (position["size"] * (price / position["entry"] - 1) if position and position["side"] == "long" else position["size"] * (position["entry"] / price - 1) if position else 0)
            equity_curve.append({"date": date, "equity": round(equity, 2)})
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        if position:
            price = closes[-1]
            pnl_pct = (price / position["entry"] - 1) * 100 if position["side"] == "long" else (position["entry"] / price - 1) * 100
            exit_pnl = position["size"] * pnl_pct / 100
            capital += exit_pnl
            trades.append({"side": position["side"], "entry": position["entry"], "exit": price, "pnl": round(exit_pnl, 2), "pnl_pct": round(pnl_pct, 2), "reason": "end"})
            daily_returns.append(pnl_pct)
        
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        
        if len(daily_returns) > 1:
            avg_ret = statistics.mean(daily_returns)
            std_ret = statistics.stdev(daily_returns) or 0.01
            sharpe = avg_ret / std_ret * (252 ** 0.5)
            neg_rets = [r for r in daily_returns if r < 0]
            downside_std = statistics.stdev(neg_rets) if len(neg_rets) > 1 else 0.01
            sortino = avg_ret / downside_std * (252 ** 0.5)
        else:
            sharpe = sortino = 0
        
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        total_return = (capital - req.initial_capital) / req.initial_capital * 100
        calmar = total_return / max_dd if max_dd > 0 else 0
        
        return {
            "strategy": req.strategy, "ticker": req.ticker, "period": f"{req.start_date} → {req.end_date}", "candles": len(prices),
            "trades": {"total": len(trades), "winning": len(wins), "losing": len(losses), "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0},
            "returns": {"total_pct": round(total_return, 2), "final_capital": round(capital, 2), "gross_profit": round(gross_profit, 2), "gross_loss": round(gross_loss, 2)},
            "risk": {"max_drawdown_pct": round(max_dd, 2), "sharpe": round(sharpe, 2), "sortino": round(sortino, 2), "calmar": round(calmar, 2), "profit_factor": round(profit_factor, 2)},
            "avg": {"win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0, "loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0},
            "equity_curve": equity_curve[::max(1, len(equity_curve)//100)],
            "recent_trades": trades[-20:]
        }
    
    async def run(self, req: BacktestRequest) -> dict:
        logger.info(f"🚀 Backtest: {req.strategy} on {req.ticker}")
        prices = await self.fetch_data(req.ticker, req.start_date, req.end_date)
        return self.run_backtest(prices, req)
    
    async def run_multi(self, req: MultiBacktestRequest) -> dict:
        results = []
        for ticker in req.tickers:
            r = await self.run(BacktestRequest(ticker=ticker, strategy=req.strategy, start_date=req.start_date, end_date=req.end_date, initial_capital=req.initial_capital / len(req.tickers), position_size_pct=req.position_size_pct))
            if "error" not in r:
                results.append({"ticker": ticker, **r})
        if not results:
            return {"error": "No valid results"}
        total_return = sum(r["returns"]["total_pct"] for r in results) / len(results)
        avg_sharpe = sum(r["risk"]["sharpe"] for r in results) / len(results)
        return {"strategy": req.strategy, "tickers": req.tickers, "aggregate": {"avg_return_pct": round(total_return, 2), "avg_sharpe": round(avg_sharpe, 2)}, "by_ticker": results}

engine = BacktestEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start(); yield; await engine.stop()

app = FastAPI(title="Backtest v2.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0", "strategies": list(STRATEGIES.keys())}

@app.post("/run")
async def run(req: BacktestRequest):
    return await engine.run(req)

@app.post("/multi")
async def multi(req: MultiBacktestRequest):
    return await engine.run_multi(req)

@app.get("/strategies")
async def strategies():
    return [{"name": n, "desc": f.__doc__ or n} for n, f in STRATEGIES.items()]

@app.post("/compare")
async def compare(ticker: str = "SBER", start: str = "2024-01-01", end: str = "2025-01-01"):
    results = []
    for s in STRATEGIES:
        r = await engine.run(BacktestRequest(ticker=ticker, strategy=s, start_date=start, end_date=end))
        if "error" not in r:
            results.append({"strategy": s, "return_pct": r["returns"]["total_pct"], "trades": r["trades"]["total"], "win_rate": r["trades"]["win_rate"], "max_dd": r["risk"]["max_drawdown_pct"], "sharpe": r["risk"]["sharpe"]})
    return sorted(results, key=lambda x: x["sharpe"], reverse=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
