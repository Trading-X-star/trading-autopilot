#!/usr/bin/env python3
"""Trading Terminal v7.1 - Tinkoff Format Fix"""

import os, json, asyncio, logging, time, csv, io, statistics
from datetime import datetime
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, Deque, List
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
log = logging.getLogger("terminal")

METRICS = {
    "v": Counter("t_views", "", ["p"]),
    "api": Counter("t_api", "", ["e", "s"]),
    "lat": Histogram("t_lat", "", ["e"]),
    "ws": Gauge("t_ws", "")
}


def tinkoff_to_float(money) -> float:
    """Convert Tinkoff MoneyValue to float"""
    if money is None:
        return 0.0
    if isinstance(money, (int, float)):
        return float(money)
    if isinstance(money, dict):
        units = int(money.get('units', 0) or 0)
        nano = int(money.get('nano', 0) or 0)
        return units + nano / 1_000_000_000
    return 0.0


class C:
    STRATEGY = os.getenv("STRATEGY_URL", "http://strategy:8005")
    EXECUTOR = os.getenv("EXECUTOR_URL", "http://executor:8007")
    DATAFEED = os.getenv("DATAFEED_URL", "http://datafeed:8006")
    RISK = os.getenv("RISK_MANAGER_URL", "http://risk-manager:8001")
    ORCH = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
    BRAIN = os.getenv("BRAIN_URL", "http://scheduler:8009")
    REDIS = os.getenv("REDIS_URL", "redis://redis:6379/0")
    TTL = int(os.getenv("CACHE_TTL", "2"))
    WS_INT = float(os.getenv("WS_INTERVAL", "1.5"))
    TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "8"))
    SVC = {"orch": ORCH, "strat": STRATEGY, "exec": EXECUTOR, "feed": DATAFEED, "risk": RISK, "brain": BRAIN}


class CS(str, Enum):
    OK = "ok"
    FAIL = "fail"
    HALF = "half"


@dataclass
class Alert:
    id: str
    sev: str
    title: str
    msg: str
    ts: str
    src: str
    def d(self): return asdict(self)


class Cache:
    def __init__(self, ttl=2):
        self._d, self._ttl, self._h, self._m = {}, ttl, 0, 0

    async def get(self, k):
        e = self._d.get(k)
        if not e: self._m += 1; return None, 0
        v, x = e
        if time.time() < x: self._h += 1; return v, 1
        del self._d[k]; self._m += 1; return None, 0

    async def set(self, k, v, t=None):
        if len(self._d) > 500: del self._d[min(self._d, key=lambda x: self._d[x][1])]
        self._d[k] = (v, time.time() + (t or self._ttl))

    def inv(self, p):
        for k in [k for k in self._d if k.startswith(p)]: del self._d[k]

    def st(self):
        t = self._h + self._m
        return {"h": self._h, "m": self._m, "r": round(self._h / t, 2) if t else 0}


class CB:
    def __init__(self, th=5, rec=30):
        self.th, self.rec, self._f, self._u = th, rec, {}, {}

    def st(self, n):
        u = self._u.get(n)
        if not u: return CS.OK
        return CS.FAIL if time.time() < u else CS.HALF

    def fail(self, n):
        self._f[n] = self._f.get(n, 0) + 1
        if self._f[n] >= self.th: self._u[n] = time.time() + self.rec

    def ok(self, n): self._f[n] = 0; self._u.pop(n, None)
    def all(self): return {n: self.st(n).value for n in C.SVC}


class WS:
    def __init__(self): self._c, self._l = {}, asyncio.Lock()

    async def conn(self, ws):
        await ws.accept()
        async with self._l: self._c[ws] = time.time()
        METRICS["ws"].set(len(self._c))

    async def disc(self, ws):
        async with self._l: self._c.pop(ws, None)
        METRICS["ws"].set(len(self._c))

    async def send(self, d):
        if not self._c: return 0
        m, dead = json.dumps(d, default=str), []
        async with self._l: cs = list(self._c)
        for ws in cs:
            try: await asyncio.wait_for(ws.send_text(m), 5)
            except: dead.append(ws)
        if dead:
            async with self._l:
                for w in dead: self._c.pop(w, None)
            METRICS["ws"].set(len(self._c))
        return len(cs) - len(dead)

    @property
    def n(self): return len(self._c)


class Alerts:
    def __init__(self): self._a, self._l = deque(maxlen=100), asyncio.Lock()
    async def add(self, a):
        async with self._l: self._a.appendleft(a)
    async def get(self, n=20):
        async with self._l: return [x.d() for x in list(self._a)[:n]]


class Terminal:
    def __init__(self):
        self.redis = self.http = None
        self.ws, self.cache, self.cb, self.alerts = WS(), Cache(C.TTL), CB(), Alerts()
        self._run, self._tasks = False, []
        self._hist = {k: deque(maxlen=200) for k in ["dd", "pnl", "eq", "vol", "wr", "fg"]}
        self._sparks, self._tkstats, self._hourly = {}, {}, defaultdict(list)
        self._start, self._peak_eq = 0.0, 0

    async def start(self):
        self._start = time.time()
        self.redis = aioredis.from_url(C.REDIS, decode_responses=True)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(C.TIMEOUT, connect=5))
        self._run = True
        self._tasks = [asyncio.create_task(self._loop()), asyncio.create_task(self._sub())]
        log.info("✅ Terminal v7.1 started")

    async def stop(self):
        self._run = False
        for t in self._tasks: t.cancel()
        for ws in list(self.ws._c):
            try: await ws.close(1001)
            except: pass
        if self.http: await self.http.aclose()
        if self.redis: await self.redis.close()

    async def _f(self, url, ep):
        svc = ep.split("_")[0]
        if self.cb.st(svc) == CS.FAIL: return {}
        v, hit = await self.cache.get(ep)
        if hit: return v
        try:
            r = await self.http.get(url)
            if r.status_code >= 400: self.cb.fail(svc); return {}
            self.cb.ok(svc)
            d = r.json()
            await self.cache.set(ep, d)
            return d
        except: self.cb.fail(svc); return {}

    def _parse_portfolio(self, port):
        """Parse Tinkoff portfolio format"""
        if not port or not isinstance(port, dict):
            return {"balance": 0, "total_value": 0, "unrealized_pnl": 0}
        
        if 'totalAmountPortfolio' in port:
            total = tinkoff_to_float(port.get('totalAmountPortfolio'))
            cash = tinkoff_to_float(port.get('totalAmountCurrencies'))
            shares = tinkoff_to_float(port.get('totalAmountShares'))
            pnl = tinkoff_to_float(port.get('expectedYield'))
            return {"balance": cash, "total_value": total, "unrealized_pnl": pnl, "shares_value": shares}
        
        return {"balance": port.get('balance', 0), "total_value": port.get('total_value', 0), 
                "unrealized_pnl": port.get('unrealized_pnl', 0)}

    def _parse_positions(self, pos_data):
        """Parse Tinkoff positions format"""
        if not pos_data: return []
        
        # If already list, check format
        if isinstance(pos_data, list):
            positions = pos_data
        elif isinstance(pos_data, dict):
            positions = pos_data.get('positions', [])
        else:
            return []
        
        result = []
        for p in positions:
            if not isinstance(p, dict): continue
            ticker = p.get('ticker', '')
            if ticker in ('RUB000UTSTOM', ''): continue  # Skip cash
            
            qty = tinkoff_to_float(p.get('quantity', p.get('balance', 0)))
            if qty == 0: continue
            
            avg_price = tinkoff_to_float(p.get('averagePositionPrice', p.get('avg_price', 0)))
            cur_price = tinkoff_to_float(p.get('currentPrice', p.get('current_price', avg_price)))
            
            market_value = qty * cur_price
            pnl = (cur_price - avg_price) * qty if avg_price > 0 else 0
            pnl_pct = ((cur_price / avg_price) - 1) * 100 if avg_price > 0 else 0
            
            result.append({
                "ticker": ticker,
                "quantity": qty,
                "avg_price": avg_price,
                "current_price": cur_price,
                "market_value": market_value,
                "unrealized_pnl": pnl,
                "pnl_pct": pnl_pct
            })
        return result

    def _parse_trades(self, trades_data):
        """Parse trades format"""
        if not trades_data: return []
        if isinstance(trades_data, list): return trades_data
        if isinstance(trades_data, dict): return trades_data.get('trades', [])
        return []

    async def data(self):
        res = await asyncio.gather(
            self._f(f"{C.STRATEGY}/scan", "strat_s"),
            self._f(f"{C.DATAFEED}/prices", "feed_p"),
            self._f(f"{C.EXECUTOR}/portfolio", "exec_pf"),
            self._f(f"{C.EXECUTOR}/trades?limit=100", "exec_tr"),
            self._f(f"{C.RISK}/health", "risk_h"),
            self._f(f"{C.DATAFEED}/macro", "feed_m"),
            self._f(f"{C.ORCH}/health", "orch_h"),
            self._f(f"{C.EXECUTOR}/positions", "exec_ps"),
            self._f(f"{C.BRAIN}/sentiment", "brain_fg"),
            self._f(f"{C.BRAIN}/config", "brain_cfg"),
            self._f(f"{C.STRATEGY}/health", "strat_health"),
            return_exceptions=True
        )

        def sf(v): return {} if isinstance(v, (Exception, type(None))) else v
        sig, prices, port_raw, trades_raw, risk, macro, state, pos_raw, sentiment, brain_cfg, strat_health = map(sf, res)

        sig_l = sig if isinstance(sig, list) else sig.get("signals", [])
        port_d = self._parse_portfolio(port_raw)
        pos_l = self._parse_positions(port_raw)
        tr_l = self._parse_trades(trades_raw)
        risk_d = risk if isinstance(risk, dict) else {}
        macro_d = macro if isinstance(macro, dict) else {}
        state_d = state if isinstance(state, dict) else {}
        prices_d = prices if isinstance(prices, dict) else {}
        sentiment_d = sentiment if isinstance(sentiment, dict) else {}
        brain_cfg_d = brain_cfg if isinstance(brain_cfg, dict) else {}
        strat_health_d = sf(res[10]) if len(res) > 10 else {}

        # Sparklines & sectors
        sectors = defaultdict(list)
        for tk, pd in prices_d.items():
            if isinstance(pd, dict):
                p = pd.get("price", 0)
                ch = pd.get("change", pd.get("change_pct", 0))
                if tk not in self._sparks: self._sparks[tk] = deque(maxlen=50)
                self._sparks[tk].append(p)
                sec = pd.get("sector", "other")
                sectors[sec].append(ch)

        sector_perf = {k: round(sum(v) / len(v), 2) if v else 0 for k, v in sectors.items()}

        # Core metrics
        dd = float(risk_d.get("drawdown_pct", risk_d.get("drawdown", 0)))
        self._hist["dd"].append(dd)

        pnls = [t.get("pnl", 0) or 0 for t in tr_l if isinstance(t, dict)]
        total_pnl = sum(pnls)
        self._hist["pnl"].append(total_pnl)

        eq = float(port_d.get("total_value", 0))
        self._hist["eq"].append(eq)
        self._peak_eq = max(self._peak_eq, eq) if eq > 0 else self._peak_eq

        vol = sum(abs(t.get("quantity", 0) * t.get("price", 0)) for t in tr_l[-20:] if isinstance(t, dict))
        self._hist["vol"].append(vol)

        fg_val = sentiment_d.get("value", 50)
        self._hist["fg"].append(fg_val)

        # Stats
        w = sum(1 for p in pnls if p > 0)
        l = sum(1 for p in pnls if p < 0)
        wr = w / len(pnls) if pnls else 0
        self._hist["wr"].append(wr)

        aw = sum(p for p in pnls if p > 0) / w if w else 0
        al = abs(sum(p for p in pnls if p < 0) / l) if l else 0
        pf = aw / al if al else 0

        avg = sum(pnls) / len(pnls) if pnls else 0
        std = statistics.stdev(pnls) if len(pnls) > 1 else 1
        sharpe = avg / max(.01, std)

        neg = [p for p in pnls if p < 0]
        neg_std = statistics.stdev(neg) if len(neg) > 1 else 1
        sortino = avg / max(.01, neg_std)
        calmar = avg / max(.01, dd) if dd > 0 else 0

        streak = 0
        for p in reversed(pnls[:20]):
            if p > 0:
                if streak >= 0: streak += 1
                else: break
            elif p < 0:
                if streak <= 0: streak -= 1
                else: break

        hr = datetime.now().hour
        self._hourly[hr].append(total_pnl)
        hourly_avg = {h: round(sum(v[-10:]) / len(v[-10:]), 0) if v else 0 for h, v in self._hourly.items()}

        for t in tr_l:
            tk = t.get("ticker")
            if tk:
                if tk not in self._tkstats: self._tkstats[tk] = {"pnl": 0, "n": 0, "w": 0, "vol": 0}
                self._tkstats[tk]["pnl"] += t.get("pnl", 0) or 0
                self._tkstats[tk]["n"] += 1
                self._tkstats[tk]["vol"] += abs(t.get("quantity", 0) * t.get("price", 0))
                if (t.get("pnl", 0) or 0) > 0: self._tkstats[tk]["w"] += 1

        top_tk = sorted(self._tkstats.items(), key=lambda x: x[1]["pnl"], reverse=True)[:8]

        long_exp = sum(p.get("market_value", 0) for p in pos_l if p.get("quantity", 0) > 0)
        short_exp = abs(sum(p.get("market_value", 0) for p in pos_l if p.get("quantity", 0) < 0))

        return {
            "t": "u",
            "ts": datetime.utcnow().isoformat() + "Z",
            "signals": sig_l,
            "prices": prices_d,
            "port": port_d,
            "pos": pos_l,
            "trades": tr_l,
            "risk": risk_d,
            "macro": macro_d,
            "state": state_d,
            "sparks": {k: list(v) for k, v in list(self._sparks.items())[:30]},
            "brain": {
                "fg": fg_val,
                "fgEmo": sentiment_d.get("emotion", "neutral"),
                "fgComp": sentiment_d.get("components", {}),
                "regime": brain_cfg_d.get("regime", "sideways"),
                "minConf": brain_cfg_d.get("min_confidence", 0.45),
                "maxPos": brain_cfg_d.get("max_position_pct", 0.1),
                "maxTrades": brain_cfg_d.get("max_daily_trades", 20),
                "fgHist": list(self._hist["fg"])[-50:],
                "model": strat_health_d.get("ml_info", {}),
            },
            "m": {
                "dd": round(dd, 2),
                "maxDd": round(max(self._hist["dd"]) if self._hist["dd"] else 0, 2),
                "pnl": round(total_pnl, 0),
                "eq": round(eq, 0),
                "peakEq": round(self._peak_eq, 0),
                "wr": round(wr, 3),
                "pf": round(pf, 2),
                "sharpe": round(sharpe, 2),
                "sortino": round(sortino, 2),
                "calmar": round(calmar, 2),
                "n": len(tr_l),
                "w": w, "l": l,
                "streak": streak,
                "avgWin": round(aw, 0),
                "avgLoss": round(al, 0),
                "sigN": len([x for x in sig_l if x.get("signal", 0) != 0]),
                "posN": len(pos_l),
                "vol": round(vol, 0),
                "longExp": round(long_exp, 0),
                "shortExp": round(short_exp, 0),
                "netExp": round(long_exp - short_exp, 0),
                "grossExp": round(long_exp + short_exp, 0),
                "hist": {k: list(v)[-100:] for k, v in self._hist.items()},
                "topTk": top_tk,
                "sectors": sector_perf,
                "hourly": hourly_avg,
            },
            "alerts": await self.alerts.get(15),
            "sys": {"cb": self.cb.all(), "cache": self.cache.st(), "ws": self.ws.n, "up": round(time.time() - self._start)},
        }

    async def _loop(self):
        while self._run:
            try:
                if self.ws.n > 0: await self.ws.send(await self.data())
                await asyncio.sleep(C.WS_INT)
            except asyncio.CancelledError: break
            except: await asyncio.sleep(5)

    async def _sub(self):
        while self._run:
            try:
                ps = self.redis.pubsub()
                await ps.subscribe("trades", "signals", "alerts")
                async for msg in ps.listen():
                    if not self._run: break
                    if msg["type"] != "message": continue
                    try:
                        d = json.loads(msg["data"])
                        ch = msg["channel"]
                        if ch == "trades": self.cache.inv("exec_")
                        elif ch == "signals": self.cache.inv("strat_")
                        elif ch == "alerts":
                            await self.alerts.add(Alert(str(time.time()), d.get("severity", "info"),
                                d.get("title", ""), d.get("message", ""), datetime.utcnow().isoformat(), d.get("source", "")))
                        await self.ws.send({"t": "e", "ch": ch, "d": d})
                    except: pass
            except asyncio.CancelledError: break
            except: await asyncio.sleep(5)


term = Terminal()

@asynccontextmanager
async def lifespan(app):
    await term.start()
    yield
    await term.stop()

app = FastAPI(title="Trading Terminal", version="7.1", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health(): return {"ok": 1, "v": "7.1", "up": round(time.time() - term._start), "ws": term.ws.n}

@app.get("/metrics")
async def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/data")
async def api_data(): return await term.data()

@app.get("/api/export")
async def export(fmt: str = "csv"):
    t = await term._f(f"{C.EXECUTOR}/trades?limit=2000", "exec_exp")
    tl = t if isinstance(t, list) else t.get("trades", [])
    if fmt == "json": return {"trades": tl}
    out = io.StringIO()
    if tl:
        w = csv.DictWriter(out, fieldnames=tl[0].keys())
        w.writeheader(); w.writerows(tl)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment;filename=trades_{datetime.now():%Y%m%d}.csv"})

@app.post("/api/brain/correct")
async def brain_correct():
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{C.BRAIN}/correct")
            return r.json()
    except: return {"error": "failed"}

@app.post("/api/brain/retrain")
async def brain_retrain():
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{C.BRAIN}/retrain")
            return r.json()
    except: return {"error": "failed"}

@app.get("/api/brain/analysis/{ticker}")
async def brain_analysis(ticker: str):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{C.BRAIN}/analysis/{ticker}")
            return r.json()
    except: return {"error": "failed"}

@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await term.ws.conn(ws)
    try:
        await ws.send_json(await term.data())
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), 25)
                if msg == "ping": await ws.send_text("pong")
                elif msg.startswith("{"):
                    c = json.loads(msg)
                    if c.get("a") == "r": await ws.send_json(await term.data())
            except asyncio.TimeoutError: await ws.send_json({"t": "hb"})
    except: pass
    finally: await term.ws.disc(ws)

@app.get("/", response_class=HTMLResponse)
async def index():
    METRICS["v"].labels(p="i").inc()
    return HTML

HTML = '''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Trading Terminal v7.1</title>
<style>
:root{--bg:#0a0a12;--bg2:#0e0e18;--bg3:#141420;--bg4:#1a1a28;--brd:#252535;--txt:#d0d0e0;--txt2:#606075;--g:#00e676;--r:#ff5555;--y:#ffd740;--b:#40c4ff;--p:#e040fb;--o:#ff9100}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:ui-monospace,'SF Mono',Monaco,monospace;background:var(--bg);color:var(--txt);overflow-x:hidden;font-size:12px}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:var(--brd);border-radius:2px}
.terminal{display:grid;grid-template-rows:36px 32px 1fr 24px;height:100vh}
.hdr{background:var(--bg2);display:flex;align-items:center;padding:0 12px;border-bottom:1px solid var(--brd);gap:12px}
.logo{font-weight:700;color:var(--g);text-shadow:0 0 10px var(--g)}
.hdr-r{margin-left:auto;display:flex;align-items:center;gap:12px;font-size:11px;color:var(--txt2)}
.dot{width:6px;height:6px;border-radius:50%}.dot.on{background:var(--g);box-shadow:0 0 8px var(--g)}.dot.off{background:var(--r)}
.hdr button{background:none;border:none;color:var(--txt2);cursor:pointer;padding:4px 8px;font-size:12px}
.hdr button:hover{color:var(--txt)}
.bar{background:var(--bg2);display:flex;align-items:center;padding:0 12px;gap:8px;border-bottom:1px solid var(--brd)}
.bar input{background:var(--bg3);border:1px solid var(--brd);color:var(--txt);padding:4px 8px;border-radius:4px;width:120px;font-size:11px}
.bar select{background:var(--bg3);border:1px solid var(--brd);color:var(--txt);padding:3px 6px;border-radius:4px;font-size:11px}
.tabs{display:flex;gap:4px;margin-left:12px}
.tab{padding:6px 12px;border-radius:4px;cursor:pointer;color:var(--txt2);font-size:11px}.tab:hover{background:var(--bg3)}.tab.on{background:var(--b);color:#fff}
.main{display:grid;grid-template-columns:180px 1fr 200px;overflow:hidden}
@media(max-width:900px){.main{grid-template-columns:1fr}.side,.aside{display:none!important}}
.side,.aside{background:var(--bg2);border-right:1px solid var(--brd);overflow-y:auto;padding:8px}
.aside{border-right:none;border-left:1px solid var(--brd)}
.sec{margin-bottom:12px}.sec-h{font-size:9px;color:var(--txt2);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.content{overflow:auto;padding:8px}
.page{display:none}.page.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-bottom:8px}
.card{background:var(--bg2);border:1px solid var(--brd);border-radius:8px;padding:10px}.card:hover{border-color:var(--b)}
.card-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.card-t{font-size:9px;color:var(--txt2);text-transform:uppercase}
.badge{background:var(--bg4);padding:2px 6px;border-radius:4px;font-size:9px}.badge.g{color:var(--g)}.badge.r{color:var(--r)}.badge.y{color:var(--y)}.badge.b{color:var(--b)}
.val{font-size:20px;font-weight:700}.val.g{color:var(--g)}.val.r{color:var(--r)}.val.y{color:var(--y)}.val.b{color:var(--b)}.val.p{color:var(--p)}.val.o{color:var(--o)}
.lbl{font-size:10px;color:var(--txt2);margin-top:2px}
.row{display:flex;gap:6px;margin-top:6px}
.mini{flex:1;background:var(--bg3);padding:6px;border-radius:4px;text-align:center}
.mini-v{font-weight:600;font-size:12px}.mini-l{font-size:9px;color:var(--txt2)}
canvas{width:100%;height:50px;display:block;margin-top:6px}
.sig{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;background:var(--bg3);border-radius:4px;margin-bottom:3px;cursor:pointer;font-size:11px}.sig:hover{background:var(--bg4)}
.sig-tk{font-weight:600}
.tag{padding:2px 6px;border-radius:3px;font-size:9px;font-weight:600}.tag.buy{background:rgba(0,230,118,.15);color:var(--g)}.tag.sell{background:rgba(255,85,85,.15);color:var(--r)}
table{width:100%;border-collapse:collapse;font-size:11px}
th,td{padding:6px;text-align:left;border-bottom:1px solid var(--brd)}
th{color:var(--txt2);font-size:9px;text-transform:uppercase}
.pos{background:var(--bg3);padding:6px;border-radius:4px;margin-bottom:4px;font-size:11px}
.pos-h{display:flex;justify-content:space-between}.pos-tk{font-weight:600}
.alert{background:var(--bg3);padding:5px 8px;border-radius:4px;margin-bottom:3px;border-left:2px solid var(--brd);font-size:10px}.alert.e{border-color:var(--r)}.alert.w{border-color:var(--y)}
.svc{display:inline-flex;align-items:center;gap:4px;background:var(--bg3);padding:3px 8px;border-radius:4px;font-size:10px;margin:2px}.svc::before{content:'';width:5px;height:5px;border-radius:50%}.svc.ok::before{background:var(--g)}.svc.fail::before{background:var(--r)}
.heatmap{display:grid;grid-template-columns:repeat(auto-fill,minmax(60px,1fr));gap:4px}
.heat{padding:6px;border-radius:4px;text-align:center;font-size:10px}
.gauge{position:relative;width:100%;height:80px;overflow:hidden}
.gauge-bg{position:absolute;width:160px;height:80px;left:50%;transform:translateX(-50%);border-radius:80px 80px 0 0;background:linear-gradient(90deg,#ff4444 0%,#ff8800 25%,#888 50%,#88cc00 75%,#00cc00 100%)}
.gauge-mask{position:absolute;width:120px;height:60px;left:50%;bottom:0;transform:translateX(-50%);background:var(--bg2);border-radius:60px 60px 0 0}
.gauge-needle{position:absolute;width:4px;height:50px;left:50%;bottom:10px;transform-origin:bottom center;background:var(--txt);border-radius:2px;transition:transform .5s}
.gauge-val{position:absolute;bottom:5px;width:100%;text-align:center;font-size:18px;font-weight:700}
.comp-bar{height:6px;background:var(--bg4);border-radius:3px;margin:4px 0;overflow:hidden}
.comp-fill{height:100%;border-radius:3px;transition:width .3s}
.brain-btn{background:var(--bg3);border:1px solid var(--brd);color:var(--txt);padding:6px 12px;border-radius:4px;cursor:pointer;font-size:11px;margin:2px}.brain-btn:hover{background:var(--bg4);border-color:var(--b)}
.footer{background:var(--bg2);border-top:1px solid var(--brd);display:flex;align-items:center;padding:0 12px;font-size:10px;color:var(--txt2);gap:16px}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:1000;justify-content:center;align-items:center}#modal.on{display:flex}
.modal-box{background:var(--bg2);border:1px solid var(--brd);border-radius:10px;padding:16px;max-width:500px;width:95%;max-height:80vh;overflow-y:auto}
.modal-h{display:flex;justify-content:space-between;margin-bottom:12px}.modal-t{font-weight:600;font-size:14px}.modal-x{cursor:pointer;opacity:.5}.modal-x:hover{opacity:1}
#toast{position:fixed;bottom:30px;right:12px;z-index:999}.toast{background:var(--bg3);border:1px solid var(--brd);padding:8px 12px;border-radius:6px;margin-top:6px;font-size:11px}
.g{color:var(--g)}.r{color:var(--r)}.y{color:var(--y)}.b{color:var(--b)}.p{color:var(--p)}.o{color:var(--o)}
</style>
</head>
<body>
<div class="terminal">
<div class="hdr">
<span class="logo">◉ TERMINAL v7.1</span>
<div class="hdr-r"><span id="tm">--:--:--</span><span id="regime" class="badge">--</span><div class="dot off" id="ws"></div><button onclick="refresh()">↻</button></div>
</div>
<div class="bar">
<input type="text" id="q" placeholder="Search..." oninput="filter()">
<select id="f" onchange="filter()"><option value="">All</option><option value="buy">BUY</option><option value="sell">SELL</option></select>
<div class="tabs">
<div class="tab on" data-p="dash">Dashboard</div>
<div class="tab" data-p="brain">🧠 Model</div>
<div class="tab" data-p="signals">Signals</div>
<div class="tab" data-p="trades">Trades</div>
</div>
</div>
<div class="main">
<div class="side">
<div class="sec"><div class="sec-h">🎭 Fear & Greed</div>
<div class="gauge"><div class="gauge-bg"></div><div class="gauge-mask"></div><div class="gauge-needle" id="fgNeedle"></div><div class="gauge-val" id="fgVal">50</div></div>
<div style="text-align:center;font-size:10px;color:var(--txt2)" id="fgEmo">Neutral</div>
</div>
<div class="sec"><div class="sec-h">Positions</div><div id="positions"></div></div>
<div class="sec"><div class="sec-h">Top Tickers</div><div id="topTk"></div></div>
</div>
<div class="content">
<div class="page on" id="p-dash">
<div class="grid">
<div class="card"><div class="card-h"><span class="card-t">Balance</span></div><div class="val b" id="bal">--</div><div class="lbl">P&L <span id="pnl" class="g">--</span></div></div>
<div class="card"><div class="card-h"><span class="card-t">Drawdown</span><span class="badge" id="rLvl">--</span></div><div class="val" id="dd">0%</div><div class="lbl">Max <span id="mdd">0%</span></div></div>
<div class="card"><div class="card-h"><span class="card-t">Win Rate</span></div><div class="val g" id="wr">0%</div><div class="row"><div class="mini"><div class="mini-v g" id="wN">0</div><div class="mini-l">Wins</div></div><div class="mini"><div class="mini-v r" id="lN">0</div><div class="mini-l">Loss</div></div></div></div>
<div class="card"><div class="card-h"><span class="card-t">Performance</span></div><div class="row"><div class="mini"><div class="mini-v" id="sh">0</div><div class="mini-l">Sharpe</div></div><div class="mini"><div class="mini-v" id="pf">0</div><div class="mini-l">P.Factor</div></div></div></div>
</div>
<div class="grid">
<div class="card"><div class="card-t">Equity</div><canvas id="eqC"></canvas></div>
<div class="card"><div class="card-t">Drawdown</div><canvas id="ddC"></canvas></div>
<div class="card"><div class="card-t">Fear & Greed</div><canvas id="fgC"></canvas></div>
</div>
<div class="card"><div class="card-h"><span class="card-t">Signals</span><span class="badge" id="sc">0</span></div><div id="sigs" style="max-height:200px;overflow-y:auto"></div></div>
</div>
<div class="page" id="p-brain">
<div class="grid">
<div class="card" style="grid-column:span 2">
<div class="card-h"><span class="card-t">🎭 Fear & Greed Index</span><span class="badge b" id="fgBadge">50</span></div>
<div class="row">
<div class="mini"><div class="mini-v p" id="fgMain">50</div><div class="mini-l">Current</div></div>
<div class="mini"><div class="mini-v" id="fgEmoMain">Neutral</div><div class="mini-l">Emotion</div></div>
<div class="mini"><div class="mini-v b" id="regimeMain">sideways</div><div class="mini-l">Regime</div></div>
</div>
<canvas id="fgHist" style="height:60px;margin-top:10px"></canvas>
</div>
<div class=\"card\">\n<div class=\"card-t\">🤖 Model Status</div>\n<div style=\"margin-top:8px\">\n<div style=\"display:flex;justify-content:space-between;margin:4px 0\"><span class=\"txt2\">Version</span><span id=\"modelVer\" class=\"b\">--</span></div>\n<div style=\"display:flex;justify-content:space-between;margin:4px 0\"><span class=\"txt2\">Accuracy</span><span id=\"modelAcc\" class=\"g\">--</span></div>\n<div style=\"display:flex;justify-content:space-between;margin:4px 0\"><span class=\"txt2\">Features</span><span id=\"modelFeat\">--</span></div>\n<div style=\"display:flex;justify-content:space-between;margin:4px 0\"><span class=\"txt2\">Ready</span><span id=\"modelReady\">--</span></div>\n</div>\n</div>\n<div class=\"card\">\n<div class=\"card-t\">Model Config</div>
<div style="margin-top:8px">
<div style="display:flex;justify-content:space-between;margin:4px 0"><span class="txt2">Min Confidence</span><span id="cfgConf">45%</span></div>
<div style="display:flex;justify-content:space-between;margin:4px 0"><span class="txt2">Max Position</span><span id="cfgPos">10%</span></div>
<div style="display:flex;justify-content:space-between;margin:4px 0"><span class="txt2">Max Trades/Day</span><span id="cfgTrades">20</span></div>
</div>
</div>
</div>
<div class="card"><div class="card-t">F&G Components</div><div id="fgComps" style="margin-top:8px"></div></div>
<div class="grid" style="margin-top:8px">
<div class="card">
<div class="card-t">🔍 Analyze Ticker</div>
<div style="margin-top:8px">
<select id="analysisTk" style="width:100%;padding:6px;background:var(--bg3);border:1px solid var(--brd);color:var(--txt);border-radius:4px">
<option>SBER</option><option>GAZP</option><option>LKOH</option><option>GMKN</option><option>NVTK</option><option>ROSN</option><option>VTBR</option><option>YNDX</option><option>TCSG</option><option>MGNT</option>
</select>
<button class="brain-btn" style="width:100%;margin-top:6px" onclick="analyzeTk()">Analyze</button>
</div>
</div>
<div class="card">
<div class="card-t">⚡ Actions</div>
<div style="margin-top:8px">
<button class="brain-btn" style="width:100%" onclick="brainCorrect()">🔧 Auto-Correct</button>
<button class="brain-btn" style="width:100%;margin-top:4px" onclick="brainRetrain()">🔄 Retrain Model</button>
</div>
</div>
</div>
<div class="card" style="margin-top:8px"><div class="card-t">📊 Analysis Result</div><div id="analysisResult" style="margin-top:8px;font-size:11px">Select ticker and click Analyze</div></div>
</div>
<div class="page" id="p-signals"><div class="card"><div class="card-t">All Signals</div><div id="allSigs" style="max-height:70vh;overflow-y:auto"></div></div></div>
<div class="page" id="p-trades"><div class="card"><div class="card-h"><span class="card-t">Trades</span><button onclick="exp()" style="font-size:10px;background:var(--bg3);border:1px solid var(--brd);color:var(--txt);padding:3px 8px;border-radius:4px;cursor:pointer">Export</button></div><div style="overflow:auto;max-height:70vh"><table><thead><tr><th>Ticker</th><th>Side</th><th>Qty</th><th>Price</th><th>P&L</th><th>Time</th></tr></thead><tbody id="tbl"></tbody></table></div></div></div>
</div>
<div class="aside">
<div class="sec"><div class="sec-h">Alerts</div><div id="alerts"></div></div>
<div class="sec"><div class="sec-h">Services</div><div id="svcs"></div></div>
<div class="sec"><div class="sec-h">System</div><div style="font-size:10px;color:var(--txt2)"><div>WS: <span id="wsN">0</span></div><div>Cache: <span id="cacheR">0%</span></div><div>Uptime: <span id="up">0s</span></div></div></div>
</div>
</div>
<div class="footer"><span>◉ Terminal v7.1</span><span>F&G: <b id="fFg" class="y">50</b></span><span>Signals: <b id="fSig">0</b></span><span>P&L: <b id="fPnl" class="g">0</b></span></div>
</div>
<div id="modal"><div class="modal-box"><div class="modal-h"><span class="modal-t" id="mT">--</span><span class="modal-x" onclick="closeM()">×</span></div><div id="mB"></div></div></div>
<div id="toast"></div>
<script>
let ws,D,q='',flt='';
const $=id=>document.getElementById(id);
const fmt=(v,s)=>v==null||v===0?'--':(s&&v>0?'+':'')+Math.round(v).toLocaleString('ru')+' ₽';

function connect(){
ws=new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
ws.onopen=()=>{$('ws').className='dot on';setInterval(()=>ws?.readyState===1&&ws.send('ping'),20000)};
ws.onclose=()=>{$('ws').className='dot off';setTimeout(connect,3000)};
ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.t==='u'){D=d;render(d)}};
}

function render(d){
$('tm').textContent=new Date().toLocaleTimeString();
const p=d.port||{},m=d.m||{},sys=d.sys||{},brain=d.brain||{};

const bal=p.total_value||0;
$('bal').textContent=bal>0?fmt(bal):'--';
const pnl=p.unrealized_pnl||m.pnl||0;
$('pnl').textContent=fmt(pnl,true);$('pnl').className=pnl>=0?'g':'r';
$('fPnl').textContent=fmt(pnl,true);$('fPnl').className=pnl>=0?'g':'r';

const dd=m.dd||0;
$('dd').textContent=dd.toFixed(1)+'%';$('dd').className='val '+(dd>5?'r':dd>2?'y':'g');
$('mdd').textContent=(m.maxDd||0).toFixed(1)+'%';
$('rLvl').textContent=dd>5?'HIGH':dd>2?'MED':'LOW';$('rLvl').className='badge '+(dd>5?'r':dd>2?'y':'g');

$('wr').textContent=((m.wr||0)*100).toFixed(0)+'%';
$('wN').textContent=m.w||0;$('lN').textContent=m.l||0;
$('sh').textContent=(m.sharpe||0).toFixed(2);
$('pf').textContent=(m.pf||0).toFixed(2);

$('wsN').textContent=sys.ws||0;
$('cacheR').textContent=((sys.cache?.r||0)*100).toFixed(0)+'%';
$('up').textContent=fmtT(sys.up||0);
$('fSig').textContent=m.sigN||0;

const fg=brain.fg||50;
const fgEmo=brain.fgEmo||'neutral';
$('fgVal').textContent=Math.round(fg);
$('fgMain').textContent=Math.round(fg);
$('fgBadge').textContent=Math.round(fg);
$('fFg').textContent=Math.round(fg);
$('fgEmo').textContent=fgEmo;
$('fgEmoMain').textContent=fgEmo;
$('fgNeedle').style.transform=`rotate(${(fg/100)*180-90}deg)`;

const fgColor=fg<30?'r':fg<45?'o':fg<55?'y':fg<70?'b':'g';
$('fgVal').className='gauge-val '+fgColor;
$('fgMain').className='mini-v '+fgColor;
$('fFg').className=fgColor;

const regime=brain.regime||'sideways';
$('regime').textContent=regime;
$('regimeMain').textContent=regime;
const regimeColors={'crisis':'r','trending_down':'o','sideways':'y','trending_up':'g','high_vol':'p'};
$('regime').className='badge '+(regimeColors[regime]||'');
$('regimeMain').className='mini-v '+(regimeColors[regime]||'b');

$('cfgConf').textContent=((brain.minConf||0.45)*100).toFixed(0)+'%';
$('cfgPos').textContent=((brain.maxPos||0.1)*100).toFixed(0)+'%';
$('cfgTrades').textContent=brain.maxTrades||20;
const model=brain.model||{};
$('modelVer').textContent=model.version||'--';
$('modelAcc').textContent=model.accuracy?((model.accuracy*100).toFixed(1)+'%'):'--';
$('modelAcc').className=model.accuracy>0.5?'g':model.accuracy>0.4?'y':'r';
$('modelFeat').textContent=model.features||'--';
$('modelReady').innerHTML=model.ready?'<span class=\"g\">✓ Yes</span>':'<span class=\"r\">✗ No</span>';

const comps=brain.fgComp||{};
$('fgComps').innerHTML=Object.entries(comps).map(([k,v])=>{
const c=v<30?'#ff5555':v<45?'#ff9100':v<55?'#888':v<70?'#40c4ff':'#00e676';
return`<div style="margin:6px 0"><div style="display:flex;justify-content:space-between;font-size:10px"><span>${k}</span><span>${Math.round(v)}</span></div><div class="comp-bar"><div class="comp-fill" style="width:${v}%;background:${c}"></div></div></div>`;
}).join('');

const h=m.hist||{};
line($('eqC'),h.eq||[]);
line($('ddC'),h.dd||[],1);
line($('fgC'),brain.fgHist||[],0,'#ffd740');
line($('fgHist'),brain.fgHist||[],0,'#e040fb');

renderSigs(d.signals||[]);
renderTrades(d.trades||[]);
renderPos(d.pos||[]);
renderAlerts(d.alerts||[]);
renderSvcs(sys.cb||{});
renderTop(m.topTk||[]);
}

function renderSigs(sigs){
let f=sigs.filter(s=>!q||s.ticker?.toLowerCase().includes(q));
if(flt==='buy')f=f.filter(s=>s.signal>0);else if(flt==='sell')f=f.filter(s=>s.signal<0);
$('sc').textContent=f.length;
const html=s=>`<div class="sig" onclick="showTk('${s.ticker}')"><span class="sig-tk">${s.ticker}</span><span class="tag ${s.signal>0?'buy':'sell'}">${s.signal>0?'BUY':'SELL'} ${((s.confidence||0)*100).toFixed(0)}%</span></div>`;
$('sigs').innerHTML=f.slice(0,15).map(html).join('');
$('allSigs').innerHTML=f.map(html).join('');
}

function renderTrades(t){
$('tbl').innerHTML=t.filter(x=>!q||x.ticker?.toLowerCase().includes(q)).slice(0,100).map(x=>`<tr><td><b>${x.ticker||'--'}</b></td><td><span class="tag ${x.side==='buy'?'buy':'sell'}">${x.side||'--'}</span></td><td>${x.quantity||0}</td><td>${fmt(x.price)}</td><td class="${(x.pnl||0)>=0?'g':'r'}">${fmt(x.pnl,1)}</td><td style="color:var(--txt2)">${x.timestamp?new Date(x.timestamp).toLocaleTimeString():'--'}</td></tr>`).join('');
}

function renderPos(p){
$('positions').innerHTML=p.length?p.map(x=>{const pnl=x.unrealized_pnl||0;return`<div class="pos"><div class="pos-h"><span class="pos-tk">${x.ticker}</span><span class="${pnl>=0?'g':'r'}">${fmt(pnl,1)}</span></div><div style="font-size:9px;color:var(--txt2)">${x.quantity} × ${fmt(x.current_price)}</div></div>`}).join(''):'<div style="color:var(--txt2)">No positions</div>';
}

function renderTop(tk){$('topTk').innerHTML=tk.length?tk.slice(0,5).map(([n,s])=>`<div class="pos"><div class="pos-h"><span class="pos-tk">${n}</span><span class="${s.pnl>=0?'g':'r'}">${fmt(s.pnl,1)}</span></div></div>`).join(''):'--';}
function renderAlerts(a){$('alerts').innerHTML=a.length?a.slice(0,6).map(x=>`<div class="alert ${x.sev==='error'?'e':x.sev==='warning'?'w':''}">${x.title}</div>`).join(''):'<div style="color:var(--txt2)">No alerts</div>';}
function renderSvcs(cb){$('svcs').innerHTML=Object.entries(cb).map(([n,s])=>`<span class="svc ${s}">${n}</span>`).join('');}

function line(c,data,inv=0,clr=null){
if(!c||!data.length)return;
const ctx=c.getContext('2d'),w=c.width=c.offsetWidth*2,h=c.height=c.offsetHeight*2;
ctx.scale(2,2);const W=w/2,H=h/2;
const max=Math.max(...data)||1,min=Math.min(...data);
ctx.clearRect(0,0,W,H);ctx.beginPath();
data.forEach((v,i)=>{const x=i/(data.length-1)*W,y=H-((v-min)/(max-min||1))*H*.85;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
ctx.strokeStyle=clr||(inv?'#ff5555':'#00e676');ctx.lineWidth=1.5;ctx.stroke();
}

function fmtT(s){return s<60?s+'s':s<3600?Math.round(s/60)+'m':Math.round(s/3600)+'h'}
function refresh(){ws?.readyState===1&&ws.send('{"a":"r"}')}
function filter(){q=$('q').value.toLowerCase();flt=$('f').value;D&&(renderSigs(D.signals||[]),renderTrades(D.trades||[]))}
function showTk(tk){$('mT').textContent=tk;const s=D?.signals?.find(x=>x.ticker===tk);$('mB').innerHTML=s?`<div><span class="tag ${s.signal>0?'buy':'sell'}">${s.signal>0?'BUY':'SELL'}</span> ${((s.confidence||0)*100).toFixed(0)}%<div style="margin-top:8px">Price: ${fmt(s.price)}</div></div>`:'No data';$('modal').classList.add('on');}
function closeM(){$('modal').classList.remove('on')}
function exp(){window.open('/api/export?fmt=csv')}

async function brainCorrect(){
toast('Running auto-correction...');
const r=await fetch('/api/brain/correct',{method:'POST'}).then(r=>r.json());
if(r.corrections?.length){toast('✅ '+r.corrections.length+' corrections','g');}else{toast('No corrections needed','y');}
}

async function brainRetrain(){
toast('Starting retrain...');
const r=await fetch('/api/brain/retrain',{method:'POST'}).then(r=>r.json());
if(r.triggered){toast('✅ Retrain started','g');}else{toast('Retrain on cooldown','y');}
}

async function analyzeTk(){
const tk=$('analysisTk').value;
$('analysisResult').innerHTML='Loading...';
const r=await fetch('/api/brain/analysis/'+tk).then(r=>r.json());
if(r.error){$('analysisResult').innerHTML='Error: '+r.error;return;}
const liq=r.liquidity||{};
const corr=r.correlation||{};
const patterns=r.patterns||[];
$('analysisResult').innerHTML=`
<div class="grid" style="grid-template-columns:1fr 1fr 1fr;gap:8px">
<div class="mini"><div class="mini-v b">${(liq.liquidity_score||0).toFixed(1)}%</div><div class="mini-l">Liquidity</div></div>
<div class="mini"><div class="mini-v">${((liq.bid_ask_spread_pct||0)*100).toFixed(2)}%</div><div class="mini-l">Spread</div></div>
<div class="mini"><div class="mini-v y">${(corr.beta||1).toFixed(2)}</div><div class="mini-l">Beta</div></div>
</div>
<div style="margin-top:10px"><b>Correlations:</b></div>
<div style="font-size:10px;color:var(--txt2)">${Object.entries(corr.correlations||{}).slice(0,5).map(([k,v])=>`${k}: ${v}`).join(', ')||'--'}</div>
<div style="margin-top:10px"><b>Patterns:</b></div>
<div>${patterns.length?patterns.map(p=>`<span class="tag ${p.signal}">${p.name} (${((p.strength||0)*100).toFixed(0)}%)</span> `).join(''):'<span style="color:var(--txt2)">No patterns</span>'}</div>`;
}

function toast(msg,c=''){const t=document.createElement('div');t.className='toast '+c;t.textContent=msg;$('toast').appendChild(t);setTimeout(()=>t.remove(),3000);}

document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));document.querySelectorAll('.page').forEach(x=>x.classList.remove('on'));t.classList.add('on');$('p-'+t.dataset.p).classList.add('on')});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeM();if(e.key==='r')refresh()});
$('modal').onclick=e=>e.target===$('modal')&&closeM();
connect();
</script>
</body>
</html>'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
