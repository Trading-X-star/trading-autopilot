#!/usr/bin/env python3
"""Trading Terminal v10.0 - With Production/Sandbox Switch"""

import os, json, asyncio, logging, time, statistics
from datetime import datetime
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Response, WebSocket, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
log = logging.getLogger("terminal")

def tinkoff_to_float(money) -> float:
    if money is None: return 0.0
    if isinstance(money, (int, float)): return float(money)
    if isinstance(money, dict):
        return int(money.get('units', 0) or 0) + int(money.get('nano', 0) or 0) / 1e9
    return 0.0

class C:
    STRATEGY = os.getenv("STRATEGY_URL", "http://strategy:8005")
    EXECUTOR = os.getenv("EXECUTOR_URL", "http://executor:8007")
    DATAFEED = os.getenv("DATAFEED_URL", "http://datafeed:8006")
    RISK = os.getenv("RISK_MANAGER_URL", "http://risk-manager:8001")
    ORCH = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
    BRAIN = os.getenv("BRAIN_URL", "http://scheduler:8009")
    AUTOMATION = os.getenv("AUTOMATION_URL", "http://automation-v2:8030")
    REDIS = os.getenv("REDIS_URL", "redis://redis:6379/0")

class Cache:
    def __init__(self, ttl=2):
        self._d, self._ttl = {}, ttl
    async def get(self, k):
        e = self._d.get(k)
        if not e: return None, 0
        v, x = e
        if time.time() < x: return v, 1
        del self._d[k]; return None, 0
    async def set(self, k, v, t=None):
        if len(self._d) > 500: del self._d[min(self._d, key=lambda x: self._d[x][1])]
        self._d[k] = (v, time.time() + (t or self._ttl))

class WS:
    def __init__(self): self._c, self._l = {}, asyncio.Lock()
    async def conn(self, ws):
        await ws.accept()
        async with self._l: self._c[ws] = time.time()
    async def disc(self, ws):
        async with self._l: self._c.pop(ws, None)
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
        return len(cs) - len(dead)
    @property
    def n(self): return len(self._c)

class Terminal:
    def __init__(self):
        self.redis = self.http = None
        self.ws, self.cache = WS(), Cache(2)
        self._run, self._tasks = False, []
        self._hist = {k: deque(maxlen=500) for k in ["dd", "pnl", "eq", "fg"]}
        self._sparks, self._tkstats = {}, {}
        self._start, self._peak_eq = 0.0, 0
        self._alerts = deque(maxlen=50)

    async def start(self):
        self._start = time.time()
        self.redis = aioredis.from_url(C.REDIS, decode_responses=True)
        self.http = httpx.AsyncClient(timeout=8)
        self._run = True
        self._tasks = [asyncio.create_task(self._loop()), asyncio.create_task(self._sub())]
        log.info("✅ Terminal v10.0 started")

    async def stop(self):
        self._run = False
        for t in self._tasks: t.cancel()
        if self.http: await self.http.aclose()
        if self.redis: await self.redis.close()

    async def _f(self, url, ep):
        v, hit = await self.cache.get(ep)
        if hit: return v
        try:
            r = await self.http.get(url)
            if r.status_code >= 400: return {}
            d = r.json()
            await self.cache.set(ep, d)
            return d
        except: return {}

    def _parse_portfolio(self, port):
        if not port: return {"balance": 0, "total_value": 0, "unrealized_pnl": 0}
        if 'totalAmountPortfolio' in port:
            return {"balance": tinkoff_to_float(port.get('totalAmountCurrencies')),
                    "total_value": tinkoff_to_float(port.get('totalAmountPortfolio')),
                    "unrealized_pnl": tinkoff_to_float(port.get('expectedYield'))}
        return {"balance": port.get('balance', 0), "total_value": port.get('total_value', 0), 
                "unrealized_pnl": port.get('unrealized_pnl', 0)}

    def _parse_positions(self, pos_data):
        if not pos_data: return []
        positions = pos_data if isinstance(pos_data, list) else pos_data.get('positions', [])
        if isinstance(positions, dict):
            positions = [{"ticker": k, "quantity": v.get("quantity", 0), "avg_price": v.get("avg_price", 0)} for k, v in positions.items() if v.get("quantity", 0) != 0]
        result = []
        for p in positions:
            if not isinstance(p, dict): continue
            ticker = p.get('ticker', '')
            if ticker in ('RUB000UTSTOM', ''): continue
            qty = tinkoff_to_float(p.get('quantity', p.get('balance', 0)))
            if qty == 0: continue
            avg = tinkoff_to_float(p.get('averagePositionPrice', p.get('avg_price', 0)))
            cur = tinkoff_to_float(p.get('currentPrice', p.get('current_price', avg)))
            pnl = (cur - avg) * qty if avg > 0 else 0
            result.append({"ticker": ticker, "quantity": qty, "avg_price": avg, "current_price": cur,
                          "market_value": qty * cur, "unrealized_pnl": pnl, "pnl_pct": ((cur/avg)-1)*100 if avg>0 else 0})
        return result

    async def data(self):
        res = await asyncio.gather(
            self._f(f"{C.STRATEGY}/scan", "strat_s"),
            self._f(f"{C.DATAFEED}/prices", "feed_p"),
            self._f(f"{C.EXECUTOR}/portfolio", "exec_pf"),
            self._f(f"{C.EXECUTOR}/trades?limit=200", "exec_tr"),
            self._f(f"{C.EXECUTOR}/mode", "exec_mode"),
            self._f(f"{C.RISK}/health", "risk_h"),
            self._f(f"{C.BRAIN}/sentiment", "brain_fg"),
            self._f(f"{C.BRAIN}/config", "brain_cfg"),
            self._f(f"{C.AUTOMATION}/status", "auto_st"),
            self._f(f"{C.AUTOMATION}/positions", "auto_pos"),
            return_exceptions=True
        )
        def sf(v): return {} if isinstance(v, (Exception, type(None))) else v
        sig, prices, port_raw, trades_raw, mode_raw, risk, sentiment, brain_cfg, auto_st, auto_pos = map(sf, res)

        sig_l = sig if isinstance(sig, list) else sig.get("signals", [])
        port_d = self._parse_portfolio(port_raw)
        pos_l = self._parse_positions(port_raw)
        tr_l = trades_raw if isinstance(trades_raw, list) else trades_raw.get("trades", [])

        for tk, pd in (prices if isinstance(prices, dict) else {}).items():
            if isinstance(pd, dict):
                if tk not in self._sparks: self._sparks[tk] = deque(maxlen=50)
                self._sparks[tk].append(pd.get("price", 0))

        dd = float(risk.get("drawdown_pct", risk.get("drawdown", 0)))
        self._hist["dd"].append(dd)

        pnls = [t.get("pnl", 0) or 0 for t in tr_l if isinstance(t, dict)]
        total_pnl = sum(pnls)
        self._hist["pnl"].append(total_pnl)

        eq = float(port_d.get("total_value", 0))
        self._hist["eq"].append(eq)
        self._peak_eq = max(self._peak_eq, eq) if eq > 0 else self._peak_eq

        fg_val = sentiment.get("value", 50)
        self._hist["fg"].append(fg_val)

        w = sum(1 for p in pnls if p > 0)
        l = sum(1 for p in pnls if p < 0)
        wr = w / len(pnls) if pnls else 0
        aw = sum(p for p in pnls if p > 0) / w if w else 0
        al = abs(sum(p for p in pnls if p < 0) / l) if l else 0
        pf = aw / al if al else 0
        sharpe = (sum(pnls)/len(pnls)) / max(.01, statistics.stdev(pnls) if len(pnls)>1 else 1) if pnls else 0

        for t in tr_l:
            tk = t.get("ticker")
            if tk:
                if tk not in self._tkstats: self._tkstats[tk] = {"pnl": 0, "n": 0, "w": 0}
                self._tkstats[tk]["pnl"] += t.get("pnl", 0) or 0
                self._tkstats[tk]["n"] += 1
                if (t.get("pnl", 0) or 0) > 0: self._tkstats[tk]["w"] += 1

        top_tk = sorted(self._tkstats.items(), key=lambda x: x[1]["pnl"], reverse=True)[:10]
        long_exp = sum(p.get("market_value", 0) for p in pos_l if p.get("quantity", 0) > 0)
        short_exp = abs(sum(p.get("market_value", 0) for p in pos_l if p.get("quantity", 0) < 0))

        # Trading mode info
        trading_mode = mode_raw.get("mode", "sandbox")
        is_sandbox = mode_raw.get("sandbox", True)

        return {
            "t": "u", "ts": datetime.utcnow().isoformat() + "Z",
            "signals": sig_l, "prices": prices, "port": port_d, "pos": pos_l, "trades": tr_l,
            "sparks": {k: list(v) for k, v in list(self._sparks.items())[:30]},
            "brain": {"fg": fg_val, "fgEmo": sentiment.get("emotion", "neutral"), "regime": brain_cfg.get("regime", "sideways"),
                     "minConf": brain_cfg.get("min_confidence", 0.45), "fgHist": list(self._hist["fg"])[-100:]},
            "auto": {"enabled": auto_st.get("enabled", False), "regime": auto_st.get("regime", "unknown"),
                    "positions": auto_st.get("positions", 0), "daily_trades": auto_st.get("daily_trades", 0),
                    "jobs": auto_st.get("jobs", []), "pos_details": auto_pos if isinstance(auto_pos, dict) else {}},
            "mode": {"sandbox": is_sandbox, "mode": trading_mode, "limits": mode_raw.get("limits", {})},
            "m": {"dd": round(dd, 2), "maxDd": round(max(self._hist["dd"]) if self._hist["dd"] else 0, 2),
                 "pnl": round(total_pnl, 0), "eq": round(eq, 0), "peakEq": round(self._peak_eq, 0),
                 "wr": round(wr, 3), "pf": round(pf, 2), "sharpe": round(sharpe, 2),
                 "n": len(tr_l), "w": w, "l": l,
                 "sigN": len([x for x in sig_l if x.get("signal", 0) != 0]), "posN": len(pos_l),
                 "longExp": round(long_exp, 0), "shortExp": round(short_exp, 0),
                 "hist": {k: list(v)[-200:] for k, v in self._hist.items()}, "topTk": top_tk},
            "alerts": list(self._alerts),
            "sys": {"ws": self.ws.n, "up": round(time.time() - self._start)},
        }

    async def _loop(self):
        while self._run:
            try:
                if self.ws.n > 0: await self.ws.send(await self.data())
                await asyncio.sleep(1.5)
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
                        if ch == "alerts":
                            self._alerts.appendleft({"t": d.get("title", ""), "s": d.get("severity", "info"), "ts": datetime.now().isoformat()})
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

app = FastAPI(title="Trading Terminal", version="10.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health(): return {"ok": 1, "v": "10.0"}

@app.get("/api/data")
async def api_data(): return await term.data()

@app.post("/api/mode/switch")
async def switch_mode(sandbox: bool = True):
    """Switch between sandbox and production mode"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{C.EXECUTOR}/mode/switch", json={"sandbox": sandbox})
            return r.json()
    except Exception as e: return {"error": str(e)}

@app.get("/api/mode")
async def get_mode():
    """Get current trading mode"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{C.EXECUTOR}/mode")
            return r.json()
    except Exception as e: return {"error": str(e), "sandbox": True, "mode": "unknown"}

@app.post("/api/auto/toggle")
async def auto_toggle(enabled: bool = True):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{C.AUTOMATION}/toggle?enabled={str(enabled).lower()}")
            return r.json()
    except Exception as e: return {"error": str(e)}

@app.post("/api/auto/cycle")
async def auto_force_cycle():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{C.AUTOMATION}/force_cycle")
            return r.json()
    except Exception as e: return {"error": str(e)}

@app.post("/api/order")
async def place_order(ticker: str, side: str, qty: int = 1):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{C.EXECUTOR}/order", json={"ticker": ticker, "direction": side.upper(), "quantity": qty, "order_type": "MARKET"})
            return r.json()
    except Exception as e: return {"error": str(e)}

@app.post("/api/close")
async def close_position(ticker: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{C.EXECUTOR}/close", json={"ticker": ticker})
            return r.json()
    except Exception as e: return {"error": str(e)}

@app.get("/api/ticker/{ticker}")
async def ticker_info(ticker: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            prices = await client.get(f"{C.DATAFEED}/prices")
            p = prices.json().get(ticker, {}) if prices.status_code == 200 else {}
            stats = term._tkstats.get(ticker, {})
            spark = list(term._sparks.get(ticker, []))
            return {"ticker": ticker, "price": p, "stats": stats, "spark": spark}
    except Exception as e: return {"error": str(e)}

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
async def index(): return HTML

HTML = r'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Trading Terminal v10.0</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
<style>
:root{--bg:#0a0a12;--bg2:#0e0e18;--bg3:#141420;--bg4:#1a1a28;--brd:#252535;--txt:#d0d0e0;--txt2:#606075;--g:#00e676;--r:#ff5555;--y:#ffd740;--b:#40c4ff;--p:#e040fb;--o:#ff9100;--glow:0 0 20px}
[data-theme="light"]{--bg:#f5f5f7;--bg2:#ffffff;--bg3:#e8e8ed;--bg4:#d5d5dc;--brd:#c5c5d0;--txt:#1a1a2e;--txt2:#6a6a80}
*{margin:0;padding:0;box-sizing:border-box}
body{font:12px ui-monospace,'SF Mono',Monaco,monospace;background:var(--bg);color:var(--txt);overflow:hidden;touch-action:pan-y}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:var(--brd);border-radius:3px}
::selection{background:var(--b);color:#fff}

.app{display:grid;grid-template-rows:40px 36px 1fr 28px;height:100vh;height:100dvh}
.hdr{background:var(--bg2);display:flex;align-items:center;padding:0 12px;border-bottom:1px solid var(--brd);gap:12px}
.logo{font-weight:700;font-size:14px;color:var(--g);text-shadow:var(--glow) var(--g);cursor:pointer;transition:all .3s}
.logo:hover{transform:scale(1.05);filter:brightness(1.2)}
.hdr-c{display:flex;align-items:center;gap:8px;margin-left:auto}
.hdr-c>*{cursor:pointer;padding:6px 10px;border-radius:6px;transition:all .2s}
.hdr-c>*:hover{background:var(--bg3)}

/* MODE SWITCH - NEW */
.mode-switch{display:flex;align-items:center;gap:8px;padding:4px 12px;border-radius:8px;border:2px solid var(--brd);cursor:pointer;transition:all .3s;user-select:none}
.mode-switch.sandbox{border-color:var(--y);background:rgba(255,215,64,.1)}
.mode-switch.production{border-color:var(--r);background:rgba(255,85,85,.15)}
.mode-switch:hover{transform:scale(1.05)}
.mode-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px}
.mode-switch.sandbox .mode-label{color:var(--y)}
.mode-switch.production .mode-label{color:var(--r)}
.mode-dot{width:10px;height:10px;border-radius:50%;animation:pulse-dot 2s infinite}
.mode-switch.sandbox .mode-dot{background:var(--y)}
.mode-switch.production .mode-dot{background:var(--r)}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.2)}}

/* Warning banner for production */
.prod-warning{display:none;background:linear-gradient(90deg,var(--r),var(--o));color:#fff;text-align:center;padding:4px;font-size:10px;font-weight:700;animation:flash 1s infinite}
.prod-warning.on{display:block}
@keyframes flash{0%,100%{opacity:1}50%{opacity:.7}}

.tape{background:var(--bg2);border-bottom:1px solid var(--brd);overflow:hidden;position:relative}
.tape-inner{display:flex;animation:scroll 30s linear infinite;white-space:nowrap}
.tape:hover .tape-inner{animation-play-state:paused}
@keyframes scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.tape-item{padding:8px 20px;display:inline-flex;align-items:center;gap:8px;cursor:pointer;transition:background .2s}
.tape-item:hover{background:var(--bg3)}
.tape-tk{font-weight:600}
.tape-ch{font-size:11px}

.main{display:grid;grid-template-columns:1fr;overflow:hidden;position:relative}
.panel{background:var(--bg);overflow:auto;padding:10px;position:relative}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:10px}
.card{background:var(--bg2);border:1px solid var(--brd);border-radius:12px;padding:14px;transition:all .3s;cursor:default;position:relative;overflow:hidden}
.card:hover{border-color:var(--b);transform:translateY(-3px);box-shadow:0 8px 30px rgba(0,0,0,.3)}
.card-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.card-t{font-size:9px;color:var(--txt2);text-transform:uppercase;letter-spacing:1px}
.badge{background:var(--bg4);padding:3px 10px;border-radius:6px;font-size:9px;font-weight:600;transition:all .3s}
.badge.g{background:rgba(0,230,118,.15);color:var(--g)}.badge.r{background:rgba(255,85,85,.15);color:var(--r)}.badge.y{background:rgba(255,215,64,.15);color:var(--y)}.badge.b{background:rgba(64,196,255,.15);color:var(--b)}
.val{font-size:26px;font-weight:700;transition:all .5s}
.val.g{color:var(--g)}.val.r{color:var(--r)}.val.y{color:var(--y)}.val.b{color:var(--b)}.val.p{color:var(--p)}
.lbl{font-size:10px;color:var(--txt2);margin-top:4px}
.row{display:flex;gap:8px;margin-top:8px}
.mini{flex:1;background:var(--bg3);padding:10px;border-radius:8px;text-align:center;cursor:pointer;transition:all .2s}
.mini:hover{background:var(--bg4);transform:scale(1.03)}
.mini-v{font-weight:700;font-size:15px}.mini-l{font-size:9px;color:var(--txt2);margin-top:3px}

.chart-wrap{position:relative;height:100px;margin-top:10px}
.sig{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:var(--bg3);border-radius:8px;margin-bottom:4px;cursor:pointer;transition:all .2s;border:1px solid transparent}
.sig:hover{background:var(--bg4);transform:translateX(5px);border-color:var(--b)}
.tag{padding:4px 10px;border-radius:5px;font-size:9px;font-weight:700}
.tag.buy{background:rgba(0,230,118,.2);color:var(--g)}.tag.sell{background:rgba(255,85,85,.2);color:var(--r)}

.pos{background:var(--bg3);padding:12px;border-radius:8px;margin-bottom:5px;cursor:pointer;transition:all .2s;border:1px solid transparent;position:relative}
.pos:hover{background:var(--bg4);border-color:var(--b)}
.pos.profit{border-left:3px solid var(--g)}.pos.loss{border-left:3px solid var(--r)}
.pos-h{display:flex;justify-content:space-between;align-items:center}.pos-tk{font-weight:700;font-size:13px}
.pos-close{position:absolute;right:8px;top:8px;width:20px;height:20px;border-radius:50%;background:var(--r);color:#fff;border:none;cursor:pointer;opacity:0;transition:all .2s;font-size:10px}
.pos:hover .pos-close{opacity:1}

.gauge{position:relative;width:140px;height:70px;margin:0 auto;cursor:pointer}
.gauge-bg{position:absolute;width:100%;height:100%;border-radius:70px 70px 0 0;background:conic-gradient(from 180deg,var(--r) 0deg,var(--o) 45deg,var(--y) 90deg,var(--g) 135deg,var(--g) 180deg);-webkit-mask:radial-gradient(circle at 50% 100%,transparent 45px,#000 46px);mask:radial-gradient(circle at 50% 100%,transparent 45px,#000 46px)}
.gauge-needle{position:absolute;width:4px;height:55px;left:calc(50% - 2px);bottom:0;background:linear-gradient(to top,var(--txt),transparent);border-radius:2px;transform-origin:bottom center;transition:transform .8s cubic-bezier(.68,-.55,.27,1.55)}
.gauge-val{position:absolute;bottom:-25px;width:100%;text-align:center;font-size:20px;font-weight:700}
.gauge-labels{position:absolute;width:100%;bottom:-8px;display:flex;justify-content:space-between;font-size:8px;color:var(--txt2);padding:0 5px}

.switch{position:relative;width:50px;height:26px;background:var(--bg4);border-radius:13px;cursor:pointer;transition:all .3s}
.switch.on{background:linear-gradient(135deg,var(--g),var(--b))}
.switch::after{content:'';position:absolute;width:22px;height:22px;background:#fff;border-radius:50%;top:2px;left:2px;transition:all .3s cubic-bezier(.68,-.55,.27,1.55)}
.switch.on::after{left:26px}

.btn{background:var(--bg3);border:1px solid var(--brd);color:var(--txt);padding:10px 16px;border-radius:8px;cursor:pointer;font-size:11px;font-weight:600;transition:all .2s}
.btn:hover{background:var(--bg4);border-color:var(--b);transform:translateY(-2px)}
.btn.primary{background:linear-gradient(135deg,var(--b),var(--p));border:none;color:#fff}
.btn.danger{background:var(--r);border-color:var(--r);color:#fff}
.btn.success{background:var(--g);border-color:var(--g);color:#000}

.footer{background:var(--bg2);border-top:1px solid var(--brd);display:flex;align-items:center;padding:0 12px;gap:20px;font-size:10px}
.footer>span{cursor:pointer;padding:4px 8px;border-radius:4px;transition:all .2s}
.footer .dot{width:8px;height:8px;border-radius:50%;margin-right:4px;display:inline-block}
.footer .dot.on{background:var(--g);box-shadow:0 0 8px var(--g);animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.5}}
.footer .dot.off{background:var(--r)}

.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;justify-content:center;align-items:center;backdrop-filter:blur(8px)}
.modal.on{display:flex;animation:fadeIn .2s}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modal-box{background:var(--bg2);border:1px solid var(--brd);border-radius:16px;padding:24px;max-width:450px;width:95%;max-height:85vh;overflow-y:auto}
.modal-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:15px;border-bottom:1px solid var(--brd)}
.modal-t{font-weight:700;font-size:18px}.modal-x{cursor:pointer;font-size:24px;opacity:.5;transition:all .2s;width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:8px}
.modal-x:hover{opacity:1;background:var(--bg3);transform:rotate(90deg)}

.toast-wrap{position:fixed;top:60px;right:20px;z-index:999;display:flex;flex-direction:column;gap:10px;pointer-events:none}
.toast{background:var(--bg2);border:1px solid var(--brd);padding:14px 18px;border-radius:10px;font-size:12px;display:flex;align-items:center;gap:12px;animation:toastIn .3s;pointer-events:auto;cursor:pointer;box-shadow:0 8px 30px rgba(0,0,0,.4);max-width:300px}
@keyframes toastIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}
.toast.out{animation:toastOut .3s forwards}
@keyframes toastOut{to{transform:translateX(100%);opacity:0}}
.toast.success{border-color:var(--g)}.toast.error{border-color:var(--r)}.toast.warning{border-color:var(--y)}

.timer{position:fixed;top:50px;right:20px;background:var(--bg2);border:1px solid var(--brd);border-radius:8px;padding:8px 14px;font-size:11px;z-index:50;display:flex;align-items:center;gap:8px}
.timer.closed{border-color:var(--r)}.timer.open{border-color:var(--g)}
.timer-dot{width:8px;height:8px;border-radius:50%}
.timer.closed .timer-dot{background:var(--r)}.timer.open .timer-dot{background:var(--g);animation:blink 1s infinite}

/* Mode confirmation modal */
.mode-confirm{text-align:center;padding:20px}
.mode-confirm h2{color:var(--r);margin-bottom:20px}
.mode-confirm .warning-icon{font-size:60px;margin-bottom:20px}
.mode-confirm p{color:var(--txt2);margin-bottom:20px;line-height:1.6}
.mode-confirm .limits{background:var(--bg3);padding:15px;border-radius:8px;margin:20px 0;text-align:left}
.mode-confirm .limits div{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--brd)}
.mode-confirm .limits div:last-child{border:none}
.mode-confirm .btns{display:flex;gap:10px;margin-top:20px}
.mode-confirm .btns button{flex:1;padding:15px;font-size:14px}

@media(max-width:768px){.hdr{padding:0 8px}.logo{font-size:12px}.card{padding:10px}.val{font-size:20px}}
</style>
</head>
<body>
<div class="prod-warning" id="prodWarning">⚠️ PRODUCTION MODE - REAL MONEY ⚠️</div>

<div class="app">
<div class="hdr">
<span class="logo" onclick="location.reload()">◉ TERMINAL v10.0</span>

<!-- MODE SWITCH -->
<div class="mode-switch sandbox" id="modeSwitch" onclick="showModeModal()">
  <span class="mode-dot"></span>
  <span class="mode-label" id="modeLabel">SANDBOX</span>
</div>

<div class="hdr-c">
<span onclick="toggleTheme()" title="Theme">🌓</span>
<span onclick="toggleFullscreen()" title="Fullscreen">⛶</span>
<span id="time">--:--:--</span>
<span id="regime" class="badge">--</span>
<span><span class="dot off" id="ws"></span>WS</span>
</div>
</div>

<div class="tape" id="tape"><div class="tape-inner" id="tapeInner"></div></div>

<div class="main" id="main">
<div class="panel" id="panel1">
<div class="grid" id="cards">
<div class="card" data-id="balance">
<div class="card-h"><span class="card-t">💰 Balance</span><span class="badge b" id="modeBadge">SANDBOX</span></div>
<div class="val b" id="bal">--</div>
<div class="lbl">P&L <span id="pnl" class="g">--</span></div>
</div>
<div class="card" data-id="dd">
<div class="card-h"><span class="card-t">📉 Drawdown</span><span class="badge" id="rLvl">--</span></div>
<div class="val" id="dd">0%</div>
<div class="lbl">Max <span id="mdd">0%</span></div>
</div>
<div class="card" data-id="wr">
<div class="card-h"><span class="card-t">🎯 Win Rate</span></div>
<div class="val g" id="wr">0%</div>
<div class="row">
<div class="mini"><div class="mini-v g" id="wN">0</div><div class="mini-l">Wins</div></div>
<div class="mini"><div class="mini-v r" id="lN">0</div><div class="mini-l">Loss</div></div>
</div>
</div>
<div class="card" data-id="perf">
<div class="card-h"><span class="card-t">📊 Performance</span></div>
<div class="row">
<div class="mini"><div class="mini-v" id="sh">0</div><div class="mini-l">Sharpe</div></div>
<div class="mini"><div class="mini-v" id="pf">0</div><div class="mini-l">P.Factor</div></div>
</div>
</div>
</div>

<div class="card">
<div class="card-h"><span class="card-t">📈 Equity</span></div>
<div class="chart-wrap" id="eqWrap"><canvas id="eqChart"></canvas></div>
</div>

<div class="card" style="margin-top:10px">
<div class="card-h">
<span class="card-t">⚡ Signals</span>
<span class="badge b" id="sc">0</span>
</div>
<div id="sigs" style="max-height:250px;overflow-y:auto"></div>
</div>

<div class="card" style="margin-top:10px">
<div class="card-h">
<span class="card-t">🤖 Automation</span>
<div class="switch" id="autoSwitch" onclick="toggleAuto()"></div>
</div>
<div class="row">
<div class="mini"><div class="mini-v b" id="autoRegime">--</div><div class="mini-l">Regime</div></div>
<div class="mini"><div class="mini-v g" id="autoPos">0</div><div class="mini-l">Positions</div></div>
<div class="mini"><div class="mini-v y" id="autoTrades">0</div><div class="mini-l">Today</div></div>
</div>
<div style="margin-top:10px;display:flex;gap:8px">
<button class="btn primary" style="flex:1" onclick="forceCycle()">▶ Cycle</button>
<button class="btn danger" style="flex:1" onclick="emergencyStop()">🛑 Stop</button>
</div>
</div>

<div class="card" style="margin-top:10px">
<div class="card-h"><span class="card-t">📊 Fear & Greed</span></div>
<div class="gauge">
<div class="gauge-bg"></div>
<div class="gauge-needle" id="fgN"></div>
<div class="gauge-val" id="fgV">50</div>
<div class="gauge-labels"><span>Fear</span><span>Greed</span></div>
</div>
<div style="text-align:center;margin-top:30px;font-size:11px;color:var(--txt2)" id="fgEmo">Neutral</div>
</div>

<div class="card" style="margin-top:10px">
<div class="card-h"><span class="card-t">📂 Positions</span><span class="badge" id="posN">0</span></div>
<div id="positions"></div>
</div>
</div>
</div>

<div class="footer">
<span><span class="dot" id="wsDot"></span> <span id="wsStatus">Connecting...</span></span>
<span>Mode: <b id="fMode" class="y">SANDBOX</b></span>
<span>Signals: <b id="fSig">0</b></span>
<span>Auto: <b id="fAuto">OFF</b></span>
</div>
</div>

<div class="timer" id="timer"><span class="timer-dot"></span><span id="timerText">--:--</span></div>

<div class="modal" id="modal"><div class="modal-box"><div class="modal-h"><span class="modal-t" id="mT">--</span><span class="modal-x" onclick="closeModal()">×</span></div><div id="mB"></div></div></div>

<div class="toast-wrap" id="toasts"></div>

<script>
let ws,D,charts={},currentMode='sandbox';
const $=id=>document.getElementById(id);
const fmt=(v,s)=>v==null||isNaN(v)?'--':(s&&v>0?'+':'')+Math.round(v).toLocaleString('ru')+' ₽';

function toggleTheme(){
  const t=document.documentElement.dataset.theme==='light'?'dark':'light';
  document.documentElement.dataset.theme=t;
  localStorage.setItem('theme',t);
  toast('Theme: '+t);
}
if(localStorage.getItem('theme')==='light')document.documentElement.dataset.theme='light';

function toggleFullscreen(){
  if(!document.fullscreenElement)document.documentElement.requestFullscreen();
  else document.exitFullscreen();
}

function initCharts(){
  const cfg=(id,color)=>new Chart($(id),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:color,backgroundColor:color+'33',fill:true,tension:.4,pointRadius:0,borderWidth:2}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{color:'#252535'},ticks:{font:{size:9}}}}}});
  charts.eq=cfg('eqChart','#40c4ff');
}
function updateChart(chart,data){
  chart.data.labels=data.map((_,i)=>i);
  chart.data.datasets[0].data=data;
  chart.update('none');
}

function connect(){
  ws=new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
  ws.onopen=()=>{
    $('ws').className='dot on';$('wsDot').className='dot on';$('wsStatus').textContent='Connected';
    toast('Connected','success');
  };
  ws.onclose=()=>{
    $('ws').className='dot off';$('wsDot').className='dot off';$('wsStatus').textContent='Disconnected';
    setTimeout(connect,3000);
  };
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.t==='u'){D=d;render(d)}
    else if(d.t==='e')handleEvent(d);
  };
}

function handleEvent(e){
  if(e.ch==='trades')toast('Trade: '+e.d?.ticker,'success');
  else if(e.ch==='alerts')toast(e.d?.title||'Alert',e.d?.severity||'warning');
}

function render(d){
  $('time').textContent=new Date().toLocaleTimeString();
  const p=d.port||{},m=d.m||{},brain=d.brain||{},auto=d.auto||{},mode=d.mode||{};

  // MODE DISPLAY
  currentMode=mode.sandbox?'sandbox':'production';
  const modeSwitch=$('modeSwitch');
  modeSwitch.className='mode-switch '+currentMode;
  $('modeLabel').textContent=currentMode.toUpperCase();
  $('modeBadge').textContent=currentMode.toUpperCase();
  $('modeBadge').className='badge '+(mode.sandbox?'y':'r');
  $('fMode').textContent=currentMode.toUpperCase();
  $('fMode').className=mode.sandbox?'y':'r';
  $('prodWarning').classList.toggle('on',!mode.sandbox);

  const bal=p.total_value||0;
  $('bal').textContent=fmt(bal);
  
  const pnl=p.unrealized_pnl||0;
  $('pnl').textContent=fmt(pnl,true);
  $('pnl').className=pnl>=0?'g':'r';
  
  const dd=m.dd||0;
  $('dd').textContent=dd.toFixed(1)+'%';
  $('dd').className='val '+(dd>5?'r':dd>2?'y':'g');
  $('mdd').textContent=(m.maxDd||0).toFixed(1)+'%';
  $('rLvl').textContent=dd>5?'HIGH':dd>2?'MED':'LOW';
  $('rLvl').className='badge '+(dd>5?'r':dd>2?'y':'g');

  $('wr').textContent=((m.wr||0)*100).toFixed(0)+'%';
  $('wN').textContent=m.w||0;$('lN').textContent=m.l||0;
  $('sh').textContent=(m.sharpe||0).toFixed(2);
  $('pf').textContent=(m.pf||0).toFixed(2);
  $('fSig').textContent=m.sigN||0;
  $('posN').textContent=m.posN||0;

  const fg=brain.fg||50;
  $('fgV').textContent=Math.round(fg);
  $('fgEmo').textContent=brain.fgEmo||'neutral';
  $('fgN').style.transform=`rotate(${(fg/100)*180-90}deg)`;

  $('regime').textContent=brain.regime||'--';
  const regC={'crisis':'r','trending_down':'o','sideways':'y','trending_up':'g'};
  $('regime').className='badge '+(regC[brain.regime]||'');

  $('autoSwitch').className='switch'+(auto.enabled?' on':'');
  $('autoRegime').textContent=auto.regime||'--';
  $('autoPos').textContent=auto.positions||0;
  $('autoTrades').textContent=auto.daily_trades||0;
  $('fAuto').textContent=auto.enabled?'ON':'OFF';
  $('fAuto').className=auto.enabled?'g':'r';

  const h=m.hist||{};
  if(h.eq?.length)updateChart(charts.eq,h.eq);

  renderTape(d.prices||{});
  renderSigs(d.signals||[]);
  renderPos(d.pos||[]);
  updateTimer();
}

function renderTape(prices){
  const items=Object.entries(prices).slice(0,20).map(([tk,p])=>{
    if(typeof p!=='object')return'';
    const ch=p.change||p.change_pct||0;
    return`<div class="tape-item" onclick="showTicker('${tk}')"><span class="tape-tk">${tk}</span><span class="tape-ch ${ch>=0?'g':'r'}">${ch>=0?'+':''}${ch.toFixed(2)}%</span></div>`;
  }).join('');
  $('tapeInner').innerHTML=items+items;
}

function renderSigs(sigs){
  let f=sigs.filter(s=>s.signal!=0).sort((a,b)=>Math.abs(b.confidence||0)-Math.abs(a.confidence||0));
  $('sc').textContent=f.length;
  $('sigs').innerHTML=f.slice(0,15).map(s=>`<div class="sig" onclick="quickOrder('${s.ticker}','${s.signal>0?'buy':'sell'}')">
    <span><b>${s.ticker}</b></span>
    <span class="tag ${s.signal>0?'buy':'sell'}">${s.signal>0?'BUY':'SELL'} ${((s.confidence||0)*100).toFixed(0)}%</span>
  </div>`).join('');
}

function renderPos(p){
  $('positions').innerHTML=p.length?p.map(x=>`<div class="pos ${x.unrealized_pnl>=0?'profit':'loss'}">
    <button class="pos-close" onclick="event.stopPropagation();closePos('${x.ticker}')">×</button>
    <div class="pos-h"><span class="pos-tk">${x.ticker}</span><span class="${x.unrealized_pnl>=0?'g':'r'}">${fmt(x.unrealized_pnl,1)}</span></div>
    <div style="font-size:10px;color:var(--txt2);margin-top:4px">${x.quantity} × ${fmt(x.current_price)} <span class="${x.pnl_pct>=0?'g':'r'}">(${x.pnl_pct>=0?'+':''}${x.pnl_pct?.toFixed(1)}%)</span></div>
  </div>`).join(''):'<div style="color:var(--txt2);padding:12px;text-align:center">No positions</div>';
}

function updateTimer(){
  const now=new Date();
  const h=now.getHours(),m=now.getMinutes();
  const isOpen=(now.getDay()>0&&now.getDay()<6)&&(h>=10&&(h<18||(h===18&&m<=45)));
  const timer=$('timer');
  timer.className='timer '+(isOpen?'open':'closed');
  $('timerText').textContent=isOpen?'Market Open':'Market Closed';
}

// MODE SWITCHING
function showModeModal(){
  const isSandbox=currentMode==='sandbox';
  const newMode=isSandbox?'PRODUCTION':'SANDBOX';
  const limits=D?.mode?.limits||{max_position_rub:500,max_daily_trades:10};
  
  openModal('Switch Trading Mode',`
    <div class="mode-confirm">
      <div class="warning-icon">${isSandbox?'⚠️':'✅'}</div>
      <h2>Switch to ${newMode}?</h2>
      <p>${isSandbox?
        '<b style="color:var(--r)">WARNING: Production mode uses REAL MONEY!</b><br>All trades will be executed on your real Tinkoff account.':
        'Switching to Sandbox mode. All trades will be simulated with virtual money.'
      }</p>
      ${isSandbox?`<div class="limits">
        <div><span>Max Position</span><span>${limits.max_position_rub||500} ₽</span></div>
        <div><span>Max Daily Trades</span><span>${limits.max_daily_trades||10}</span></div>
        <div><span>Min Confidence</span><span>${(limits.min_confidence||0.5)*100}%</span></div>
      </div>`:''}
      <div class="btns">
        <button class="btn" onclick="closeModal()">Cancel</button>
        <button class="btn ${isSandbox?'danger':'success'}" onclick="confirmModeSwitch(${!isSandbox})">
          ${isSandbox?'🔴 Enable PRODUCTION':'🟢 Enable SANDBOX'}
        </button>
      </div>
    </div>
  `);
}

async function confirmModeSwitch(toSandbox){
  closeModal();
  toast('Switching mode...','warning');
  
  const res=await fetch('/api/mode/switch?sandbox='+toSandbox,{method:'POST'}).then(r=>r.json());
  
  if(res.error){
    toast('Error: '+res.error,'error');
  }else{
    toast(`Switched to ${toSandbox?'SANDBOX':'PRODUCTION'}`,toSandbox?'success':'warning');
    if(!toSandbox){
      // Production warning
      setTimeout(()=>toast('⚠️ REAL MONEY MODE ACTIVE','error'),500);
    }
  }
}

async function toggleAuto(){
  const cur=D?.auto?.enabled||false;
  
  // Block automation in production without confirmation
  if(!cur && currentMode==='production'){
    if(!confirm('⚠️ Enable automation in PRODUCTION mode?\n\nThis will execute REAL trades!')){
      return;
    }
  }
  
  const res=await fetch('/api/auto/toggle?enabled='+(!cur),{method:'POST'}).then(r=>r.json());
  if(!res.error)toast(res.enabled?'Automation ON 🚀':'Automation OFF','success');
  else toast(res.error,'error');
}

async function forceCycle(){
  if(currentMode==='production' && !confirm('Execute cycle in PRODUCTION mode?'))return;
  toast('Starting cycle...');
  await fetch('/api/auto/cycle',{method:'POST'});
  toast('Cycle started','success');
}

async function emergencyStop(){
  if(!confirm('⚠️ Stop all trading?'))return;
  await fetch('/api/auto/toggle?enabled=false',{method:'POST'});
  toast('Emergency stop!','error');
}

async function quickOrder(tk,side){
  if(currentMode==='production' && !confirm(`Execute ${side.toUpperCase()} ${tk} in PRODUCTION?`))return;
  const res=await fetch(`/api/order?ticker=${tk}&side=${side}&qty=1`,{method:'POST'}).then(r=>r.json());
  toast(res.error||`${side.toUpperCase()} ${tk}`);
}

async function closePos(tk){
  if(!confirm(`Close ${tk}?`))return;
  await fetch(`/api/close?ticker=${tk}`,{method:'POST'});
  toast(`Closed ${tk}`);
}

async function showTicker(tk){
  const res=await fetch(`/api/ticker/${tk}`).then(r=>r.json());
  openModal(tk,`
    <div class="row">
      <div class="mini"><div class="mini-v">${fmt(res.price?.price)}</div><div class="mini-l">Price</div></div>
      <div class="mini"><div class="mini-v ${(res.price?.change||0)>=0?'g':'r'}">${((res.price?.change||0)).toFixed(2)}%</div><div class="mini-l">Change</div></div>
    </div>
    <div style="margin-top:14px;display:flex;gap:10px">
      <button class="btn success" style="flex:1" onclick="quickOrder('${tk}','buy');closeModal()">🟢 BUY</button>
      <button class="btn danger" style="flex:1" onclick="quickOrder('${tk}','sell');closeModal()">🔴 SELL</button>
    </div>
  `);
}

function openModal(title,content){$('mT').textContent=title;$('mB').innerHTML=content;$('modal').classList.add('on')}
function closeModal(){$('modal').classList.remove('on')}

function toast(msg,type='info'){
  const t=document.createElement('div');
  t.className='toast '+type;
  t.innerHTML=`<span>${type==='success'?'✅':type==='error'?'❌':type==='warning'?'⚠️':'ℹ️'}</span><span>${msg}</span>`;
  t.onclick=()=>{t.classList.add('out');setTimeout(()=>t.remove(),300)};
  $('toasts').appendChild(t);
  setTimeout(()=>{t.classList.add('out');setTimeout(()=>t.remove(),300)},4000);
}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape')closeModal();
});

initCharts();
connect();
setInterval(updateTimer,60000);
</script>
</body>
</html>'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)