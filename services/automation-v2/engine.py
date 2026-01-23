#!/usr/bin/env python3
"""Trading Autopilot - Advanced Automation Engine v2.0"""
import asyncio,logging,os,httpx
from datetime import datetime,time,timedelta
from typing import Dict,List
from dataclasses import dataclass,field
from enum import Enum
from fastapi import FastAPI,BackgroundTasks
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import uvicorn
import numpy as np

logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
log=logging.getLogger(__name__)
app=FastAPI(title="Automation Engine v2.0")
scheduler=AsyncIOScheduler()

class MarketRegime(Enum):
    BULL_STRONG="bull_strong"
    BULL_WEAK="bull_weak"
    BEAR_STRONG="bear_strong"
    BEAR_WEAK="bear_weak"
    SIDEWAYS="sideways"
    HIGH_VOL="high_vol"
    CRASH="crash"

@dataclass
class Config:
    max_position_risk:float=0.02
    max_position_size:float=0.10
    max_daily_trades:int=30
    ml_weak:float=0.25
    default_stop:float=0.05
    trailing_activation:float=0.03
    trailing_distance:float=0.02
    breakeven_at:float=0.02
    tp_r1:float=1.5
    tp_r2:float=2.5
    tp_r3:float=4.0
    max_hold_days:int=20
    partial_exits:tuple=(0.33,0.33,0.34)

REGIME_ADJ={
    MarketRegime.BULL_STRONG:{"size":1.3,"stop":0.8,"max_pos":20},
    MarketRegime.BULL_WEAK:{"size":1.0,"stop":1.0,"max_pos":15},
    MarketRegime.BEAR_STRONG:{"size":0.5,"stop":1.5,"max_pos":5},
    MarketRegime.BEAR_WEAK:{"size":0.7,"stop":1.2,"max_pos":10},
    MarketRegime.SIDEWAYS:{"size":0.8,"stop":1.0,"max_pos":12},
    MarketRegime.HIGH_VOL:{"size":0.4,"stop":2.0,"max_pos":5},
    MarketRegime.CRASH:{"size":0.0,"stop":0.0,"max_pos":0},
}

CFG=Config()
SERVICES={"orchestrator":"http://orchestrator:8000","strategy":"http://strategy:8005","executor":"http://executor:8007","risk":"http://risk-manager:8001","datafeed":"http://datafeed:8006","killswitch":"http://kill-switch:8020","searxng":"http://searxng:8080"}

@dataclass
class Position:
    ticker:str
    direction:str
    qty:int
    entry:float
    current:float
    stop:float
    tps:List[float]
    opened:datetime
    confidence:float
    high:float=0
    low:float=float('inf')
    trailing:bool=False
    breakeven:bool=False
    exits:List[Dict]=field(default_factory=list)
    
    @property
    def pnl_pct(self):
        return(self.current/self.entry-1)*(1 if self.direction=="BUY" else -1)
    
    @property
    def r_multiple(self):
        risk=abs(self.entry-self.stop)
        return self.pnl_pct*self.entry/risk if risk>0 else 0

class ServiceClient:
    def __init__(self):
        self.client=httpx.AsyncClient(timeout=30)
        self.cache={}
        self.cache_exp={}
    
    async def call(self,svc,ep,method="GET",data=None,cache=0):
        key=f"{svc}:{ep}:{data}"
        if cache>0 and key in self.cache and datetime.now()<self.cache_exp.get(key,datetime.min):
            return self.cache[key]
        url=f"{SERVICES[svc]}{ep}"
        try:
            r=await self.client.get(url) if method=="GET" else await self.client.post(url,json=data)
            result=r.json() if r.status_code==200 else {"error":r.text}
            if cache>0 and "error" not in result:
                self.cache[key]=result
                self.cache_exp[key]=datetime.now()+timedelta(seconds=cache)
            return result
        except Exception as e:
            return {"error":str(e)}

class RegimeAnalyzer:
    def __init__(self,svc):
        self.svc=svc
    
    async def detect(self):
        data=await self.svc.call("datafeed","/prices",cache=60)
        prices=data.get("prices",[])
        if not prices or len(prices)<50:
            return MarketRegime.SIDEWAYS
        if len(prices)>=5 and prices[-1]/prices[-5]-1<-0.10:
            return MarketRegime.CRASH
        sma20=np.mean(prices[-20:])
        sma50=np.mean(prices[-50:])
        diff=(sma20-sma50)/sma50
        returns=np.diff(prices[-20:])/np.array(prices[-21:-1])
        vol=np.std(returns)*np.sqrt(252)
        if vol>0.50:
            return MarketRegime.HIGH_VOL
        if diff>0.06:
            return MarketRegime.BULL_STRONG
        if diff>0.03:
            return MarketRegime.BULL_WEAK
        if diff<-0.06:
            return MarketRegime.BEAR_STRONG
        if diff<-0.03:
            return MarketRegime.BEAR_WEAK
        return MarketRegime.SIDEWAYS

class PositionManager:
    def __init__(self,svc):
        self.svc=svc
        self.positions={}
    
    async def update_prices(self):
        if not self.positions:
            return
        quotes=await self.svc.call("datafeed","/prices")
        for t,pos in self.positions.items():
            if t in quotes:
                pos.current=quotes[t] if isinstance(quotes[t], (int, float)) else quotes[t].get("price", pos.current)
                pos.high=max(pos.high,pos.current)
                pos.low=min(pos.low,pos.current)
    
    async def manage(self,ticker):
        pos=self.positions.get(ticker)
        if not pos:
            return []
        actions=[]
        # Stop loss
        if(pos.direction=="BUY" and pos.current<=pos.stop)or(pos.direction=="SELL" and pos.current>=pos.stop):
            return[{"action":"CLOSE","ticker":ticker,"reason":"STOP_LOSS","pnl":pos.pnl_pct}]
        # Breakeven
        if not pos.breakeven and pos.pnl_pct>=CFG.breakeven_at:
            buf=pos.entry*0.001
            pos.stop=pos.entry+buf if pos.direction=="BUY" else pos.entry-buf
            pos.breakeven=True
            actions.append({"action":"UPDATE_STOP","ticker":ticker,"stop":pos.stop,"reason":"BREAKEVEN"})
        # Trailing activation
        if pos.breakeven and not pos.trailing and pos.pnl_pct>=CFG.trailing_activation:
            pos.trailing=True
            actions.append({"action":"TRAILING_ON","ticker":ticker})
        # Trailing update
        if pos.trailing:
            trail=pos.entry*CFG.trailing_distance
            new_stop=pos.high-trail if pos.direction=="BUY" else pos.low+trail
            if(pos.direction=="BUY" and new_stop>pos.stop)or(pos.direction=="SELL" and new_stop<pos.stop):
                pos.stop=new_stop
                actions.append({"action":"UPDATE_STOP","ticker":ticker,"stop":new_stop,"reason":"TRAILING"})
        # Take profit
        exits_done=len(pos.exits)
        if exits_done<len(pos.tps):
            tp=pos.tps[exits_done]
            if(pos.direction=="BUY" and pos.current>=tp)or(pos.direction=="SELL" and pos.current<=tp):
                exit_qty=int(pos.qty*CFG.partial_exits[exits_done])
                pos.exits.append({"price":pos.current,"qty":exit_qty,"r":pos.r_multiple})
                pos.qty-=exit_qty
                actions.append({"action":"PARTIAL_EXIT","ticker":ticker,"qty":exit_qty,"reason":f"TP{exits_done+1}","r":pos.r_multiple})
        # Time exit
        if(datetime.now()-pos.opened).days>=CFG.max_hold_days:
            actions.append({"action":"CLOSE","ticker":ticker,"reason":"MAX_TIME"})
        return actions
    
    async def execute(self,action):
        ticker=action["ticker"]
        if action["action"] in["CLOSE","PARTIAL_EXIT"]:
            pos=self.positions.get(ticker)
            if not pos:
                return False
            qty=action.get("qty",pos.qty)
            direction="SELL" if pos.direction=="BUY" else "BUY"
            result=await self.svc.call("executor","/order","POST",{"ticker":ticker,"side":direction.lower(),"quantity":qty,"order_type":"MARKET","reason":action["reason"]})
            if result.get("status")=="executed":
                if action["action"]=="CLOSE":
                    del self.positions[ticker]
                log.info(f"✓ {action['action']} {ticker}: {action['reason']}")
                return True
        elif action["action"]=="UPDATE_STOP":
            log.info(f"📍 Stop {ticker}: {action['stop']:.2f} ({action['reason']})")
            return True
        return False

    async def sync_from_executor(self):
        """Sync positions from executor"""
        try:
            portfolio = await self.svc.call("executor", "/portfolio")
            exec_positions = portfolio.get("positions", {})
            if isinstance(exec_positions, dict):
                for ticker, data in exec_positions.items():
                    qty = data.get("quantity", 0)
                    if qty != 0 and ticker not in self.positions:
                        self.positions[ticker] = Position(
                            ticker=ticker,
                            direction="BUY" if qty > 0 else "SELL",
                            qty=abs(qty),
                            entry=data.get("avg_price", 0),
                            current=data.get("avg_price", 0),
                            stop=0, tps=[0,0,0],
                            opened=datetime.now(),
                            confidence=0.5,
                            high=data.get("avg_price", 0),
                            low=data.get("avg_price", 0)
                        )
                        log.info(f"📥 Synced {ticker}: {qty} @ {data.get('avg_price', 0)}")
        except Exception as e:
            log.warning(f"Sync error: {e}")

class PositionSizer:
    @staticmethod
    def calculate(portfolio,confidence,stop_pct,regime,current_positions,correlation):
        risk_amt=portfolio*CFG.max_position_risk
        base=risk_amt/stop_pct if stop_pct>0 else 0
        base=min(base,portfolio*CFG.max_position_size)
        regime_adj=REGIME_ADJ.get(regime,{"size":1.0,"max_pos":15})
        conf_mult=0.5+confidence*0.5
        regime_mult=regime_adj["size"]
        corr_mult=1.0-(correlation*0.5) if correlation>0.5 else 1.0
        max_pos=regime_adj.get("max_pos",15)
        pos_mult=1.0-(current_positions/max_pos)*0.3 if current_positions>0 else 1.0
        return max(0,base*conf_mult*regime_mult*corr_mult*pos_mult)

class NewsAnalyzer:
    POS=["рост","прибыль","увеличение","дивиденды","успех","рекорд"]
    NEG=["падение","убыток","снижение","риск","санкции","кризис","долг"]
    
    def __init__(self,svc):
        self.svc=svc
        self.analyzed=set()
    
    async def check(self,ticker):
        try:
            news=await self.svc.call("searxng","/search","GET",{"q":f"{ticker} акции","format":"json"})
            pos=neg=0
            alerts=[]
            for a in news.get("results",[])[:10]:
                url=a.get("url","")
                if url in self.analyzed:
                    continue
                self.analyzed.add(url)
                text=(a.get("title","")+a.get("content","")).lower()
                p=sum(1 for w in self.POS if w in text)
                n=sum(1 for w in self.NEG if w in text)
                if n>p and n>=2:
                    alerts.append({"ticker":ticker,"title":a.get("title","")[:50]})
                pos+=p
                neg+=n
            return{"sentiment":(pos-neg)/max(pos+neg,1),"alerts":alerts}
        except:
            return{"sentiment":0,"alerts":[]}

class TradingEngine:
    def __init__(self):
        self.svc=ServiceClient()
        self.regime=RegimeAnalyzer(self.svc)
        self.positions=PositionManager(self.svc)
        self.news=NewsAnalyzer(self.svc)
        self.current_regime=MarketRegime.SIDEWAYS
        self.daily_trades=0
        self.enabled=os.getenv("AUTO_TRADE","true").lower()=="true"
    
    def is_open(self):
        now=datetime.now()
        return now.weekday()<5 and time(10,0)<=now.time()<=time(18,45)
    
    async def update_regime(self):
        new=await self.regime.detect()
        if new!=self.current_regime:
            log.info(f"🔄 Regime: {self.current_regime.value} → {new.value}")
            self.current_regime=new
            if new==MarketRegime.CRASH:
                log.critical("🚨 CRASH!")
                await self.svc.call("killswitch","/activate","POST")
    
    async def run_cycle(self):
        if not self.enabled or not self.is_open():
            return
        log.info("="*60)
        log.info(f"🔄 CYCLE {datetime.now().strftime('%H:%M:%S')}")
        await self.update_regime()
        if self.current_regime==MarketRegime.CRASH:
            return
        ks=await self.svc.call("killswitch","/status")
        if ks.get("active"):
            log.warning("⛔ Kill switch")
            return
        await self.positions.sync_from_executor()
        await self.positions.update_prices()
        for t in list(self.positions.positions.keys()):
            for a in await self.positions.manage(t):
                await self.positions.execute(a)
        log.info(f"📂 Positions: {len(self.positions.positions)}")
        signals=await self.svc.call("strategy","/scan")
        if "error" in signals:
            return
        sig_list = signals if isinstance(signals, list) else signals.get("signals", [])
        good=sorted([s for s in sig_list if s.get("confidence",0)>=CFG.ml_weak],key=lambda x:x["confidence"],reverse=True)
        log.info(f"📡 Signals: {len(good)}")
        regime_adj=REGIME_ADJ.get(self.current_regime,{})
        max_pos=regime_adj.get("max_pos",15)
        executed=0
        for sig in good[:10]:
            if len(self.positions.positions)>=max_pos or self.daily_trades>=CFG.max_daily_trades:
                break
            if sig["ticker"] in self.positions.positions:
                continue
            news=await self.news.check(sig["ticker"])
            if news["sentiment"]<-0.5:
                log.info(f"⚠️ {sig['ticker']} skip news")
                continue
            portfolio=await self.svc.call("orchestrator","/portfolio")
            pv=portfolio.get("total_value",100000)
            entry=sig.get("price",100)
            stop_pct=CFG.default_stop*regime_adj.get("stop",1.0)
            stop_price=entry*(1-stop_pct) if sig["direction"]=="BUY" else entry*(1+stop_pct)
            risk=abs(entry-stop_price)
            pos_val=PositionSizer.calculate(pv,sig["confidence"],stop_pct,self.current_regime,len(self.positions.positions),sig.get("correlation",0.3))
            qty=int(pos_val/entry) if entry>0 else 0
            if qty<=0:
                continue
            risk_ok=await self.svc.call("risk","/check","POST",{"ticker":sig["ticker"],"side":sig["direction"].lower(),"quantity":qty,"price":entry})
            if not risk_ok.get("approved",False):
                continue
            tps=[entry+risk*r*(1 if sig["direction"]=="BUY" else -1) for r in[CFG.tp_r1,CFG.tp_r2,CFG.tp_r3]]
            result=await self.svc.call("executor","/order","POST",{"ticker":sig["ticker"],"side":sig["direction"].lower(),"quantity":qty,"order_type":"LIMIT","price":entry,"stop_loss":stop_price,"take_profit":tps[0]})
            if result.get("status")=="executed":
                self.positions.positions[sig["ticker"]]=Position(ticker=sig["ticker"],direction=sig["direction"],qty=qty,entry=entry,current=entry,stop=stop_price,tps=tps,opened=datetime.now(),confidence=sig["confidence"],high=entry,low=entry)
                self.daily_trades+=1
                executed+=1
                log.info(f"{'🟢' if sig['direction']=='BUY' else '🔴'} {sig['direction']} {qty} {sig['ticker']} @ {entry:.2f}")
                log.info(f"   Stop: {stop_price:.2f} | TP: {tps[0]:.2f}, {tps[1]:.2f}, {tps[2]:.2f}")
        log.info(f"✅ Executed: {executed} | Today: {self.daily_trades}")
    
    async def check_risks(self):
        st=await self.svc.call("risk","/status")
        dd=st.get("current_drawdown",0)
        if dd>0.15:
            log.critical(f"🚨 DD {dd:.1%}")
            await self.svc.call("killswitch","/activate","POST")
        elif dd>0.10:
            log.warning(f"⚠️ DD: {dd:.1%}")
    
    async def check_news(self):
        for t in self.positions.positions:
            n=await self.news.check(t)
            for a in n.get("alerts",[]):
                log.warning(f"📰 {a['ticker']}: {a['title']}")
    
    def reset(self):
        self.daily_trades=0
        self.news.analyzed.clear()
        log.info("🔄 Reset")

engine=TradingEngine()

@app.on_event("startup")
async def startup():
    scheduler.add_job(engine.run_cycle,CronTrigger(day_of_week="mon-fri",hour="10-18",minute="*/5"),id="trading")
    scheduler.add_job(engine.check_risks,CronTrigger(minute="*"),id="risk")
    scheduler.add_job(engine.check_news,CronTrigger(day_of_week="mon-fri",hour="9-19",minute="*/15"),id="news")
    scheduler.add_job(engine.reset,CronTrigger(hour=0,minute=0),id="reset")
    scheduler.start()
    log.info("🚀 Engine v2.0 started")

@app.get("/health")
async def health():
    return{"status":"healthy","version":"2.0"}

@app.get("/status")
async def status():
    return{"enabled":engine.enabled,"regime":engine.current_regime.value,"positions":len(engine.positions.positions),"daily_trades":engine.daily_trades,"jobs":[{"id":j.id,"next":str(j.next_run_time)} for j in scheduler.get_jobs()]}

@app.get("/positions")
async def positions():
    return{t:{"pnl":f"{p.pnl_pct:.2%}","r":f"{p.r_multiple:.1f}","trailing":p.trailing,"breakeven":p.breakeven} for t,p in engine.positions.positions.items()}

@app.post("/toggle")
async def toggle(enabled:bool=True):
    engine.enabled=enabled
    return{"enabled":engine.enabled}

@app.post("/force_cycle")
async def force_cycle(bg:BackgroundTasks):
    bg.add_task(engine.run_cycle)
    return{"status":"started"}

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8030)
