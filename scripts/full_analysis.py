#!/usr/bin/env python3
"""
Trading Autopilot - Ultimate System Analysis v4.0
Полный углублённый анализ всей торговой системы с ML и предиктивной аналитикой
"""

import asyncio
import aiohttp
import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple
import statistics
import math

# === CONFIGURATION ===
SERVICES = {
    "dashboard":   ("http://localhost:8080",  "🖥️",  "UI"),
    "strategy":    ("http://localhost:8005",  "🧠",  "ML/Signals"),
    "executor":    ("http://localhost:8007",  "⚡",  "Trading"),
    "datafeed":    ("http://localhost:8006",  "📡",  "Market Data"),
    "automation":  ("http://localhost:8030",  "🤖",  "Auto Trading"),
    "risk":        ("http://localhost:8001",  "🛡️",  "Risk Mgmt"),
    "scheduler":   ("http://localhost:8009",  "⏰",  "Jobs"),
    "orchestrator":("http://localhost:8000",  "🎯",  "Coordinator"),
}

# === TERMINAL GRAPHICS ===
class Term:
    """Advanced terminal graphics"""
    # Colors
    BLACK = '\033[30m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    RESET = '\033[0m'
    
    # Background
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    
    @staticmethod
    def rgb(r, g, b): 
        return f'\033[38;2;{r};{g};{b}m'
    
    @staticmethod
    def bg_rgb(r, g, b): 
        return f'\033[48;2;{r};{g};{b}m'
    
    @staticmethod
    def gradient(text: str, start: Tuple[int,int,int], end: Tuple[int,int,int]) -> str:
        """Apply gradient color to text"""
        result = ""
        n = len(text)
        for i, char in enumerate(text):
            r = int(start[0] + (end[0] - start[0]) * i / max(n-1, 1))
            g = int(start[1] + (end[1] - start[1]) * i / max(n-1, 1))
            b = int(start[2] + (end[2] - start[2]) * i / max(n-1, 1))
            result += f'\033[38;2;{r};{g};{b}m{char}'
        return result + Term.RESET

class Box:
    """Unicode box drawing"""
    TL = '╭'; TR = '╮'; BL = '╰'; BR = '╯'
    H = '─'; V = '│'
    LT = '├'; RT = '┤'; TT = '┬'; BT = '┴'; X = '┼'
    
    # Double
    D_TL = '╔'; D_TR = '╗'; D_BL = '╚'; D_BR = '╝'
    D_H = '═'; D_V = '║'
    
    @staticmethod
    def wrap(content: List[str], width: int = 60, title: str = None, color: str = Term.CYAN) -> str:
        """Wrap content in a box"""
        lines = []
        
        # Top
        if title:
            title_part = f" {title} "
            padding = width - len(title_part) - 2
            lines.append(f"{color}{Box.TL}{Box.H}{title_part}{Box.H * padding}{Box.TR}{Term.RESET}")
        else:
            lines.append(f"{color}{Box.TL}{Box.H * (width-2)}{Box.TR}{Term.RESET}")
        
        # Content
        for line in content:
            # Strip ANSI for length calculation
            clean_len = len(line.encode().decode('unicode_escape').replace('\033[', '').split('m')[-1]) if '\033[' in line else len(line)
            visible_len = len(line) - (len(line) - clean_len)
            padding = width - 4 - min(len(line.replace('\033[', '').split('m')[-1]), 50)
            lines.append(f"{color}{Box.V}{Term.RESET} {line:<{width-4}} {color}{Box.V}{Term.RESET}")
        
        # Bottom
        lines.append(f"{color}{Box.BL}{Box.H * (width-2)}{Box.BR}{Term.RESET}")
        
        return '\n'.join(lines)

class Charts:
    """ASCII chart generators"""
    
    BLOCKS = ' ▁▂▃▄▅▆▇█'
    BARS_H = '▏▎▍▌▋▊▉█'
    DOTS = '⠀⠁⠂⠃⠄⠅⠆⠇⡀⡁⡂⡃⡄⡅⡆⡇⠈⠉⠊⠋⠌⠍⠎⠏⡈⡉⡊⡋⡌⡍⡎⡏⠐⠑⠒⠓⠔⠕⠖⠗⡐⡑⡒⡓⡔⡕⡖⡗⠘⠙⠚⠛⠜⠝⠞⠟⡘⡙⡚⡛⡜⡝⡞⡟⠠⠡⠢⠣⠤⠥⠦⠧⡠⡡⡢⡣⡤⡥⡦⡧⠨⠩⠪⠫⠬⠭⠮⠯⡨⡩⡪⡫⡬⡭⡮⡯⠰⠱⠲⠳⠴⠵⠶⠷⡰⡱⡲⡳⡴⡵⡶⡷⠸⠹⠺⠻⠼⠽⠾⠿⡸⡹⡺⡻⡼⡽⡾⡿⢀⢁⢂⢃⢄⢅⢆⢇⣀⣁⣂⣃⣄⣅⣆⣇⢈⢉⢊⢋⢌⢍⢎⢏⣈⣉⣊⣋⣌⣍⣎⣏⢐⢑⢒⢓⢔⢕⢖⢗⣐⣑⣒⣓⣔⣕⣖⣗⢘⢙⢚⢛⢜⢝⢞⢟⣘⣙⣚⣛⣜⣝⣞⣟⢠⢡⢢⢣⢤⢥⢦⢧⣠⣡⣢⣣⣤⣥⣦⣧⢨⢩⢪⢫⢬⢭⢮⢯⣨⣩⣪⣫⣬⣭⣮⣯⢰⢱⢲⢳⢴⢵⢶⢷⣰⣱⣲⣳⣴⣵⣶⣷⢸⢹⢺⢻⢼⢽⢾⢿⣸⣹⣺⣻⣼⣽⣾⣿'
    
    @staticmethod
    def sparkline(data: List[float], width: int = 30, color_pos: str = Term.GREEN, color_neg: str = Term.RED) -> str:
        """Generate sparkline with colors"""
        if not data or len(data) < 2:
            return Term.GRAY + '─' * width + Term.RESET
        
        # Resample
        step = len(data) / width
        resampled = [data[min(int(i * step), len(data)-1)] for i in range(width)]
        
        mn, mx = min(resampled), max(resampled)
        rng = mx - mn if mx != mn else 1
        
        result = ""
        blocks = Charts.BLOCKS
        prev = resampled[0]
        
        for v in resampled:
            idx = int((v - mn) / rng * (len(blocks) - 1))
            color = color_pos if v >= prev else color_neg
            result += color + blocks[idx]
            prev = v
        
        return result + Term.RESET
    
    @staticmethod
    def bar_chart(data: Dict[str, float], width: int = 40, color: str = Term.CYAN) -> List[str]:
        """Horizontal bar chart"""
        if not data:
            return []
        
        max_val = max(abs(v) for v in data.values()) or 1
        max_label = max(len(k) for k in data.keys())
        
        lines = []
        for label, value in data.items():
            bar_width = int(abs(value) / max_val * (width - max_label - 15))
            bar = '█' * bar_width + '▌' * (1 if bar_width > 0 else 0)
            
            c = Term.GREEN if value >= 0 else Term.RED
            lines.append(f"  {label:<{max_label}} {c}{bar:<{width-max_label-10}}{Term.RESET} {value:>8.1f}")
        
        return lines
    
    @staticmethod
    def heatmap(data: List[List[float]], labels_x: List[str] = None, labels_y: List[str] = None) -> List[str]:
        """Generate heatmap"""
        if not data:
            return []
        
        mn = min(min(row) for row in data)
        mx = max(max(row) for row in data)
        rng = mx - mn if mx != mn else 1
        
        # Color gradient from blue to red
        def get_color(v):
            norm = (v - mn) / rng
            if norm < 0.5:
                r, g, b = 50, 50, int(150 + norm * 200)
            else:
                r, g, b = int((norm - 0.5) * 400), 50, int(250 - (norm - 0.5) * 400)
            return Term.bg_rgb(min(r, 255), min(g, 255), min(b, 255))
        
        lines = []
        
        # Header
        if labels_x:
            header = "     " + "".join(f"{l[:3]:^5}" for l in labels_x)
            lines.append(Term.GRAY + header + Term.RESET)
        
        for i, row in enumerate(data):
            label = labels_y[i][:4] if labels_y and i < len(labels_y) else f"{i:2d}"
            line = f"{Term.GRAY}{label:>4}{Term.RESET} "
            
            for v in row:
                color = get_color(v)
                line += f"{color} {v:>3.0f} {Term.RESET}"
            
            lines.append(line)
        
        return lines
    
    @staticmethod
    def gauge(value: float, max_val: float = 100, width: int = 30, 
              thresholds: Tuple[float, float] = (30, 70)) -> str:
        """Semicircle gauge representation"""
        pct = min(value / max_val, 1.0) if max_val > 0 else 0
        
        # Color based on thresholds
        if value < thresholds[0]:
            color = Term.GREEN
        elif value < thresholds[1]:
            color = Term.YELLOW
        else:
            color = Term.RED
        
        filled = int(pct * width)
        return f"{color}{'▓' * filled}{Term.GRAY}{'░' * (width - filled)}{Term.RESET} {value:5.1f}%"
    
    @staticmethod
    def mini_chart(data: List[float], width: int = 10, height: int = 3) -> List[str]:
        """Multi-line mini chart"""
        if not data or len(data) < 2:
            return ['─' * width] * height
        
        # Resample
        step = len(data) / width
        resampled = [data[min(int(i * step), len(data)-1)] for i in range(width)]
        
        mn, mx = min(resampled), max(resampled)
        rng = mx - mn if mx != mn else 1
        
        lines = []
        for row in range(height):
            threshold_low = 1 - (row + 1) / height
            threshold_high = 1 - row / height
            
            line = ""
            for v in resampled:
                norm = (v - mn) / rng
                if norm >= threshold_high:
                    line += '█'
                elif norm >= threshold_low:
                    line += '▄'
                else:
                    line += ' '
            lines.append(line)
        
        return lines

class Formatters:
    """Value formatters"""
    
    @staticmethod
    def money(v: float, symbol: str = "₽", colored: bool = False) -> str:
        if v is None:
            return f"{symbol}--"
        
        if abs(v) >= 1_000_000_000:
            formatted = f"{symbol}{v/1_000_000_000:.2f}B"
        elif abs(v) >= 1_000_000:
            formatted = f"{symbol}{v/1_000_000:.2f}M"
        elif abs(v) >= 1_000:
            formatted = f"{symbol}{v/1_000:.1f}K"
        else:
            formatted = f"{symbol}{v:,.0f}"
        
        if colored and v != 0:
            color = Term.GREEN if v > 0 else Term.RED
            return f"{color}{formatted}{Term.RESET}"
        return formatted
    
    @staticmethod
    def pct(v: float, colored: bool = True) -> str:
        if v is None:
            return "--%"
        sign = "+" if v > 0 else ""
        formatted = f"{sign}{v:.1f}%"
        if colored and v != 0:
            color = Term.GREEN if v > 0 else Term.RED
            return f"{color}{formatted}{Term.RESET}"
        return formatted
    
    @staticmethod
    def duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"
    
    @staticmethod
    def number(v: float, precision: int = 0) -> str:
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        elif abs(v) >= 1_000:
            return f"{v/1_000:.1f}K"
        else:
            return f"{v:,.{precision}f}"
    
    @staticmethod
    def trend_arrow(current: float, previous: float) -> str:
        if previous == 0:
            return "→"
        change = (current - previous) / abs(previous) * 100
        if change > 5:
            return f"{Term.GREEN}▲{Term.RESET}"
        elif change > 1:
            return f"{Term.GREEN}↗{Term.RESET}"
        elif change < -5:
            return f"{Term.RED}▼{Term.RESET}"
        elif change < -1:
            return f"{Term.RED}↘{Term.RESET}"
        else:
            return f"{Term.YELLOW}→{Term.RESET}"

# === OUTPUT HELPERS ===
def print_header(title: str, icon: str = ""):
    gradient_title = Term.gradient(f" {title} ", (0, 200, 255), (150, 50, 255))
    width = 66
    print(f"\n{Term.BOLD}{Term.CYAN}{Box.D_TL}{Box.D_H * width}{Box.D_TR}")
    print(f"{Box.D_V}{gradient_title:^{width + 20}}{Term.CYAN}{Box.D_V}")
    print(f"{Box.D_BL}{Box.D_H * width}{Box.D_BR}{Term.RESET}\n")

def print_section(title: str):
    print(f"\n  {Term.BOLD}{Term.CYAN}▸ {title}{Term.RESET}")
    print(f"  {Term.GRAY}{'─' * 50}{Term.RESET}")

def ok(msg: str):     print(f"  {Term.GREEN}✅ {msg}{Term.RESET}")
def fail(msg: str):   print(f"  {Term.RED}❌ {msg}{Term.RESET}")
def warn(msg: str):   print(f"  {Term.YELLOW}⚠️  {msg}{Term.RESET}")
def info(msg: str):   print(f"  {Term.CYAN}ℹ️  {msg}{Term.RESET}")
def debug(msg: str):  print(f"  {Term.GRAY}   {msg}{Term.RESET}")

# === MAIN ANALYZER ===
class SystemAnalyzer:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool = None
        self.results: Dict[str, Any] = defaultdict(dict)
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        self.recommendations: List[Dict] = []
        self.metrics: Dict[str, Any] = {}
        self.history: Dict[str, List] = defaultdict(list)
        self.start_time = datetime.now()
        
    async def start(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            connector=aiohttp.TCPConnector(limit=30)
        )
        
        try:
            import asyncpg
            self.db_pool = await asyncpg.create_pool(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", 5432)),
                user=os.getenv("DB_USER", "trading"),
                password=os.getenv("DB_PASSWORD", "trading123"),
                database=os.getenv("DB_NAME", "trading"),
                min_size=1, max_size=5,
                command_timeout=15
            )
        except:
            pass
            
    async def stop(self):
        if self.session: await self.session.close()
        if self.db_pool: await self.db_pool.close()
        
    async def fetch(self, url: str) -> Optional[dict]:
        try:
            async with self.session.get(url) as r:
                return await r.json() if r.status == 200 else {"_error": r.status}
        except asyncio.TimeoutError:
            return {"_error": "timeout"}
        except Exception as e:
            return {"_error": str(e)[:50]}

    async def fetch_all(self, urls: Dict[str, str]) -> Dict[str, Any]:
        tasks = {k: self.fetch(v) for k, v in urls.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(tasks.keys(), results))

    def add_issue(self, msg: str, severity: str = "high", component: str = "system"):
        self.issues.append({"msg": msg, "severity": severity, "component": component})
    
    def add_warning(self, msg: str, component: str = "system"):
        self.warnings.append({"msg": msg, "component": component})
    
    def add_recommendation(self, msg: str, priority: str = "medium"):
        self.recommendations.append({"msg": msg, "priority": priority})

    # =========================================================================
    # 1. SERVICES HEALTH
    # =========================================================================
    async def analyze_services(self):
        print_header("SERVICES HEALTH", "🔌")
        
        urls = {name: f"{cfg[0]}/health" for name, cfg in SERVICES.items()}
        results = await self.fetch_all(urls)
        
        online, degraded, offline = 0, 0, 0
        
        print(f"  {'Service':<14} {'Status':<12} {'Latency':<10} {'Details':<30}")
        print(f"  {Term.GRAY}{'─' * 68}{Term.RESET}")
        
        for name, (url, icon, desc) in SERVICES.items():
            start = datetime.now()
            health = results.get(name, {})
            latency = (datetime.now() - start).total_seconds() * 1000
            
            if isinstance(health, Exception) or health.get("_error"):
                offline += 1
                err = str(health.get("_error", health))[:25]
                print(f"  {icon} {name:<12} {Term.RED}{'OFFLINE':<12}{Term.RESET} {'-':>8}   {Term.GRAY}{err}{Term.RESET}")
                self.add_issue(f"{name} is offline", "critical", name)
            else:
                status = health.get("status", health.get("ok", "?"))
                
                # Extract details
                details = []
                for key in ["mode", "version", "enabled", "ml_available"]:
                    if key in health:
                        v = health[key]
                        if isinstance(v, bool):
                            v = "✓" if v else "✗"
                        details.append(f"{key}={v}")
                
                if status in ["healthy", "ok", 1, True]:
                    online += 1
                    status_str = f"{Term.GREEN}{'HEALTHY':<12}{Term.RESET}"
                else:
                    degraded += 1
                    status_str = f"{Term.YELLOW}{'DEGRADED':<12}{Term.RESET}"
                
                latency_color = Term.GREEN if latency < 100 else Term.YELLOW if latency < 500 else Term.RED
                detail_str = ", ".join(details)[:30] if details else ""
                
                print(f"  {icon} {name:<12} {status_str} {latency_color}{latency:>6.0f}ms{Term.RESET}   {Term.GRAY}{detail_str}{Term.RESET}")
                self.results["services"][name] = health
        
        # Summary bar
        total = len(SERVICES)
        print(f"\n  {Term.GREEN}{'█' * online}{Term.YELLOW}{'█' * degraded}{Term.RED}{'█' * offline}{Term.GRAY}{'░' * (20 - online - degraded - offline)}{Term.RESET} {online}/{total} online")
        
        self.metrics["services"] = {"online": online, "degraded": degraded, "offline": offline, "total": total}

    # =========================================================================
    # 2. PORTFOLIO DEEP ANALYSIS
    # =========================================================================
    async def analyze_portfolio(self):
        print_header("PORTFOLIO ANALYSIS", "💰")
        
        portfolio = await self.fetch(f"{SERVICES['executor'][0]}/portfolio")
        
        if not portfolio or portfolio.get("_error"):
            warn("Portfolio data not available")
            return
        
        def parse_money(v):
            if isinstance(v, dict):
                return int(v.get('units', 0)) + int(v.get('nano', 0)) / 1e9
            return float(v) if v else 0
        
        # Parse portfolio
        if 'totalAmountPortfolio' in portfolio:
            total = parse_money(portfolio['totalAmountPortfolio'])
            cash = parse_money(portfolio.get('totalAmountCurrencies', 0))
            pnl = parse_money(portfolio.get('expectedYield', 0))
            mode = "LIVE API"
        else:
            total = portfolio.get('total_value', 0)
            cash = portfolio.get('cash', 0)
            pnl = portfolio.get('unrealized_pnl', 0)
            mode = portfolio.get('mode', 'SIMULATION').upper()
        
        invested = total - cash
        cash_pct = (cash / total * 100) if total > 0 else 0
        
        self.metrics["portfolio"] = {
            "total": total, "cash": cash, "pnl": pnl, "invested": invested
        }
        
        # Main metrics box
        print_section("Overview")
        
        pnl_formatted = Formatters.money(pnl, colored=True)
        pnl_pct = (pnl / (total - pnl) * 100) if total != pnl else 0
        
        print(f"""
    ╭────────────────────────────────────────────────────────────╮
    │  📊 Mode          │  {mode:<40} │
    │  💰 Total Value   │  {Formatters.money(total):<40} │
    │  💵 Cash          │  {Formatters.money(cash)} ({cash_pct:.1f}%){' ' * 28}│
    │  📈 Invested      │  {Formatters.money(invested):<40} │
    │  💹 Unrealized    │  {pnl_formatted} ({Formatters.pct(pnl_pct)}){' ' * 23}│
    ╰────────────────────────────────────────────────────────────╯""")
        
        # Allocation chart
        if invested > 0:
            print_section("Allocation")
            alloc_data = {"Cash": cash_pct, "Invested": 100 - cash_pct}
            
            cash_bar = int(cash_pct / 5)
            inv_bar = 20 - cash_bar
            print(f"    {Term.CYAN}{'█' * cash_bar}{Term.RESET}{Term.MAGENTA}{'█' * inv_bar}{Term.RESET}")
            print(f"    {Term.CYAN}■ Cash {cash_pct:.0f}%{Term.RESET}  {Term.MAGENTA}■ Invested {100-cash_pct:.0f}%{Term.RESET}")
        
        # Positions
        positions = portfolio.get('positions', [])
        real_pos = [p for p in positions 
                    if p.get('ticker') not in ('RUB000UTSTOM', '', None)
                    and parse_money(p.get('quantity', p.get('balance', 0))) != 0]
        
        self.metrics["positions"] = len(real_pos)
        
        if real_pos:
            print_section(f"Positions ({len(real_pos)})")
            
            # Sort by absolute PnL
            sorted_pos = sorted(real_pos, 
                key=lambda x: abs(parse_money(x.get('expectedYield', 0))), 
                reverse=True)[:12]
            
            print(f"    {'Ticker':<8} {'Qty':>8} {'Value':>12} {'P&L':>12} {'%':>8} {'Chart'}")
            print(f"    {Term.GRAY}{'─' * 60}{Term.RESET}")
            
            for p in sorted_pos:
                tk = p.get('ticker', '?')[:8]
                qty = parse_money(p.get('quantity', p.get('balance', 0)))
                avg = parse_money(p.get('averagePositionPrice', 0))
                cur = parse_money(p.get('currentPrice', avg))
                pos_pnl = parse_money(p.get('expectedYield', 0)) or (cur - avg) * qty
                value = qty * cur
                pnl_pct = ((cur / avg) - 1) * 100 if avg > 0 else 0
                
                # Mini chart placeholder (would need historical data)
                mini = Charts.gauge(50 + pnl_pct, 100, 8, (40, 60))
                
                pnl_str = Formatters.money(pos_pnl, colored=True)
                pct_str = Formatters.pct(pnl_pct)
                
                print(f"    {tk:<8} {qty:>8.0f} {Formatters.money(value):>12} {pnl_str:>20} {pct_str:>12}")
            
            if len(real_pos) > 12:
                print(f"    {Term.GRAY}... and {len(real_pos) - 12} more positions{Term.RESET}")
            
            # Position diversification warning
            if len(real_pos) < 3 and invested > 100000:
                self.add_warning("Low diversification - only {len(real_pos)} positions")
        else:
            info("No open positions")

    # =========================================================================
    # 3. SIGNALS DEEP ANALYSIS
    # =========================================================================
    async def analyze_signals(self):
        print_header("SIGNALS ANALYSIS", "🎯")
        
        signals_raw = await self.fetch(f"{SERVICES['strategy'][0]}/scan")
        
        if not signals_raw or signals_raw.get("_error"):
            warn("Signals not available")
            return
        
        signals = signals_raw if isinstance(signals_raw, list) else signals_raw.get('signals', [])
        
        def get_signal_type(s):
            sig = s.get('signal')
            if sig in [1, 'buy', 'BUY']: return 'buy'
            if sig in [-1, 'sell', 'SELL']: return 'sell'
            return 'hold'
        
        buys = [s for s in signals if get_signal_type(s) == 'buy']
        sells = [s for s in signals if get_signal_type(s) == 'sell']
        holds = [s for s in signals if get_signal_type(s) == 'hold']
        
        self.metrics["signals"] = {"buy": len(buys), "sell": len(sells), "hold": len(holds)}
        
        print_section("Signal Distribution")
        
        total = len(signals)
        buy_pct = len(buys) / total * 100 if total > 0 else 0
        sell_pct = len(sells) / total * 100 if total > 0 else 0
        hold_pct = len(holds) / total * 100 if total > 0 else 0
        
        # Visual distribution
        buy_bar = int(buy_pct / 5)
        sell_bar = int(sell_pct / 5)
        hold_bar = 20 - buy_bar - sell_bar
        
        print(f"""
    Total Scanned: {total}
    
    {Term.GREEN}█{'█' * buy_bar}{Term.RESET}{Term.RED}{'█' * sell_bar}{Term.RESET}{Term.GRAY}{'█' * hold_bar}{Term.RESET}
    {Term.GREEN}■ BUY  {len(buys):>3} ({buy_pct:>4.1f}%){Term.RESET}
    {Term.RED}■ SELL {len(sells):>3} ({sell_pct:>4.1f}%){Term.RESET}
    {Term.GRAY}■ HOLD {len(holds):>3} ({hold_pct:>4.1f}%){Term.RESET}""")
        
        # Confidence analysis
        all_confs = [float(s.get('confidence', 0)) for s in signals if get_signal_type(s) != 'hold']
        
        if all_confs:
            print_section("Confidence Analysis")
            
            avg_conf = statistics.mean(all_confs)
            med_conf = statistics.median(all_confs)
            std_conf = statistics.stdev(all_confs) if len(all_confs) > 1 else 0
            
            # Histogram
            buckets = [0] * 10
            for c in all_confs:
                buckets[min(int(c * 10), 9)] += 1
            
            max_b = max(buckets) or 1
            
            print(f"    {'Range':<12} {'Count':>6} {'Distribution'}")
            print(f"    {Term.GRAY}{'─' * 45}{Term.RESET}")
            
            for i, count in enumerate(buckets):
                pct_range = f"{i*10:>2}-{(i+1)*10:<2}%"
                bar_len = int(count / max_b * 25)
                
                # Color by quality
                color = Term.RED if i < 3 else Term.YELLOW if i < 5 else Term.GREEN if i < 8 else Term.CYAN
                bar = color + '▓' * bar_len + Term.RESET
                
                print(f"    {pct_range:<12} {count:>6} {bar}")
            
            print(f"\n    📊 Mean: {avg_conf:.0%}  Median: {med_conf:.0%}  StdDev: {std_conf:.2f}")
            
            if avg_conf < 0.45:
                self.add_warning("Low average signal confidence", "strategy")
                self.add_recommendation("Consider retraining ML model or adjusting thresholds")
        
        # Top signals
        def show_signals(sigs: List[dict], title: str, color: str, limit: int = 5):
            if not sigs:
                return
            print_section(title)
            
            sorted_sigs = sorted(sigs, key=lambda x: -float(x.get('confidence', 0)))[:limit]
            
            for s in sorted_sigs:
                conf = float(s.get('confidence', 0))
                price = float(s.get('price', 0))
                
                # Confidence bar
                conf_bar = f"{color}{'█' * int(conf * 15)}{Term.GRAY}{'░' * (15 - int(conf * 15))}{Term.RESET}"
                
                # Additional indicators
                extras = []
                if 'rsi' in s: extras.append(f"RSI:{s['rsi']:.0f}")
                if 'volume_ratio' in s: extras.append(f"Vol:{s['volume_ratio']:.1f}x")
                if 'trend' in s: extras.append(f"T:{s['trend']}")
                
                extra_str = f" {Term.GRAY}[{', '.join(extras)}]{Term.RESET}" if extras else ""
                
                print(f"    {s['ticker']:<8} {conf_bar} {conf:>5.0%}  {Term.GRAY}₽{price:>10.2f}{Term.RESET}{extra_str}")
        
        show_signals(buys, "🟢 Top BUY Signals", Term.GREEN)
        show_signals(sells, "🔴 Top SELL Signals", Term.RED)

    # =========================================================================
    # 4. RISK ANALYSIS
    # =========================================================================
    async def analyze_risk(self):
        print_header("RISK ANALYSIS", "🛡️")
        
        risk = await self.fetch(f"{SERVICES['risk'][0]}/health")
        
        if not risk or risk.get("_error"):
            warn("Risk manager not available")
            self.add_issue("Risk manager offline", "critical", "risk")
            return
        
        dd = float(risk.get('drawdown_pct', risk.get('drawdown', 0)))
        blocked = risk.get('trading_blocked', False)
        balance = float(risk.get('balance', 0))
        peak = float(risk.get('peak_balance', balance))
        daily_loss = float(risk.get('daily_loss', 0))
        max_dd = float(risk.get('max_drawdown_limit', 10))
        daily_limit = float(risk.get('daily_loss_limit', 50000))
        
        self.metrics["risk"] = {
            "drawdown": dd, "blocked": blocked, "daily_loss": daily_loss
        }
        
        print_section("Risk Metrics")
        
        # Drawdown gauge
        dd_severity = "LOW" if dd < 2 else "MEDIUM" if dd < 5 else "HIGH" if dd < 8 else "CRITICAL"
        dd_color = Term.GREEN if dd < 2 else Term.YELLOW if dd < 5 else Term.rgb(255, 165, 0) if dd < 8 else Term.RED
        
        dd_bar = Charts.gauge(dd, max_dd, 30, (max_dd * 0.3, max_dd * 0.7))
        daily_bar = Charts.gauge(abs(daily_loss), daily_limit, 30, (daily_limit * 0.3, daily_limit * 0.7))
        
        print(f"""
    ╭─────────────────────────────────────────────────────────────────╮
    │  📉 Drawdown        {dd_bar}   │
    │     Level: {dd_color}{dd_severity:<10}{Term.RESET}   Current: {dd:.2f}%   Max: {max_dd}%              │
    ├─────────────────────────────────────────────────────────────────┤
    │  📅 Daily Loss      {daily_bar}   │
    │     Used: {Formatters.money(abs(daily_loss))}   Limit: {Formatters.money(daily_limit)}                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  🏔️  Peak Balance    {Formatters.money(peak):<45} │
    │  💰 Current         {Formatters.money(balance):<45} │
    ╰─────────────────────────────────────────────────────────────────╯""")
        
        # Status
        print_section("Trading Status")
        
        if blocked:
            fail("🚨 TRADING IS BLOCKED!")
            print(f"    {Term.RED}{Term.BOLD}Risk limits exceeded - manual intervention required{Term.RESET}")
            self.add_issue("Trading blocked by risk manager", "critical", "risk")
        elif dd > max_dd * 0.8:
            warn(f"Drawdown {dd:.1f}% approaching limit ({max_dd}%)")
            self.add_warning(f"High drawdown: {dd:.1f}%", "risk")
        elif dd > max_dd * 0.5:
            warn(f"Drawdown {dd:.1f}% - monitor closely")
        else:
            ok(f"All risk metrics within limits")

    # =========================================================================
    # 5. TRADE HISTORY & PERFORMANCE
    # =========================================================================
    async def analyze_trades(self):
        print_header("TRADING PERFORMANCE", "📈")
        
        trades_raw = await self.fetch(f"{SERVICES['executor'][0]}/trades?limit=500")
        
        if not trades_raw or trades_raw.get("_error"):
            info("No trades history available")
            return
        
        trades = trades_raw if isinstance(trades_raw, list) else trades_raw.get('trades', [])
        
        if not trades:
            info("No trades executed yet")
            print(f"\n    {Term.CYAN}System is ready for trading. Enable automation to start.{Term.RESET}")
            return
        
        # Calculate metrics
        pnls = [float(t.get('pnl', 0) or 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        
        win_rate = len(wins) / len([p for p in pnls if p != 0]) if any(p != 0 for p in pnls) else 0
        avg_win = statistics.mean(wins) if wins else 0
        avg_loss = statistics.mean(losses) if losses else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if pnls else 0
        
        # Max drawdown from trades
        cumulative = []
        running = 0
        peak_cum = 0
        max_dd_trades = 0
        for p in pnls:
            running += p
            cumulative.append(running)
            if running > peak_cum:
                peak_cum = running
            dd = (peak_cum - running) / peak_cum * 100 if peak_cum > 0 else 0
            max_dd_trades = max(max_dd_trades, dd)
        
        self.metrics["trades"] = {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "expectancy": expectancy
        }
        
        print_section("Performance Metrics")
        
        pf_color = Term.GREEN if profit_factor > 1.5 else Term.YELLOW if profit_factor > 1 else Term.RED
        wr_color = Term.GREEN if win_rate > 0.55 else Term.YELLOW if win_rate > 0.45 else Term.RED
        
        print(f"""
    ╭───────────────────────────────────────────────────────────────╮
    │                      TRADING STATISTICS                       │
    ├───────────────────────────────────────────────────────────────┤
    │  Total Trades     │ {len(trades):>10}      │  Win Rate    │ {wr_color}{win_rate:>7.1%}{Term.RESET}  │
    │  Winning Trades   │ {Term.GREEN}{len(wins):>10}{Term.RESET}      │  Profit Factor │ {pf_color}{profit_factor:>6.2f}{Term.RESET}  │
    │  Losing Trades    │ {Term.RED}{len(losses):>10}{Term.RESET}      │  Expectancy  │ {Formatters.money(expectancy):>8}│
    ├───────────────────────────────────────────────────────────────┤
    │  Gross Profit     │ {Term.GREEN}{Formatters.money(gross_profit):>10}{Term.RESET}      │  Avg Win     │ {Term.GREEN}{Formatters.money(avg_win):>8}{Term.RESET}│
    │  Gross Loss       │ {Term.RED}{Formatters.money(gross_loss):>10}{Term.RESET}      │  Avg Loss    │ {Term.RED}{Formatters.money(avg_loss):>8}{Term.RESET}│
    │  Net P&L          │ {Formatters.money(total_pnl, colored=True):>18}      │  Max DD      │ {max_dd_trades:>6.1f}% │
    ╰───────────────────────────────────────────────────────────────╯""")
        
        # Equity curve
        if len(cumulative) > 5:
            print_section("Equity Curve")
            sparkline = Charts.sparkline(cumulative, 60)
            print(f"    {sparkline}")
            print(f"    {Term.GRAY}{'─' * 60}{Term.RESET}")
            print(f"    {Term.GRAY}Start{' ' * 50}Latest{Term.RESET}")
            print(f"    {Formatters.money(0)}{' ' * 45}{Formatters.money(cumulative[-1], colored=True)}")
        
        # By ticker analysis
        by_ticker = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in trades:
            tk = t.get('ticker', 'UNK')
            by_ticker[tk]["count"] += 1
            pnl_val = float(t.get('pnl', 0) or 0)
            by_ticker[tk]["pnl"] += pnl_val
            if pnl_val > 0:
                by_ticker[tk]["wins"] += 1
        
        if by_ticker:
            print_section("Performance by Ticker")
            
            sorted_tickers = sorted(by_ticker.items(), key=lambda x: x[1]["pnl"], reverse=True)
            
            print(f"    {'Ticker':<10} {'Trades':>8} {'Win%':>8} {'P&L':>14}")
            print(f"    {Term.GRAY}{'─' * 42}{Term.RESET}")
            
            for tk, data in sorted_tickers[:10]:
                wr = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
                wr_c = Term.GREEN if wr > 55 else Term.YELLOW if wr > 45 else Term.RED
                print(f"    {tk:<10} {data['count']:>8} {wr_c}{wr:>7.0f}%{Term.RESET} {Formatters.money(data['pnl'], colored=True):>20}")
        
        # Performance warnings
        if win_rate < 0.4:
            self.add_warning(f"Low win rate: {win_rate:.0%}", "trading")
            self.add_recommendation("Review signal quality and entry criteria", "high")
        
        if profit_factor < 1:
            self.add_issue(f"Negative profit factor: {profit_factor:.2f}", "high", "trading")

    # =========================================================================
    # 6. AUTOMATION STATUS
    # =========================================================================
    async def analyze_automation(self):
        print_header("AUTOMATION STATUS", "🤖")
        
        status = await self.fetch(f"{SERVICES['automation'][0]}/status")
        
        if not status or status.get("_error"):
            warn("Automation service not available")
            return
        
        enabled = status.get('enabled', False)
        regime = status.get('regime', 'unknown')
        positions = status.get('positions', 0)
        daily_trades = status.get('daily_trades', 0)
        max_positions = status.get('max_positions', 10)
        last_cycle = status.get('last_cycle', 'N/A')
        cycle_interval = status.get('cycle_interval', 300)
        
        self.metrics["automation"] = {
            "enabled": enabled, "regime": regime, "positions": positions
        }
        
        # Status display
        status_color = Term.GREEN if enabled else Term.RED
        status_text = "🟢 ACTIVE" if enabled else "🔴 STOPPED"
        
        # Regime colors
        regime_colors = {
            "trending": (Term.GREEN, "📈"), "bullish": (Term.GREEN, "🐂"),
            "sideways": (Term.YELLOW, "➡️"), "ranging": (Term.YELLOW, "↔️"),
            "volatile": (Term.rgb(255, 165, 0), "⚡"), "bearish": (Term.RED, "🐻"),
        }
        regime_color, regime_icon = regime_colors.get(regime.lower(), (Term.WHITE, "❓"))
        
        pos_bar = Charts.gauge(positions, max_positions, 20, (max_positions * 0.5, max_positions * 0.8))
        
        print(f"""
    ╭────────────────────────────────────────────────────────────────╮
    │  Status          │  {status_color}{status_text:<42}{Term.RESET}│
    │  Market Regime   │  {regime_icon} {regime_color}{regime:<39}{Term.RESET}│
    │  Positions       │  {pos_bar} {positions}/{max_positions}       │
    │  Daily Trades    │  {daily_trades:<43}│
    │  Cycle Interval  │  {cycle_interval}s ({cycle_interval/60:.1f} min){' ' * 28}│
    │  Last Cycle      │  {str(last_cycle)[:40]:<43}│
    ╰────────────────────────────────────────────────────────────────╯""")
        
        if not enabled:
            print_section("Quick Start")
            print(f"    {Term.CYAN}Enable automation:{Term.RESET}")
            print(f"    curl -X POST 'http://localhost:8030/toggle?enabled=true'")
            print(f"\n    {Term.CYAN}Force immediate cycle:{Term.RESET}")
            print(f"    curl -X POST 'http://localhost:8030/force_cycle'")

    # =========================================================================
    # 7. ML MODEL ANALYSIS
    # =========================================================================
    async def analyze_ml(self):
        print_header("ML MODEL ANALYSIS", "🧠")
        
        # Get ML info from strategy
        strat_health = self.results["services"].get("strategy", {})
        
        if not strat_health:
            warn("Strategy service data not available")
            return
        
        ml_available = strat_health.get("ml_available", False)
        ml_info = strat_health.get("ml_info", {})
        
        if not ml_available:
            warn("ML model not loaded")
            self.add_recommendation("Train ML model for better signal quality", "high")
            return
        
        accuracy = ml_info.get("accuracy", 0)
        features = ml_info.get("features", 0)
        samples = ml_info.get("training_samples", 0)
        last_trained = ml_info.get("last_trained", "unknown")
        
        print_section("Model Status")
        
        acc_bar = Charts.gauge(accuracy * 100, 100, 25, (50, 70))
        
        print(f"""
    ╭────────────────────────────────────────────────────────────╮
    │  Status          │  {Term.GREEN}LOADED{Term.RESET}                              │
    │  Accuracy        │  {acc_bar}      │
    │  Features        │  {features:<40}│
    │  Training Size   │  {Formatters.number(samples)} samples{' ' * 27}│
    │  Last Trained    │  {str(last_trained)[:38]:<40}│
    ╰────────────────────────────────────────────────────────────╯""")
        
        if accuracy < 0.55:
            self.add_warning(f"ML accuracy below 55%: {accuracy:.1%}", "ml")
            self.add_recommendation("Consider retraining with more data or feature engineering")

    # =========================================================================
    # 8. LATENCY & PERFORMANCE
    # =========================================================================
    async def analyze_latency(self):
        print_header("SYSTEM PERFORMANCE", "⚡")
        
        endpoints = [
            ("strategy", "/health", "Strategy Health"),
            ("strategy", "/scan", "Signal Scan"),
            ("executor", "/portfolio", "Portfolio"),
            ("datafeed", "/prices", "Price Feed"),
            ("automation", "/status", "Automation"),
        ]
        
        print_section("API Latency")
        print(f"    {'Endpoint':<20} {'Avg':>10} {'P95':>10} {'Status':<15}")
        print(f"    {Term.GRAY}{'─' * 58}{Term.RESET}")
        
        for svc, ep, name in endpoints:
            url = f"{SERVICES[svc][0]}{ep}"
            times = []
            
            for _ in range(10):
                start = datetime.now()
                r = await self.fetch(url)
                elapsed = (datetime.now() - start).total_seconds() * 1000
                if r and not r.get("_error"):
                    times.append(elapsed)
            
            if times:
                avg_t = statistics.mean(times)
                p95_t = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]
                
                if avg_t < 100:
                    status = f"{Term.GREEN}✓ Excellent{Term.RESET}"
                elif avg_t < 300:
                    status = f"{Term.YELLOW}○ Good{Term.RESET}"
                elif avg_t < 500:
                    status = f"{Term.rgb(255,165,0)}△ Slow{Term.RESET}"
                else:
                    status = f"{Term.RED}✗ Critical{Term.RESET}"
                    self.add_warning(f"Slow endpoint: {name} ({avg_t:.0f}ms)", "performance")
                
                latency_color = Term.GREEN if avg_t < 100 else Term.YELLOW if avg_t < 300 else Term.RED
                print(f"    {name:<20} {latency_color}{avg_t:>8.0f}ms{Term.RESET} {p95_t:>8.0f}ms {status}")
            else:
                print(f"    {name:<20} {Term.RED}{'FAILED':>10}{Term.RESET}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    def print_summary(self):
        print_header("SYSTEM HEALTH REPORT", "📊")
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Critical Issues
        critical = [i for i in self.issues if i["severity"] == "critical"]
        high = [i for i in self.issues if i["severity"] == "high"]
        
        if critical:
            print_section(f"🚨 Critical Issues ({len(critical)})")
            for i in critical:
                print(f"    {Term.RED}● {i['msg']}{Term.RESET} [{i['component']}]")
        
        if high:
            print_section(f"❌ High Priority Issues ({len(high)})")
            for i in high:
                print(f"    {Term.rgb(255,165,0)}● {i['msg']}{Term.RESET} [{i['component']}]")
        
        # Warnings
        if self.warnings:
            print_section(f"⚠️  Warnings ({len(self.warnings)})")
            for w in self.warnings[:5]:
                print(f"    {Term.YELLOW}○ {w['msg']}{Term.RESET}")
            if len(self.warnings) > 5:
                print(f"    {Term.GRAY}... and {len(self.warnings) - 5} more{Term.RESET}")
        
        # Recommendations
        if self.recommendations:
            print_section(f"💡 Recommendations ({len(self.recommendations)})")
            for r in self.recommendations[:5]:
                print(f"    {Term.CYAN}→ {r['msg']}{Term.RESET}")
        
        if not self.issues and not self.warnings:
            ok("All systems operational!")
        
        # Health Score
        print_section("Overall Health Score")
        
        base_score = 100
        score = base_score
        score -= len(critical) * 25
        score -= len(high) * 15
        score -= len(self.warnings) * 5
        score = max(0, min(100, score))
        
        if score >= 90:
            grade, color = "A+", Term.GREEN
            status = "EXCELLENT"
            emoji = "🌟"
        elif score >= 80:
            grade, color = "A", Term.GREEN
            status = "HEALTHY"
            emoji = "✅"
        elif score >= 70:
            grade, color = "B", Term.YELLOW
            status = "GOOD"
            emoji = "👍"
        elif score >= 60:
            grade, color = "C", Term.YELLOW
            status = "FAIR"
            emoji = "⚠️"
        elif score >= 50:
            grade, color = "D", Term.rgb(255, 165, 0)
            status = "POOR"
            emoji = "⚡"
        else:
            grade, color = "F", Term.RED
            status = "CRITICAL"
            emoji = "🚨"
        
        # Visual score bar
        bar_full = int(score / 5)
        bar = f"{color}{'█' * bar_full}{Term.GRAY}{'░' * (20 - bar_full)}{Term.RESET}"
        
        print(f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║      {bar}  {color}{score:>3.0f}%{Term.RESET}                      ║
    ║                                                                ║
    ║              Grade: {color}{Term.BOLD}{grade}{Term.RESET}     Status: {color}{status}{Term.RESET}  {emoji}          ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝""")
        
        # Key metrics summary
        print_section("Key Metrics")
        
        svc = self.metrics.get("services", {})
        port = self.metrics.get("portfolio", {})
        risk = self.metrics.get("risk", {})
        trades = self.metrics.get("trades", {})
        signals = self.metrics.get("signals", {})
        auto = self.metrics.get("automation", {})
        
        print(f"""
    Services:      {svc.get('online', 0)}/{svc.get('total', 0)} online
    Portfolio:     {Formatters.money(port.get('total', 0))}
    Positions:     {self.metrics.get('positions', 0)}
    Drawdown:      {risk.get('drawdown', 0):.1f}%
    Signals:       {signals.get('buy', 0)} BUY / {signals.get('sell', 0)} SELL
    Trades:        {trades.get('total', 0)} (WR: {trades.get('win_rate', 0):.0%})
    Automation:    {'ON' if auto.get('enabled') else 'OFF'} ({auto.get('regime', 'N/A')})""")
        
        # Quick actions
        print_section("Quick Actions")
        print(f"""
    {Term.GRAY}1.{Term.RESET} Dashboard        {Term.CYAN}http://localhost:8080{Term.RESET}
    {Term.GRAY}2.{Term.RESET} Enable Auto      {Term.CYAN}curl -X POST 'localhost:8030/toggle?enabled=true'{Term.RESET}
    {Term.GRAY}3.{Term.RESET} Force Cycle      {Term.CYAN}curl -X POST 'localhost:8030/force_cycle'{Term.RESET}
    {Term.GRAY}4.{Term.RESET} View Logs        {Term.CYAN}docker compose logs -f --tail=100{Term.RESET}
    {Term.GRAY}5.{Term.RESET} Restart All      {Term.CYAN}docker compose restart{Term.RESET}
        """)
        
        print(f"  {Term.GRAY}Analysis completed in {elapsed:.1f}s at {datetime.now().strftime('%H:%M:%S')}{Term.RESET}\n")

    # =========================================================================
    # MAIN
    # =========================================================================
    async def run(self):
        # Animated header
        title = "TRADING AUTOPILOT - SYSTEM ANALYSIS v4.0"
        gradient_title = Term.gradient(title, (0, 200, 255), (200, 50, 255))
        
        print(f"""
{Term.BOLD}
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   🔍  {gradient_title}  ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝{Term.RESET}

    📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    🖥️  Starting comprehensive analysis...
        """)
        
        await self.start()
        
        try:
            sections = [
                ("Services", self.analyze_services),
                ("Portfolio", self.analyze_portfolio),
                ("Signals", self.analyze_signals),
                ("Risk", self.analyze_risk),
                ("Trades", self.analyze_trades),
                ("Automation", self.analyze_automation),
                ("ML Model", self.analyze_ml),
                ("Performance", self.analyze_latency),
            ]
            
            for name, func in sections:
                try:
                    await func()
                except Exception as e:
                    warn(f"Error analyzing {name}: {e}")
            
            self.print_summary()
            
        except KeyboardInterrupt:
            print(f"\n{Term.YELLOW}Analysis interrupted by user{Term.RESET}")
        except Exception as e:
            print(f"\n{Term.RED}Fatal error: {e}{Term.RESET}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()


if __name__ == "__main__":
    analyzer = SystemAnalyzer()
    asyncio.run(analyzer.run())
