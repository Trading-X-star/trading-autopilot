#!/usr/bin/env python3
"""
Trading Autopilot - Full System Analysis
Полный углублённый анализ всей системы
"""

import asyncio
import aiohttp
import asyncpg
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# === CONFIGURATION ===
SERVICES = {
    "strategy": "http://strategy:8005",
    "executor": "http://executor:8007", 
    "datafeed": "http://datafeed:8006",
    "dashboard": "http://dashboard:8080",
    "backtest": "http://backtest:8015",
}

DB_DSN = "postgresql://trading:trading123@postgres:5432/trading"
REDIS_URL = "redis://redis:6379/0"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def ok(msg): print(f"{Colors.GREEN}✅ {msg}{Colors.END}")
def fail(msg): print(f"{Colors.RED}❌ {msg}{Colors.END}")
def warn(msg): print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")
def info(msg): print(f"{Colors.CYAN}ℹ️  {msg}{Colors.END}")
def header(msg): print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}\n{msg}\n{'='*60}{Colors.END}")
def subheader(msg): print(f"\n{Colors.BOLD}--- {msg} ---{Colors.END}")

class SystemAnalyzer:
    def __init__(self):
        self.session = None
        self.db_pool = None
        self.results = defaultdict(dict)
        self.issues = []
        self.recommendations = []
        
    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        try:
            self.db_pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=3)
        except Exception as e:
            warn(f"Database connection failed: {e}")
            
    async def stop(self):
        if self.session: await self.session.close()
        if self.db_pool: await self.db_pool.close()
        
    async def fetch_json(self, url):
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except: pass
        return None

    # =========================================================================
    # 1. SERVICES HEALTH CHECK
    # =========================================================================
    async def analyze_services(self):
        header("1. SERVICES HEALTH CHECK")
        
        for name, base_url in SERVICES.items():
            health = await self.fetch_json(f"{base_url}/health")
            
            if health:
                status = health.get("status", "unknown")
                if status in ["healthy", "ok"]:
                    ok(f"{name:12} @ {base_url} — {status}")
                    self.results["services"][name] = {"status": "healthy", "data": health}
                else:
                    warn(f"{name:12} @ {base_url} — {status}")
                    self.results["services"][name] = {"status": "degraded", "data": health}
            else:
                fail(f"{name:12} @ {base_url} — OFFLINE")
                self.results["services"][name] = {"status": "offline"}
                self.issues.append(f"Service {name} is offline")
        
        # Detailed service info
        subheader("Service Details")
        
        # Strategy
        if strat := self.results["services"].get("strategy", {}).get("data"):
            ml_info = strat.get("ml_info", {})
            print(f"   Strategy Mode: {strat.get('strategy_mode', 'unknown')}")
            print(f"   ML Available:  {strat.get('ml_available', False)}")
            print(f"   ML Accuracy:   {ml_info.get('accuracy', 0):.1%}")
            print(f"   ML Features:   {ml_info.get('features', 0)}")
            
            if ml_info.get('accuracy', 0) < 0.5:
                self.recommendations.append("ML accuracy below 50% - consider retraining with more data")
        
        # Executor
        if exec_data := self.results["services"].get("executor", {}).get("data"):
            print(f"   Executor Mode:    {exec_data.get('mode', 'unknown')}")
            print(f"   Auto Execute:     {exec_data.get('auto_execute', False)}")
            print(f"   Daily Trades:     {exec_data.get('daily_trades', 0)}")
            print(f"   Tinkoff Connected:{exec_data.get('tinkoff_connected', False)}")
            
            if not exec_data.get('tinkoff_connected'):
                self.issues.append("Tinkoff API not connected")

    # =========================================================================
    # 2. DATABASE ANALYSIS
    # =========================================================================
    async def analyze_database(self):
        header("2. DATABASE ANALYSIS")
        
        if not self.db_pool:
            fail("Database not available")
            return
            
        async with self.db_pool.acquire() as conn:
            # Tables overview
            subheader("Tables Overview")
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """)
            print(f"   Tables found: {len(tables)}")
            
            for table in tables:
                tname = table['tablename']
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {tname}")
                size = await conn.fetchval(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{tname}'))
                """)
                print(f"   {tname:20} — {count:>10,} rows — {size}")
                self.results["database"][tname] = {"rows": count, "size": size}
            
            # Candles analysis
            subheader("Candles Data Quality")
            candles_stats = await conn.fetch("""
                SELECT 
                    ticker,
                    COUNT(*) as cnt,
                    MIN(date) as min_date,
                    MAX(date) as max_date,
                    COUNT(DISTINCT date) as unique_days
                FROM ohlcv_daily
                GROUP BY ticker
                ORDER BY cnt DESC
                LIMIT 10
            """)
            
            print(f"   {'Ticker':<8} {'Count':>10} {'From':>12} {'To':>12} {'Days':>8}")
            print(f"   {'-'*50}")
            for row in candles_stats:
                print(f"   {row['ticker']:<8} {row['cnt']:>10,} {str(row['min_date'])[:10]:>12} {str(row['max_date'])[:10]:>12} {row['unique_days']:>8}")
            
            total_candles = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_daily")
            self.results["database"]["total_candles"] = total_candles
            
            if total_candles < 100000:
                self.recommendations.append(f"Only {total_candles:,} candles - consider loading more historical data")
            
            # Features analysis
            subheader("Features Data Quality")
            features_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(signal_class) as with_labels,
                    AVG(CASE WHEN signal_class = 1 THEN 1.0 ELSE 0 END) as buy_ratio,
                    AVG(CASE WHEN signal_class = -1 THEN 1.0 ELSE 0 END) as sell_ratio,
                    COUNT(DISTINCT ticker) as tickers
                FROM features
            """)
            
            if features_stats:
                print(f"   Total features:    {features_stats['total']:,}")
                print(f"   With labels:       {features_stats['with_labels']:,}")
                print(f"   Buy signals:       {features_stats['buy_ratio']:.1%}")
                print(f"   Sell signals:      {features_stats['sell_ratio']:.1%}")
                print(f"   Hold signals:      {1 - features_stats['buy_ratio'] - features_stats['sell_ratio']:.1%}")
                print(f"   Unique tickers:    {features_stats['tickers']}")
                
                # Check for class imbalance
                buy_r, sell_r = features_stats['buy_ratio'], features_stats['sell_ratio']
                if abs(buy_r - sell_r) > 0.1:
                    self.recommendations.append(f"Class imbalance detected: Buy={buy_r:.1%}, Sell={sell_r:.1%}")
            
            # Null values check
            subheader("Data Quality - Null Values")
            null_check = await conn.fetch("""
                SELECT 
                    'rsi_14' as col, COUNT(*) FILTER (WHERE rsi_14 IS NULL) as nulls FROM features
                UNION ALL
                SELECT 'macd', COUNT(*) FILTER (WHERE macd IS NULL) FROM features
                UNION ALL
                SELECT 'sma_200', COUNT(*) FILTER (WHERE sma_200 IS NULL) FROM features
                UNION ALL
                SELECT 'volume_ratio', COUNT(*) FILTER (WHERE volume_ratio IS NULL) FROM features
            """)
            
            total = features_stats['total'] if features_stats else 1
            for row in null_check:
                pct = row['nulls'] / total * 100 if total > 0 else 0
                status = "✓" if pct < 5 else "⚠" if pct < 20 else "✗"
                print(f"   {status} {row['col']:15} — {row['nulls']:,} nulls ({pct:.1f}%)")
                if pct > 20:
                    self.issues.append(f"High null rate in {row['col']}: {pct:.1f}%")

    # =========================================================================
    # 3. TRADING PERFORMANCE ANALYSIS
    # =========================================================================
    async def analyze_trading(self):
        header("3. TRADING PERFORMANCE ANALYSIS")
        
        # Get trades from executor
        trades = await self.fetch_json(f"{SERVICES['executor']}/trades?limit=1000")
        
        if not trades:
            warn("No trades data available")
            return
            
        subheader("Trade Statistics")
        print(f"   Total trades: {len(trades)}")
        
        if len(trades) == 0:
            info("No trades executed yet")
            return
        
        # Analyze trades
        buys = [t for t in trades if t.get('side') == 'buy']
        sells = [t for t in trades if t.get('side') == 'sell']
        
        print(f"   Buy orders:  {len(buys)}")
        print(f"   Sell orders: {len(sells)}")
        
        # By ticker
        by_ticker = defaultdict(list)
        for t in trades:
            by_ticker[t.get('ticker', 'UNK')].append(t)
        
        print(f"\n   Trades by ticker:")
        for ticker, ticker_trades in sorted(by_ticker.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"      {ticker}: {len(ticker_trades)}")
        
        # Time analysis
        if trades and trades[0].get('executed_at'):
            dates = [t['executed_at'][:10] for t in trades if t.get('executed_at')]
            unique_days = len(set(dates))
            print(f"\n   Trading days: {unique_days}")
            print(f"   Avg trades/day: {len(trades)/max(unique_days,1):.1f}")
        
        # Portfolio
        subheader("Portfolio Status")
        portfolio = await self.fetch_json(f"{SERVICES['executor']}/portfolio")
        
        if portfolio:
            total = portfolio.get('total_value') or portfolio.get('totalAmountPortfolio', {}).get('units', 0)
            cash = portfolio.get('cash') or portfolio.get('totalAmountCurrencies', {}).get('units', 0)
            print(f"   Total value: ₽{float(total):,.0f}")
            print(f"   Cash:        ₽{float(cash):,.0f}")
            
            positions = portfolio.get('positions', [])
            if positions:
                print(f"   Positions:   {len(positions)}")
                for pos in positions[:5]:
                    ticker = pos.get('ticker') or pos.get('figi', 'UNK')
                    qty = pos.get('quantity') or pos.get('balance', {}).get('units', 0)
                    print(f"      {ticker}: {qty}")

    # =========================================================================
    # 4. SIGNALS ANALYSIS
    # =========================================================================
    async def analyze_signals(self):
        header("4. SIGNALS ANALYSIS")
        
        # Current signals
        signals = await self.fetch_json(f"{SERVICES['strategy']}/scan")
        
        if not signals:
            warn("No signals data available")
            return
            
        subheader("Current Signals")
        
        buys = [s for s in signals if s.get('signal') == 'buy']
        sells = [s for s in signals if s.get('signal') == 'sell']
        holds = [s for s in signals if s.get('signal') == 'hold']
        
        print(f"   Total scanned: {len(signals)}")
        print(f"   BUY signals:   {len(buys)}")
        print(f"   SELL signals:  {len(sells)}")
        print(f"   HOLD signals:  {len(holds)}")
        
        # Top signals
        if buys:
            print(f"\n   Top BUY signals:")
            for s in sorted(buys, key=lambda x: -x.get('confidence', 0))[:5]:
                print(f"      {s['ticker']:6} conf={s.get('confidence', 0):.2f} price=₽{s.get('price', 0):.2f}")
        
        if sells:
            print(f"\n   Top SELL signals:")
            for s in sorted(sells, key=lambda x: -x.get('confidence', 0))[:5]:
                print(f"      {s['ticker']:6} conf={s.get('confidence', 0):.2f} price=₽{s.get('price', 0):.2f}")
        
        # Confidence distribution
        subheader("Confidence Distribution")
        confidences = [s.get('confidence', 0) for s in signals if s.get('signal') != 'hold']
        
        if confidences:
            buckets = {'0.0-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-1.0': 0}
            for c in confidences:
                if c < 0.3: buckets['0.0-0.3'] += 1
                elif c < 0.5: buckets['0.3-0.5'] += 1
                elif c < 0.7: buckets['0.5-0.7'] += 1
                else: buckets['0.7-1.0'] += 1
            
            for bucket, count in buckets.items():
                bar = '█' * (count * 2)
                print(f"   {bucket}: {count:3} {bar}")
            
            avg_conf = statistics.mean(confidences)
            print(f"\n   Avg confidence: {avg_conf:.2f}")
            
            if avg_conf < 0.5:
                self.recommendations.append("Average signal confidence is low - consider stricter filters")

    # =========================================================================
    # 5. PAIR TRADING ANALYSIS
    # =========================================================================
    async def analyze_pairs(self):
        header("5. PAIR TRADING ANALYSIS")
        
        # Pairs status
        pairs_status = await self.fetch_json(f"{SERVICES['executor']}/pairs/status")
        pairs_signals = await self.fetch_json(f"{SERVICES['strategy']}/pairs/scan")
        
        if not pairs_status:
            warn("Pair trading not available")
            return
            
        subheader("Pairs Status")
        print(f"   Enabled:        {pairs_status.get('enabled', False)}")
        print(f"   Open positions: {pairs_status.get('positions', 0)}")
        print(f"   Total PnL:      ₽{pairs_status.get('total_pnl', 0):,.0f}")
        
        # Open pairs
        open_pairs = pairs_status.get('open_pairs', [])
        if open_pairs:
            print(f"\n   Open pairs:")
            for p in open_pairs:
                pair = p.get('pair', ['?', '?'])
                print(f"      {pair[0]}/{pair[1]} — Z={p.get('entry_zscore', 0):.2f}, conf={p.get('confidence', 0):.2f}")
        
        # History
        history = pairs_status.get('history', [])
        if history:
            print(f"\n   Trade history: {len(history)} trades")
            total_pnl = sum(h.get('total_pnl', 0) for h in history)
            wins = sum(1 for h in history if h.get('total_pnl', 0) > 0)
            print(f"      Win rate: {wins/len(history)*100:.1f}%")
            print(f"      Total PnL: ₽{total_pnl:,.0f}")
        
        # Pairs signals
        if pairs_signals:
            subheader("Pair Signals")
            signals = pairs_signals.get('signals', [])
            all_pairs = pairs_signals.get('pairs', [])
            
            print(f"   Monitored pairs: {len(all_pairs)}")
            print(f"   Active signals:  {len(signals)}")
            
            if all_pairs:
                print(f"\n   Z-scores:")
                for p in all_pairs[:10]:
                    pair = p.get('pair', ['?', '?'])
                    z = p.get('zscore', 0)
                    bar = '█' * min(int(abs(z) * 5), 20)
                    direction = '↑' if z > 0 else '↓' if z < 0 else '='
                    print(f"      {pair[0]:5}/{pair[1]:5} Z={z:+.2f} {direction} {bar}")

    # =========================================================================
    # 6. ML MODEL ANALYSIS
    # =========================================================================
    async def analyze_ml(self):
        header("6. ML MODEL ANALYSIS")
        
        if not self.db_pool:
            warn("Database not available for ML analysis")
            return
            
        async with self.db_pool.acquire() as conn:
            # Feature correlations with target
            subheader("Feature-Target Correlations")
            
            correlations = await conn.fetch("""
                SELECT 
                    'rsi_14' as feature,
                    CORR(rsi_14, CASE WHEN signal_class = 1 THEN 1 ELSE 0 END) as corr_buy
                FROM features WHERE signal_class IS NOT NULL
                UNION ALL
                SELECT 'macd_hist', CORR(macd_hist, CASE WHEN signal_class = 1 THEN 1 ELSE 0 END) FROM features WHERE signal_class IS NOT NULL
                UNION ALL
                SELECT 'return_5d', CORR(return_5d, CASE WHEN signal_class = 1 THEN 1 ELSE 0 END) FROM features WHERE signal_class IS NOT NULL
                UNION ALL
                SELECT 'bb_pct', CORR(bb_pct, CASE WHEN signal_class = 1 THEN 1 ELSE 0 END) FROM features WHERE signal_class IS NOT NULL
                UNION ALL
                SELECT 'volatility_20', CORR(volatility_20, CASE WHEN signal_class = 1 THEN 1 ELSE 0 END) FROM features WHERE signal_class IS NOT NULL
            """)
            
            print(f"   {'Feature':<15} {'Corr with BUY':>15}")
            print(f"   {'-'*32}")
            for row in sorted(correlations, key=lambda x: abs(x['corr_buy'] or 0), reverse=True):
                corr = row['corr_buy'] or 0
                bar = '█' * int(abs(corr) * 50)
                sign = '+' if corr > 0 else '-' if corr < 0 else ' '
                print(f"   {row['feature']:<15} {sign}{abs(corr):.4f} {bar}")
            
            # Signal accuracy by confidence bucket
            subheader("Historical Signal Accuracy")
            
            accuracy_by_conf = await conn.fetch("""
                WITH signal_outcomes AS (
                    SELECT 
                        ticker, date, signal_class,
                        target_5d,
                        CASE 
                            WHEN signal_class = 1 AND target_5d > 0.01 THEN 1
                            WHEN signal_class = -1 AND target_5d < -0.01 THEN 1
                            WHEN signal_class = 0 AND ABS(target_5d) < 0.01 THEN 1
                            ELSE 0
                        END as correct
                    FROM features
                    WHERE signal_class IS NOT NULL AND target_5d IS NOT NULL
                )
                SELECT 
                    signal_class,
                    COUNT(*) as total,
                    AVG(correct) as accuracy,
                    AVG(target_5d) as avg_return
                FROM signal_outcomes
                GROUP BY signal_class
                ORDER BY signal_class
            """)
            
            print(f"   {'Signal':<8} {'Count':>10} {'Accuracy':>10} {'Avg Return':>12}")
            print(f"   {'-'*42}")
            for row in accuracy_by_conf:
                sig_name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}.get(row['signal_class'], '?')
                print(f"   {sig_name:<8} {row['total']:>10,} {row['accuracy']*100:>9.1f}% {row['avg_return']*100:>+11.2f}%")

    # =========================================================================
    # 7. LATENCY & PERFORMANCE
    # =========================================================================
    async def analyze_performance(self):
        header("7. LATENCY & PERFORMANCE")
        
        subheader("API Response Times")
        
        endpoints = [
            ("strategy", "/health"),
            ("strategy", "/scan"),
            ("strategy", "/analyze/SBER"),
            ("executor", "/health"),
            ("executor", "/portfolio"),
            ("datafeed", "/prices"),
        ]
        
        for service, endpoint in endpoints:
            base_url = SERVICES.get(service)
            if not base_url:
                continue
                
            url = f"{base_url}{endpoint}"
            times = []
            
            for _ in range(3):
                start = datetime.now()
                result = await self.fetch_json(url)
                elapsed = (datetime.now() - start).total_seconds() * 1000
                if result:
                    times.append(elapsed)
            
            if times:
                avg = statistics.mean(times)
                status = "✓" if avg < 100 else "⚠" if avg < 500 else "✗"
                print(f"   {status} {service:10} {endpoint:20} — {avg:6.1f}ms")
                
                if avg > 500:
                    self.issues.append(f"Slow response: {service}{endpoint} ({avg:.0f}ms)")
            else:
                print(f"   ✗ {service:10} {endpoint:20} — FAILED")

    # =========================================================================
    # 8. CONFIGURATION AUDIT
    # =========================================================================
    async def analyze_config(self):
        header("8. CONFIGURATION AUDIT")
        
        subheader("Environment Variables")
        
        # Check executor config
        exec_health = self.results["services"].get("executor", {}).get("data", {})
        
        configs = {
            "Mode": exec_health.get("mode", "unknown"),
            "Auto Execute": exec_health.get("auto_execute", False),
            "Tinkoff Connected": exec_health.get("tinkoff_connected", False),
            "Account ID": exec_health.get("account_id", "not set")[:20] + "..." if exec_health.get("account_id") else "not set",
        }
        
        for key, value in configs.items():
            status = "✓" if value and value != "unknown" else "⚠"
            print(f"   {status} {key:20}: {value}")
        
        # Risk checks
        subheader("Risk Configuration")
        
        risk_items = [
            ("Daily trade limit", "20", "Prevents overtrading"),
            ("Min confidence", "0.55", "Filters weak signals"),
            ("Max position size", "10%", "Diversification"),
            ("Stop loss", "5%", "Loss protection"),
        ]
        
        for item, value, desc in risk_items:
            print(f"   • {item}: {value} — {desc}")

    # =========================================================================
    # SUMMARY & RECOMMENDATIONS
    # =========================================================================
    def print_summary(self):
        header("9. SUMMARY & RECOMMENDATIONS")
        
        # Issues
        subheader("Issues Found")
        if self.issues:
            for issue in self.issues:
                fail(issue)
        else:
            ok("No critical issues found")
        
        # Recommendations
        subheader("Recommendations")
        if self.recommendations:
            for rec in self.recommendations:
                warn(rec)
        else:
            ok("System is well configured")
        
        # Overall health score
        subheader("Overall Health Score")
        
        total_checks = 10
        passed = total_checks - len(self.issues)
        score = passed / total_checks * 100
        
        bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
        
        if score >= 80:
            color = Colors.GREEN
            status = "HEALTHY"
        elif score >= 60:
            color = Colors.YELLOW
            status = "DEGRADED"
        else:
            color = Colors.RED
            status = "CRITICAL"
        
        print(f"\n   {color}{bar} {score:.0f}% — {status}{Colors.END}")
        
        # Quick actions
        subheader("Quick Actions")
        print("""
   1. Check logs:    docker-compose logs -f strategy executor
   2. Restart all:   docker-compose restart
   3. View dashboard: http://dashboard:8080
   4. Run backtest:  curl -X POST http://backtest:8015/run
   5. Retrain ML:    docker exec strategy python /app/train_model_v2.py
        """)

    async def run(self):
        print(f"\n{Colors.BOLD}🔍 TRADING AUTOPILOT - FULL SYSTEM ANALYSIS{Colors.END}")
        print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        await self.start()
        
        try:
            await self.analyze_services()
            await self.analyze_database()
            await self.analyze_trading()
            await self.analyze_signals()
            await self.analyze_pairs()
            await self.analyze_ml()
            await self.analyze_performance()
            await self.analyze_config()
            self.print_summary()
        finally:
            await self.stop()
        
        print(f"\n   Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    analyzer = SystemAnalyzer()
    asyncio.run(analyzer.run())
