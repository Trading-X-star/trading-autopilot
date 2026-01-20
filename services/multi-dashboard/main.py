#!/usr/bin/env python3
"""Multi-Dashboard - Web UI for multi-account trading system"""
import asyncio
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("multi-dashboard")

ACCOUNT_MANAGER_URL = os.getenv("ACCOUNT_MANAGER_URL", "http://account-manager:8020")


class Dashboard:
    def __init__(self):
        self.redis = None
        self.http = None

    async def start(self):
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )
        self.http = httpx.AsyncClient(timeout=10.0)
        logger.info("✅ Dashboard started")

    async def stop(self):
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        logger.info("🛑 Dashboard stopped")

    async def get_overview(self) -> dict:
        """Get system overview"""
        # Trading state
        trading = await self.redis.get("system:trading") == "1"

        # Get accounts
        try:
            resp = await self.http.get(f"{ACCOUNT_MANAGER_URL}/accounts")
            accounts = resp.json()
        except:
            accounts = []

        # Aggregate stats
        total_value = 0
        total_positions = 0

        for acc in accounts:
            total_value += acc.get("current_balance", 0)
            positions = await self.redis.hgetall(f"positions:{acc['id']}")
            total_positions += len(positions)

        # Recent alerts
        alerts = await self.redis.xrevrange("stream:alerts", count=5)
        recent_alerts = [{"id": a[0], **a[1]} for a in alerts]

        return {
            "trading_enabled": trading,
            "accounts_count": len(accounts),
            "total_value": round(total_value, 2),
            "total_positions": total_positions,
            "recent_alerts": recent_alerts,
            "timestamp": datetime.now().isoformat()
        }

    async def get_accounts_summary(self) -> list:
        """Get all accounts with positions"""
        try:
            resp = await self.http.get(f"{ACCOUNT_MANAGER_URL}/accounts")
            accounts = resp.json()
        except:
            return []

        result = []
        for acc in accounts:
            positions = await self.redis.hgetall(f"positions:{acc['id']}")
            pos_list = []
            total_pnl = 0

            for ticker, data in positions.items():
                pos = json.loads(data)
                price_data = await self.redis.get(f"price:{ticker}")
                current = json.loads(price_data).get("close", pos["avg_price"]) if price_data else pos["avg_price"]
                pnl = (current - pos["avg_price"]) * pos["quantity"]
                pnl_pct = ((current / pos["avg_price"]) - 1) * 100 if pos["avg_price"] > 0 else 0
                total_pnl += pnl

                pos_list.append({
                    "ticker": ticker,
                    "quantity": pos["quantity"],
                    "avg_price": pos["avg_price"],
                    "current_price": round(current, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2)
                })

            result.append({
                "id": acc["id"],
                "name": acc["name"],
                "risk_profile": acc.get("risk_profile", "balanced"),
                "balance": acc.get("current_balance", 0),
                "is_main": acc.get("is_main", False),
                "is_active": acc.get("is_active", True),
                "positions": pos_list,
                "positions_count": len(pos_list),
                "unrealized_pnl": round(total_pnl, 2)
            })

        return result


# Initialize
dashboard = Dashboard()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await dashboard.start()
    yield
    await dashboard.stop()


app = FastAPI(
    title="Trading Dashboard",
    version="1.0.0",
    lifespan=lifespan
)


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Autopilot</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <style>
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        [x-cloak] { display: none !important; }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div x-data="dashboard()" x-init="init()" class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="flex justify-between items-center mb-8">
            <h1 class="text-3xl font-bold">🤖 Trading Autopilot</h1>
            <div class="flex items-center gap-4">
                <span x-text="overview.timestamp ? new Date(overview.timestamp).toLocaleTimeString() : ''" class="text-gray-400"></span>
                <button @click="toggleTrading()" 
                        :class="overview.trading_enabled ? 'bg-green-600' : 'bg-red-600'"
                        class="px-4 py-2 rounded-lg font-semibold">
                    <span x-text="overview.trading_enabled ? '● TRADING ON' : '○ TRADING OFF'"></span>
                </button>
            </div>
        </div>

        <!-- Stats Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Accounts</div>
                <div class="text-2xl font-bold" x-text="overview.accounts_count || 0"></div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Total Value</div>
                <div class="text-2xl font-bold" x-text="formatMoney(overview.total_value)"></div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Positions</div>
                <div class="text-2xl font-bold" x-text="overview.total_positions || 0"></div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Status</div>
                <div class="text-2xl font-bold" :class="overview.trading_enabled ? 'text-green-500' : 'text-red-500'"
                     x-text="overview.trading_enabled ? 'Active' : 'Stopped'"></div>
            </div>
        </div>

        <!-- Accounts -->
        <div class="mb-8">
            <h2 class="text-xl font-semibold mb-4">📊 Accounts</h2>
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <template x-for="account in accounts" :key="account.id">
                    <div class="bg-gray-800 rounded-lg p-4">
                        <div class="flex justify-between items-start mb-3">
                            <div>
                                <div class="font-semibold text-lg" x-text="account.name"></div>
                                <div class="text-sm text-gray-400">
                                    <span x-text="account.risk_profile"></span>
                                    <span x-show="account.is_main" class="ml-2 text-yellow-500">⭐ Main</span>
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="font-bold" x-text="formatMoney(account.balance)"></div>
                                <div :class="account.unrealized_pnl >= 0 ? 'positive' : 'negative'" 
                                     x-text="formatPnL(account.unrealized_pnl)"></div>
                            </div>
                        </div>

                        <!-- Positions -->
                        <div x-show="account.positions.length > 0" class="mt-3 border-t border-gray-700 pt-3">
                            <div class="text-sm text-gray-400 mb-2">Positions:</div>
                            <template x-for="pos in account.positions" :key="pos.ticker">
                                <div class="flex justify-between text-sm py-1">
                                    <span>
                                        <span class="font-medium" x-text="pos.ticker"></span>
                                        <span class="text-gray-500" x-text="'x' + pos.quantity"></span>
                                    </span>
                                    <span :class="pos.pnl >= 0 ? 'positive' : 'negative'">
                                        <span x-text="formatPnL(pos.pnl)"></span>
                                        <span class="text-gray-500" x-text="'(' + pos.pnl_pct.toFixed(1) + '%)'"></span>
                                    </span>
                                </div>
                            </template>
                        </div>
                        <div x-show="account.positions.length === 0" class="text-gray-500 text-sm mt-3">
                            No positions
                        </div>
                    </div>
                </template>
            </div>

            <!-- Add Account Button -->
            <button x-show="accounts.length < 3" @click="showAddAccount = true"
                    class="mt-4 px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700">
                + Add Account
            </button>
        </div>

        <!-- Recent Alerts -->
        <div>
            <h2 class="text-xl font-semibold mb-4">🔔 Recent Alerts</h2>
            <div class="bg-gray-800 rounded-lg p-4">
                <template x-for="alert in overview.recent_alerts || []" :key="alert.id">
                    <div class="flex justify-between py-2 border-b border-gray-700 last:border-0">
                        <div>
                            <span :class="{
                                'text-red-400': alert.severity === 'error',
                                'text-yellow-400': alert.severity === 'warning',
                                'text-blue-400': alert.severity === 'info'
                            }" x-text="alert.name"></span>
                            <span class="text-gray-400 ml-2" x-text="alert.message"></span>
                        </div>
                    </div>
                </template>
                <div x-show="!overview.recent_alerts?.length" class="text-gray-500">No recent alerts</div>
            </div>
        </div>

        <!-- Add Account Modal -->
        <div x-show="showAddAccount" x-cloak class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
            <div class="bg-gray-800 rounded-lg p-6 w-96">
                <h3 class="text-xl font-semibold mb-4">New Account</h3>
                <input x-model="newAccount.name" placeholder="Account Name" 
                       class="w-full bg-gray-700 rounded px-3 py-2 mb-3">
                <select x-model="newAccount.risk_profile" class="w-full bg-gray-700 rounded px-3 py-2 mb-3">
                    <option value="conservative">Conservative</option>
                    <option value="balanced">Balanced</option>
                    <option value="aggressive">Aggressive</option>
                </select>
                <input x-model="newAccount.initial_balance" type="number" placeholder="Initial Balance"
                       class="w-full bg-gray-700 rounded px-3 py-2 mb-4">
                <div class="flex gap-2">
                    <button @click="createAccount()" class="flex-1 bg-green-600 rounded py-2">Create</button>
                    <button @click="showAddAccount = false" class="flex-1 bg-gray-600 rounded py-2">Cancel</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        function dashboard() {
            return {
                overview: {},
                accounts: [],
                showAddAccount: false,
                newAccount: { name: '', risk_profile: 'balanced', initial_balance: 1000000 },

                async init() {
                    await this.refresh();
                    setInterval(() => this.refresh(), 10000);
                },

                async refresh() {
                    try {
                        const [ov, acc] = await Promise.all([
                            fetch('/api/overview').then(r => r.json()),
                            fetch('/api/accounts').then(r => r.json())
                        ]);
                        this.overview = ov;
                        this.accounts = acc;
                    } catch(e) { console.error(e); }
                },

                async toggleTrading() {
                    const action = this.overview.trading_enabled ? 'stop' : 'start';
                    await fetch(`/api/trading/${action}`, { method: 'POST' });
                    await this.refresh();
                },

                async createAccount() {
                    await fetch('/api/accounts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(this.newAccount)
                    });
                    this.showAddAccount = false;
                    this.newAccount = { name: '', risk_profile: 'balanced', initial_balance: 1000000 };
                    await this.refresh();
                },

                formatMoney(v) { return (v || 0).toLocaleString('ru-RU', {maximumFractionDigits: 0}) + ' ₽'; },
                formatPnL(v) { return (v >= 0 ? '+' : '') + (v || 0).toLocaleString('ru-RU', {maximumFractionDigits: 0}) + ' ₽'; }
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/overview")
async def api_overview():
    return await dashboard.get_overview()


@app.get("/api/accounts")
async def api_accounts():
    return await dashboard.get_accounts_summary()


@app.post("/api/trading/{action}")
async def api_trading(action: str):
    if action == "start":
        await dashboard.redis.set("system:trading", "1")
    else:
        await dashboard.redis.set("system:trading", "0")
    return {"trading": action == "start"}


@app.post("/api/accounts")
async def api_create_account(request: Request):
    data = await request.json()
    async with dashboard.http as client:
        resp = await client.post(f"{ACCOUNT_MANAGER_URL}/accounts", json=data)
        return resp.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8022)
