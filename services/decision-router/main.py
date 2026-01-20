#!/usr/bin/env python3
"""Decision Router - Route signals to accounts based on rules"""
import asyncio
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("decision-router")

# Metrics
SIGNALS_ROUTED = Counter("signals_routed_total", "Signals routed", ["signal", "account_id"])
SIGNALS_SKIPPED = Counter("signals_skipped_total", "Signals skipped", ["reason"])


class RoutingRule(BaseModel):
    account_id: str
    min_confidence: float = 0.5
    allowed_signals: list[str] = ["strong_buy", "buy", "sell", "strong_sell"]
    max_positions: int = 10
    tickers: list[str] | None = None  # None = all tickers


class DecisionRouter:
    def __init__(self):
        self.redis = None
        self.http = None
        self.running = False
        self.rules: dict[str, RoutingRule] = {}
        self.account_manager_url = os.getenv("ACCOUNT_MANAGER_URL", "http://account-manager:8020")

    async def start(self):
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )
        self.http = httpx.AsyncClient(timeout=10.0)

        # Load rules from Redis
        await self._load_rules()

        # Auto-create rules for existing accounts
        await self._sync_accounts()

        self.running = True
        asyncio.create_task(self._signal_consumer())

        logger.info(f"✅ Decision Router started ({len(self.rules)} rules)")

    async def stop(self):
        self.running = False
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        logger.info("🛑 Decision Router stopped")

    async def _load_rules(self):
        """Load routing rules from Redis"""
        keys = await self.redis.keys("routing_rule:*")
        for key in keys:
            data = await self.redis.get(key)
            if data:
                rule_data = json.loads(data)
                self.rules[rule_data["account_id"]] = RoutingRule(**rule_data)

    async def _save_rule(self, rule: RoutingRule):
        """Save rule to Redis"""
        await self.redis.set(f"routing_rule:{rule.account_id}", rule.model_dump_json())

    async def _sync_accounts(self):
        """Create default rules for accounts without rules"""
        try:
            response = await self.http.get(f"{self.account_manager_url}/accounts")
            accounts = response.json()

            for account in accounts:
                if account["id"] not in self.rules:
                    # Create default rule based on risk profile
                    profile = account.get("risk_profile", "balanced")

                    if profile == "conservative":
                        rule = RoutingRule(
                            account_id=account["id"],
                            min_confidence=0.7,
                            allowed_signals=["strong_buy", "strong_sell"],
                            max_positions=5
                        )
                    elif profile == "aggressive":
                        rule = RoutingRule(
                            account_id=account["id"],
                            min_confidence=0.3,
                            allowed_signals=["strong_buy", "buy", "sell", "strong_sell"],
                            max_positions=15
                        )
                    else:  # balanced
                        rule = RoutingRule(
                            account_id=account["id"],
                            min_confidence=0.5,
                            allowed_signals=["strong_buy", "buy", "sell", "strong_sell"],
                            max_positions=10
                        )

                    self.rules[account["id"]] = rule
                    await self._save_rule(rule)
                    logger.info(f"📝 Created rule for {account['name']} ({profile})")

        except Exception as e:
            logger.error(f"Failed to sync accounts: {e}")

    async def _signal_consumer(self):
        """Consume signals from Redis stream"""
        last_id = "0"

        while self.running:
            try:
                # Read from stream
                result = await self.redis.xread(
                    {"stream:signals": last_id},
                    count=10,
                    block=5000
                )

                if not result:
                    continue

                for stream_name, messages in result:
                    for msg_id, data in messages:
                        last_id = msg_id
                        await self._process_signal(data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(5)

    async def _process_signal(self, signal: dict):
        """Process and route signal to appropriate accounts"""
        ticker = signal.get("ticker", "")
        signal_type = signal.get("signal", "")
        confidence = float(signal.get("confidence", 0))
        price = float(signal.get("price", 0))

        logger.info(f"📨 Signal: {ticker} {signal_type} ({confidence:.0%})")

        # Check trading enabled
        trading = await self.redis.get("system:trading")
        if trading != "1":
            SIGNALS_SKIPPED.labels(reason="trading_disabled").inc()
            return

        routed_count = 0

        for account_id, rule in self.rules.items():
            # Check if account should receive this signal
            should_route, reason = await self._should_route(signal, rule)

            if not should_route:
                SIGNALS_SKIPPED.labels(reason=reason).inc()
                continue

            # Route to account
            await self._route_to_account(account_id, ticker, signal_type, confidence, price, rule)
            SIGNALS_ROUTED.labels(signal=signal_type, account_id=account_id).inc()
            routed_count += 1

        if routed_count > 0:
            logger.info(f"✅ Routed {ticker} to {routed_count} account(s)")

    async def _should_route(self, signal: dict, rule: RoutingRule) -> tuple[bool, str]:
        """Check if signal should be routed to account"""
        ticker = signal.get("ticker", "")
        signal_type = signal.get("signal", "")
        confidence = float(signal.get("confidence", 0))

        # Check confidence threshold
        if confidence < rule.min_confidence:
            return False, "low_confidence"

        # Check signal type
        if signal_type not in rule.allowed_signals:
            return False, "signal_not_allowed"

        # Check ticker filter
        if rule.tickers and ticker not in rule.tickers:
            return False, "ticker_not_allowed"

        # Check max positions
        positions = await self.redis.hgetall(f"positions:{rule.account_id}")
        if len(positions) >= rule.max_positions and signal_type in ("buy", "strong_buy"):
            # Allow if adding to existing position
            if ticker not in positions:
                return False, "max_positions"

        return True, ""

    async def _route_to_account(self, account_id: str, ticker: str, signal_type: str, 
                                confidence: float, price: float, rule: RoutingRule):
        """Execute signal for account"""
        # Get account info
        try:
            response = await self.http.get(f"{self.account_manager_url}/accounts/{account_id}")
            account = response.json()
        except Exception as e:
            logger.error(f"Failed to get account {account_id}: {e}")
            return

        if not account.get("is_active", True):
            return

        # Determine action
        if signal_type in ("strong_buy", "buy"):
            side = "buy"
        elif signal_type in ("strong_sell", "sell"):
            side = "sell"
        else:
            return

        # Calculate position size (simplified)
        balance = account.get("current_balance", account.get("balance", 1000000))

        # Size based on confidence and signal strength
        if signal_type.startswith("strong"):
            size_pct = 0.08  # 8%
        else:
            size_pct = 0.05  # 5%

        size_pct *= confidence  # Scale by confidence
        position_value = balance * size_pct
        quantity = int(position_value / price) if price > 0 else 0

        if quantity < 1:
            return

        # Check if selling existing position
        if side == "sell":
            positions = await self.redis.hget(f"positions:{account_id}", ticker)
            if positions:
                pos = json.loads(positions)
                quantity = min(quantity, pos.get("quantity", 0))
                if quantity < 1:
                    return
            else:
                return  # No position to sell

        # Send to risk manager then execution
        decision = {
            "account_id": account_id,
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "price": price,
            "signal": signal_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }

        # Publish decision
        await self.redis.xadd("stream:decisions", {
            k: str(v) for k, v in decision.items()
        }, maxlen=10000)

        logger.info(f"🎯 Decision: {side.upper()} {ticker} x{quantity} for {account_id}")

    async def set_rule(self, rule: RoutingRule) -> dict:
        """Set routing rule for account"""
        self.rules[rule.account_id] = rule
        await self._save_rule(rule)
        logger.info(f"📝 Rule updated for {rule.account_id}")
        return rule.model_dump()

    async def get_rule(self, account_id: str) -> dict | None:
        """Get routing rule"""
        rule = self.rules.get(account_id)
        return rule.model_dump() if rule else None

    async def get_all_rules(self) -> list:
        """Get all routing rules"""
        return [rule.model_dump() for rule in self.rules.values()]

    async def delete_rule(self, account_id: str) -> dict:
        """Delete routing rule"""
        if account_id in self.rules:
            del self.rules[account_id]
            await self.redis.delete(f"routing_rule:{account_id}")
        return {"deleted": account_id}


# Initialize
svc = DecisionRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(
    title="Decision Router",
    description="Route signals to accounts",
    version="1.0.0",
    lifespan=lifespan
)
# ============================================================
# METRICS ENDPOINT (fixed - no 307 redirects)
# ============================================================
@app.get("/metrics")
@app.get("/metrics/")
async def prometheus_metrics():
    from fastapi import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# OLD: metrics mount removed


@app.get("/health")
async def health():
    return {"status": "healthy", "rules_count": len(svc.rules)}


@app.get("/rules")
async def list_rules():
    """List all routing rules"""
    return await svc.get_all_rules()


@app.get("/rules/{account_id}")
async def get_rule(account_id: str):
    """Get rule for account"""
    rule = await svc.get_rule(account_id)
    if rule:
        return rule
    return {"error": "Rule not found"}


@app.post("/rules")
async def set_rule(rule: RoutingRule):
    """Set routing rule"""
    return await svc.set_rule(rule)


@app.delete("/rules/{account_id}")
async def delete_rule(account_id: str):
    """Delete routing rule"""
    return await svc.delete_rule(account_id)


@app.post("/sync")
async def sync_accounts():
    """Sync rules with accounts"""
    await svc._sync_accounts()
    return {"synced": True, "rules_count": len(svc.rules)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)
