#!/usr/bin/env python3
"""Pair Trading API - добавляется к strategy"""
import json
import redis

r = redis.from_url("redis://redis:6379/0", decode_responses=True)

from pair_trading import pair_strategy, PAIRS

async def get_prices():
    prices = {}
    for ta, tb in PAIRS:
        for t in [ta, tb]:
            data = r.zrevrange(f"history:{t}", 0, 100)
            if data:
                prices[t] = [json.loads(d)["price"] for d in data][::-1]
    return prices

def scan_pairs(prices):
    return pair_strategy.get_signals(prices)
