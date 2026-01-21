-- Batch price update for multiple tickers
-- KEYS = ticker keys (price:TICKER)
-- ARGV = prices (JSON array)

local prices = cjson.decode(ARGV[1])
local updated = 0

for i, key in ipairs(KEYS) do
    local price = prices[i]
    if price then
        redis.call('SET', key, cjson.encode(price))
        redis.call('EXPIRE', key, 60)  -- 1 minute TTL
        updated = updated + 1
    end
end

return updated
