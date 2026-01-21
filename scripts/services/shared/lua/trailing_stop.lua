-- Atomic trailing stop update
-- KEYS[1] = stop key
-- ARGV[1] = current_price
-- ARGV[2] = trailing_pct
-- ARGV[3] = breakeven_trigger_pct
-- ARGV[4] = breakeven_offset_pct

local stop_data = redis.call('GET', KEYS[1])
if not stop_data then
    return nil
end

local stop = cjson.decode(stop_data)
local current_price = tonumber(ARGV[1])
local trailing_pct = tonumber(ARGV[2])
local breakeven_trigger = tonumber(ARGV[3])
local breakeven_offset = tonumber(ARGV[4])

local entry_price = tonumber(stop.entry_price)
local highest_price = tonumber(stop.highest_price) or entry_price
local stop_price = tonumber(stop.stop_price)
local breakeven_active = stop.breakeven_active or false

local updated = false
local triggered = false
local trigger_type = nil

-- Update highest price
if current_price > highest_price then
    highest_price = current_price
    stop.highest_price = highest_price
    updated = true
end

-- Calculate new trailing stop
local new_stop = highest_price * (1 - trailing_pct / 100)
if new_stop > stop_price then
    stop.stop_price = new_stop
    stop_price = new_stop
    updated = true
end

-- Check breakeven activation
local profit_pct = ((current_price / entry_price) - 1) * 100
if not breakeven_active and profit_pct >= breakeven_trigger then
    local breakeven_price = entry_price * (1 + breakeven_offset / 100)
    if breakeven_price > stop_price then
        stop.stop_price = breakeven_price
        stop_price = breakeven_price
        stop.breakeven_active = true
        updated = true
    end
end

-- Check if stop triggered
if current_price <= stop_price then
    triggered = true
    trigger_type = stop.breakeven_active and 'breakeven' or 'trailing'
end

if updated then
    stop.updated_at = redis.call('TIME')[1]
    redis.call('SET', KEYS[1], cjson.encode(stop))
end

return cjson.encode({
    updated = updated,
    triggered = triggered,
    trigger_type = trigger_type,
    stop_price = stop_price,
    highest_price = highest_price,
    profit_pct = profit_pct
})
