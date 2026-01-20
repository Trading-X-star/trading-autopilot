-- ============================================================
-- TRADING AUTOPILOT DATABASE INIT v2
-- PostgreSQL 16+
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- MARKET DATA
-- ============================================================
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(20,4),
    high DECIMAL(20,4),
    low DECIMAL(20,4),
    close DECIMAL(20,4),
    volume BIGINT,
    value DECIMAL(20,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_market_ticker ON market_data(ticker);
CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(date DESC);
CREATE INDEX IF NOT EXISTS idx_market_ticker_date ON market_data(ticker, date DESC);

-- ============================================================
-- FEATURES (ML)
-- ============================================================
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    
    -- Price data
    open DECIMAL(20,4),
    high DECIMAL(20,4),
    low DECIMAL(20,4),
    close DECIMAL(20,4),
    volume BIGINT,
    
    -- Moving averages
    sma_5 DECIMAL(20,4),
    sma_10 DECIMAL(20,4),
    sma_20 DECIMAL(20,4),
    sma_50 DECIMAL(20,4),
    sma_200 DECIMAL(20,4),
    ema_12 DECIMAL(20,4),
    ema_26 DECIMAL(20,4),
    
    -- Indicators
    rsi_14 DECIMAL(10,4),
    macd DECIMAL(20,6),
    macd_signal DECIMAL(20,6),
    macd_hist DECIMAL(20,6),
    bb_upper DECIMAL(20,4),
    bb_middle DECIMAL(20,4),
    bb_lower DECIMAL(20,4),
    bb_width DECIMAL(10,6),
    bb_pct DECIMAL(10,6),
    atr_14 DECIMAL(20,4),
    adx_14 DECIMAL(10,4),
    
    -- Returns
    return_1d DECIMAL(10,6),
    return_5d DECIMAL(10,6),
    return_10d DECIMAL(10,6),
    return_20d DECIMAL(10,6),
    
    -- Volatility
    volatility_20 DECIMAL(10,6),
    
    -- Volume
    volume_sma_20 DECIMAL(20,2),
    volume_ratio DECIMAL(10,4),
    
    -- Position
    pct_from_high DECIMAL(10,6),
    pct_from_low DECIMAL(10,6),
    
    -- Target (for training)
    target_1d DECIMAL(10,6),
    target_5d DECIMAL(10,6),
    signal_class SMALLINT,  -- -1=sell, 0=hold, 1=buy
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_features_ticker ON features(ticker);
CREATE INDEX IF NOT EXISTS idx_features_date ON features(date DESC);
CREATE INDEX IF NOT EXISTS idx_features_ticker_date ON features(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_features_signal ON features(signal_class) WHERE signal_class IS NOT NULL;

-- ============================================================
-- MACRO DATA (CBR)
-- ============================================================
CREATE TABLE IF NOT EXISTS macro_daily (
    date DATE PRIMARY KEY,
    usd_rate DECIMAL(10,4),
    eur_rate DECIMAL(10,4),
    cny_rate DECIMAL(10,4),
    key_rate DECIMAL(6,2),
    usd_change_1d DECIMAL(10,6),
    usd_change_5d DECIMAL(10,6),
    usd_change_20d DECIMAL(10,6),
    usd_volatility DECIMAL(10,6),
    eur_change_5d DECIMAL(10,6),
    rate_change DECIMAL(6,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_daily(date DESC);

-- ============================================================
-- NEWS DATA (MOEX)
-- ============================================================
CREATE TABLE IF NOT EXISTS news (
    id VARCHAR(50) PRIMARY KEY,
    title TEXT NOT NULL,
    published TIMESTAMPTZ NOT NULL,
    category VARCHAR(50),
    sentiment DECIMAL(5,4),
    url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news(published DESC);
CREATE INDEX IF NOT EXISTS idx_news_title_trgm ON news USING gin(title gin_trgm_ops);

CREATE TABLE IF NOT EXISTS news_tickers (
    news_id VARCHAR(50) REFERENCES news(id) ON DELETE CASCADE,
    ticker VARCHAR(20) NOT NULL,
    PRIMARY KEY (news_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_news_tickers_ticker ON news_tickers(ticker);

CREATE TABLE IF NOT EXISTS news_sentiment_daily (
    date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    sentiment_avg DECIMAL(5,4),
    sentiment_sum DECIMAL(8,4),
    news_count INTEGER,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_daily ON news_sentiment_daily(date DESC, ticker);

-- ============================================================
-- TRADING ACCOUNTS
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_accounts (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    broker VARCHAR(50) DEFAULT 'tinkoff',
    account_type VARCHAR(20) DEFAULT 'sandbox',  -- sandbox, live
    status VARCHAR(20) DEFAULT 'active',         -- active, suspended, closed
    risk_profile VARCHAR(20) DEFAULT 'balanced', -- conservative, balanced, aggressive
    initial_balance DECIMAL(20,2),
    current_balance DECIMAL(20,2),
    currency VARCHAR(10) DEFAULT 'RUB',
    risk_limits JSONB DEFAULT '{
        "max_position_pct": 0.10,
        "max_daily_loss_pct": 0.02,
        "max_drawdown_pct": 0.08,
        "max_open_positions": 10
    }'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- POSITIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) REFERENCES trading_accounts(id),
    ticker VARCHAR(20) NOT NULL,
    quantity INTEGER DEFAULT 0,
    avg_price DECIMAL(20,4),
    current_price DECIMAL(20,4),
    market_value DECIMAL(20,2),
    unrealized_pnl DECIMAL(20,2),
    unrealized_pnl_pct DECIMAL(10,4),
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(account_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_positions_account ON positions(account_id);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);

-- ============================================================
-- TRADES
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(100) UNIQUE,
    account_id VARCHAR(50) REFERENCES trading_accounts(id),
    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- buy, sell
    quantity INTEGER NOT NULL,
    price DECIMAL(20,4) NOT NULL,
    commission DECIMAL(20,4) DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'RUB',
    
    -- Execution details
    order_id VARCHAR(100),
    source VARCHAR(50),  -- signal, manual, stop_loss, take_profit, trailing_stop
    signal_confidence DECIMAL(5,4),
    
    -- Result (for closed positions)
    realized_pnl DECIMAL(20,2),
    realized_pnl_pct DECIMAL(10,4),
    
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);

-- ============================================================
-- ORDERS
-- ============================================================
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) UNIQUE NOT NULL,
    account_id VARCHAR(50) REFERENCES trading_accounts(id),
    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- buy, sell
    order_type VARCHAR(20) DEFAULT 'market',  -- market, limit, stop
    quantity INTEGER NOT NULL,
    price DECIMAL(20,4),
    stop_price DECIMAL(20,4),
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending',  -- pending, submitted, filled, partially_filled, cancelled, rejected
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price DECIMAL(20,4),
    
    -- Source
    source VARCHAR(50),  -- signal, manual, orchestrator
    signal_confidence DECIMAL(5,4),
    reason TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_orders_account ON orders(account_id);
CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at DESC);

-- ============================================================
-- SIGNALS
-- ============================================================
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    signal VARCHAR(10) NOT NULL,  -- BUY, SELL, HOLD
    confidence DECIMAL(5,4) NOT NULL,
    model_version VARCHAR(50),
    
    -- Features snapshot
    rsi_14 DECIMAL(10,4),
    macd_hist DECIMAL(20,6),
    price DECIMAL(20,4),
    volume_ratio DECIMAL(10,4),
    
    -- Processing
    processed BOOLEAN DEFAULT FALSE,
    order_id VARCHAR(100),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_unprocessed ON signals(processed) WHERE processed = FALSE;

-- ============================================================
-- STOP ORDERS (Trailing, Stop-Loss, Take-Profit)
-- ============================================================
CREATE TABLE IF NOT EXISTS stop_orders (
    id SERIAL PRIMARY KEY,
    stop_id VARCHAR(100) UNIQUE NOT NULL,
    account_id VARCHAR(50) REFERENCES trading_accounts(id),
    ticker VARCHAR(20) NOT NULL,
    
    -- Type
    stop_type VARCHAR(20) NOT NULL,  -- stop_loss, take_profit, trailing_stop
    
    -- Prices
    entry_price DECIMAL(20,4) NOT NULL,
    stop_price DECIMAL(20,4) NOT NULL,
    trigger_price DECIMAL(20,4),  -- For trailing: activation price
    highest_price DECIMAL(20,4),  -- For trailing: tracked high
    trail_pct DECIMAL(6,4),       -- For trailing: trail percentage
    
    -- Config
    quantity INTEGER NOT NULL,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',  -- active, triggered, cancelled, expired
    triggered_at TIMESTAMPTZ,
    triggered_price DECIMAL(20,4),
    order_id VARCHAR(100),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    
    UNIQUE(account_id, ticker, stop_type)
);

CREATE INDEX IF NOT EXISTS idx_stop_orders_account ON stop_orders(account_id);
CREATE INDEX IF NOT EXISTS idx_stop_orders_ticker ON stop_orders(ticker);
CREATE INDEX IF NOT EXISTS idx_stop_orders_active ON stop_orders(status) WHERE status = 'active';

-- ============================================================
-- ORCHESTRATOR TABLES
-- ============================================================
CREATE TABLE IF NOT EXISTS orchestrator_decisions (
    id SERIAL PRIMARY KEY,
    decision_type VARCHAR(50) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    market_state JSONB,
    portfolio_state JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orch_decisions_date ON orchestrator_decisions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orch_decisions_type ON orchestrator_decisions(decision_type);

CREATE TABLE IF NOT EXISTS position_actions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    action VARCHAR(30) NOT NULL,  -- STOP_LOSS, TAKE_PROFIT, TRAILING_STOP, NEWS_EXIT, etc.
    reason TEXT,
    entry_price DECIMAL(20,4),
    exit_price DECIMAL(20,4),
    quantity INTEGER,
    pnl DECIMAL(20,4),
    pnl_pct DECIMAL(10,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pos_actions_date ON position_actions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pos_actions_ticker ON position_actions(ticker);
CREATE INDEX IF NOT EXISTS idx_pos_actions_action ON position_actions(action);

CREATE TABLE IF NOT EXISTS filtered_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    signal VARCHAR(10) NOT NULL,
    original_confidence DECIMAL(5,4),
    adjusted_confidence DECIMAL(5,4),
    filters_passed JSONB,
    filters_failed JSONB,
    forwarded BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_filtered_signals_date ON filtered_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_filtered_signals_forwarded ON filtered_signals(forwarded);

-- ============================================================
-- PORTFOLIO SNAPSHOTS
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50),
    total_value DECIMAL(20,4),
    cash DECIMAL(20,4),
    invested DECIMAL(20,4),
    positions_count INTEGER,
    daily_return DECIMAL(10,6),
    cumulative_return DECIMAL(10,6),
    drawdown DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,4),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_ts ON portfolio_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_account ON portfolio_snapshots(account_id, timestamp DESC);

-- ============================================================
-- SCHEDULER & JOBS
-- ============================================================
CREATE TABLE IF NOT EXISTS scheduler_jobs (
    id SERIAL PRIMARY KEY,
    job_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,  -- running, success, failed
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    duration_sec INTEGER,
    records_processed INTEGER,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_name_date ON scheduler_jobs(job_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON scheduler_jobs(status);

-- ============================================================
-- MODEL TRAINING LOG
-- ============================================================
CREATE TABLE IF NOT EXISTS model_training_log (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'ensemble',
    accuracy DECIMAL(6,4),
    accuracy_std DECIMAL(6,4),
    precision_score DECIMAL(6,4),
    recall_score DECIMAL(6,4),
    f1_score DECIMAL(6,4),
    n_folds INTEGER,
    n_features INTEGER,
    n_samples INTEGER,
    feature_importance JSONB,
    hyperparameters JSONB,
    training_time_sec INTEGER,
    model_path TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_log_version ON model_training_log(version);
CREATE INDEX IF NOT EXISTS idx_model_log_date ON model_training_log(created_at DESC);

-- ============================================================
-- BACKTEST RESULTS
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    
    -- Period
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    
    -- Config
    initial_capital DECIMAL(20,2),
    config JSONB,
    
    -- Results
    final_value DECIMAL(20,4),
    total_return DECIMAL(10,6),
    annual_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,6),
    max_drawdown_duration INTEGER,  -- days
    win_rate DECIMAL(6,4),
    profit_factor DECIMAL(10,4),
    
    -- Trades
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_trade_return DECIMAL(10,6),
    avg_win DECIMAL(10,6),
    avg_loss DECIMAL(10,6),
    
    -- Details
    equity_curve JSONB,
    trades_log JSONB,
    monthly_returns JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_model ON backtest_results(model_version);

-- ============================================================
-- ALERTS & NOTIFICATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(100) UNIQUE,
    severity VARCHAR(20) NOT NULL,  -- info, warning, error, critical
    category VARCHAR(50),           -- trading, risk, system, market
    title VARCHAR(200) NOT NULL,
    message TEXT,
    source VARCHAR(50),
    metadata JSONB,
    
    -- Status
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    
    -- Delivery
    telegram_sent BOOLEAN DEFAULT FALSE,
    telegram_sent_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_unacked ON alerts(acknowledged) WHERE acknowledged = FALSE;

-- ============================================================
-- RISK EVENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,  -- drawdown_breach, daily_loss_breach, position_limit, etc.
    severity VARCHAR(20) NOT NULL,
    
    -- Details
    ticker VARCHAR(20),
    account_id VARCHAR(50),
    threshold DECIMAL(10,6),
    actual_value DECIMAL(10,6),
    
    -- Action taken
    action_taken VARCHAR(50),  -- mode_change, position_closed, trading_stopped
    action_details JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_events_date ON risk_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_type ON risk_events(event_type);

-- ============================================================
-- SYSTEM STATE
-- ============================================================
CREATE TABLE IF NOT EXISTS system_state (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Initialize system state
INSERT INTO system_state (key, value) VALUES 
    ('trading_enabled', 'false'::jsonb),
    ('trading_mode', '"normal"'::jsonb),
    ('market_regime', '"unknown"'::jsonb),
    ('last_health_check', 'null'::jsonb)
ON CONFLICT (key) DO NOTHING;

-- ============================================================
-- TICKERS METADATA
-- ============================================================
CREATE TABLE IF NOT EXISTS tickers (
    ticker VARCHAR(20) PRIMARY KEY,
    name VARCHAR(200),
    sector VARCHAR(50),
    industry VARCHAR(100),
    market_cap DECIMAL(20,2),
    currency VARCHAR(10) DEFAULT 'RUB',
    lot_size INTEGER DEFAULT 1,
    min_price_increment DECIMAL(10,6),
    isin VARCHAR(20),
    figi VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Initialize default tickers
INSERT INTO tickers (ticker, name, sector) VALUES 
    ('SBER', 'Сбербанк', 'banks'),
    ('GAZP', 'Газпром', 'oil_gas'),
    ('LKOH', 'Лукойл', 'oil_gas'),
    ('GMKN', 'Норникель', 'metals'),
    ('NVTK', 'Новатэк', 'oil_gas'),
    ('ROSN', 'Роснефть', 'oil_gas'),
    ('VTBR', 'ВТБ', 'banks'),
    ('MTSS', 'МТС', 'telecom'),
    ('MGNT', 'Магнит', 'retail'),
    ('TATN', 'Татнефть', 'oil_gas'),
    ('SNGS', 'Сургутнефтегаз', 'oil_gas'),
    ('NLMK', 'НЛМК', 'metals'),
    ('ALRS', 'Алроса', 'metals'),
    ('CHMF', 'Северсталь', 'metals'),
    ('MAGN', 'ММК', 'metals'),
    ('PLZL', 'Полюс', 'metals'),
    ('YNDX', 'Яндекс', 'tech'),
    ('POLY', 'Полиметалл', 'metals'),
    ('MOEX', 'Мосбиржа', 'finance'),
    ('FIVE', 'X5 Group', 'retail'),
    ('TCSG', 'Тинькофф', 'banks'),
    ('AFKS', 'АФК Система', 'holdings'),
    ('IRAO', 'Интер РАО', 'energy'),
    ('HYDR', 'РусГидро', 'energy'),
    ('PHOR', 'ФосАгро', 'chemicals'),
    ('RUAL', 'Русал', 'metals'),
    ('AFLT', 'Аэрофлот', 'transport'),
    ('CBOM', 'МКБ', 'banks'),
    ('RTKM', 'Ростелеком', 'telecom'),
    ('FEES', 'ФСК ЕЭС', 'energy')
ON CONFLICT (ticker) DO NOTHING;

-- ============================================================
-- FUNCTIONS
-- ============================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at
DO $$
DECLARE
    t text;
BEGIN
    FOR t IN 
        SELECT table_name 
        FROM information_schema.columns 
        WHERE column_name = 'updated_at' 
          AND table_schema = 'public'
    LOOP
        EXECUTE format('
            DROP TRIGGER IF EXISTS update_%I_updated_at ON %I;
            CREATE TRIGGER update_%I_updated_at
            BEFORE UPDATE ON %I
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        ', t, t, t, t);
    END LOOP;
END;
$$;

-- Function to calculate portfolio metrics
CREATE OR REPLACE FUNCTION calc_portfolio_metrics(p_account_id VARCHAR)
RETURNS TABLE (
    total_value DECIMAL,
    cash DECIMAL,
    invested DECIMAL,
    unrealized_pnl DECIMAL,
    positions_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(a.current_balance, 0) + COALESCE(SUM(p.market_value), 0) as total_value,
        COALESCE(a.current_balance, 0) as cash,
        COALESCE(SUM(p.market_value), 0) as invested,
        COALESCE(SUM(p.unrealized_pnl), 0) as unrealized_pnl,
        COUNT(p.id)::INTEGER as positions_count
    FROM trading_accounts a
    LEFT JOIN positions p ON a.id = p.account_id AND p.quantity > 0
    WHERE a.id = p_account_id
    GROUP BY a.id, a.current_balance;
END;
$$ LANGUAGE plpgsql;

-- Function to get daily PnL
CREATE OR REPLACE FUNCTION get_daily_pnl(p_account_id VARCHAR, p_date DATE DEFAULT CURRENT_DATE)
RETURNS DECIMAL AS $$
DECLARE
    v_pnl DECIMAL;
BEGIN
    SELECT COALESCE(SUM(
        CASE WHEN side = 'sell' THEN price * quantity
             ELSE -price * quantity END
    ), 0) INTO v_pnl
    FROM trades
    WHERE account_id = p_account_id
      AND DATE(timestamp) = p_date;
    
    RETURN v_pnl;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- VIEWS
-- ============================================================

-- Active positions view
CREATE OR REPLACE VIEW v_active_positions AS
SELECT 
    p.*,
    t.name as ticker_name,
    t.sector,
    ROUND(p.unrealized_pnl_pct * 100, 2) as pnl_pct_display
FROM positions p
JOIN tickers t ON p.ticker = t.ticker
WHERE p.quantity > 0;

-- Recent trades view
CREATE OR REPLACE VIEW v_recent_trades AS
SELECT 
    tr.*,
    t.name as ticker_name,
    t.sector
FROM trades tr
JOIN tickers t ON tr.ticker = t.ticker
ORDER BY tr.timestamp DESC
LIMIT 100;

-- Daily summary view
CREATE OR REPLACE VIEW v_daily_summary AS
SELECT 
    DATE(timestamp) as trade_date,
    account_id,
    COUNT(*) as trades_count,
    SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as buys,
    SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as sells,
    SUM(price * quantity) as volume,
    SUM(CASE WHEN side = 'sell' THEN price * quantity ELSE -price * quantity END) as net_pnl
FROM trades
GROUP BY DATE(timestamp), account_id
ORDER BY trade_date DESC;

-- Model performance view
CREATE OR REPLACE VIEW v_model_performance AS
SELECT 
    version,
    accuracy,
    accuracy_std,
    n_samples,
    n_features,
    training_time_sec,
    created_at,
    RANK() OVER (ORDER BY accuracy DESC) as accuracy_rank
FROM model_training_log
ORDER BY created_at DESC;

-- ============================================================
-- CLEANUP & MAINTENANCE
-- ============================================================

-- Partition old data (optional, for large datasets)
-- CREATE TABLE market_data_2024 PARTITION OF market_data FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Vacuum and analyze
ANALYZE market_data;
ANALYZE features;
ANALYZE trades;
ANALYZE positions;

-- ============================================================
-- GRANTS (if using separate roles)
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading;

-- ============================================================
-- DONE
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE 'Trading Autopilot database initialized successfully!';
    RAISE NOTICE 'Tables: %', (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public');
    RAISE NOTICE 'Indexes: %', (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public');
END;
$$;
