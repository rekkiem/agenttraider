-- ─────────────────────────────────────────────────────────────
-- AgentTrader CL — Schema PostgreSQL
-- Ejecutar una vez al inicializar la base de datos.
-- Requiere extensión TimescaleDB (opcional pero recomendado).
-- ─────────────────────────────────────────────────────────────

-- Intentar cargar TimescaleDB (continuar si no está disponible)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'TimescaleDB no disponible, continuando sin él';
END;
$$;

-- ── PRECIOS RAW (scraping diario) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS prices_raw (
    ticker      VARCHAR(20)    NOT NULL,
    last_price  NUMERIC(14,2),
    change_pct  NUMERIC(8,4),
    bid         NUMERIC(14,2),
    ask         NUMERIC(14,2),
    volume      BIGINT         DEFAULT 0,
    timestamp   TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, timestamp)
);

-- Convertir a hypertable si TimescaleDB está disponible
DO $$
BEGIN
    PERFORM create_hypertable('prices_raw', 'timestamp', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Hypertable no creada (TimescaleDB no disponible)';
END;
$$;

CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices_raw (ticker, timestamp DESC);

-- ── ESTADO DEL PORTAFOLIO ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_state (
    ticker       VARCHAR(20)   PRIMARY KEY,
    shares       INTEGER       NOT NULL DEFAULT 0,
    avg_cost     NUMERIC(14,2),
    last_update  TIMESTAMPTZ   DEFAULT NOW()
);

-- Cargar cartera inicial
INSERT INTO portfolio_state (ticker, shares) VALUES
    ('AGUAS-A',   1000),
    ('ENJOY',     150000),
    ('LTM',       300),
    ('SOCOVESA',  5000),
    ('VAPORES',   1570)
ON CONFLICT (ticker) DO NOTHING;

-- ── SEÑALES GENERADAS ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals_log (
    id               BIGSERIAL      PRIMARY KEY,
    ticker           VARCHAR(20)    NOT NULL,
    signal           VARCHAR(10)    NOT NULL,  -- BUY / SELL / HOLD
    probability      NUMERIC(6,4),
    confidence       VARCHAR(10),
    suggested_size   INTEGER,
    current_price    NUMERIC(14,2),
    stop_loss        NUMERIC(14,2),
    take_profit      NUMERIC(14,2),
    risk_pct         NUMERIC(8,6),
    expected_return  NUMERIC(8,4),
    approved         BOOLEAN        DEFAULT FALSE,
    rejection_reason VARCHAR(100),
    reasoning        JSONB,
    email_sent       BOOLEAN        DEFAULT FALSE,
    created_at       TIMESTAMPTZ    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker   ON signals_log (ticker, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_created  ON signals_log (created_at DESC);

-- ── PREDICCIONES POR MODELO ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_predictions (
    id           BIGSERIAL    PRIMARY KEY,
    ticker       VARCHAR(20)  NOT NULL,
    model_name   VARCHAR(50)  NOT NULL,
    prediction   NUMERIC(6,4),
    auc_score    NUMERIC(6,4),
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

-- ── MÉTRICAS DE RIESGO CARTERA ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS risk_metrics (
    id                    BIGSERIAL    PRIMARY KEY,
    portfolio_vol_ann     NUMERIC(8,4),
    portfolio_return_ann  NUMERIC(8,4),
    var_95_daily_clp      NUMERIC(16,0),
    sharpe_portfolio      NUMERIC(8,4),
    daily_pnl_clp         NUMERIC(16,0),
    calculated_at         TIMESTAMPTZ  DEFAULT NOW()
);

-- ── LOGS DE EJECUCIÓN ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS execution_log (
    id           BIGSERIAL    PRIMARY KEY,
    run_date     DATE         NOT NULL,
    status       VARCHAR(20)  NOT NULL,  -- SUCCESS / PARTIAL / FAILED
    tickers_ok   INTEGER      DEFAULT 0,
    tickers_fail INTEGER      DEFAULT 0,
    signals_buy  INTEGER      DEFAULT 0,
    signals_sell INTEGER      DEFAULT 0,
    signals_hold INTEGER      DEFAULT 0,
    duration_sec NUMERIC(8,2),
    error_msg    TEXT,
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

-- ── VISTA ÚTIL: ÚLTIMA SEÑAL POR TICKER ──────────────────────────────
CREATE OR REPLACE VIEW v_latest_signals AS
SELECT DISTINCT ON (ticker)
    ticker, signal, probability, confidence,
    suggested_size, current_price, stop_loss,
    take_profit, risk_pct, expected_return,
    approved, created_at
FROM signals_log
ORDER BY ticker, created_at DESC;

COMMENT ON TABLE prices_raw       IS 'Precios diarios obtenidos por scraping de Security';
COMMENT ON TABLE portfolio_state  IS 'Estado actual del portafolio del usuario';
COMMENT ON TABLE signals_log      IS 'Historial de señales generadas por el agente';
COMMENT ON TABLE model_predictions IS 'Predicciones individuales por modelo ML';
COMMENT ON TABLE risk_metrics     IS 'Métricas de riesgo calculadas diariamente';
