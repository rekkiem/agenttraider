"""
AgentTrader CL — Configuración Central
Edita este archivo con tus datos antes de desplegar.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── BASE DE DATOS ─────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "agenttrader"),
    "user":     os.getenv("DB_USER", "trader"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# ── EMAIL ─────────────────────────────────────────────────────────────
EMAIL_CONFIG = {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "user":      os.getenv("GMAIL_USER", ""),
    "password":  os.getenv("GMAIL_APP_PASS", ""),  # Google App Password
    "to":        os.getenv("ALERT_EMAIL", "perfumes.arhom@gmail.com"),
}

# ── CARTERA ACTIVA ────────────────────────────────────────────────────
# avg_cost en CLP por acción. Completar con tus precios reales de compra.
PORTFOLIO = {
    "AGUAS-A":  {"shares": 1_000,   "avg_cost": None},
    "ENJOY":    {"shares": 150_000, "avg_cost": None},
    "LTM":      {"shares": 300,     "avg_cost": None},
    "SOCOVESA": {"shares": 5_000,   "avg_cost": None},
    "VAPORES":  {"shares": 1_570,   "avg_cost": None},
}

# ── VALOR TOTAL CARTERA EN CLP ────────────────────────────────────────
# Actualizar manualmente o conectar a tu broker
PORTFOLIO_VALUE_CLP = float(os.getenv("PORTFOLIO_VALUE_CLP", "10000000"))

# ── UNIVERSO DE ANÁLISIS ──────────────────────────────────────────────
WATCHLIST = [
    "AGUAS-A", "ENJOY", "LTM", "SOCOVESA", "VAPORES",
    "FALABELLA", "ENELCHILE", "COPEC", "CHILE", "BCI",
    "CENCOSUD", "CMPC", "COLBUN", "RIPLEY", "CCU",
]

# ── PARÁMETROS DE RIESGO ──────────────────────────────────────────────
RISK_PARAMS = {
    "max_position_pct":      0.05,   # Máx 5% del capital por operación
    "max_sector_pct":        0.25,   # Máx 25% en un sector
    "max_daily_loss_pct":    0.02,   # Stop diario en -2%
    "max_monthly_loss_pct":  0.07,   # Stop mensual en -7%
    "min_liquidity_ratio":   3.0,    # Orden máx = 33% del vol. diario
    "min_signal_probability": 0.55,  # Umbral mínimo para operar
    "atr_stop_multiplier":   2.0,    # Stop loss = 2x ATR
    "atr_tp_multiplier":     3.5,    # Take profit = 3.5x ATR
}

# ── SCHEDULER ─────────────────────────────────────────────────────────
SCHEDULER = {
    "analysis_hour":   15,
    "analysis_minute": 45,
    "retrain_day":     "sat",
    "retrain_hour":    8,
    "timezone":        "America/Santiago",
}

# ── MODELOS ───────────────────────────────────────────────────────────
MODEL_PATH = "saved_models/ensemble_latest.pkl"
FEATURE_COLS = [
    "rsi_14", "macd", "macd_hist", "bb_pct", "bb_width",
    "atr_14", "volume_ratio", "dist_support", "dist_resist",
    "momentum_score", "return_5d", "return_10d", "return_20d",
    "vol_20d_ann", "vol_regime", "trend_short", "trend_long",
    "golden_cross", "obv",
]

# ── SCRAPER ───────────────────────────────────────────────────────────
SECURITY_URL = "https://mercadosenlinea.inversionessecurity.cl/www/resumen.html"
SCRAPER_TIMEOUT_MS = 30_000
