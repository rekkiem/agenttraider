"""
AgentTrader CL — Gestión de Riesgo
Validación de señales, VaR de cartera y límites operacionales.
"""

import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
from datetime import date
from loguru import logger

from config import DB_CONFIG, RISK_PARAMS
from engine.decision import TradingSignal


class RiskManager:
    """
    Capa de validación de riesgo que actúa como guardián
    antes de que cualquier señal sea aprobada para operar.

    Checks implementados:
    - Límite de pérdida diaria
    - Límite de pérdida mensual
    - Riesgo máximo por operación (5% capital)
    - Liquidez: no superar 33% del volumen diario
    - Concentración sectorial
    - Probabilidad mínima de señal
    """

    SECTOR_MAP = {
        "AGUAS-A":   "utilities",
        "ENJOY":     "consumer",
        "LTM":       "transportation",
        "SOCOVESA":  "real_estate",
        "VAPORES":   "transportation",
        "FALABELLA": "retail",
        "CENCOSUD":  "retail",
        "RIPLEY":    "retail",
        "COPEC":     "energy",
        "ENELCHILE": "utilities",
        "COLBUN":    "utilities",
        "CMPC":      "materials",
        "SQM-B":     "materials",
        "BCI":       "financials",
        "CHILE":     "financials",
        "SANTANDER": "financials",
        "ITAUCL":    "financials",
    }

    def __init__(
        self,
        portfolio_value: float,
        params: dict = None,
    ):
        self.portfolio_value = portfolio_value
        self.params          = params or RISK_PARAMS
        self.daily_pnl       = self._load_daily_pnl()
        self.monthly_pnl     = self._load_monthly_pnl()

    # ── VALIDACIÓN PRINCIPAL ──────────────────────────────────────

    def validate_signal(
        self,
        signal: TradingSignal,
        avg_daily_volume: float,
        current_sector_exposure: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Ejecuta todos los checks de riesgo sobre una señal.
        Retorna (aprobado: bool, razón: str).

        Si se aprueba con tamaño reducido (por liquidez),
        el signal.suggested_size ya viene modificado.
        """

        # Check 1: Límite de pérdida diaria
        max_daily_loss = -self.params["max_daily_loss_pct"] * self.portfolio_value
        if self.daily_pnl < max_daily_loss:
            return False, "DAILY_LOSS_LIMIT_REACHED"

        # Check 2: Límite de pérdida mensual
        max_monthly_loss = -self.params["max_monthly_loss_pct"] * self.portfolio_value
        if self.monthly_pnl < max_monthly_loss:
            return False, "MONTHLY_LOSS_LIMIT_REACHED"

        # Check 3: Probabilidad mínima
        min_prob = self.params["min_signal_probability"]
        if signal.signal in ("BUY", "SELL") and signal.probability < min_prob:
            return False, f"LOW_PROBABILITY_{signal.probability:.1%}"

        # Check 4: Riesgo por operación
        if signal.risk_pct > self.params["max_position_pct"]:
            return False, f"POSITION_RISK_EXCEEDED_{signal.risk_pct:.1%}"

        # Check 5: Liquidez (crítico en Chile)
        if avg_daily_volume > 0 and signal.suggested_size > 0:
            order_value      = signal.suggested_size * signal.current_price
            max_order_value  = avg_daily_volume / self.params["min_liquidity_ratio"]

            if order_value > max_order_value:
                max_shares = int(max_order_value / signal.current_price)
                if max_shares <= 0:
                    return False, "INSUFFICIENT_LIQUIDITY"

                logger.warning(
                    f"{signal.ticker}: Orden reducida por liquidez "
                    f"{signal.suggested_size:,} → {max_shares:,} acciones"
                )
                signal.suggested_size = max_shares

                # Recalcular riesgo con nuevo tamaño
                risk_per_share    = abs(signal.current_price - signal.stop_loss)
                signal.risk_pct   = (risk_per_share * max_shares) / self.portfolio_value

        # Check 6: Orden vacía después de ajuste liquidez
        if signal.suggested_size <= 0:
            return False, "ZERO_SIZE_AFTER_LIQUIDITY_ADJUSTMENT"

        # Check 7: Concentración sectorial
        sector = self.SECTOR_MAP.get(signal.ticker, "other")
        new_sector_exp = current_sector_exposure + signal.risk_pct
        if new_sector_exp > self.params["max_sector_pct"]:
            return False, f"SECTOR_LIMIT_{sector}_{new_sector_exp:.1%}"

        return True, "APPROVED"

    # ── VaR CARTERA ───────────────────────────────────────────────

    def compute_portfolio_var(
        self,
        positions: dict,    # {ticker: shares}
        price_data: dict,   # {ticker: pd.Series de precios close}
        confidence: float = 0.95,
    ) -> dict:
        """
        VaR paramétrico de la cartera completa usando correlaciones reales.
        Retorna volatilidad, VaR diario en CLP y Sharpe.
        """
        returns_dict = {}
        for ticker, series in price_data.items():
            if ticker in positions and positions[ticker] > 0:
                ret = series.pct_change().dropna()
                if len(ret) >= 20:
                    returns_dict[ticker] = ret

        if not returns_dict:
            return {"error": "Sin datos para calcular VaR"}

        returns_df = pd.DataFrame(returns_dict).dropna()
        total_value = sum(
            positions.get(t, 0) * price_data[t].iloc[-1]
            for t in returns_dict
            if t in price_data
        )

        if total_value <= 0:
            return {"error": "Valor de cartera = 0"}

        weights = np.array([
            positions.get(t, 0) * price_data[t].iloc[-1] / total_value
            for t in returns_df.columns
        ])

        cov_matrix   = returns_df.cov() * 252
        port_vol_ann = float(np.sqrt(weights @ cov_matrix.values @ weights))
        port_ret_ann = float((returns_df @ weights).mean() * 252)
        port_returns = (returns_df @ weights)
        var_daily    = float(np.percentile(port_returns.values, (1 - confidence) * 100))
        sharpe       = (port_ret_ann - 0.06) / port_vol_ann if port_vol_ann > 0 else 0.0

        return {
            "portfolio_vol_ann":      round(port_vol_ann, 4),
            "portfolio_return_ann":   round(port_ret_ann, 4),
            "var_95_daily_pct":       round(var_daily, 4),
            "var_95_daily_clp":       round(var_daily * self.portfolio_value, 0),
            "sharpe_portfolio":       round(sharpe, 3),
            "confidence":             confidence,
            "tickers_included":       list(returns_df.columns),
        }

    # ── HELPERS PRIVADOS ──────────────────────────────────────────

    def _load_daily_pnl(self) -> float:
        """Carga P&L del día desde DB. Retorna 0 si no hay datos."""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur  = conn.cursor()
            cur.execute(
                """
                SELECT COALESCE(SUM(daily_pnl_clp), 0)
                FROM risk_metrics
                WHERE calculated_at::date = CURRENT_DATE
                """
            )
            result = cur.fetchone()[0]
            cur.close()
            conn.close()
            return float(result)
        except Exception:
            return 0.0

    def _load_monthly_pnl(self) -> float:
        """Carga P&L del mes desde DB."""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur  = conn.cursor()
            cur.execute(
                """
                SELECT COALESCE(SUM(daily_pnl_clp), 0)
                FROM risk_metrics
                WHERE DATE_TRUNC('month', calculated_at) = DATE_TRUNC('month', CURRENT_DATE)
                """
            )
            result = cur.fetchone()[0]
            cur.close()
            conn.close()
            return float(result)
        except Exception:
            return 0.0

    def save_risk_metrics(self, metrics: dict):
        """Persiste métricas de riesgo en DB."""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur  = conn.cursor()
            cur.execute(
                """
                INSERT INTO risk_metrics
                    (portfolio_vol_ann, portfolio_return_ann,
                     var_95_daily_clp, sharpe_portfolio, daily_pnl_clp)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    metrics.get("portfolio_vol_ann"),
                    metrics.get("portfolio_return_ann"),
                    metrics.get("var_95_daily_clp"),
                    metrics.get("sharpe_portfolio"),
                    metrics.get("daily_pnl_clp", 0),
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error guardando métricas de riesgo: {e}")
