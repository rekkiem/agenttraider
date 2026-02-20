"""
AgentTrader CL — Motor de Decisión
Convierte probabilidades del ensemble en señales
BUY / SELL / HOLD con tamaño, stops y justificación.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from config import RISK_PARAMS, PORTFOLIO_VALUE_CLP
from models.predictive import ModelEnsemble, run_prophet_forecast


# ── ESTRUCTURA DE SEÑAL ───────────────────────────────────────────────

@dataclass
class TradingSignal:
    ticker:          str
    signal:          str            # BUY / SELL / HOLD
    probability:     float          # 0.0 – 1.0
    confidence:      str            # LOW / MEDIUM / HIGH
    suggested_size:  int            # Acciones
    current_price:   float          # CLP
    stop_loss:       float          # CLP
    take_profit:     float          # CLP
    risk_pct:        float          # Fracción del portafolio en riesgo
    expected_return: float          # % esperado hasta take profit
    reasoning:       dict = field(default_factory=dict)

    def summary(self) -> str:
        arrow = "▲" if self.signal == "BUY" else ("▼" if self.signal == "SELL" else "━")
        return (
            f"{arrow} {self.ticker} | {self.signal} | P={self.probability:.1%} "
            f"| Size={self.suggested_size:,} | SL=${self.stop_loss:,.0f} "
            f"| TP=${self.take_profit:,.0f} | Conf={self.confidence}"
        )


# ── MOTOR DE DECISIÓN ─────────────────────────────────────────────────

class DecisionEngine:
    """
    Genera señales de trading para un ticker dado.

    Flujo:
    1. Obtiene probabilidad del ensemble ML
    2. Ajusta con dirección de Prophet
    3. Aplica penalización si hay pérdida acumulada alta
    4. Determina señal y tamaño
    5. Calcula stops con ATR
    """

    def __init__(
        self,
        model: ModelEnsemble,
        portfolio_value: float = PORTFOLIO_VALUE_CLP,
        risk_params: dict = None,
    ):
        self.model           = model
        self.portfolio_value = portfolio_value
        self.risk_params     = risk_params or RISK_PARAMS

    def generate_signal(
        self,
        ticker: str,
        df: pd.DataFrame,
        current_price: float,
        portfolio_shares: int = 0,
        avg_cost: Optional[float] = None,
    ) -> TradingSignal:
        """
        Genera una señal completa para el ticker dado.

        Args:
            ticker:           Nemotécnico (ej: 'AGUAS-A')
            df:               DataFrame con features calculados (FeatureEngineer)
            current_price:    Precio actual en CLP
            portfolio_shares: Acciones actualmente en cartera
            avg_cost:         Precio promedio de compra en cartera (opcional)
        """

        # ── 1. PROBABILIDAD ENSEMBLE ──────────────────────────────
        try:
            ensemble_prob = self.model.predict_proba(df)
        except Exception as e:
            logger.error(f"{ticker}: Error en predicción ML: {e}")
            ensemble_prob = 0.5

        # ── 2. PROPHET DIRECTION ──────────────────────────────────
        prophet = run_prophet_forecast(df, periods=5)
        prophet_prob = 0.60 if prophet["prophet_direction"] == 1 else 0.40

        # ── 3. PROBABILIDAD COMBINADA ─────────────────────────────
        if prophet["prophet_available"]:
            raw_prob = ensemble_prob * 0.75 + prophet_prob * 0.25
        else:
            raw_prob = ensemble_prob

        # ── 4. AJUSTE POR P&L NO REALIZADO ───────────────────────
        unrealized_pct = 0.0
        if avg_cost and portfolio_shares > 0 and avg_cost > 0:
            unrealized_pct = (current_price - avg_cost) / avg_cost

        # Penalizar compras adicionales cuando ya estamos en pérdida severa
        buy_penalty = 1.0
        if unrealized_pct < -0.20:
            buy_penalty = 0.75   # -20%: castigar fuerte
        elif unrealized_pct < -0.10:
            buy_penalty = 0.90   # -10%: castigar moderado

        # ── 5. SEÑAL FINAL ────────────────────────────────────────
        min_prob = self.risk_params["min_signal_probability"]

        if raw_prob >= min_prob:
            adjusted_prob = raw_prob * buy_penalty
            signal = "BUY"
        elif raw_prob <= (1 - min_prob):
            adjusted_prob = 1 - raw_prob
            signal = "SELL"
        else:
            adjusted_prob = 0.50
            signal = "HOLD"

        # ── 6. STOPS BASADOS EN ATR ───────────────────────────────
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else current_price * 0.02

        stop_mult = self.risk_params["atr_stop_multiplier"]
        tp_mult   = self.risk_params["atr_tp_multiplier"]

        if signal == "BUY":
            stop_loss   = current_price - stop_mult * atr
            take_profit = current_price + tp_mult   * atr
        elif signal == "SELL":
            stop_loss   = current_price + stop_mult * atr
            take_profit = current_price - tp_mult   * atr
        else:
            stop_loss   = current_price - stop_mult * atr
            take_profit = current_price + tp_mult   * atr

        # ── 7. TAMAÑO DE LA ORDEN ─────────────────────────────────
        suggested_size = self._compute_size(
            signal, adjusted_prob, current_price, stop_loss, portfolio_shares
        )

        # ── 8. MÉTRICAS DE RIESGO ─────────────────────────────────
        risk_per_share = abs(current_price - stop_loss)
        risk_pct = (risk_per_share * suggested_size) / self.portfolio_value if self.portfolio_value > 0 else 0
        expected_return = (take_profit - current_price) / current_price * 100

        # ── 9. CONFIANZA ──────────────────────────────────────────
        if adjusted_prob >= 0.75:
            confidence = "HIGH"
        elif adjusted_prob >= 0.60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # ── 10. REASONING ─────────────────────────────────────────
        last = df.iloc[-1]
        reasoning = {
            "ensemble_prob":      round(ensemble_prob, 4),
            "prophet_direction":  prophet["prophet_direction"],
            "prophet_upside_pct": prophet["prophet_upside_pct"],
            "rsi_14":             round(float(last.get("rsi_14", 50)), 2),
            "macd_hist":          round(float(last.get("macd_hist", 0)), 4),
            "bb_pct":             round(float(last.get("bb_pct", 0.5)), 4),
            "volume_ratio":       round(float(last.get("volume_ratio", 1)), 2),
            "momentum_score":     round(float(last.get("momentum_score", 0)), 4),
            "vol_20d_ann":        round(float(last.get("vol_20d_ann", 0)), 4),
            "trend_short":        int(last.get("trend_short", 0)),
            "trend_long":         int(last.get("trend_long", 0)),
            "atr_14":             round(atr, 2),
            "unrealized_pnl_pct": round(unrealized_pct * 100, 2),
            "buy_penalty":        buy_penalty,
        }

        signal_obj = TradingSignal(
            ticker          = ticker,
            signal          = signal,
            probability     = round(adjusted_prob, 4),
            confidence      = confidence,
            suggested_size  = suggested_size,
            current_price   = round(current_price, 2),
            stop_loss       = round(stop_loss, 2),
            take_profit     = round(take_profit, 2),
            risk_pct        = round(risk_pct, 4),
            expected_return = round(expected_return, 2),
            reasoning       = reasoning,
        )

        logger.info(signal_obj.summary())
        return signal_obj

    def _compute_size(
        self,
        signal: str,
        prob: float,
        price: float,
        stop_loss: float,
        current_shares: int,
    ) -> int:
        """
        Escala el tamaño según probabilidad (Kelly ajustado).

        Probabilidad  → Fracción del máximo permitido
        55–60%        → 25%
        60–75%        → 50%
        75–85%        → 75%
        >85%          → 100%
        """
        max_capital = self.portfolio_value * self.risk_params["max_position_pct"]

        if prob >= 0.85:
            size_frac = 1.00
        elif prob >= 0.75:
            size_frac = 0.75
        elif prob >= 0.60:
            size_frac = 0.50
        else:
            size_frac = 0.25

        capital_to_deploy = max_capital * size_frac
        shares = int(capital_to_deploy / price) if price > 0 else 0

        if signal == "SELL":
            shares = min(shares, current_shares)

        return max(shares, 0)

    def evaluate_portfolio_position(
        self,
        ticker: str,
        current_price: float,
        avg_cost: float,
        shares: int,
        signal_prob: float,
    ) -> dict:
        """
        Evalúa si conviene promediar a la baja, cortar pérdidas o mantener.
        """
        if not avg_cost or avg_cost <= 0:
            return {"recommendation": "UNKNOWN", "reason": "Sin precio promedio"}

        loss_pct = (current_price - avg_cost) / avg_cost
        valor_posicion = shares * current_price

        if loss_pct < -0.25:
            rec    = "CUT_LOSS"
            reason = f"Pérdida severa {loss_pct:.1%} supera -25%. Cortar y rotar capital."
        elif loss_pct < -0.15 and signal_prob < 0.60:
            rec    = "CUT_LOSS"
            reason = f"Pérdida {loss_pct:.1%} con baja probabilidad ({signal_prob:.0%}). Salir."
        elif loss_pct < -0.08 and signal_prob >= 0.68:
            rec    = "AVERAGE_DOWN"
            reason = f"Pérdida moderada {loss_pct:.1%} con señal fuerte ({signal_prob:.0%}). Promediar."
        elif loss_pct > 0.20 and signal_prob < 0.50:
            rec    = "TAKE_PROFIT"
            reason = f"Ganancia {loss_pct:.1%} con señal débil. Asegurar ganancias."
        else:
            rec    = "HOLD"
            reason = f"Posición dentro de parámetros normales ({loss_pct:.1%})."

        return {
            "ticker":            ticker,
            "unrealized_pnl_pct": round(loss_pct * 100, 2),
            "valor_posicion_clp": round(valor_posicion, 0),
            "recommendation":    rec,
            "reason":            reason,
            "signal_prob":       round(signal_prob, 4),
        }
