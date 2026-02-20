"""
AgentTrader CL — Feature Engineering
Calcula todos los indicadores técnicos y cuantitativos
sobre los datos OHLCV históricos.
"""

import pandas as pd
import numpy as np
from loguru import logger


class FeatureEngineer:
    """
    Transforma un DataFrame OHLCV crudo en features
    listos para entrenamiento y predicción.

    Input esperado: columnas [open, high, low, close, volume]
    con índice tipo DatetimeIndex o columna 'date'.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_all(self) -> pd.DataFrame:
        """
        Pipeline completo de features.
        Retorna DataFrame con todos los indicadores calculados.
        """
        df = self.df

        # Verificar columnas mínimas
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en DataFrame: {missing}")

        df = df.copy()

        # ── RSI ───────────────────────────────────────────────────
        df["rsi_14"] = self._rsi(df["close"], 14)

        # ── MACD ──────────────────────────────────────────────────
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # ── EMAs ──────────────────────────────────────────────────
        for w in [9, 20, 50, 200]:
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

        # ── BANDAS DE BOLLINGER ───────────────────────────────────
        sma20        = df["close"].rolling(20).mean()
        std20        = df["close"].rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
        # Posición dentro de la banda: 0 = piso, 1 = techo
        df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ── ATR ───────────────────────────────────────────────────
        df["atr_14"] = self._atr(df, 14)

        # ── VOLUMEN ───────────────────────────────────────────────
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"]  = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

        # OBV (On Balance Volume)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()

        # ── SOPORTES Y RESISTENCIAS DINÁMICOS ────────────────────
        df["support_20"]    = df["low"].rolling(20).min()
        df["resistance_20"] = df["high"].rolling(20).max()
        df["dist_support"]  = (df["close"] - df["support_20"]) / df["close"]
        df["dist_resist"]   = (df["resistance_20"] - df["close"]) / df["close"]

        # ── MOMENTUM MULTI-PERÍODO ────────────────────────────────
        for lag in [5, 10, 20, 60]:
            df[f"return_{lag}d"] = df["close"].pct_change(lag)

        df["momentum_score"] = (
            df["return_5d"]  * 0.40
            + df["return_10d"] * 0.30
            + df["return_20d"] * 0.20
            + df["return_60d"] * 0.10
        )

        # ── VOLATILIDAD ───────────────────────────────────────────
        df["log_return"]  = np.log(df["close"] / df["close"].shift(1))
        df["vol_20d_ann"] = df["log_return"].rolling(20).std() * np.sqrt(252)
        df["vol_60d_ann"] = df["log_return"].rolling(60).std() * np.sqrt(252)
        # Régimen: 1 si volatilidad corta > larga (entorno más volátil)
        df["vol_regime"]  = (df["vol_20d_ann"] > df["vol_60d_ann"]).astype(int)

        # ── SEÑALES DERIVADAS ─────────────────────────────────────
        df["golden_cross"] = (
            (df["ema_20"] > df["ema_50"]) &
            (df["ema_20"].shift(1) <= df["ema_50"].shift(1))
        ).astype(int)

        df["death_cross"] = (
            (df["ema_20"] < df["ema_50"]) &
            (df["ema_20"].shift(1) >= df["ema_50"].shift(1))
        ).astype(int)

        df["trend_short"] = np.where(df["close"] > df["ema_20"],  1, -1)
        df["trend_long"]  = np.where(df["close"] > df["ema_200"], 1, -1)

        # ── TARGET: retorno positivo en 5 días ───────────────────
        # (solo para entrenamiento, será NaN en datos recientes)
        df["target_5d"] = (df["close"].shift(-5) > df["close"]).astype(float)

        result = df.dropna(subset=["rsi_14", "macd", "bb_pct", "atr_14", "vol_20d_ann"])
        logger.debug(f"Features calculados: {len(result)} filas válidas de {len(df)}")
        return result

    # ── HELPERS PRIVADOS ──────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.where(delta > 0, 0.0)
        loss  = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()


# ── MÉTRICAS DE PORTAFOLIO ────────────────────────────────────────────

def compute_sharpe(returns: pd.Series, rf_annual: float = 0.06) -> float:
    """Sharpe ratio anualizado. rf_annual = tasa libre de riesgo CLP."""
    rf_daily = rf_annual / 252
    excess   = returns - rf_daily
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def compute_var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR paramétrico diario (retorno, no monto)."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def compute_portfolio_correlation(price_dict: dict) -> pd.DataFrame:
    """
    Calcula la matriz de correlación de retornos entre activos.
    price_dict: {ticker: pd.Series de precios de cierre}
    """
    returns = pd.DataFrame({
        t: s.pct_change().dropna()
        for t, s in price_dict.items()
    })
    return returns.corr()
