"""
AgentTrader CL — Modelos Predictivos
Ensemble de LogisticRegression + XGBoost + RandomForest
con validación temporal (TimeSeriesSplit) y reentrenamiento semanal.
Prophet se usa como capa separada de tendencia.
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from loguru import logger

from config import FEATURE_COLS, MODEL_PATH

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet no instalado. Usando solo ensemble ML.")


# ── ENSEMBLE PRINCIPAL ────────────────────────────────────────────────

class ModelEnsemble:
    """
    Ensemble ponderado de 3 clasificadores binarios.
    Target: subida en 5 días (1) o baja/plano (0).

    Los pesos se ajustan automáticamente según AUC de validación.
    """

    def __init__(self):
        self.scaler   = StandardScaler()
        self.logreg   = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        self.xgb_clf  = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        self.weights     = {"logreg": 0.25, "xgb": 0.50, "rf": 0.25}
        self.is_trained  = False
        self.last_scores = {}
        self.trained_at  = None

    def train(self, df: pd.DataFrame) -> dict:
        """
        Entrena el ensemble con validación cruzada temporal.
        Retorna métricas de validación.

        df debe contener FEATURE_COLS + 'target_5d'.
        """
        # Filtrar filas con target válido (excluir últimas 5 filas sin target)
        df_train = df.dropna(subset=FEATURE_COLS + ["target_5d"])

        if len(df_train) < 100:
            raise ValueError(f"Datos insuficientes para entrenar: {len(df_train)} filas")

        X = df_train[FEATURE_COLS].values
        y = df_train["target_5d"].values.astype(int)

        logger.info(f"Entrenando con {len(X)} muestras. Balance: {y.mean():.1%} positivos")

        tscv = TimeSeriesSplit(n_splits=5)
        auc_logreg, auc_xgb, auc_rf = [], [], []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            X_tr_sc  = self.scaler.fit_transform(X_tr)
            X_val_sc = self.scaler.transform(X_val)

            self.logreg.fit(X_tr_sc, y_tr)
            self.xgb_clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.rf.fit(X_tr, y_tr)

            auc_logreg.append(roc_auc_score(y_val, self.logreg.predict_proba(X_val_sc)[:, 1]))
            auc_xgb.append(roc_auc_score(y_val, self.xgb_clf.predict_proba(X_val)[:, 1]))
            auc_rf.append(roc_auc_score(y_val, self.rf.predict_proba(X_val)[:, 1]))

            logger.debug(f"Fold {fold+1}: LR={auc_logreg[-1]:.3f} XGB={auc_xgb[-1]:.3f} RF={auc_rf[-1]:.3f}")

        self.last_scores = {
            "logreg_auc": float(np.mean(auc_logreg)),
            "xgb_auc":    float(np.mean(auc_xgb)),
            "rf_auc":     float(np.mean(auc_rf)),
        }
        logger.info(f"AUC promedio: {self.last_scores}")

        # Pesos proporcionales al AUC
        total = sum(self.last_scores.values())
        self.weights = {
            "logreg": self.last_scores["logreg_auc"] / total,
            "xgb":    self.last_scores["xgb_auc"]    / total,
            "rf":     self.last_scores["rf_auc"]     / total,
        }

        # Entrenar final con todos los datos
        X_sc = self.scaler.fit_transform(X)
        self.logreg.fit(X_sc, y)
        self.xgb_clf.fit(X, y)
        self.rf.fit(X, y)

        self.is_trained = True
        self.trained_at = datetime.now()
        logger.success("Entrenamiento completado")
        return self.last_scores

    def predict_proba(self, df: pd.DataFrame) -> float:
        """
        Probabilidad de alza (última fila del df).
        Retorna float entre 0 y 1.
        """
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Llama a train() primero.")

        X = df[FEATURE_COLS].dropna().values
        if len(X) == 0:
            raise ValueError("Sin datos válidos para predicción")

        X_last   = X[-1:]
        X_last_sc = self.scaler.transform(X_last)

        p_lr  = float(self.logreg.predict_proba(X_last_sc)[0, 1])
        p_xgb = float(self.xgb_clf.predict_proba(X_last)[0, 1])
        p_rf  = float(self.rf.predict_proba(X_last)[0, 1])

        ensemble = (
            self.weights["logreg"] * p_lr
            + self.weights["xgb"]  * p_xgb
            + self.weights["rf"]   * p_rf
        )

        logger.debug(f"Predicciones: LR={p_lr:.3f} XGB={p_xgb:.3f} RF={p_rf:.3f} → Ensemble={ensemble:.3f}")
        return float(ensemble)

    def feature_importance(self) -> pd.DataFrame:
        """Importancia de features según XGBoost."""
        if not self.is_trained:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": self.xgb_clf.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Modelo guardado en {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "ModelEnsemble":
        model = joblib.load(path)
        logger.info(f"Modelo cargado desde {path} (entrenado: {model.trained_at})")
        return model


# ── PROPHET (TENDENCIA) ───────────────────────────────────────────────

def run_prophet_forecast(df: pd.DataFrame, periods: int = 5) -> dict:
    """
    Modelo Prophet para capturar tendencia y estacionalidad.
    Retorna dirección esperada y upside potencial.

    Si Prophet no está disponible, retorna resultado neutro.
    """
    if not PROPHET_AVAILABLE:
        return {
            "prophet_direction":  0,
            "prophet_upside_pct": 0.0,
            "prophet_ci_width":   0.0,
            "prophet_available":  False,
        }

    try:
        prophet_df = df[["close"]].copy().reset_index()
        if "date" in prophet_df.columns:
            prophet_df = prophet_df.rename(columns={"date": "ds", "close": "y"})
        else:
            prophet_df.columns = ["ds", "y"]

        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        # Solo usar últimos 2 años para velocidad
        prophet_df = prophet_df.tail(504)

        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=True,
        )
        # Silenciar logs de Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        model.fit(prophet_df)

        future   = model.make_future_dataframe(periods=periods, freq="B")
        forecast = model.predict(future)

        last_actual = float(df["close"].iloc[-1])
        pred_5d     = float(forecast["yhat"].iloc[-1])
        lower_5d    = float(forecast["yhat_lower"].iloc[-1])
        upper_5d    = float(forecast["yhat_upper"].iloc[-1])

        return {
            "prophet_direction":  1 if pred_5d > last_actual else 0,
            "prophet_upside_pct": round((pred_5d - last_actual) / last_actual * 100, 2),
            "prophet_ci_width":   round((upper_5d - lower_5d) / last_actual * 100, 2),
            "prophet_available":  True,
        }

    except Exception as e:
        logger.warning(f"Prophet falló: {e}. Usando resultado neutro.")
        return {
            "prophet_direction":  0,
            "prophet_upside_pct": 0.0,
            "prophet_ci_width":   0.0,
            "prophet_available":  False,
        }


# ── ENTRENAMIENTO INICIAL (SCRIPT STANDALONE) ─────────────────────────

def train_from_scratch(tickers: list, period: str = "5y") -> ModelEnsemble:
    """
    Entrena el ensemble desde cero con datos históricos de todos los tickers.
    Útil para la primera ejecución.
    """
    from ingestion.scraper import get_historical_data
    from models.features import FeatureEngineer

    all_featured = []

    for ticker in tickers:
        logger.info(f"Descargando histórico de {ticker}...")
        hist = get_historical_data(ticker, period=period)
        if hist.empty:
            logger.warning(f"Sin datos para {ticker}, omitiendo")
            continue

        fe = FeatureEngineer(hist)
        try:
            featured = fe.compute_all()
            featured["ticker"] = ticker
            all_featured.append(featured)
        except Exception as e:
            logger.error(f"Error en features de {ticker}: {e}")
            continue

    if not all_featured:
        raise RuntimeError("Sin datos suficientes para entrenar")

    combined = pd.concat(all_featured, ignore_index=True)
    logger.info(f"Dataset combinado: {len(combined)} filas, {len(tickers)} tickers")

    model = ModelEnsemble()
    scores = model.train(combined)
    model.save()

    logger.success(f"Modelo inicial entrenado. Scores: {scores}")
    return model
