"""
AgentTrader CL — Orquestador Principal
Punto de entrada del sistema. Gestiona el scheduler,
el pipeline diario y el reentrenamiento semanal.
"""

import asyncio
import time
from datetime import datetime
from loguru import logger
import pytz
import psycopg2
import psycopg2.extras
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import (
    PORTFOLIO, WATCHLIST, PORTFOLIO_VALUE_CLP,
    RISK_PARAMS, SCHEDULER, MODEL_PATH,
)
from ingestion.scraper import (
    scrape_sync, get_historical_data, get_multiple_historical,
    save_prices_to_db, save_signal_to_db, get_portfolio_from_db,
)
from models.features import FeatureEngineer, compute_sharpe
from models.predictive import ModelEnsemble, train_from_scratch
from engine.decision import DecisionEngine
from risk.manager import RiskManager
from notifications.email_sender import send_alert, send_system_alert

SANTIAGO_TZ = pytz.timezone(SCHEDULER["timezone"])


# ══════════════════════════════════════════════════════════════════════
# PIPELINE DIARIO
# ══════════════════════════════════════════════════════════════════════

def run_daily_analysis():
    """
    Pipeline principal de análisis. Se ejecuta a las 15:45 CLT.

    Pasos:
    1. Scraping de precios actuales
    2. Cargar modelo entrenado
    3. Por cada ticker: calcular features → señal → validar riesgo
    4. Enviar alertas por email
    5. Guardar resultados en DB
    """
    run_start = time.time()
    now_str   = datetime.now(SANTIAGO_TZ).strftime("%Y-%m-%d %H:%M")
    logger.info("=" * 60)
    logger.info(f"ANÁLISIS DIARIO — {now_str} CLT")
    logger.info("=" * 60)

    # ── 1. SCRAPING ───────────────────────────────────────────────
    market_df = None
    try:
        market_df = scrape_sync()
        save_prices_to_db(market_df)
        logger.info(f"Precios obtenidos: {len(market_df)} instrumentos")
    except Exception as e:
        logger.error(f"SCRAPING FALLÓ: {e}")
        send_system_alert(
            "Scraping falló",
            f"Error al obtener precios de Security: {e}\n\nFecha: {now_str}"
        )
        # Continuar el análisis sin precios del scraping (usará yfinance)

    # ── 2. CARGAR MODELO ──────────────────────────────────────────
    try:
        model = ModelEnsemble.load(MODEL_PATH)
    except FileNotFoundError:
        logger.warning("Modelo no encontrado. Entrenando desde cero (puede tardar 2-5 min)...")
        try:
            model = train_from_scratch(WATCHLIST, period="3y")
        except Exception as e:
            logger.error(f"Entrenamiento inicial falló: {e}")
            send_system_alert("Entrenamiento falló", str(e))
            return

    # ── 3. CARTERA ACTUAL (DB o config) ──────────────────────────
    portfolio = get_portfolio_from_db() or PORTFOLIO

    # ── 4. MOTOR DE DECISIÓN Y RIESGO ────────────────────────────
    engine  = DecisionEngine(model, PORTFOLIO_VALUE_CLP, RISK_PARAMS)
    risk_mgr = RiskManager(PORTFOLIO_VALUE_CLP, RISK_PARAMS)

    results = {
        "signals":     [],
        "ok":          0,
        "fail":        0,
        "buy_count":   0,
        "sell_count":  0,
        "hold_count":  0,
    }

    # Datos históricos (para calcular VaR de cartera)
    price_series = {}

    # ── 5. ANALIZAR CADA TICKER ───────────────────────────────────
    for ticker in list(portfolio.keys()) + [t for t in WATCHLIST if t not in portfolio]:
        ticker = ticker.upper()
        logger.info(f"\n── Analizando {ticker} ──")

        try:
            # Histórico
            hist_df = get_historical_data(ticker, period="2y")
            if hist_df.empty:
                logger.warning(f"{ticker}: Sin datos históricos")
                results["fail"] += 1
                continue

            price_series[ticker] = hist_df["close"]

            # Calcular features
            fe = FeatureEngineer(hist_df)
            featured_df = fe.compute_all()

            if len(featured_df) < 50:
                logger.warning(f"{ticker}: Muy pocos datos ({len(featured_df)} filas)")
                results["fail"] += 1
                continue

            # Precio actual: scraping primero, yfinance como fallback
            current_price = None
            if market_df is not None and not market_df.empty:
                row = market_df[market_df["ticker"] == ticker]
                if not row.empty and row["last_price"].iloc[0] is not None:
                    current_price = float(row["last_price"].iloc[0])

            if current_price is None:
                current_price = float(hist_df["close"].iloc[-1])
                logger.debug(f"{ticker}: Usando precio histórico ${current_price:,.2f}")

            # Info de posición actual
            pos_info = portfolio.get(ticker, {"shares": 0, "avg_cost": None})
            current_shares = int(pos_info.get("shares", 0))
            avg_cost       = pos_info.get("avg_cost")

            # Volumen diario promedio
            avg_daily_vol = float(hist_df["volume"].tail(20).mean()) * current_price

            # Generar señal
            signal = engine.generate_signal(
                ticker           = ticker,
                df               = featured_df,
                current_price    = current_price,
                portfolio_shares = current_shares,
                avg_cost         = avg_cost,
            )

            # Evaluación de posición actual (si tenemos avg_cost)
            portfolio_eval = None
            if avg_cost and current_shares > 0:
                portfolio_eval = engine.evaluate_portfolio_position(
                    ticker, current_price, avg_cost,
                    current_shares, signal.probability
                )
                logger.info(
                    f"{ticker} P/L: {portfolio_eval['unrealized_pnl_pct']:.2f}% "
                    f"→ {portfolio_eval['recommendation']}"
                )

            # Validar riesgo
            approved, reason = risk_mgr.validate_signal(signal, avg_daily_vol)

            # Guardar en DB
            save_signal_to_db(signal, approved, reason)

            if approved:
                results["signals"].append(signal)
                results["ok"] += 1

                if signal.signal == "BUY":
                    results["buy_count"] += 1
                elif signal.signal == "SELL":
                    results["sell_count"] += 1
                else:
                    results["hold_count"] += 1

                # Enviar email para BUY/SELL aprobados
                if signal.signal in ("BUY", "SELL"):
                    sent = send_alert(signal, portfolio_eval)
                    if sent:
                        logger.success(f"{ticker}: Alerta enviada ({signal.signal})")
            else:
                logger.warning(f"{ticker}: Señal rechazada — {reason}")
                results["fail"] += 1

        except Exception as e:
            logger.error(f"{ticker}: Error inesperado — {e}", exc_info=True)
            results["fail"] += 1
            continue

    # ── 6. MÉTRICAS DE RIESGO DE CARTERA ─────────────────────────
    portfolio_positions = {
        t: int(d.get("shares", 0))
        for t, d in portfolio.items()
        if int(d.get("shares", 0)) > 0
    }
    if portfolio_positions and price_series:
        var_metrics = risk_mgr.compute_portfolio_var(
            portfolio_positions, price_series
        )
        risk_mgr.save_risk_metrics(var_metrics)
        logger.info(
            f"VaR Cartera (95%, 1d): "
            f"${var_metrics.get('var_95_daily_clp', 0):,.0f} CLP | "
            f"Vol: {var_metrics.get('portfolio_vol_ann', 0):.1%} | "
            f"Sharpe: {var_metrics.get('sharpe_portfolio', 0):.2f}"
        )

    # ── 7. RESUMEN FINAL ──────────────────────────────────────────
    duration = round(time.time() - run_start, 1)
    logger.info("=" * 60)
    logger.info(
        f"ANÁLISIS COMPLETADO — {duration}s | "
        f"OK={results['ok']} FAIL={results['fail']} | "
        f"BUY={results['buy_count']} SELL={results['sell_count']} HOLD={results['hold_count']}"
    )
    logger.info("=" * 60)

    # Log en DB
    _log_execution(results, duration)


# ══════════════════════════════════════════════════════════════════════
# REENTRENAMIENTO SEMANAL
# ══════════════════════════════════════════════════════════════════════

def run_weekly_retrain():
    """
    Reentrenamiento del ensemble cada sábado.
    Descarga 5 años históricos y reentrena con todos los datos.
    """
    logger.info("REENTRENAMIENTO SEMANAL — Iniciando")
    try:
        model = train_from_scratch(WATCHLIST, period="5y")
        logger.success(
            f"Reentrenamiento OK. Scores: {model.last_scores}"
        )
        send_system_alert(
            "Reentrenamiento semanal OK",
            f"Scores de validación:\n{model.last_scores}\n\nFecha: {datetime.now()}"
        )
    except Exception as e:
        logger.error(f"Reentrenamiento falló: {e}")
        send_system_alert("Reentrenamiento falló", str(e))


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _log_execution(results: dict, duration: float):
    """Registra el resultado de la ejecución en la DB."""
    try:
        import psycopg2
        from config import DB_CONFIG
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute(
            """
            INSERT INTO execution_log
                (run_date, status, tickers_ok, tickers_fail,
                 signals_buy, signals_sell, signals_hold, duration_sec)
            VALUES (CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                "SUCCESS" if results["fail"] == 0 else "PARTIAL",
                results["ok"], results["fail"],
                results["buy_count"], results["sell_count"],
                results["hold_count"], duration,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error registrando ejecución en DB: {e}")


# ══════════════════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════════════════

def start_scheduler():
    """
    Inicia el scheduler de APScheduler.
    Análisis diario: 15:45 CLT (Lun-Vie)
    Reentrenamiento: Sábado 08:00 CLT
    """
    scheduler = BlockingScheduler(timezone=SANTIAGO_TZ)

    # Análisis diario
    scheduler.add_job(
        run_daily_analysis,
        CronTrigger(
            hour=SCHEDULER["analysis_hour"],
            minute=SCHEDULER["analysis_minute"],
            day_of_week="mon-fri",
            timezone=SANTIAGO_TZ,
        ),
        id="daily_analysis",
        name="Análisis Diario",
        misfire_grace_time=300,
        coalesce=True,
        max_instances=1,
    )

    # Reentrenamiento semanal
    scheduler.add_job(
        run_weekly_retrain,
        CronTrigger(
            day_of_week=SCHEDULER["retrain_day"],
            hour=SCHEDULER["retrain_hour"],
            minute=0,
            timezone=SANTIAGO_TZ,
        ),
        id="weekly_retrain",
        name="Reentrenamiento Semanal",
        misfire_grace_time=1800,
        coalesce=True,
        max_instances=1,
    )

    logger.info("Scheduler iniciado:")
    logger.info(f"  → Análisis diario: {SCHEDULER['analysis_hour']}:{SCHEDULER['analysis_minute']:02d} CLT (Lun-Vie)")
    logger.info(f"  → Reentrenamiento: Sábado {SCHEDULER['retrain_hour']}:00 CLT")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler detenido por el usuario")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgentTrader CL")
    parser.add_argument(
        "--mode",
        choices=["scheduler", "run-now", "train", "backtest"],
        default="scheduler",
        help=(
            "scheduler: Iniciar con schedule automático\n"
            "run-now:   Ejecutar análisis una vez ahora mismo\n"
            "train:     Entrenar/reentrenar modelos ahora\n"
            "backtest:  Correr backtesting y mostrar métricas"
        ),
    )
    args = parser.parse_args()

    if args.mode == "scheduler":
        start_scheduler()

    elif args.mode == "run-now":
        logger.info("Ejecutando análisis manual...")
        run_daily_analysis()

    elif args.mode == "train":
        logger.info("Entrenando modelos desde cero...")
        run_weekly_retrain()

    elif args.mode == "backtest":
        from backtesting.engine import Backtester
        logger.info("Cargando datos para backtesting...")

        price_data = get_multiple_historical(list(PORTFOLIO.keys()), period="5y")

        # Generar señales históricas (simplificado para backtest)
        import pandas as pd
        all_signals = []

        for ticker, hist_df in price_data.items():
            fe = FeatureEngineer(hist_df)
            try:
                featured = fe.compute_all()
            except Exception as e:
                logger.warning(f"{ticker}: {e}")
                continue

            # Cargar modelo o entrenar
            try:
                model = ModelEnsemble.load(MODEL_PATH)
            except FileNotFoundError:
                logger.info("Entrenando modelo para backtest...")
                combined_data = pd.concat(
                    [FeatureEngineer(df).compute_all() for df in price_data.values()
                     if not df.empty],
                    ignore_index=True
                )
                model = ModelEnsemble()
                model.train(combined_data)
                model.save()

            engine = DecisionEngine(model, PORTFOLIO_VALUE_CLP, RISK_PARAMS)

            # Generar señal por fecha (walk-forward simplificado)
            for i in range(100, len(featured)):
                window = featured.iloc[:i]
                try:
                    price = float(window["close"].iloc[-1])
                    sig   = engine.generate_signal(ticker, window, price)
                    all_signals.append({
                        "date":        featured.index[i],
                        "ticker":      ticker,
                        "signal":      sig.signal,
                        "probability": sig.probability,
                    })
                except Exception:
                    continue

        if all_signals:
            signals_df = pd.DataFrame(all_signals)
            signals_df["date"] = pd.to_datetime(signals_df["date"])

            bt = Backtester(initial_capital=PORTFOLIO_VALUE_CLP)
            metrics = bt.run(signals_df, price_data)

            print("\n" + "=" * 50)
            print("RESULTADOS BACKTESTING")
            print("=" * 50)
            for k, v in metrics.items():
                print(f"  {k:<35}: {v}")
        else:
            logger.error("Sin señales para backtest")
