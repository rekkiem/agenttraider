"""
AgentTrader CL — Data Ingestion v2
Estrategia multi-fuente con 4 orígenes independientes de datos:

  1. yfinance  — Yahoo Finance con workarounds para API 2024/2025
  2. Stooq     — stooq.com (independiente de Yahoo), CSV directo con fechas
  3. pandas_datareader Stooq — implementación alternativa del mismo Stooq
  4. Alpha Vantage — API premium gratuita (25 req/día, requiere API key)

Si TODAS fallan: error claro con instrucciones de qué configurar.
"""

import asyncio
import time
import os
import warnings
from datetime import datetime, date, timedelta
from io import StringIO

import requests
import psycopg2
import psycopg2.extras
import pandas as pd
import yfinance as yf
from loguru import logger
from playwright.async_api import async_playwright

from config import DB_CONFIG, SECURITY_URL, SCRAPER_TIMEOUT_MS

warnings.filterwarnings("ignore", category=FutureWarning)

# ── MAPA DE TICKERS ───────────────────────────────────────────────────
# Cada ticker tiene sus alias en Yahoo (.SN), ADR (NYSE), y Stooq (.CL)
TICKER_MAP: dict[str, dict] = {
    "AGUAS-A":   {"yf": ["AGUAS-A.SN"], "stooq": "aguas-a.cl",  "av": "AGUAS-A"},
    "ENJOY":     {"yf": ["ENJOY.SN"],    "stooq": "enjoy.cl",    "av": "ENJOY"},
    "LTM":       {"yf": ["LTM.SN", "LTM"],  "stooq": "ltm.cl",  "av": "LTM"},
    "SOCOVESA":  {"yf": ["SOCOVESA.SN"], "stooq": "socovesa.cl", "av": "SOCOVESA"},
    "VAPORES":   {"yf": ["VAPORES.SN"],  "stooq": "vapores.cl",  "av": "VAPORES"},
    "FALABELLA": {"yf": ["FALABELLA.SN"],"stooq": "falabella.cl","av": "FALABELLA"},
    "ENELCHILE": {"yf": ["ENELCHILE.SN","ENIC"],"stooq": "enelchile.cl","av": "ENELCHILE"},
    "ENELAM":    {"yf": ["ENELAM.SN"],   "stooq": "enelam.cl",   "av": "ENELAM"},
    "COPEC":     {"yf": ["COPEC.SN"],    "stooq": "copec.cl",    "av": "COPEC"},
    "CHILE":     {"yf": ["CHILE.SN"],    "stooq": "chile.cl",    "av": "CHILE"},
    "BCI":       {"yf": ["BCI.SN"],      "stooq": "bci.cl",      "av": "BCI"},
    "CENCOSUD":  {"yf": ["CENCOSUD.SN","CNCO"],"stooq": "cencosud.cl","av": "CENCOSUD"},
    "CMPC":      {"yf": ["CMPC.SN"],     "stooq": "cmpc.cl",     "av": "CMPC"},
    "COLBUN":    {"yf": ["COLBUN.SN"],   "stooq": "colbun.cl",   "av": "COLBUN"},
    "RIPLEY":    {"yf": ["RIPLEY.SN"],   "stooq": "ripley.cl",   "av": "RIPLEY"},
    "CCU":       {"yf": ["CCU.SN","CCU"],"stooq": "ccu.cl",      "av": "CCU"},
    "SQM-B":     {"yf": ["SQM-B.SN","SQM"],"stooq": "sqm-b.cl", "av": "SQM"},
    "PARAUCO":   {"yf": ["PARAUCO.SN"],  "stooq": "parauco.cl",  "av": "PARAUCO"},
    "ANDINA-B":  {"yf": ["ANDINA-B.SN"],"stooq": "andina-b.cl", "av": "ANDINA-B"},
    "SONDA":     {"yf": ["SONDA.SN"],    "stooq": "sonda.cl",    "av": "SONDA"},
    "BESALCO":   {"yf": ["BESALCO.SN"],  "stooq": "besalco.cl",  "av": "BESALCO"},
    "SALFACORP": {"yf": ["SALFACORP.SN"],"stooq": "salfacorp.cl","av": "SALFACORP"},
    "FORUS":     {"yf": ["FORUS.SN"],    "stooq": "forus.cl",    "av": "FORUS"},
}

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")


# ═══════════════════════════════════════════════════════════════════════
# FUENTE 1: yfinance
# ═══════════════════════════════════════════════════════════════════════

def _yf_session() -> requests.Session:
    """Sesión HTTP con headers de navegador para evitar bloqueos."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-CL,es;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control":   "no-cache",
        "Pragma":          "no-cache",
    })
    return s


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas y tipos de un DataFrame OHLCV."""
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # yfinance a veces retorna MultiIndex (ticker, field)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]

    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    required = {"open", "high", "low", "close", "volume"}
    missing   = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df[df["close"] > 0]
    return df.sort_index()


def _source_yfinance(ticker: str, days: int) -> pd.DataFrame:
    """
    Fuente 1: yfinance con Ticker().history() y sesión propia.
    Usa fechas explícitas en vez de 'period' para evitar bugs de yfinance 0.2.40.
    """
    aliases  = TICKER_MAP.get(ticker, {}).get("yf", [f"{ticker}.SN"])
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days)
    session  = _yf_session()

    for alias in aliases:
        try:
            time.sleep(0.4)
            t  = yf.Ticker(alias, session=session)

            # Intentar primero con start/end (más estable en 0.2.40)
            df = t.history(
                start=str(start_dt),
                end=str(end_dt),
                interval="1d",
                auto_adjust=True,
                actions=False,
                timeout=15,
            )
            if df.empty:
                # Segundo intento con period
                df = t.history(
                    period="2y",
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    timeout=15,
                )
            if df.empty:
                raise ValueError("DataFrame vacío")

            result = _normalize_df(df)
            if len(result) < 30:
                raise ValueError(f"Muy pocos datos: {len(result)} filas")

            return result

        except Exception as e:
            logger.debug(f"yfinance '{alias}' falló: {e}")
            continue

    raise ConnectionError(f"yfinance: todos los alias fallaron para {ticker}")


# ═══════════════════════════════════════════════════════════════════════
# FUENTE 2: Stooq CSV directo (con fechas en URL)
# ═══════════════════════════════════════════════════════════════════════

def _source_stooq_direct(ticker: str, days: int) -> pd.DataFrame:
    """
    Fuente 2: Stooq CSV con parámetros de fecha explícitos.
    URL: https://stooq.com/q/d/l/?s=aguas-a.cl&d1=20230101&d2=20260221&i=d

    Diferencia clave vs implementación anterior: incluye d1/d2.
    Sin fechas, Stooq devuelve solo el último dato.
    """
    stooq_sym = TICKER_MAP.get(ticker, {}).get("stooq", f"{ticker.lower()}.cl")
    end_dt    = date.today()
    start_dt  = end_dt - timedelta(days=days)
    d1        = start_dt.strftime("%Y%m%d")
    d2        = end_dt.strftime("%Y%m%d")

    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&d1={d1}&d2={d2}&i=d"

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer":    "https://stooq.com/",
        "Accept":     "text/html,application/xhtml+xml,*/*",
    })

    resp = s.get(url, timeout=20)
    resp.raise_for_status()

    text = resp.text.strip()
    if len(text) < 30 or "No data" in text or "<html" in text.lower():
        raise ValueError(f"Stooq sin datos para {stooq_sym}: respuesta={text[:80]!r}")

    df = pd.read_csv(StringIO(text))
    df.columns = [c.lower().strip() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError(f"Stooq: sin columna 'date'. Columnas: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return _normalize_df(df)


# ═══════════════════════════════════════════════════════════════════════
# FUENTE 3: pandas_datareader + Stooq
# ═══════════════════════════════════════════════════════════════════════

def _source_pdr_stooq(ticker: str, days: int) -> pd.DataFrame:
    """
    Fuente 3: pandas_datareader con backend Stooq.
    Implementación más robusta y con retry automático.
    Requiere: pip install pandas-datareader
    """
    try:
        import pandas_datareader as pdr
    except ImportError:
        raise ImportError("pandas_datareader no instalado. Ejecutar: pip install pandas-datareader")

    stooq_sym = TICKER_MAP.get(ticker, {}).get("stooq", f"{ticker.lower()}.cl").upper()
    end_dt    = date.today()
    start_dt  = end_dt - timedelta(days=days)

    df = pdr.get_data_stooq(stooq_sym, start=start_dt, end=end_dt)

    if df.empty:
        raise ValueError(f"pandas_datareader Stooq: sin datos para {stooq_sym}")

    return _normalize_df(df)


# ═══════════════════════════════════════════════════════════════════════
# FUENTE 4: Alpha Vantage
# ═══════════════════════════════════════════════════════════════════════

def _source_alpha_vantage(ticker: str, days: int) -> pd.DataFrame:
    """
    Fuente 4: Alpha Vantage API (gratuita, 25 requests/día).
    Requiere ALPHA_VANTAGE_KEY en .env
    Obtener clave gratuita en: https://www.alphavantage.co/support/#api-key

    NOTA: Alpha Vantage tiene cobertura limitada de la Bolsa de Santiago.
    Funciona mejor para LTM (ADR NYSE) y otros tickers con cotización en EEUU.
    """
    if not ALPHA_VANTAGE_KEY:
        raise ValueError(
            "Alpha Vantage no configurado. "
            "Obtén tu clave gratuita en https://www.alphavantage.co/support/#api-key "
            "y agrégala como ALPHA_VANTAGE_KEY en tu .env"
        )

    av_sym = TICKER_MAP.get(ticker, {}).get("av", ticker)
    url    = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={av_sym}"
        f"&outputsize=full"
        f"&apikey={ALPHA_VANTAGE_KEY}"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        raise ValueError(f"Alpha Vantage rate limit: {data['Note'][:80]}")
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Alpha Vantage: respuesta inesperada para {av_sym}")

    ts      = data["Time Series (Daily)"]
    records = []
    cutoff  = date.today() - timedelta(days=days)

    for date_str, vals in ts.items():
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        if d < cutoff:
            continue
        records.append({
            "date":   pd.Timestamp(date_str),
            "open":   float(vals.get("1. open",  0)),
            "high":   float(vals.get("2. high",  0)),
            "low":    float(vals.get("3. low",   0)),
            "close":  float(vals.get("5. adjusted close", vals.get("4. close", 0))),
            "volume": float(vals.get("6. volume", 0)),
        })

    if not records:
        raise ValueError(f"Alpha Vantage: sin datos para {av_sym}")

    df = pd.DataFrame(records).set_index("date").sort_index()
    return _normalize_df(df)


# ═══════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: multi-source con fallbacks
# ═══════════════════════════════════════════════════════════════════════

def get_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Descarga datos históricos OHLCV con 4 fuentes independientes.

    Orden de prioridad:
      1. yfinance  (Yahoo Finance con sesión propia + start/end params)
      2. Stooq directo CSV con fechas (independiente de Yahoo)
      3. pandas_datareader + Stooq (si está instalado)
      4. Alpha Vantage (si ALPHA_VANTAGE_KEY está en .env)

    Retorna DataFrame con columnas [open, high, low, close, volume]
    o DataFrame vacío si todas las fuentes fallan.
    """
    period_days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}.get(period, 730)

    sources = [
        ("yfinance",           lambda: _source_yfinance(ticker, period_days)),
        ("Stooq-directo",      lambda: _source_stooq_direct(ticker, period_days)),
        ("pandas_datareader",  lambda: _source_pdr_stooq(ticker, period_days)),
        ("Alpha Vantage",      lambda: _source_alpha_vantage(ticker, period_days)),
    ]

    for name, fetch_fn in sources:
        try:
            df = fetch_fn()
            if not df.empty and len(df) >= 30:
                logger.info(f"{ticker} ✓ [{name}]: {len(df)} sesiones")
                return df
            else:
                logger.debug(f"{ticker}: {name} devolvió datos insuficientes ({len(df)} filas)")
        except ImportError as e:
            logger.debug(f"{ticker}: {name} no disponible — {e}")
        except Exception as e:
            logger.debug(f"{ticker}: {name} falló — {type(e).__name__}: {str(e)[:100]}")
        time.sleep(0.3)

    # ── TODAS LAS FUENTES FALLARON ────────────────────────────────────
    logger.error(
        f"\n{'═'*60}\n"
        f"  {ticker}: SIN DATOS — todas las fuentes fallaron.\n"
        f"  Soluciones posibles:\n"
        f"  1. Verificar conexión a internet\n"
        f"  2. Desactivar VPN si está activa\n"
        f"  3. Instalar pandas-datareader: pip install pandas-datareader\n"
        f"  4. Configurar ALPHA_VANTAGE_KEY en .env\n"
        f"     (gratis en https://www.alphavantage.co/support/#api-key)\n"
        f"{'═'*60}"
    )
    return pd.DataFrame()


def get_multiple_historical(tickers: list, period: str = "2y") -> dict:
    """Descarga históricos de múltiples tickers. Retorna dict {ticker: df}."""
    result = {}
    for t in tickers:
        df = get_historical_data(t, period)
        if not df.empty:
            result[t] = df
        else:
            logger.warning(f"Sin datos para {t} — se excluye del análisis")
    return result


# ═══════════════════════════════════════════════════════════════════════
# SCRAPER SECURITY (Playwright)
# ═══════════════════════════════════════════════════════════════════════

async def scrape_security_market() -> pd.DataFrame:
    """
    Scrapea el portal de Inversiones Security con Playwright headless.
    Retorna DataFrame con: ticker, last_price, change_pct, bid, ask, volume, timestamp
    """
    records = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        try:
            logger.info(f"Accediendo a {SECURITY_URL}")
            await page.goto(SECURITY_URL, timeout=SCRAPER_TIMEOUT_MS, wait_until="networkidle")

            try:
                await page.wait_for_selector("table", timeout=15_000)
            except Exception:
                await page.wait_for_selector("[class*='tabla'], [class*='table'], .resumen", timeout=10_000)

            rows = await page.query_selector_all("table tbody tr")
            if not rows:
                rows = await page.query_selector_all("tr")

            logger.info(f"Filas encontradas: {len(rows)}")

            def clean_num(s: str) -> str:
                return s.strip().replace(".", "").replace(",", ".").replace("%", "").replace("$", "").replace(" ", "")

            for row in rows:
                cells = await row.query_selector_all("td")
                if len(cells) < 5:
                    continue
                try:
                    ticker = (await cells[0].inner_text()).strip().upper()
                    last   = clean_num(await cells[1].inner_text())
                    chg    = clean_num(await cells[2].inner_text())
                    bid    = clean_num(await cells[3].inner_text()) if len(cells) > 3 else ""
                    ask    = clean_num(await cells[4].inner_text()) if len(cells) > 4 else ""
                    volstr = clean_num(await cells[5].inner_text()) if len(cells) > 5 else "0"

                    if not ticker or ticker in ("NEMOTÉCNICO", "NEMO", "ACCIÓN", "TICKER", "-", ""):
                        continue

                    records.append({
                        "ticker":     ticker,
                        "last_price": float(last) if last else None,
                        "change_pct": float(chg)  if chg  else None,
                        "bid":        float(bid)   if bid  else None,
                        "ask":        float(ask)   if ask  else None,
                        "volume":     int(float(volstr)) if volstr else 0,
                        "timestamp":  datetime.now(),
                    })
                except (ValueError, IndexError) as e:
                    logger.debug(f"Fila ignorada: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error en scraping Security: {e}")
            raise
        finally:
            await browser.close()

    df = pd.DataFrame(records)
    logger.info(f"Scraping OK — {len(df)} instrumentos")
    return df


def scrape_sync() -> pd.DataFrame:
    """Wrapper sincrónico."""
    return asyncio.run(scrape_security_market())


# ═══════════════════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════════════════

def save_prices_to_db(df: pd.DataFrame):
    if df.empty:
        return
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        inserted = 0
        for _, row in df.iterrows():
            if row.get("last_price") is None:
                continue
            cur.execute(
                """
                INSERT INTO prices_raw (ticker, last_price, change_pct, bid, ask, volume, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, timestamp) DO NOTHING
                """,
                (row["ticker"], row["last_price"], row["change_pct"],
                 row.get("bid"), row.get("ask"), row.get("volume", 0), row["timestamp"]),
            )
            inserted += 1
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"DB: {inserted} precios guardados")
    except Exception as e:
        logger.error(f"Error guardando precios: {e}")
        raise


def save_signal_to_db(signal, approved: bool, rejection_reason: str = ""):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute(
            """
            INSERT INTO signals_log
                (ticker, signal, probability, confidence, suggested_size,
                 current_price, stop_loss, take_profit, risk_pct,
                 expected_return, approved, rejection_reason, reasoning)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (signal.ticker, signal.signal, signal.probability, signal.confidence,
             signal.suggested_size, signal.current_price, signal.stop_loss,
             signal.take_profit, signal.risk_pct, signal.expected_return,
             approved, rejection_reason, psycopg2.extras.Json(signal.reasoning)),
        )
        signal_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return signal_id
    except Exception as e:
        logger.error(f"Error guardando señal: {e}")
        return None


def get_portfolio_from_db() -> dict:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("SELECT ticker, shares, avg_cost FROM portfolio_state")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {r[0]: {"shares": r[1], "avg_cost": r[2]} for r in rows}
    except Exception as e:
        logger.error(f"Error leyendo portafolio: {e}")
        return {}
