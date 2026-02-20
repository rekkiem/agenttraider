"""
AgentTrader CL — Data Ingestion
Scraper principal de Mercados en Línea (Inversiones Security)
+ fallback histórico vía yfinance.
"""

import asyncio
import psycopg2
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger
from playwright.async_api import async_playwright

from config import DB_CONFIG, SECURITY_URL, SCRAPER_TIMEOUT_MS


# ── SCRAPER PRINCIPAL ─────────────────────────────────────────────────

async def scrape_security_market() -> pd.DataFrame:
    """
    Scrapea el portal de Inversiones Security con Playwright headless.
    Retorna DataFrame con columnas:
        ticker, last_price, change_pct, bid, ask, volume, timestamp
    """
    records = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        try:
            logger.info(f"Accediendo a {SECURITY_URL}")
            await page.goto(
                SECURITY_URL,
                timeout=SCRAPER_TIMEOUT_MS,
                wait_until="networkidle",
            )

            # Esperar a que cargue la tabla principal
            try:
                await page.wait_for_selector("table", timeout=15_000)
            except Exception:
                # Algunos portales renderizan con divs; intentar selector alternativo
                await page.wait_for_selector("[class*='tabla'], [class*='table'], .resumen", timeout=10_000)

            rows = await page.query_selector_all("table tbody tr")
            if not rows:
                rows = await page.query_selector_all("tr")

            logger.info(f"Filas encontradas en tabla: {len(rows)}")

            for row in rows:
                cells = await row.query_selector_all("td")
                if len(cells) < 5:
                    continue
                try:
                    def clean_num(s: str) -> str:
                        return s.strip().replace(".", "").replace(",", ".").replace("%", "").replace("$", "").replace(" ", "")

                    ticker = (await cells[0].inner_text()).strip().upper()
                    last   = clean_num(await cells[1].inner_text())
                    chg    = clean_num(await cells[2].inner_text())
                    bid    = clean_num(await cells[3].inner_text()) if len(cells) > 3 else ""
                    ask    = clean_num(await cells[4].inner_text()) if len(cells) > 4 else ""
                    volstr = clean_num(await cells[5].inner_text()) if len(cells) > 5 else "0"

                    if not ticker or ticker in ("NEMOTÉCNICO", "NEMO", "ACCIÓN", "-"):
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
            logger.error(f"Error durante scraping: {e}")
            raise
        finally:
            await browser.close()

    df = pd.DataFrame(records)
    logger.info(f"Scraping OK — {len(df)} instrumentos")
    return df


def scrape_sync() -> pd.DataFrame:
    """Wrapper sincrónico para llamar desde el scheduler."""
    return asyncio.run(scrape_security_market())


# ── DATOS HISTÓRICOS (yfinance fallback) ──────────────────────────────

def get_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Descarga datos históricos OHLCV desde Yahoo Finance.
    Tickers chilenos terminan en .SN (ej: AGUAS-A.SN)

    Manejo especial:
    - LTM cotiza también en NYSE como LTM
    - SQM-B → SQM en NYSE
    """
    yf_ticker = f"{ticker}.SN"

    # Algunos tickers tienen alias en yfinance
    alias_map = {
        "LTM.SN":    "LTM",     # Latam Airlines ADR
        "SQM-B.SN":  "SQM",
        "ENELCHILE.SN": "ENIC",
    }
    yf_ticker = alias_map.get(yf_ticker, yf_ticker)

    try:
        df = yf.download(
            yf_ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(f"Sin datos para {yf_ticker}")

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() 
                      for c in df.columns]
        df.index.name = "date"
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        logger.info(f"{ticker}: {len(df)} sesiones descargadas ({period})")
        return df

    except Exception as e:
        logger.error(f"Error descargando {ticker} ({yf_ticker}): {e}")
        return pd.DataFrame()


def get_multiple_historical(tickers: list, period: str = "2y") -> dict:
    """Descarga históricos de múltiples tickers. Retorna dict {ticker: df}."""
    result = {}
    for t in tickers:
        df = get_historical_data(t, period)
        if not df.empty:
            result[t] = df
        else:
            logger.warning(f"Sin datos históricos para {t}")
    return result


# ── GUARDAR EN DB ─────────────────────────────────────────────────────

def save_prices_to_db(df: pd.DataFrame):
    """Inserta precios del scraping en tabla prices_raw."""
    if df.empty:
        logger.warning("DataFrame vacío, nada que guardar")
        return

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()

        inserted = 0
        for _, row in df.iterrows():
            if row["last_price"] is None:
                continue
            cur.execute(
                """
                INSERT INTO prices_raw
                    (ticker, last_price, change_pct, bid, ask, volume, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, timestamp) DO NOTHING
                """,
                (
                    row["ticker"], row["last_price"], row["change_pct"],
                    row["bid"],    row["ask"],         row["volume"],
                    row["timestamp"],
                ),
            )
            inserted += 1

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"DB: {inserted} precios guardados")

    except Exception as e:
        logger.error(f"Error guardando en DB: {e}")
        raise


def save_signal_to_db(signal, approved: bool, rejection_reason: str = ""):
    """Persiste una señal en signals_log."""
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
            (
                signal.ticker, signal.signal, signal.probability,
                signal.confidence, signal.suggested_size,
                signal.current_price, signal.stop_loss, signal.take_profit,
                signal.risk_pct, signal.expected_return,
                approved, rejection_reason,
                psycopg2.extras.Json(signal.reasoning),
            ),
        )
        signal_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return signal_id

    except Exception as e:
        logger.error(f"Error guardando señal en DB: {e}")
        return None


def get_portfolio_from_db() -> dict:
    """Lee el estado actual del portafolio desde la DB."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("SELECT ticker, shares, avg_cost FROM portfolio_state")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {r[0]: {"shares": r[1], "avg_cost": r[2]} for r in rows}
    except Exception as e:
        logger.error(f"Error leyendo portafolio de DB: {e}")
        return {}
