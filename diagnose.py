"""
AgentTrader CL — Diagnóstico v2
Testea las 4 fuentes de datos independientemente y muestra cuál funciona.
Ejecutar: python diagnose.py

Si todo falla: instrucciones específicas de qué instalar o configurar.
"""

import os
import sys
import time
import requests
from datetime import date, datetime, timedelta
from io import StringIO

# ── Silence yfinance warnings ─────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

TICKERS = ["AGUAS-A", "ENJOY", "LTM", "SOCOVESA", "VAPORES",
           "FALABELLA", "COPEC", "CMPC"]

TICKER_MAP = {
    "AGUAS-A":   {"yf": ["AGUAS-A.SN"], "stooq": "aguas-a.cl"},
    "ENJOY":     {"yf": ["ENJOY.SN"],    "stooq": "enjoy.cl"},
    "LTM":       {"yf": ["LTM.SN", "LTM"], "stooq": "ltm.cl"},
    "SOCOVESA":  {"yf": ["SOCOVESA.SN"], "stooq": "socovesa.cl"},
    "VAPORES":   {"yf": ["VAPORES.SN"],  "stooq": "vapores.cl"},
    "FALABELLA": {"yf": ["FALABELLA.SN"],"stooq": "falabella.cl"},
    "COPEC":     {"yf": ["COPEC.SN"],    "stooq": "copec.cl"},
    "CMPC":      {"yf": ["CMPC.SN"],     "stooq": "cmpc.cl"},
}

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")


# ══════════════════════════════════════════════════════════════════════
# TEST FUENTE 1: yfinance con start/end
# ══════════════════════════════════════════════════════════════════════

def test_yfinance(ticker: str) -> tuple[bool, str]:
    try:
        import yfinance as yf
        aliases  = TICKER_MAP.get(ticker, {}).get("yf", [f"{ticker}.SN"])
        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=365)

        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "es-CL,es;q=0.9",
        })

        for alias in aliases:
            try:
                t  = yf.Ticker(alias, session=s)
                df = t.history(start=str(start_dt), end=str(end_dt),
                               interval="1d", auto_adjust=True, actions=False, timeout=15)
                if df.empty:
                    df = t.history(period="1y", interval="1d", auto_adjust=True, actions=False, timeout=15)
                if not df.empty and len(df) >= 30:
                    return True, f"{alias} → {len(df)} filas"
            except Exception:
                continue
        return False, "sin datos en todos los alias"
    except Exception as e:
        return False, str(e)[:60]


# ══════════════════════════════════════════════════════════════════════
# TEST FUENTE 2: Stooq CSV directo con fechas
# ══════════════════════════════════════════════════════════════════════

def test_stooq_direct(ticker: str) -> tuple[bool, str]:
    try:
        sym      = TICKER_MAP.get(ticker, {}).get("stooq", f"{ticker.lower()}.cl")
        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=365)
        d1, d2   = start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
        url      = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&d2={d2}&i=d"

        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer":    "https://stooq.com/",
        })
        resp = s.get(url, timeout=15)
        resp.raise_for_status()
        text = resp.text.strip()

        if len(text) < 30 or "No data" in text or "<html" in text.lower():
            return False, f"respuesta inválida: {text[:60]!r}"

        import pandas as pd
        df = pd.read_csv(StringIO(text))
        df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            return False, f"sin columna 'close': {df.columns.tolist()}"
        if len(df) < 10:
            return False, f"muy pocas filas: {len(df)}"

        return True, f"{sym} → {len(df)} filas"
    except Exception as e:
        return False, str(e)[:80]


# ══════════════════════════════════════════════════════════════════════
# TEST FUENTE 3: pandas_datareader + Stooq
# ══════════════════════════════════════════════════════════════════════

def test_pdr_stooq(ticker: str) -> tuple[bool, str]:
    try:
        import pandas_datareader as pdr
        sym      = TICKER_MAP.get(ticker, {}).get("stooq", f"{ticker.lower()}.cl").upper()
        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=365)
        df       = pdr.get_data_stooq(sym, start=start_dt, end=end_dt)
        if df.empty or len(df) < 10:
            return False, f"sin datos para {sym}"
        return True, f"{sym} → {len(df)} filas"
    except ImportError:
        return False, "pandas_datareader no instalado → pip install pandas-datareader"
    except Exception as e:
        return False, str(e)[:80]


# ══════════════════════════════════════════════════════════════════════
# TEST FUENTE 4: Alpha Vantage
# ══════════════════════════════════════════════════════════════════════

def test_alpha_vantage(ticker: str) -> tuple[bool, str]:
    if not ALPHA_VANTAGE_KEY:
        return False, "ALPHA_VANTAGE_KEY no configurado en .env"
    try:
        url  = (f"https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY_ADJUSTED"
                f"&symbol={ticker}&outputsize=compact"
                f"&apikey={ALPHA_VANTAGE_KEY}")
        resp = requests.get(url, timeout=20)
        data = resp.json()
        if "Time Series (Daily)" in data:
            rows = len(data["Time Series (Daily)"])
            return True, f"{ticker} → {rows} filas"
        if "Note" in data:
            return False, "rate limit alcanzado (25 req/día free)"
        return False, str(data)[:60]
    except Exception as e:
        return False, str(e)[:80]


# ══════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════

def run():
    print("\n" + "═" * 70)
    print("  AgentTrader CL — Diagnóstico de Fuentes de Datos v2")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 70)

    source_results = {
        "yfinance":          {"ok": 0, "fail": 0},
        "Stooq-directo":     {"ok": 0, "fail": 0},
        "pandas_datareader": {"ok": 0, "fail": 0},
        "Alpha Vantage":     {"ok": 0, "fail": 0},
    }

    CARTERA = {"AGUAS-A", "ENJOY", "LTM", "SOCOVESA", "VAPORES"}

    for ticker in TICKERS:
        label = "🔵 CARTERA" if ticker in CARTERA else "⚪ WATCH  "
        print(f"\n{label} {ticker}")
        print("  " + "─" * 60)

        # yfinance
        ok, msg = test_yfinance(ticker)
        icon = "✅" if ok else "❌"
        print(f"  {icon} [1] yfinance:          {msg}")
        source_results["yfinance"]["ok" if ok else "fail"] += 1
        time.sleep(0.3)

        # Stooq directo
        ok2, msg2 = test_stooq_direct(ticker)
        icon2 = "✅" if ok2 else "❌"
        print(f"  {icon2} [2] Stooq directo:    {msg2}")
        source_results["Stooq-directo"]["ok" if ok2 else "fail"] += 1
        time.sleep(0.3)

        # pandas_datareader (solo primer ticker para no demorar)
        if ticker == TICKERS[0]:
            ok3, msg3 = test_pdr_stooq(ticker)
            icon3 = "✅" if ok3 else "❌"
            print(f"  {icon3} [3] pandas_datareader: {msg3}")
            source_results["pandas_datareader"]["ok" if ok3 else "fail"] += 1

        # Alpha Vantage (solo primer ticker)
        if ticker == TICKERS[0]:
            ok4, msg4 = test_alpha_vantage(ticker)
            icon4 = "✅" if ok4 else "❌"
            print(f"  {icon4} [4] Alpha Vantage:    {msg4}")
            source_results["Alpha Vantage"]["ok" if ok4 else "fail"] += 1

    # ── RESUMEN ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  RESUMEN POR FUENTE")
    print("─" * 70)
    any_working = False
    best_source = None
    for src, counts in source_results.items():
        total = counts["ok"] + counts["fail"]
        if total == 0:
            continue
        pct = counts["ok"] / total * 100
        bar = "█" * counts["ok"] + "░" * counts["fail"]
        icon = "✅" if counts["ok"] > 0 else "❌"
        print(f"  {icon}  {src:<22} [{bar}] {counts['ok']}/{total} ({pct:.0f}%)")
        if counts["ok"] > 0 and best_source is None:
            best_source = src
            any_working = True

    print("\n" + "─" * 70)

    if any_working:
        print(f"\n  🎉 FUENTE FUNCIONAL DETECTADA: {best_source}")
        print(f"     El agente usará esta fuente automáticamente.")
        print(f"\n  → Puedes proceder con: python main.py --mode train")
    else:
        print("\n  ⛔ NINGUNA FUENTE DE DATOS FUNCIONA")
        print("\n  PLAN DE ACCIÓN (en orden de facilidad):\n")
        print("  PASO 1 — Instalar pandas-datareader:")
        print("    pip install pandas-datareader")
        print()
        print("  PASO 2 — Si sigue fallando, obtener API key gratuita:")
        print("    → https://www.alphavantage.co/support/#api-key")
        print("    → Agregar en .env:  ALPHA_VANTAGE_KEY=tu_clave_aqui")
        print()
        print("  PASO 3 — Si estás detrás de VPN o proxy corporativo:")
        print("    → Desactivar VPN durante la descarga inicial")
        print("    → O configurar proxy en requests.Session()")
        print()
        print("  PASO 4 — Bajar versión de yfinance (más compatible):")
        print("    pip install yfinance==0.2.36 --force-reinstall")

    print("\n" + "═" * 70 + "\n")


if __name__ == "__main__":
    run()
