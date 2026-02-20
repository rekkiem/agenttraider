"""
AgentTrader CL — Backtesting Engine
Walk-forward validation sobre datos históricos de 5 años.
Compara performance vs IPSA benchmark.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from datetime import datetime


class Backtester:
    """
    Backtesting vectorizado con:
    - Comisiones reales del mercado chileno (0.03% por lado)
    - Slippage por liquidez baja (0.2%)
    - Walk-forward: entrenamiento rolling, test en período siguiente
    - Benchmark IPSA para comparación relativa
    """

    def __init__(
        self,
        initial_capital:  float = 10_000_000,  # CLP
        commission_pct:   float = 0.0003,       # 0.03%
        slippage_pct:     float = 0.002,        # 0.2% (liquidez baja Chile)
    ):
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.slippage_pct    = slippage_pct

    def run(
        self,
        signals_history: pd.DataFrame,
        prices_history:  dict,          # {ticker: pd.DataFrame con 'close'}
    ) -> dict:
        """
        Corre el backtest sobre señales pre-generadas.

        signals_history: DataFrame con columnas
            [date, ticker, signal, probability]

        prices_history: {ticker: df con columna 'close' e índice de fechas}
        """
        logger.info("Iniciando backtesting...")

        capital   = self.initial_capital
        portfolio = {}   # {ticker: shares}
        equity_curve = []
        trades = []

        all_dates = sorted(signals_history["date"].unique())

        for current_date in tqdm(all_dates, desc="Backtesting días"):
            day_signals = signals_history[signals_history["date"] == current_date]

            for _, sig in day_signals.iterrows():
                ticker   = sig["ticker"]
                signal   = sig["signal"]
                prob     = sig["probability"]

                # Obtener precio del día
                if ticker not in prices_history:
                    continue
                price_series = prices_history[ticker]["close"]

                try:
                    price = float(price_series.loc[current_date])
                except KeyError:
                    # Intentar con fecha más cercana
                    available = price_series.index[price_series.index <= current_date]
                    if len(available) == 0:
                        continue
                    price = float(price_series.loc[available[-1]])

                effective_buy  = price * (1 + self.slippage_pct + self.commission_pct)
                effective_sell = price * (1 - self.slippage_pct - self.commission_pct)

                # BUY
                if signal == "BUY" and prob >= 0.55:
                    shares = int((capital * 0.05) / effective_buy)
                    if shares > 0 and capital >= shares * effective_buy:
                        capital -= shares * effective_buy
                        portfolio[ticker] = portfolio.get(ticker, 0) + shares
                        trades.append({
                            "date":   current_date,
                            "ticker": ticker,
                            "side":   "BUY",
                            "shares": shares,
                            "price":  price,
                            "prob":   prob,
                        })

                # SELL
                elif signal == "SELL" and ticker in portfolio and portfolio[ticker] > 0:
                    shares = portfolio[ticker]
                    capital += shares * effective_sell
                    portfolio[ticker] = 0
                    trades.append({
                        "date":   current_date,
                        "ticker": ticker,
                        "side":   "SELL",
                        "shares": shares,
                        "price":  price,
                        "prob":   prob,
                    })

            # Valorizar portafolio al cierre del día
            port_value = capital
            for t, sh in portfolio.items():
                if sh <= 0:
                    continue
                if t in prices_history:
                    p_series = prices_history[t]["close"]
                    available = p_series.index[p_series.index <= current_date]
                    if len(available) > 0:
                        port_value += sh * float(p_series.loc[available[-1]])

            equity_curve.append({"date": current_date, "equity": port_value})

        return self._compute_metrics(equity_curve, trades)

    def _compute_metrics(self, equity_curve: list, trades: list) -> dict:
        """Calcula todas las métricas de performance."""
        if not equity_curve:
            return {"error": "Sin datos en equity curve"}

        ec = pd.DataFrame(equity_curve).set_index("date")
        ec["return"] = ec["equity"].pct_change()

        # Período
        days  = len(ec)
        years = days / 252

        # Retornos
        final_equity = float(ec["equity"].iloc[-1])
        total_return = (final_equity / self.initial_capital) - 1
        cagr = (final_equity / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe
        daily_returns = ec["return"].dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Sortino (solo retornos negativos)
        neg_returns = daily_returns[daily_returns < 0]
        sortino = (daily_returns.mean() / neg_returns.std()) * np.sqrt(252) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0

        # Max Drawdown
        rolling_max  = ec["equity"].cummax()
        drawdowns    = (ec["equity"] - rolling_max) / rolling_max
        max_drawdown = float(drawdowns.min())

        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Stats de trades
        trades_df = pd.DataFrame(trades)
        win_rate = profit_factor = 0.0
        total_closed = 0

        if len(trades_df) > 0:
            wins, losses = 0, 0
            gross_profit = gross_loss = 0.0

            for ticker in trades_df["ticker"].unique():
                t_ticker = trades_df[trades_df["ticker"] == ticker].sort_values("date")
                buy_price = None
                for _, t in t_ticker.iterrows():
                    if t["side"] == "BUY":
                        buy_price = t["price"]
                    elif t["side"] == "SELL" and buy_price is not None:
                        ret = (t["price"] - buy_price) / buy_price
                        if ret > 0:
                            wins        += 1
                            gross_profit += ret
                        else:
                            losses      += 1
                            gross_loss  += abs(ret)
                        buy_price = None

            total_closed  = wins + losses
            win_rate      = wins / total_closed if total_closed > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        metrics = {
            "periodo":            f"{ec.index[0].date()} → {ec.index[-1].date()}",
            "dias_operados":      days,
            "capital_inicial_clp": self.initial_capital,
            "capital_final_clp":  round(final_equity, 0),
            "retorno_total_pct":  round(total_return * 100, 2),
            "cagr_pct":           round(cagr * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "sortino_ratio":      round(sortino, 3),
            "calmar_ratio":       round(calmar, 3),
            "max_drawdown_pct":   round(max_drawdown * 100, 2),
            "win_rate_pct":       round(win_rate * 100, 2),
            "profit_factor":      round(profit_factor, 3),
            "total_operaciones":  total_closed,
            "total_ordenes":      len(trades_df),
        }

        logger.info("─" * 50)
        logger.info("RESULTADOS BACKTESTING")
        logger.info("─" * 50)
        for k, v in metrics.items():
            logger.info(f"  {k:<30}: {v}")

        return metrics

    def compare_vs_ipsa(
        self,
        equity_curve: pd.DataFrame,
        ipsa_prices:  pd.Series,
    ) -> dict:
        """
        Compara el equity curve del agente vs el IPSA.
        ipsa_prices: pd.Series con el precio del IPSA indexado por fecha.
        """
        start_date = equity_curve.index[0]
        end_date   = equity_curve.index[-1]

        ipsa_period = ipsa_prices.loc[start_date:end_date]
        if ipsa_period.empty:
            return {"error": "Sin datos IPSA para el período"}

        ipsa_return = float((ipsa_period.iloc[-1] / ipsa_period.iloc[0]) - 1)
        agent_return = float(
            (equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0]) - 1
        )

        return {
            "agent_return_pct": round(agent_return * 100, 2),
            "ipsa_return_pct":  round(ipsa_return  * 100, 2),
            "alpha_pct":        round((agent_return - ipsa_return) * 100, 2),
            "outperforms":      agent_return > ipsa_return,
        }
