"""
AgentTrader CL — Módulo de Notificaciones
Envía alertas por email vía SMTP cuando el agente
detecta oportunidades de compra o venta.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from loguru import logger

from config import EMAIL_CONFIG
from engine.decision import TradingSignal


# ── CONSTRUCTOR DE EMAIL ──────────────────────────────────────────────

def build_email(signal: TradingSignal, portfolio_eval: dict = None) -> tuple[str, str]:
    """
    Construye el asunto y cuerpo HTML del correo de alerta.
    Retorna (subject, html_body).
    """
    is_buy  = signal.signal == "BUY"
    color   = "#1B5E20" if is_buy else "#B71C1C"
    emoji   = "🟢" if is_buy else "🔴"
    verb    = "Compra" if is_buy else "Venta"
    arrow   = "▲ COMPRAR" if is_buy else "▼ VENDER"

    subject = f"{emoji} {verb} {signal.ticker} | P={signal.probability:.0%} | Conf={signal.confidence}"

    r = signal.reasoning
    capital_involucrado = signal.suggested_size * signal.current_price

    # Sección de evaluación de posición actual
    portfolio_section = ""
    if portfolio_eval:
        rec_color = {
            "CUT_LOSS":     "#C62828",
            "AVERAGE_DOWN": "#1565C0",
            "TAKE_PROFIT":  "#2E7D32",
            "HOLD":         "#555555",
        }.get(portfolio_eval.get("recommendation", "HOLD"), "#555555")

        portfolio_section = f"""
        <h3 style="margin-top:24px;">📂 Evaluación Posición Actual</h3>
        <table border="1" cellpadding="8" cellspacing="0" width="100%"
               style="border-collapse:collapse;">
            <tr><td style="width:50%">P/L No Realizado</td>
                <td><b style="color:{rec_color}">{portfolio_eval.get('unrealized_pnl_pct',0):.2f}%</b></td></tr>
            <tr><td>Valor Posición Actual</td>
                <td>${portfolio_eval.get('valor_posicion_clp',0):,.0f} CLP</td></tr>
            <tr><td>Recomendación</td>
                <td><b style="color:{rec_color}">{portfolio_eval.get('recommendation','N/A')}</b></td></tr>
            <tr><td>Justificación</td>
                <td>{portfolio_eval.get('reason','')}</td></tr>
        </table>
        """

    # RSI interpretation
    rsi = r.get("rsi_14", 50)
    rsi_label = (
        "Sobrevendido ✅ (señal alcista)"   if rsi < 35 else
        "Sobrecomprado ⚠️ (señal bajista)" if rsi > 70 else
        "Zona neutral"
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: Arial, sans-serif; background:#f5f5f5; margin:0; padding:20px; }}
  .container {{ max-width:700px; margin:auto; background:#fff;
                border-radius:8px; overflow:hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,.15); }}
  .header {{ background:{color}; color:#fff; padding:20px 24px; }}
  .header h1 {{ margin:0; font-size:22px; }}
  .header p  {{ margin:4px 0 0; opacity:.85; font-size:14px; }}
  .body {{ padding:24px; }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:16px; }}
  td {{ padding:8px 10px; border:1px solid #e0e0e0; font-size:14px; }}
  tr:nth-child(even) {{ background:#fafafa; }}
  td:first-child {{ font-weight:bold; width:45%; color:#333; }}
  h3 {{ color:#333; margin-top:20px; margin-bottom:8px; font-size:15px; }}
  .badge {{ display:inline-block; padding:4px 10px; border-radius:12px;
            font-weight:bold; font-size:13px; }}
  .footer {{ background:#f5f5f5; padding:14px 24px; font-size:12px; color:#888;
             border-top:1px solid #e0e0e0; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>{emoji} SEÑAL {signal.signal}: {signal.ticker}</h1>
    <p>AgentTrader CL | {datetime.now().strftime('%d %b %Y %H:%M')} CLT</p>
  </div>

  <div class="body">

    <h3>📊 Resumen de la Operación</h3>
    <table>
      <tr><td>Acción</td>
          <td><b>{signal.ticker}</b></td></tr>
      <tr><td>Señal</td>
          <td><span class="badge" style="background:{color};color:#fff;">{arrow}</span></td></tr>
      <tr><td>Probabilidad de Éxito</td>
          <td><b style="font-size:18px;">{signal.probability:.1%}</b></td></tr>
      <tr><td>Nivel de Confianza</td>
          <td>{signal.confidence}</td></tr>
      <tr><td>Precio Actual</td>
          <td>${signal.current_price:,.2f} CLP</td></tr>
      <tr><td>Cantidad Recomendada</td>
          <td><b>{signal.suggested_size:,} acciones</b></td></tr>
      <tr><td>Capital Involucrado</td>
          <td>${capital_involucrado:,.0f} CLP</td></tr>
      <tr><td>Stop Loss</td>
          <td style="color:#C62828;"><b>${signal.stop_loss:,.2f} CLP</b></td></tr>
      <tr><td>Take Profit</td>
          <td style="color:#1B5E20;"><b>${signal.take_profit:,.2f} CLP</b></td></tr>
      <tr><td>Retorno Esperado</td>
          <td>{signal.expected_return:.2f}%</td></tr>
      <tr><td>Riesgo sobre Cartera</td>
          <td>{signal.risk_pct:.2%}</td></tr>
    </table>

    <h3>🔬 Justificación Técnica</h3>
    <table>
      <tr><td>RSI (14)</td>
          <td>{rsi:.1f} — {rsi_label}</td></tr>
      <tr><td>MACD Histograma</td>
          <td>{r.get('macd_hist',0):.4f} {"✅ Alcista" if r.get('macd_hist',0)>0 else "⚠️ Bajista"}</td></tr>
      <tr><td>Posición en Bollinger</td>
          <td>{r.get('bb_pct',0.5):.1%}
            {"(cerca del piso ✅)" if r.get('bb_pct',0.5)<0.2
              else "(cerca del techo ⚠️)" if r.get('bb_pct',0.5)>0.8
              else "(zona media)"}</td></tr>
      <tr><td>Volumen vs Promedio 20d</td>
          <td>{r.get('volume_ratio',1):.2f}x
            {"🔔 Volumen anómalo" if r.get('volume_ratio',1)>1.5 else ""}</td></tr>
      <tr><td>Momentum (multi-período)</td>
          <td>{r.get('momentum_score',0):.4f}
            {"✅ Positivo" if r.get('momentum_score',0)>0 else "Negativo"}</td></tr>
      <tr><td>Volatilidad Anualizada</td>
          <td>{r.get('vol_20d_ann',0):.1%}</td></tr>
      <tr><td>Tendencia Corto Plazo</td>
          <td>{"▲ Sobre EMA20" if r.get('trend_short',0)>0 else "▼ Bajo EMA20"}</td></tr>
      <tr><td>Tendencia Largo Plazo</td>
          <td>{"▲ Sobre EMA200" if r.get('trend_long',0)>0 else "▼ Bajo EMA200"}</td></tr>
      <tr><td>ATR (14 períodos)</td>
          <td>${r.get('atr_14',0):.2f} CLP</td></tr>
      <tr><td>Prob. Ensemble ML</td>
          <td>{r.get('ensemble_prob',0):.1%}</td></tr>
      <tr><td>Dirección Prophet</td>
          <td>{"▲ Alza" if r.get('prophet_direction',0)==1 else "▼ Baja"}
            {r.get('prophet_upside_pct',0):+.1f}%</td></tr>
    </table>

    {portfolio_section}

  </div>

  <div class="footer">
    ⚠️ Este análisis es generado automáticamente con fines informativos.
    No constituye asesoría de inversión. Opera siempre con tu propio criterio
    y bajo la supervisión de un asesor certificado.<br>
    AgentTrader CL v1.0 | perfumes.arhom@gmail.com
  </div>

</div>
</body>
</html>
"""
    return subject, html


# ── ENVÍO DE EMAIL ────────────────────────────────────────────────────

def send_alert(signal: TradingSignal, portfolio_eval: dict = None) -> bool:
    """
    Envía correo de alerta para señales BUY o SELL.
    Retorna True si el envío fue exitoso.
    """
    if signal.signal == "HOLD":
        logger.debug(f"{signal.ticker}: HOLD — correo no enviado")
        return True

    if not EMAIL_CONFIG["user"] or not EMAIL_CONFIG["password"]:
        logger.warning("Gmail no configurado (GMAIL_USER / GMAIL_APP_PASS vacíos)")
        return False

    subject, body = build_email(signal, portfolio_eval)

    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_CONFIG["user"]
    msg["To"]      = EMAIL_CONFIG["to"]
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        with smtplib.SMTP(EMAIL_CONFIG["smtp_host"], EMAIL_CONFIG["smtp_port"]) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_CONFIG["user"], EMAIL_CONFIG["password"])
            server.sendmail(EMAIL_CONFIG["user"], EMAIL_CONFIG["to"], msg.as_string())

        logger.success(f"Email enviado: {subject[:60]}...")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "Error de autenticación SMTP. "
            "Verifica GMAIL_USER y GMAIL_APP_PASS en tu .env. "
            "Asegúrate de usar un App Password de Google, no tu contraseña normal."
        )
        return False
    except Exception as e:
        logger.error(f"Error enviando email: {e}")
        return False


def send_system_alert(subject: str, body: str) -> bool:
    """Alerta genérica del sistema (errores críticos, fallos del scheduler)."""
    if not EMAIL_CONFIG["user"]:
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"]    = EMAIL_CONFIG["user"]
    msg["To"]      = EMAIL_CONFIG["to"]
    msg["Subject"] = f"🚨 AgentTrader CL — {subject}"

    try:
        with smtplib.SMTP(EMAIL_CONFIG["smtp_host"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["user"], EMAIL_CONFIG["password"])
            server.send_message(msg)
        logger.info(f"Alerta del sistema enviada: {subject}")
        return True
    except Exception as e:
        logger.error(f"Error en alerta del sistema: {e}")
        return False
