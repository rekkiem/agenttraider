# 🤖 AgentTrader CL
### Agente Experto de Trading — Bolsa Chilena

Sistema cuantitativo de análisis y señales para el mercado accionario chileno.
Opera con datos de Inversiones Security y modelos ML ensemble.

---

## 🚀 Setup Rápido (5 pasos)

### 1. Clonar y abrir en Cursor

```bash
git clone <tu-repo> agenttrader
cd agenttrader
# Abrir en Cursor: File → Open Folder
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
playwright install chromium
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus datos reales:
# - DB_PASSWORD
# - GMAIL_USER
# - GMAIL_APP_PASS (Google App Password)
# - PORTFOLIO_VALUE_CLP
```

**Para Gmail App Password:**
1. Activa la verificación en dos pasos en tu cuenta Google
2. Ve a: Google Account → Security → App Passwords
3. Crea una app password para "Mail"
4. Copia el código de 16 caracteres en GMAIL_APP_PASS

### 4. Iniciar PostgreSQL

**Opción A — Docker (recomendado):**
```bash
docker compose up postgres -d
```

**Opción B — PostgreSQL local:**
```bash
psql -U postgres -c "CREATE DATABASE agenttrader;"
psql -U postgres -c "CREATE USER trader WITH PASSWORD 'tu_password';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE agenttrader TO trader;"
psql -U trader -d agenttrader -f sql/init.sql
```

### 5. Ejecutar el agente

```bash
# Entrenar modelos (primera vez, tarda ~5 min)
python main.py --mode train

# Análisis manual ahora mismo
python main.py --mode run-now

# Iniciar con scheduler automático (producción)
python main.py --mode scheduler

# Correr backtesting
python main.py --mode backtest
```

---

## 📁 Estructura del Proyecto

```
agenttrader/
├── main.py                     # Orquestador principal + scheduler
├── config.py                   # Configuración central
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
│
├── ingestion/
│   └── scraper.py              # Playwright + yfinance
│
├── models/
│   ├── features.py             # Feature engineering (RSI, MACD, ATR...)
│   └── predictive.py           # Ensemble ML + Prophet
│
├── engine/
│   └── decision.py             # Motor de señales BUY/SELL/HOLD
│
├── risk/
│   └── manager.py              # Validación de riesgo, VaR
│
├── notifications/
│   └── email_sender.py         # Alertas SMTP HTML
│
├── backtesting/
│   └── engine.py               # Walk-forward backtest
│
├── sql/
│   └── init.sql                # Schema PostgreSQL
│
├── saved_models/               # Modelos entrenados (ignorado por git)
└── logs/                       # Logs del sistema (ignorado por git)
```

---

## ⚙️ Configuración de la Cartera

Edita `config.py` — sección `PORTFOLIO`:

```python
PORTFOLIO = {
    "AGUAS-A":  {"shares": 1000,   "avg_cost": 350.0},   # ← Tu precio promedio
    "ENJOY":    {"shares": 150000, "avg_cost": 28.5},
    "LTM":      {"shares": 300,    "avg_cost": 7200.0},
    "SOCOVESA": {"shares": 5000,   "avg_cost": 185.0},
    "VAPORES":  {"shares": 1570,   "avg_cost": 890.0},
}

PORTFOLIO_VALUE_CLP = 15_000_000  # ← Tu capital total
```

El `avg_cost` es **crítico** para que el agente evalúe correctamente
si conviene promediar a la baja, cortar pérdidas o rotar capital.

---

## 📧 Formato de Email de Alerta

Cuando el agente detecta una señal con probabilidad ≥ 55%, envía a `perfumes.arhom@gmail.com`:

```
🟢 Compra AGUAS-A | P=72% | Conf=MEDIUM
```

El correo incluye:
- Precio actual y cantidad recomendada
- Stop Loss y Take Profit (basados en ATR)
- Breakdown técnico completo (RSI, MACD, Bollinger, Volumen)
- Evaluación P&L de la posición actual
- Recomendación: CUT_LOSS / AVERAGE_DOWN / HOLD / TAKE_PROFIT

---

## 🐳 Deploy en VPS (Producción)

```bash
# En tu VPS Ubuntu 22.04
curl -fsSL https://get.docker.com | sh
git clone <tu-repo> /opt/agenttrader
cd /opt/agenttrader
cp .env.example .env && nano .env   # Completar con datos reales

docker compose up -d

# Ver logs en tiempo real
docker compose logs -f agent

# La DB es accesible en http://tu-vps-ip:8080 (Adminer)
# Usuario: trader | Base de datos: agenttrader
```

---

## 🧪 Modos de Ejecución

| Comando | Descripción |
|---------|-------------|
| `python main.py --mode scheduler` | Scheduler automático (producción) |
| `python main.py --mode run-now`   | Análisis una vez ahora mismo |
| `python main.py --mode train`     | Entrenar/reentrenar modelos |
| `python main.py --mode backtest`  | Backtest histórico con métricas |

---

## ⚠️ Advertencias del Mercado Chileno

1. **ENJOY (150.000 acciones):** Esta posición es muy grande relativa a la liquidez diaria. El agente limitará automáticamente el tamaño de orden para no impactar el precio. La salida debe ser gradual en múltiples días.

2. **LTM:** Latam Airlines cotiza en el IPSA pero su principal liquidez está en Nueva York (ADR). Las señales pueden ser menos precisas en momentos de alta volatilidad sectorial.

3. **Horario:** El agente analiza a las 15:45 CLT, 45 minutos antes del cierre. En días de alta volatilidad, el precio al momento de ejecutar puede diferir del precio analizado.

4. **Datos:** El scraper puede fallar si Security cambia su HTML. En ese caso usa yfinance como fallback automático.

---

## 📊 Métricas del Backtesting (Targets)

| Métrica | Target |
|---------|--------|
| CAGR | > 10% anual |
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < 15% |
| Win Rate | > 52% |
| Profit Factor | > 1.5 |

---

## 🔧 Troubleshooting

**El scraper falla:**
```
Error: Timeout waiting for selector 'table'
```
→ Security puede haber cambiado su layout. Revisar `ingestion/scraper.py`
y ajustar los selectores CSS. El sistema usará yfinance automáticamente como fallback.

**Error de autenticación Gmail:**
```
SMTPAuthenticationError
```
→ Usar App Password, no tu contraseña real. Verificar que la verificación en 2 pasos esté activa.

**Modelo no entrenado:**
```
FileNotFoundError: saved_models/ensemble_latest.pkl
```
→ Ejecutar `python main.py --mode train` primero.

**PostgreSQL sin conexión:**
→ Verificar que el contenedor esté corriendo: `docker compose ps`
→ Revisar credenciales en `.env`

---

*AgentTrader CL v1.0 — Solo para uso informativo. No es asesoría de inversión.*
