import os
import json
import math
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:   # sin la librería no hay datos, pero el bot no se rompe
    yf = None

VERSION = "1.3"

# ==============================================================================
# CONFIGURACIÓN DE TELEGRAM Y WEB
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
APP_URL = os.getenv("APP_URL", "https://franciscoserranoadt.github.io/bot-quant-mercados/")

TZ_MADRID = ZoneInfo("Europe/Madrid")

# ==============================================================================
# UNIVERSO DE ACTIVOS (datos gratuitos de Yahoo Finance vía yfinance, sin clave)
# Para añadir o quitar activos basta con editar estas listas.
# ==============================================================================
CATEGORIAS = {
    "cripto":   {"name": "Cripto", "icono": "🪙"},
    "materias": {"name": "Materias primas", "icono": "🛢️"},
    "indices":  {"name": "Índices", "icono": "📊"},
    "acciones": {"name": "Acciones", "icono": "🏢"},
}

CRIPTOS = [("BTC-USD", "Bitcoin", "BTC"), ("ETH-USD", "Ethereum", "ETH"), ("SOL-USD", "Solana", "SOL"),
           ("XRP-USD", "XRP", "XRP"), ("BNB-USD", "BNB", "BNB"), ("DOGE-USD", "Dogecoin", "DOGE"),
           ("ADA-USD", "Cardano", "ADA"), ("AVAX-USD", "Avalanche", "AVAX"), ("LINK-USD", "Chainlink", "LINK"),
           ("LTC-USD", "Litecoin", "LTC")]
MATERIAS = [("GC=F", "Oro", "ORO"), ("CL=F", "Petróleo WTI", "WTI"), ("BZ=F", "Petróleo Brent", "BRENT")]
INDICES = [("^GSPC", "S&P 500", "SPX")]
# 50 mayores empresas por capitalización que cotizan en EE. UU. (acciones o ADR). Revísala de vez en cuando.
ACCIONES = [("NVDA", "Nvidia"), ("MSFT", "Microsoft"), ("AAPL", "Apple"), ("GOOGL", "Alphabet"), ("AMZN", "Amazon"),
            ("META", "Meta Platforms"), ("AVGO", "Broadcom"), ("TSM", "TSMC"), ("TSLA", "Tesla"),
            ("BRK-B", "Berkshire Hathaway"), ("JPM", "JPMorgan Chase"), ("LLY", "Eli Lilly"), ("WMT", "Walmart"),
            ("V", "Visa"), ("ORCL", "Oracle"), ("MA", "Mastercard"), ("XOM", "Exxon Mobil"), ("NFLX", "Netflix"),
            ("COST", "Costco"), ("JNJ", "Johnson & Johnson"), ("PLTR", "Palantir"), ("HD", "Home Depot"),
            ("ABBV", "AbbVie"), ("BAC", "Bank of America"), ("PG", "Procter & Gamble"), ("AMD", "AMD"),
            ("SAP", "SAP"), ("ASML", "ASML"), ("KO", "Coca-Cola"), ("CVX", "Chevron"), ("GE", "GE Aerospace"),
            ("UNH", "UnitedHealth"), ("CSCO", "Cisco"), ("TMUS", "T-Mobile US"), ("PM", "Philip Morris"),
            ("WFC", "Wells Fargo"), ("IBM", "IBM"), ("MS", "Morgan Stanley"), ("GS", "Goldman Sachs"),
            ("CRM", "Salesforce"), ("AXP", "American Express"), ("BABA", "Alibaba"), ("NVO", "Novo Nordisk"),
            ("AZN", "AstraZeneca"), ("MU", "Micron"), ("LIN", "Linde"), ("MCD", "McDonald's"), ("ABT", "Abbott"),
            ("TM", "Toyota"), ("RTX", "RTX")]

ACTIVOS = {}
for _t, _n, _s in CRIPTOS:
    ACTIVOS[_t] = {"nombre": _n, "simbolo": _s, "cat": "cripto"}
for _t, _n, _s in MATERIAS:
    ACTIVOS[_t] = {"nombre": _n, "simbolo": _s, "cat": "materias"}
for _t, _n, _s in INDICES:
    ACTIVOS[_t] = {"nombre": _n, "simbolo": _s, "cat": "indices"}
for _t, _n in ACCIONES:
    ACTIVOS[_t] = {"nombre": _n, "simbolo": _t.replace("-", "."), "cat": "acciones"}

# ==============================================================================
# PARÁMETROS DE LA ESTRATEGIA (velas de 1 hora)
# ==============================================================================
INTERVALO = "1h"
PERIODO_LIVE = "60d"            # velas para calcular la señal de cada ciclo
PERIODO_BT = "1y"               # velas para el histórico (backtest) de cada activo
BT_TTL_H = float(os.getenv("BT_TTL_H", "24"))            # el backtest se recalcula una vez al día
ATR_SL = float(os.getenv("ATR_SL", "1.5"))               # stop loss = 1,5 x ATR
ATR_TP = float(os.getenv("ATR_TP", "3.0"))               # take profit = 3 x ATR
RR = ATR_TP / ATR_SL                                      # beneficio/riesgo (2,0 por defecto)
MAX_BARRAS = int(os.getenv("MAX_BARRAS", "48"))           # vida máxima de una señal (velas de 1 h)
EV_MINIMO_PCT = float(os.getenv("EV_MINIMO_PCT", "15"))   # ventaja mínima (expectativa en % del riesgo) para Telegram
EV_MAXIMO_PCT = 150.0                                     # por encima: dato sospechoso
BT_MIN_SENALES = int(os.getenv("BT_MIN_SENALES", "10"))   # señales históricas mínimas para confiar en el backtest
BT_PRIOR = 5.0                                            # "señales virtuales" de regresión a cero
PROB_MINIMA = float(os.getenv("PROB_MINIMA", "0.30"))
COOLDOWN_H = float(os.getenv("COOLDOWN_H", "6"))          # horas sin nueva señal tras cerrar una en el mismo activo
APALANCAMIENTO_MAX = int(os.getenv("APALANCAMIENTO_MAX", "20"))      # tope de apalancamiento sugerido
PERDIDA_MARGEN_SL = float(os.getenv("PERDIDA_MARGEN_SL", "0.10"))  # tocar el stop = perder como mucho el 10 % del margen
VELAS_PREVIAS = 12          # velas anteriores a la entrada que se guardan como contexto para el gráfico de velas
LOTE_DESCARGA = 20
MAX_SENALES_GUARDADAS = 400

ESTADO_DATOS = {"descargas_fallidas": 0}


# ==============================================================================
# DESCARGA DE DATOS
# ==============================================================================
def descargar(tickers, periodo):
    datos = {}
    if yf is None:
        print("⚠️ yfinance no está instalado")
        return datos
    for k in range(0, len(tickers), LOTE_DESCARGA):
        lote = tickers[k:k + LOTE_DESCARGA]
        raw = None
        for intento in range(3):
            try:
                raw = yf.download(lote, period=periodo, interval=INTERVALO, group_by="ticker",
                                  auto_adjust=False, progress=False, threads=True)
                if raw is not None and not raw.empty:
                    break
            except Exception as e:
                print(f"⚠️ Error descargando {lote[0]}..: {e}")
            time.sleep(5 * (intento + 1))
        if raw is None or raw.empty:
            ESTADO_DATOS["descargas_fallidas"] += 1
            continue
        for t in lote:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t not in raw.columns.get_level_values(0):
                        continue
                    df = raw[t]
                else:
                    df = raw
                df = df[["Open", "High", "Low", "Close"]].dropna().astype(float)
            except Exception:
                continue
            if len(df) < 60:
                continue
            idx = pd.DatetimeIndex(df.index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            df.index = idx
            df = df[~df.index.duplicated(keep="last")].sort_index()
            datos[t] = df
        time.sleep(1.0)
    return datos


# ==============================================================================
# INDICADORES Y REGLA DE SEÑAL
# LONG: tendencia alcista (EMA20 > EMA50 > EMA200), precio sobre la EMA50, el RSI recupera 50
#       tras un retroceso y el MACD acompaña. SHORT: lo mismo al revés.
# ==============================================================================
def indicadores(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    out = pd.DataFrame(index=df.index)
    out["close"], out["high"], out["low"] = c, h, l
    out["ema20"] = c.ewm(span=20, adjust=False).mean()
    out["ema50"] = c.ewm(span=50, adjust=False).mean()
    out["ema200"] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    rsi_prev = out["rsi"].shift(1)
    alcista = (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"])
    bajista = (out["ema20"] < out["ema50"]) & (out["ema50"] < out["ema200"])
    out["long"] = alcista & (c > out["ema50"]) & (rsi_prev < 50) & (out["rsi"] >= 50) & (out["hist"] > 0)
    out["short"] = bajista & (c < out["ema50"]) & (rsi_prev > 50) & (out["rsi"] <= 50) & (out["hist"] < 0)
    out.loc[out.index[:200], ["long", "short"]] = False   # la EMA200 necesita calentamiento
    return out


def tendencia_de(row):
    if row["ema20"] > row["ema50"] > row["ema200"]:
        return "ALCISTA"
    if row["ema20"] < row["ema50"] < row["ema200"]:
        return "BAJISTA"
    return "LATERAL"


# ==============================================================================
# BACKTEST PROPIO: la misma regla aplicada al último año de cada activo
# ==============================================================================
def _simular(close, high, low, atr, i, direccion):
    e, a = close[i], atr[i]
    if not (a > 0):
        return None
    n = len(close)
    larga = direccion == "LONG"
    sl = e - ATR_SL * a if larga else e + ATR_SL * a
    tp = e + ATR_TP * a if larga else e - ATR_TP * a
    fin = min(n - 1, i + MAX_BARRAS)
    for j in range(i + 1, fin + 1):
        if larga:
            if low[j] <= sl:
                return -1.0
            if high[j] >= tp:
                return RR
        else:
            if high[j] >= sl:
                return -1.0
            if low[j] <= tp:
                return RR
    if i + MAX_BARRAS <= n - 1:
        r = ((close[fin] - e) if larga else (e - close[fin])) / (ATR_SL * a)
        return max(-1.0, min(RR, r))
    return None   # todavía sin resolver: no cuenta


def backtest_activo(df):
    ind = indicadores(df)
    close, high, low, atr = (ind[k].to_numpy() for k in ("close", "high", "low", "atr"))
    res = {}
    for direccion, col in (("LONG", "long"), ("SHORT", "short")):
        rs = [r for r in (_simular(close, high, low, atr, i, direccion) for i in np.where(ind[col].to_numpy())[0])
              if r is not None]
        n = len(rs)
        if n == 0:
            res[direccion] = {"n": 0, "p": None, "v": None, "r_medio": None}
            continue
        wins = sum(1 for r in rs if r > 0)
        res[direccion] = {"n": n, "p": round(wins / n, 4), "v": round(sum(rs) / (n + BT_PRIOR) * 100, 1),
                          "r_medio": round(sum(rs) / n, 3)}
    return res


def actualizar_backtest(previo, tickers, ahora_utc):
    try:
        t = datetime.fromisoformat((previo or {}).get("t", "").replace("Z", "+00:00"))
        if previo.get("e") and (ahora_utc - t).total_seconds() < BT_TTL_H * 3600:
            return previo
    except Exception:
        pass
    print(f"📐 Recalculando backtest de 1 año para {len(tickers)} activos...")
    datos_bt = descargar(tickers, PERIODO_BT)
    if not datos_bt:
        return previo or {"t": "", "e": {}}
    e = dict((previo or {}).get("e") or {})
    for tk, df in datos_bt.items():
        try:
            e[tk] = backtest_activo(df)
        except Exception as ex:
            print(f"   Aviso backtest {tk}: {ex}")
    print(f"📐 Backtest listo ({len(datos_bt)} activos)")
    return {"t": ahora_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "e": e}


# ==============================================================================
# UTILIDADES
# ==============================================================================
def fp(v):
    if v is None:
        return "—"
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    if abs(v) >= 1:
        return f"{v:.4f}"
    return f"{v:.6f}"


def num(v, dec=4):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, dec)
    except Exception:
        return None


def obtener_identificador_semana(fecha_madrid):
    año, semana, _ = fecha_madrid.isocalendar()
    return f"{año}-W{semana:02d}"


def obtener_fechas_limite_semana(fecha_madrid):
    lunes = fecha_madrid - timedelta(days=fecha_madrid.weekday())
    return lunes.strftime("%Y-%m-%d"), (lunes + timedelta(days=6)).strftime("%Y-%m-%d")


def display_madrid(ts_utc):
    m = ts_utc.astimezone(TZ_MADRID)
    return m.strftime("%Y-%m-%d"), m.strftime("%H:%M"), f"{m.strftime('%d/%m')} • {m.strftime('%H:%M')}"


def cargar_datos_existentes():
    if os.path.exists("datos.json"):
        try:
            with open("datos.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Aviso leyendo datos existentes: {e}")
    return {}


AVISO_RIESGO = "⚠️ <i>Operar conlleva riesgo de pérdida • Información estadística, no es asesoramiento financiero</i>"


def enviar_telegram(mensaje, url_accion=None, texto_boton="📱 Abrir en Web App"):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": f"{mensaje}\n\n{AVISO_RIESGO}",
               "parse_mode": "HTML", "disable_web_page_preview": True}
    destino = url_accion or APP_URL
    if destino:
        payload["reply_markup"] = {"inline_keyboard": [[{"text": texto_boton, "url": destino}]]}
    for intento in range(3):
        try:
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                return True
            print(f"Telegram HTTP {res.status_code}: {res.text[:160]}")
            time.sleep(1.5)
        except Exception as e:
            print(f"Error Telegram intento {intento + 1}: {e}")
            time.sleep(2)
    return False


# ==============================================================================
# CREACIÓN, SEGUIMIENTO Y LIQUIDACIÓN DE SEÑALES
# ==============================================================================
def apalancamiento_sugerido(entrada, sl):
    """Apalancamiento con el que tocar el Stop Loss cuesta como mucho el 10 % del margen de la operación.
    Así la liquidación queda unas 10 veces más lejos que el stop. Mínimo 1x, máximo APALANCAMIENTO_MAX."""
    try:
        dist = abs(float(entrada) - float(sl)) / float(entrada)
    except Exception:
        return 1
    if not dist > 0:
        return 1
    return int(max(1, min(APALANCAMIENTO_MAX, math.floor(PERDIDA_MARGEN_SL / dist))))


def crear_senal(tk, info, direccion, ts_barra, precio, atr, fila, stats, ahora_utc, semana_id):
    larga = direccion == "LONG"
    sl = precio - ATR_SL * atr if larga else precio + ATR_SL * atr
    tp = precio + ATR_TP * atr if larga else precio - ATR_TP * atr
    n_bt = (stats or {}).get("n") or 0
    p = (stats or {}).get("p")
    v = (stats or {}).get("v")
    nota = None
    alertable = False
    if n_bt < BT_MIN_SENALES or p is None or v is None:
        nota = f"Histórico insuficiente ({n_bt}/{BT_MIN_SENALES} señales en 1 año): no se envía a Telegram"
    elif v > EV_MAXIMO_PCT:
        nota = "Ventaja histórica anómala: no se envía a Telegram"
    elif v < EV_MINIMO_PCT:
        nota = f"Ventaja inferior al {EV_MINIMO_PCT:g}%: no se envía a Telegram"
    elif p < PROB_MINIMA:
        nota = "Probabilidad de acierto demasiado baja: no se envía a Telegram"
    else:
        alertable = True
    stake = 1
    if p is not None:
        kelly = p - (1 - p) / RR
        stake = min(10, max(1, round(max(0.0, kelly) * 0.25 * 40)))
    fecha, hora, display = display_madrid(ahora_utc)
    cat = CATEGORIAS[info["cat"]]
    return {
        "id": f"{tk}_{ts_barra.strftime('%Y%m%d%H')}_{direccion}",
        "ticker": tk, "nombre": info["nombre"], "simbolo": info["simbolo"],
        "categoria": info["cat"], "cat_nombre": cat["name"], "cat_icono": cat["icono"],
        "direccion": direccion, "entrada": num(precio, 6), "sl": num(sl, 6), "tp": num(tp, 6),
        "rr": round(RR, 2), "atr": num(atr, 6), "apalancamiento": apalancamiento_sugerido(precio, sl),
        "fecha": fecha, "hora": hora, "fecha_hora_display": display,
        "ts_entrada_utc": ahora_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_ms": int(ahora_utc.timestamp() * 1000),
        "barra_senal_utc": ts_barra.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "semana_id": semana_id, "estado": "ABIERTA", "acertado": None, "resultado_r": None,
        "precio_actual": num(precio, 6), "r_actual": 0.0, "max_r": 0.0, "traza": [num(precio, 6)],
        "precio_cierre": None, "motivo_cierre": None, "fecha_cierre": None, "hora_cierre": None,
        "fecha_hora_cierre_display": None, "semana_cierre": None, "ts_cierre_utc": None,
        "stake": stake, "ventaja": v, "prob": round(p * 100, 1) if p is not None else None, "n_bt": n_bt,
        "confirmado_bt": alertable, "telegram_enviado": False, "liquidada_telegram": False, "nota": nota,
        "rsi": num(fila["rsi"], 1), "tendencia": tendencia_de(fila),
    }


def mensaje_senal(s):
    larga = s["direccion"] == "LONG"
    e = s["entrada"]
    pct_tp = (s["tp"] / e - 1) * 100
    pct_sl = (s["sl"] / e - 1) * 100
    return (
        f"🎯 <b>SEÑAL CUANTITATIVA (+EV) — {'📈 LONG (COMPRA)' if larga else '📉 SHORT (VENTA)'}</b>\n\n"
        f"{s['cat_icono']} <b>Activo:</b> {s['nombre']} ({s['simbolo']}) • {s['cat_nombre']}\n"
        f"⏰ <b>Emitida:</b> {s['fecha_hora_display']}\n"
        f"🎯 <b>Entrada:</b> {fp(e)}\n"
        f"✅ <b>Take Profit:</b> {fp(s['tp'])} ({pct_tp:+.2f}%)\n"
        f"🛑 <b>Stop Loss:</b> {fp(s['sl'])} ({pct_sl:+.2f}%)\n"
        f"⚖️ <b>Riesgo/Beneficio:</b> 1:{s['rr']:g}\n"
        f"🔧 <b>Apalancamiento sugerido:</b> {s['apalancamiento']}x (máx. {APALANCAMIENTO_MAX}x) • un stop ≈ "
        f"{abs(pct_sl) * s['apalancamiento']:.1f}% del margen\n"
        f"📈 <b>Ventaja (backtest 1 año):</b> +{s['ventaja']}% por unidad arriesgada\n"
        f"🎲 <b>Prob. de acierto histórica:</b> {s['prob']}% ({s['n_bt']} señales)\n"
        f"💡 <b>Stake sugerido:</b> {s['stake']}/10 unidades de riesgo\n"
        f"📊 <b>Contexto:</b> tendencia {s['tendencia'].lower()} • RSI {s['rsi']}\n"
        f"⏳ <b>Validez máxima:</b> {MAX_BARRAS} velas de 1 h (después se cierra a mercado)\n\n"
        f"⚡ <i>Apex Quant Engine • Precios reales de mercado</i>"
    )


def cerrar_senal(s, r, motivo, precio, ts_utc, semana_id):
    fecha, hora, display = display_madrid(ts_utc)
    semana_id = obtener_identificador_semana(ts_utc.astimezone(TZ_MADRID))   # semana en la que se cerró
    s.update({"estado": "CERRADA", "resultado_r": round(float(r), 2), "acertado": bool(r > 0),
              "precio_cierre": num(precio, 6), "motivo_cierre": motivo, "fecha_cierre": fecha,
              "hora_cierre": hora, "fecha_hora_cierre_display": display, "semana_cierre": semana_id,
              "ts_cierre_utc": ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "r_actual": round(float(r), 2),
              "precio_actual": num(precio, 6)})
    if s.get("telegram_enviado") and not s.get("liquidada_telegram"):
        textos = {"TP": "Take Profit alcanzado", "SL": "Stop Loss alcanzado", "TIEMPO": "Cierre por tiempo (a mercado)"}
        icono = "✅" if s["acertado"] else "❌"
        unidades = s["stake"] * s["resultado_r"]
        enviar_telegram(
            f"{icono} <b>SEÑAL CERRADA — {s['nombre']} ({s['simbolo']}) {s['direccion']}</b>\n\n"
            f"🎯 <b>Entrada:</b> {fp(s['entrada'])} → 🏁 <b>Cierre:</b> {fp(precio)}\n"
            f"📌 <b>Motivo:</b> {textos.get(motivo, motivo)}\n"
            f"📈 <b>Resultado:</b> {s['resultado_r']:+.2f}R • {unidades:+.2f} U. (stake {s['stake']}/10)\n\n"
            f"⚡ <i>Apex Quant Engine • Disciplina a largo plazo</i>",
            url_accion=APP_URL, texto_boton="📱 Ver en la Web App")
        s["liquidada_telegram"] = True


def _sig(v):
    """Redondeo a 6 cifras significativas para que datos.json no pese de más."""
    try:
        return float(f"{float(v):.6g}")
    except Exception:
        return None


def _vela(ts, row):
    return [int(ts.timestamp()), _sig(row["Open"]), _sig(row["High"]), _sig(row["Low"]), _sig(row["Close"])]


def seguir_senal(s, df, ahora_utc, semana_id):
    ent = pd.Timestamp(datetime.fromisoformat(s["ts_entrada_utc"].replace("Z", "+00:00")))
    larga = s["direccion"] == "LONG"
    e, sl, tp = s["entrada"], s["sl"], s["tp"]
    riesgo = abs(e - sl) or 1e-12
    posteriores = df[df.index >= ent]
    traza = [e]   # recorrido del precio durante la operación (cierre de cada vela de 1 h) para el gráfico de la web
    velas = [_vela(ts, row) for ts, row in df[df.index < ent].tail(VELAS_PREVIAS).iterrows()]
    s["velas_entrada_idx"] = len(velas)   # las velas anteriores son contexto; a partir de aquí, la operación

    def fin(r, motivo, precio_fin, ts):
        traza.append(precio_fin)
        s["traza"] = [num(v, 6) for v in traza[-(MAX_BARRAS + 2):]]
        s["velas"] = velas
        return cerrar_senal(s, r, motivo, precio_fin, ts, semana_id)

    for ts, row in posteriores.iterrows():
        velas.append(_vela(ts, row))
        if larga:
            if row["Low"] <= sl:
                return fin(-1.0, "SL", sl, ts.to_pydatetime())
            if row["High"] >= tp:
                return fin(RR, "TP", tp, ts.to_pydatetime())
        else:
            if row["High"] >= sl:
                return fin(-1.0, "SL", sl, ts.to_pydatetime())
            if row["Low"] <= tp:
                return fin(RR, "TP", tp, ts.to_pydatetime())
        traza.append(float(row["Close"]))
    precio = float(df["Close"].iloc[-1])
    if not len(posteriores):
        traza.append(precio)
    if (larga and precio <= sl) or (not larga and precio >= sl):
        return fin(-1.0, "SL", sl, ahora_utc)
    if (larga and precio >= tp) or (not larga and precio <= tp):
        return fin(RR, "TP", tp, ahora_utc)
    s["traza"] = [num(v, 6) for v in traza[-(MAX_BARRAS + 2):]]
    s["velas"] = velas
    r = ((precio - e) if larga else (e - precio)) / riesgo
    s["precio_actual"] = num(precio, 6)
    s["r_actual"] = round(r, 2)
    s["max_r"] = round(max(s.get("max_r") or 0.0, r), 2)
    if len(posteriores) >= MAX_BARRAS:
        cerrar_senal(s, max(-1.0, min(RR, r)), "TIEMPO", precio, ahora_utc, semana_id)


# ==============================================================================
# RESÚMENES Y LIBRO MAYOR
# ==============================================================================
def calcular_resumen(lista):
    apostado, neto, aciertos = 0.0, 0.0, 0
    for s in lista:
        stk = float(s.get("stake") or 1)
        r = float(s.get("resultado_r") or s.get("r") or 0)
        apostado += stk
        neto += stk * r
        if r > 0:
            aciertos += 1
    n = len(lista)
    return {"finalizadas": n, "acertadas": aciertos, "falladas": n - aciertos,
            "unidades_apostadas": round(apostado, 1), "unidades_netas": round(neto, 2),
            "roi_pct": round(neto / apostado * 100, 1) if apostado > 0 else 0.0}


ARCHIVO_LEDGER = "historico_picks.json"


def actualizar_ledger(senales):
    """Libro mayor permanente (no se reinicia los lunes) con todas las señales cerradas:
    grupo TG = enviadas a Telegram; grupo CONTROL = detectadas y descartadas."""
    ledger = {"version": 1, "picks": []}
    if os.path.exists(ARCHIVO_LEDGER):
        try:
            with open(ARCHIVO_LEDGER, "r", encoding="utf-8") as f:
                ledger = json.load(f)
        except Exception:
            pass
    picks = ledger.get("picks", [])
    ids = {p.get("id") for p in picks}
    nuevas = 0
    for s in senales:
        if s.get("estado") != "CERRADA" or s.get("resultado_r") is None or s["id"] in ids:
            continue
        picks.append({"id": s["id"], "version": VERSION, "grupo": "TG" if s.get("telegram_enviado") else "CONTROL",
                      "fecha": s.get("fecha_cierre"), "mes": (s.get("fecha_cierre") or "")[:7],
                      "semana_id": s.get("semana_cierre"), "activo": f"{s['nombre']} ({s['simbolo']})",
                      "categoria": s.get("categoria"), "direccion": s.get("direccion"), "stake": s.get("stake", 1),
                      "r": s.get("resultado_r"), "acertado": bool(s.get("acertado")), "ventaja": s.get("ventaja"),
                      "motivo": s.get("motivo_cierre")})
        ids.add(s["id"])
        nuevas += 1
    if nuevas:
        ledger["picks"] = picks
        with open(ARCHIVO_LEDGER, "w", encoding="utf-8") as f:
            json.dump(ledger, f, ensure_ascii=False, separators=(",", ":"))
        print(f"📒 Libro mayor actualizado: +{nuevas} señales cerradas (total {len(picks)})")
    return picks


def resumen_semanal(senales, semana_id, ahora_madrid, enviado_para):
    if ahora_madrid.weekday() != 6 or ahora_madrid.hour < 22 or enviado_para == semana_id:
        return enviado_para
    tg = [s for s in senales if s.get("telegram_enviado") and s.get("estado") == "CERRADA" and s.get("semana_cierre") == semana_id]
    r = calcular_resumen(tg)
    if r["finalizadas"] == 0:
        return enviado_para
    pct = round(r["acertadas"] / r["finalizadas"] * 100, 1)
    sg = "+" if r["unidades_netas"] >= 0 else ""
    enviar_telegram(
        f"📊 <b>CIERRE SEMANAL OFICIAL — APEX QUANT</b> 🏆\n\n"
        f"📅 <b>Semana:</b> {semana_id} (Finalizada)\n"
        f"📈 <b>Señales oficiales cerradas:</b> <b>{r['finalizadas']}</b>\n\n"
        f"✅ <b>Ganadoras:</b> <b>{r['acertadas']}</b> ({pct}%)\n"
        f"❌ <b>Perdedoras:</b> <b>{r['falladas']}</b>\n\n"
        f"💰 <b>Rendimiento neto:</b> <b>{sg}{r['unidades_netas']} Unidades</b>\n"
        f"📈 <b>Yield / ROI semanal:</b> <b>{sg}{r['roi_pct']}%</b>\n\n"
        f"🔄 <i>Las estadísticas semanales se reinician a las 00:00 del lunes.</i>\n"
        f"⚡ <i>Apex Quant Engine • Máxima transparencia y disciplina</i>",
        url_accion=APP_URL, texto_boton="📱 Abrir Balance Completo en la App")
    return semana_id


def resumen_mensual(picks, mes_id, ahora_madrid, enviado_para):
    if (ahora_madrid + timedelta(days=1)).month == ahora_madrid.month or ahora_madrid.hour < 22 or enviado_para == mes_id:
        return enviado_para
    tg = [p for p in picks if p.get("grupo") == "TG" and p.get("mes") == mes_id]
    r = calcular_resumen(tg)
    if r["finalizadas"] == 0:
        return enviado_para
    pct = round(r["acertadas"] / r["finalizadas"] * 100, 1)
    sg = "+" if r["unidades_netas"] >= 0 else ""
    semanas = {}
    for p in tg:
        semanas.setdefault(p.get("semana_id") or "S/D", []).append(p)
    lineas = []
    for i, (sk, lst) in enumerate(sorted(semanas.items()), 1):
        rs = calcular_resumen(lst)
        lineas.append(f"▫️ <b>Semana {i} ({sk}):</b> {'+' if rs['unidades_netas'] >= 0 else ''}{rs['unidades_netas']} U. "
                      f"(ROI {'+' if rs['roi_pct'] >= 0 else ''}{rs['roi_pct']}%)")
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    enviar_telegram(
        f"🏆 <b>CIERRE MENSUAL OFICIAL — APEX QUANT</b> 📊\n\n"
        f"📅 <b>Mes:</b> <b>{meses[ahora_madrid.month - 1]} {ahora_madrid.year}</b> (Finalizado)\n"
        f"📈 <b>Señales oficiales cerradas:</b> <b>{r['finalizadas']}</b>\n\n"
        f"✅ <b>Ganadoras:</b> <b>{r['acertadas']}</b> ({pct}%)\n"
        f"❌ <b>Perdedoras:</b> <b>{r['falladas']}</b>\n\n"
        f"💰 <b>Rendimiento neto:</b> <b>{sg}{r['unidades_netas']} Unidades</b>\n"
        f"📈 <b>Yield / ROI mensual:</b> <b>{sg}{r['roi_pct']}%</b>\n\n"
        f"📊 <b>EVOLUCIÓN SEMANA A SEMANA:</b>\n" + "\n".join(lineas) + "\n\n"
        f"⚡ <i>Apex Quant Engine • Transparencia, disciplina y consistencia</i>",
        url_accion=APP_URL, texto_boton="📱 Abrir Balance Completo en la App")
    return mes_id


# ==============================================================================
# CICLO PRINCIPAL
# ==============================================================================
def ejecutar_ciclo():
    ahora_utc = datetime.now(timezone.utc).replace(microsecond=0)
    ahora_pd = pd.Timestamp(ahora_utc)
    ahora_madrid = ahora_utc.astimezone(TZ_MADRID)
    semana_id = obtener_identificador_semana(ahora_madrid)
    mes_id = ahora_madrid.strftime("%Y-%m")
    lunes, domingo = obtener_fechas_limite_semana(ahora_madrid)
    print(f"[{ahora_madrid.strftime('%H:%M')}] Iniciando ciclo Apex Quant Mercados v{VERSION} (Semana {semana_id})...")

    previos = cargar_datos_existentes()
    senales = previos.get("senales", [])
    ultima_barra = dict(previos.get("ultima_barra_senal") or {})
    activos_previos = {a["ticker"]: a for a in previos.get("activos", [])}
    if previos.get("semana_activa") and previos.get("semana_activa") != semana_id:
        print(f"🔄 ¡NUEVA SEMANA! {previos.get('semana_activa')} -> {semana_id}: métricas semanales a cero")

    tickers = list(ACTIVOS)
    datos = descargar(tickers, PERIODO_LIVE)
    n_ok = len(datos)
    estado_datos = "error" if n_ok == 0 else ("parcial" if n_ok < 0.8 * len(tickers) else "ok")
    print(f"📡 Datos de mercado: {n_ok}/{len(tickers)} activos ({estado_datos})")

    backtest = actualizar_backtest(previos.get("backtest"), tickers, ahora_utc) if n_ok else (previos.get("backtest") or {"t": "", "e": {}})

    # 1) Seguimiento y liquidación de las señales abiertas
    for s in senales:
        if s.get("estado") == "ABIERTA" and s.get("ticker") in datos:
            try:
                seguir_senal(s, datos[s["ticker"]], ahora_utc, semana_id)
            except Exception as e:
                print(f"   Aviso siguiendo {s.get('id')}: {e}")

    abiertas = {s["ticker"]: s for s in senales if s.get("estado") == "ABIERTA"}

    def en_cooldown(tk):
        for s in senales:
            if s.get("ticker") == tk and s.get("estado") == "CERRADA" and s.get("ts_cierre_utc"):
                t = datetime.fromisoformat(s["ts_cierre_utc"].replace("Z", "+00:00"))
                if (ahora_utc - t).total_seconds() < COOLDOWN_H * 3600:
                    return True
        return False

    # 2) Estado de cada activo y detección de nuevas señales
    activos_web, nuevas = [], 0
    for tk, info in ACTIVOS.items():
        cat = CATEGORIAS[info["cat"]]
        df = datos.get(tk)
        if df is None:
            prev = activos_previos.get(tk)
            if prev:
                prev["sin_datos"] = True
                activos_web.append(prev)
            continue
        cerradas = df[df.index + pd.Timedelta(hours=1) <= ahora_pd]
        if len(cerradas) < 60:
            continue
        ind = indicadores(cerradas)
        fila = ind.iloc[-1]
        precio = float(df["Close"].iloc[-1])
        ts_precio = df.index[-1]
        mercado_abierto = (ahora_pd - ts_precio) <= pd.Timedelta(minutes=75)
        ref = df[df.index <= ahora_pd - pd.Timedelta(hours=24)]
        var_24h = (precio / float(ref["Close"].iloc[-1]) - 1) * 100 if len(ref) else None
        stats = (backtest.get("e") or {}).get(tk) or {}

        if mercado_abierto and tk not in abiertas and not en_cooldown(tk) and len(ind) > 210:
            for k in (-1, -2):
                ts_b = ind.index[k]
                if ahora_pd - (ts_b + pd.Timedelta(hours=1)) > pd.Timedelta(hours=2):
                    continue
                clave = ts_b.strftime("%Y-%m-%dT%H:%M:%SZ")
                if clave <= ultima_barra.get(tk, ""):
                    continue
                row = ind.iloc[k]
                direccion = "LONG" if row["long"] else ("SHORT" if row["short"] else None)
                if not direccion:
                    continue
                s = crear_senal(tk, info, direccion, ts_b, precio, float(row["atr"]), row,
                                stats.get(direccion), ahora_utc, semana_id)
                ultima_barra[tk] = clave
                if s["confirmado_bt"]:
                    s["telegram_enviado"] = True
                    enviar_telegram(mensaje_senal(s), url_accion=APP_URL, texto_boton="📱 Abrir en Web App")
                senales.append(s)
                abiertas[tk] = s
                nuevas += 1
                print(f"   {'📣' if s['telegram_enviado'] else '🗂️'} Señal {direccion} {info['nombre']} @ {fp(precio)} "
                      f"(ventaja {s['ventaja']}%, n={s['n_bt']})")
                break

        tend = tendencia_de(fila)
        if not mercado_abierto:
            nota = "Mercado cerrado: se vigilará en la próxima sesión"
        elif tend == "LATERAL":
            nota = "Sin tendencia clara: el sistema espera alineación de medias"
        else:
            nota = f"Tendencia {tend.lower()}: esperando retroceso y recuperación del RSI 50"
        _, _, disp = display_madrid(ts_precio.to_pydatetime())
        bl, bs = stats.get("LONG") or {}, stats.get("SHORT") or {}
        activos_web.append({
            "ticker": tk, "nombre": info["nombre"], "simbolo": info["simbolo"], "categoria": info["cat"],
            "cat_nombre": cat["name"], "cat_icono": cat["icono"], "precio": num(precio, 6),
            "var_24h": num(var_24h, 2), "rsi": num(fila["rsi"], 1), "tendencia": tend,
            "atr_pct": num(float(fila["atr"]) / precio * 100, 2), "mercado_abierto": bool(mercado_abierto),
            "fecha_hora_display": disp, "nota": nota, "sin_datos": False,
            "bt_long": {"n": bl.get("n", 0), "p": bl.get("p"), "v": bl.get("v")},
            "bt_short": {"n": bs.get("n", 0), "p": bs.get("p"), "v": bs.get("v")},
        })

    # 3) Resúmenes, libro mayor y guardado
    picks = actualizar_ledger(senales)
    resumen_semanal_para = resumen_semanal(senales, semana_id, ahora_madrid, previos.get("resumen_semanal_enviado_para", ""))
    resumen_mensual_para = resumen_mensual(picks, mes_id, ahora_madrid, previos.get("resumen_mensual_enviado_para", ""))

    limite = (ahora_madrid - timedelta(weeks=6)).strftime("%Y-%m-%d")
    senales = [s for s in senales if s.get("estado") == "ABIERTA" or (s.get("fecha_cierre") or s.get("fecha", "")) >= limite]
    senales = senales[-MAX_SENALES_GUARDADAS:]

    # Las velas solo se guardan para señales abiertas y cerradas esta semana (para que datos.json no crezca)
    for s in senales:
        if s.get("estado") == "CERRADA" and s.get("semana_cierre") != semana_id:
            s.pop("velas", None)
            s.pop("velas_entrada_idx", None)

    tg_semana = [s for s in senales if s.get("telegram_enviado") and s.get("estado") == "CERRADA" and s.get("semana_cierre") == semana_id]
    datos_completos = {
        "version_bot": VERSION,
        "ultima_actualizacion": ahora_madrid.strftime("%H:%M"),
        "ultima_actualizacion_utc": ahora_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "huso_horario": "Europe/Madrid",
        "semana_activa": semana_id,
        "lunes_semana": lunes,
        "domingo_semana": domingo,
        "resumen_semanal_enviado_para": resumen_semanal_para,
        "resumen_mensual_enviado_para": resumen_mensual_para,
        "metricas_semana": calcular_resumen(tg_semana),
        "api_datos_estado": estado_datos,
        "activos_ok": n_ok,
        "activos_total": len(tickers),
        "ev_minimo": EV_MINIMO_PCT,
        "rr": round(RR, 2),
        "activos": activos_web,
        "senales": senales,
        "ultima_barra_senal": ultima_barra,
        "backtest": backtest,
        "diagnostico": {"nuevas_senales": nuevas, "abiertas": sum(1 for s in senales if s.get("estado") == "ABIERTA"),
                        "descargas_fallidas": ESTADO_DATOS["descargas_fallidas"]},
    }
    with open("datos.json", "w", encoding="utf-8") as f:
        json.dump(datos_completos, f, ensure_ascii=False, separators=(",", ":"))
    print(f"📁 datos.json guardado. Activos: {len(activos_web)} | Señales guardadas: {len(senales)} | Nuevas: {nuevas}")


if __name__ == "__main__":
    ejecutar_ciclo()
