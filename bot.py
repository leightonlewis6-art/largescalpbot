import os
import json
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from groq import Groq
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ── CONFIG ─────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
CHECK_INTERVAL   = int(os.environ.get("CHECK_INTERVAL", "3"))
SCORE_NEEDED     = int(os.environ.get("SCORE_NEEDED", "3"))
MIN_BACKTEST     = int(os.environ.get("MIN_BACKTEST", "35"))
ALLOW_ASIAN      = os.environ.get("ALLOW_ASIAN", "true").lower() == "true"

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, GROQ_API_KEY]):
    raise RuntimeError("Missing env vars - check Railway Variables tab")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)

# ── SCALPING SETTINGS ──────────────────────────────────────────────────────
SYMBOL        = "GC=F"          # XAU/USD Gold Futures
SYMBOL_NAME   = "XAU/USD Gold"
PRICE_MIN     = 3000
PRICE_MAX     = 8000
MIN_GAP_MIN   = int(os.environ.get("MIN_GAP_MIN", "30"))



# Scalping-specific SL/TP multipliers (tight, fast)
SL_ATR_MULT   = 0.8             # tight stop loss for scalping
TP1_ATR_MULT  = 0.8             # quick TP1 (1:1)
TP2_ATR_MULT  = 1.5             # TP2 (1:1.8)
TP3_ATR_MULT  = 2.5             # TP3 (1:3)

is_paused        = False
last_signal_time = None
trade_log        = []
pending_checks   = []


# ── PRICE & CANDLES ────────────────────────────────────────────────────────
def get_live_price():
    headers = {"User-Agent": "Mozilla/5.0"}
    # Try 1: Yahoo Finance GC=F
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}",
                         headers=headers, timeout=8)
        price = float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
        if PRICE_MIN < price < PRICE_MAX:
            return price
    except Exception as e:
        log.warning("Yahoo price 1 failed: " + str(e))
    # Try 2: Yahoo query2
    try:
        r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{SYMBOL}",
                         headers=headers, timeout=8)
        price = float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
        if PRICE_MIN < price < PRICE_MAX:
            return price
    except Exception as e:
        log.warning("Yahoo price 2 failed: " + str(e))
    # Try 3: goldprice.org
    try:
        r = requests.get("https://data-asg.goldprice.org/dbXRates/USD",
                         headers=headers, timeout=8)
        price = float(r.json()["items"][0]["xauPrice"])
        if PRICE_MIN < price < PRICE_MAX:
            return price
    except Exception as e:
        log.warning("goldprice.org failed: " + str(e))
    return None


def fetch_candles(interval="5m", period="2d"):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}?interval={interval}&range={period}"
    r = requests.get(url, headers=headers, timeout=10)
    result = r.json()["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "time":   pd.to_datetime(result["timestamp"], unit="s", utc=True),
        "open":   q["open"], "high": q["high"],
        "low":    q["low"],  "close": q["close"],
        "volume": q["volume"],
    }).dropna()
    return df


# ── INDICATORS ─────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_adx(df, period=14):
    high, low = df["high"], df["low"]
    atr_s    = calc_atr(df, period)
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_di  = 100 * plus_dm.ewm(span=period).mean() / (atr_s + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period).mean() / (atr_s + 1e-10)
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    return dx.ewm(span=period).mean().iloc[-1]


def calc_vwap(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol     = df["volume"].replace(0, np.nan).fillna(1)
    return (typical * vol).cumsum() / vol.cumsum()


def calc_bollinger(series, period=20, std=2):
    mid  = series.rolling(period).mean()
    band = series.rolling(period).std()
    return mid - std*band, mid, mid + std*band


def get_sr_levels(df, lookback=80):
    recent  = df.tail(lookback)
    highs   = recent["high"].nlargest(4).round(2).tolist()
    lows    = recent["low"].nsmallest(4).round(2).tolist()
    return sorted(lows), sorted(highs, reverse=True)


def get_indicators(df, live_price):
    close    = df["close"].copy()
    close.iloc[-1] = live_price

    rsi_s    = calc_rsi(close)
    ema8     = calc_ema(close, 8)
    ema21    = calc_ema(close, 21)
    ema50    = calc_ema(close, 50)
    ema200   = calc_ema(close, 200)
    atr_s    = calc_atr(df)
    vwap_s   = calc_vwap(df)
    bb_lo, bb_mid, bb_hi = calc_bollinger(close)

    avg_vol  = df["volume"].tail(20).mean()
    vol_ratio = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 and not np.isnan(avg_vol) else 1.0

    # Momentum: last 3 candles
    momentum = close.iloc[-1] - close.iloc[-4]

    # RSI divergence
    price_hi = close.iloc[-1] > close.iloc[-6:-1].max()
    price_lo = close.iloc[-1] < close.iloc[-6:-1].min()
    rsi_hi   = rsi_s.iloc[-1] > rsi_s.iloc[-6:-1].max()
    rsi_lo   = rsi_s.iloc[-1] < rsi_s.iloc[-6:-1].min()

    supports, resistances = get_sr_levels(df)

    return {
        "price":          round(live_price, 2),
        "rsi":            round(rsi_s.iloc[-1], 2),
        "ema8":           round(ema8.iloc[-1], 2),
        "ema21":          round(ema21.iloc[-1], 2),
        "ema50":          round(ema50.iloc[-1], 2),
        "ema200":         round(ema200.iloc[-1], 2),
        "bullish_cross":  ema8.iloc[-2] <= ema21.iloc[-2] and ema8.iloc[-1] > ema21.iloc[-1],
        "bearish_cross":  ema8.iloc[-2] >= ema21.iloc[-2] and ema8.iloc[-1] < ema21.iloc[-1],
        "ema_trend":      "bullish" if ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1] else
                          "bearish" if ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1] else "mixed",
        "above_ema200":   live_price > ema200.iloc[-1],
        "vwap":           round(vwap_s.iloc[-1], 2),
        "above_vwap":     live_price > vwap_s.iloc[-1],
        "bb_lo":          round(bb_lo.iloc[-1], 2),
        "bb_mid":         round(bb_mid.iloc[-1], 2),
        "bb_hi":          round(bb_hi.iloc[-1], 2),
        "bb_squeeze":     (bb_hi.iloc[-1] - bb_lo.iloc[-1]) < (bb_hi.iloc[-5] - bb_lo.iloc[-5]) * 0.7,
        "near_bb_lo":     live_price <= bb_lo.iloc[-1] * 1.001,
        "near_bb_hi":     live_price >= bb_hi.iloc[-1] * 0.999,
        "atr":            round(atr_s.iloc[-1], 2),
        "adx":            round(calc_adx(df), 2),
        "vol_ratio":      round(float(vol_ratio), 2),
        "high_volume":    float(vol_ratio) > 1.2,
        "momentum":       round(momentum, 2),
        "bullish_div":    price_lo and not rsi_lo,
        "bearish_div":    price_hi and not rsi_hi,
        "supports":       supports[:3],
        "resistances":    resistances[:3],
    }


# ── CANDLESTICK PATTERNS ───────────────────────────────────────────────────
def detect_candle_patterns(df):
    patterns = []
    if len(df) < 4:
        return patterns

    c, p, p2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    body       = abs(c["close"] - c["open"])
    rng        = c["high"] - c["low"] + 1e-10
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]

    # Pin Bar
    if lower_wick > body * 2.5 and upper_wick < body * 0.5:
        patterns.append(("BUY", "Bullish Pin Bar"))
    if upper_wick > body * 2.5 and lower_wick < body * 0.5:
        patterns.append(("SELL", "Bearish Pin Bar"))

    # Engulfing
    if (c["close"] > c["open"] and p["close"] < p["open"]
            and c["open"] <= p["close"] and c["close"] >= p["open"]):
        patterns.append(("BUY", "Bullish Engulfing"))
    if (c["close"] < c["open"] and p["close"] > p["open"]
            and c["open"] >= p["close"] and c["close"] <= p["open"]):
        patterns.append(("SELL", "Bearish Engulfing"))

    # Doji
    if body < rng * 0.08:
        patterns.append(("NEUTRAL", "Doji - indecision"))

    # Morning Star
    if (p2["close"] < p2["open"] and
            abs(p["close"] - p["open"]) < abs(p2["close"] - p2["open"]) * 0.3 and
            c["close"] > c["open"] and
            c["close"] > (p2["open"] + p2["close"]) / 2):
        patterns.append(("BUY", "Morning Star"))

    # Evening Star
    if (p2["close"] > p2["open"] and
            abs(p["close"] - p["open"]) < abs(p2["close"] - p2["open"]) * 0.3 and
            c["close"] < c["open"] and
            c["close"] < (p2["open"] + p2["close"]) / 2):
        patterns.append(("SELL", "Evening Star"))

    # Marubozu
    if upper_wick < rng * 0.04 and lower_wick < rng * 0.04:
        if c["close"] > c["open"]:
            patterns.append(("BUY", "Bullish Marubozu"))
        else:
            patterns.append(("SELL", "Bearish Marubozu"))

    # Hammer at support
    if lower_wick > rng * 0.6 and body < rng * 0.3:
        patterns.append(("BUY", "Hammer"))

    # Shooting star at resistance
    if upper_wick > rng * 0.6 and body < rng * 0.3:
        patterns.append(("SELL", "Shooting Star"))

    return patterns


# ── SCALP SESSION FILTER ───────────────────────────────────────────────────
def get_session():
    hour = datetime.now(timezone.utc).hour
    if 7 <= hour < 12:    return "London", True
    if 12 <= hour < 15:   return "Overlap", True       # best session
    if 15 <= hour < 20:   return "New York", True
    if 6 <= hour < 7:     return "London Open", True   # high volatility open
    return "Asian/Off-hours", ALLOW_ASIAN                    # lower quality session


def is_news_time():
    now  = datetime.now(timezone.utc)
    day  = now.strftime("%a")
    hour = now.hour
    min_ = now.minute
    high_impact = {"Tue":[13,14],"Wed":[13,14,18,19],"Thu":[12,13,14,18,19],"Fri":[12,13,14,15]}
    if hour in high_impact.get(day, []):
        return True, "High-impact news window"
    if min_ < 10 and hour in [8,12,13,14,15,18,19]:
        return True, "News release window (first 10 min of hour)"
    return False, ""


# ── SCALP SETUP SCORING ────────────────────────────────────────────────────
def score_scalp_setup(ind, patterns, session_ok):
    score   = 0
    reasons = []

    # Session quality
    if not session_ok:
        score -= 3
        reasons.append("Off-peak session - lower quality")
    else:
        reasons.append("Active trading session")

    # RSI (scalping zones: tighter than swing)
    rsi = ind["rsi"]
    if rsi < 32:
        score += 3; reasons.append("RSI deeply oversold (" + str(rsi) + ")")
    elif rsi < 42:
        score += 2; reasons.append("RSI oversold (" + str(rsi) + ")")
    elif rsi > 68:
        score -= 3; reasons.append("RSI deeply overbought (" + str(rsi) + ")")
    elif rsi > 58:
        score -= 2; reasons.append("RSI overbought (" + str(rsi) + ")")

    # EMA crossover (strong scalp signal)
    if ind["bullish_cross"]:
        score += 4; reasons.append("Fresh bullish EMA 8/21 cross")
    elif ind["bearish_cross"]:
        score -= 4; reasons.append("Fresh bearish EMA 8/21 cross")

    # EMA stack
    if ind["ema_trend"] == "bullish":
        score += 2; reasons.append("EMA stack bullish (8>21>50)")
    elif ind["ema_trend"] == "bearish":
        score -= 2; reasons.append("EMA stack bearish (8<21<50)")

    # EMA200 bias (major trend filter)
    if ind["above_ema200"]:
        score += 1; reasons.append("Price above EMA200 (bullish bias)")
    else:
        score -= 1; reasons.append("Price below EMA200 (bearish bias)")

    # VWAP
    if ind["above_vwap"]:
        score += 1; reasons.append("Price above VWAP")
    else:
        score -= 1; reasons.append("Price below VWAP")

    # Bollinger Bands
    if ind["near_bb_lo"]:
        score += 2; reasons.append("Price at lower Bollinger Band (buy zone)")
    if ind["near_bb_hi"]:
        score -= 2; reasons.append("Price at upper Bollinger Band (sell zone)")
    if ind["bb_squeeze"]:
        score += 1; reasons.append("Bollinger squeeze - breakout incoming")

    # ADX trend strength
    if ind["adx"] > 30:
        adj = 2 if score > 0 else -2
        score += adj
        reasons.append("Strong trend ADX " + str(ind["adx"]))
    elif ind["adx"] > 20:
        adj = 1 if score > 0 else -1
        score += adj
        reasons.append("Moderate trend ADX " + str(ind["adx"]))
    elif ind["adx"] < 15:
        score = int(score * 0.6)
        reasons.append("Weak trend ADX " + str(ind["adx"]) + " - score reduced")

    # Volume
    if ind["high_volume"]:
        adj = 1 if score > 0 else -1
        score += adj
        reasons.append("Volume " + str(ind["vol_ratio"]) + "x above average")

    # Momentum
    if ind["momentum"] > 0 and score > 0:
        score += 1; reasons.append("Positive price momentum")
    elif ind["momentum"] < 0 and score < 0:
        score -= 1; reasons.append("Negative price momentum")

    # RSI divergence
    if ind["bullish_div"]:
        score += 2; reasons.append("Bullish RSI divergence")
    if ind["bearish_div"]:
        score -= 2; reasons.append("Bearish RSI divergence")

    # Candlestick patterns
    for direction, name in patterns:
        if direction == "BUY":
            if score > 0:
                score += 2; reasons.append("Pattern confirms: " + name)
            else:
                reasons.append("Pattern conflicts (BUY): " + name)
        elif direction == "SELL":
            if score < 0:
                score -= 2; reasons.append("Pattern confirms: " + name)
            else:
                reasons.append("Pattern conflicts (SELL): " + name)
        else:
            reasons.append("Pattern: " + name)

    # S/R proximity
    price = ind["price"]
    for s in ind["supports"]:
        if s > 0 and abs(price - s) / price < 0.0015:
            score += 1; reasons.append("At support $" + str(s))
    for r in ind["resistances"]:
        if r > 0 and abs(price - r) / price < 0.0015:
            score -= 1; reasons.append("At resistance $" + str(r))

    direction  = "BUY" if score >= SCORE_NEEDED else "SELL" if score <= -SCORE_NEEDED else "NEUTRAL"
    confidence = min(95, 50 + abs(score) * 5)

    return {
        "direction":    direction,
        "score":        score,
        "confidence":   confidence,
        "reasons":      reasons,
        "is_setup":     abs(score) >= SCORE_NEEDED,
    }


# ── BACKTEST ───────────────────────────────────────────────────────────────
def backtest(df, direction):
    wins = 0; total = 0
    rsi_s    = calc_rsi(df["close"])
    ema8_s   = calc_ema(df["close"], 8)
    ema21_s  = calc_ema(df["close"], 21)
    atr_s    = calc_atr(df)

    for i in range(25, len(df) - 12):
        rsi_v  = rsi_s.iloc[i]
        e8     = ema8_s.iloc[i]
        e21    = ema21_s.iloc[i]
        atr_v  = atr_s.iloc[i]
        price  = df["close"].iloc[i]
        if atr_v == 0 or np.isnan(atr_v):
            continue
        match = (rsi_v < 45 and e8 > e21) if direction == "BUY" else (rsi_v > 55 and e8 < e21)
        if not match:
            continue
        future = df["close"].iloc[i+1:i+12]
        sl = atr_v * SL_ATR_MULT
        tp = atr_v * TP2_ATR_MULT
        hit_tp = any(future >= price + tp) if direction == "BUY" else any(future <= price - tp)
        hit_sl = any(future <= price - sl) if direction == "BUY" else any(future >= price + sl)
        if hit_tp or hit_sl:
            total += 1
            if hit_tp:
                wins += 1

    if total < 5:
        return None, total
    return round(wins / total * 100, 1), total


# ── MULTI-TIMEFRAME ────────────────────────────────────────────────────────
def get_htf_bias():
    try:
        df15 = fetch_candles("15m", "5d")
        df1h = fetch_candles("60m", "30d")
        e9_15  = calc_ema(df15["close"], 9).iloc[-1]
        e21_15 = calc_ema(df15["close"], 21).iloc[-1]
        e9_1h  = calc_ema(df1h["close"], 9).iloc[-1]
        e21_1h = calc_ema(df1h["close"], 21).iloc[-1]
        rsi_15 = calc_rsi(df15["close"]).iloc[-1]
        rsi_1h = calc_rsi(df1h["close"]).iloc[-1]
        t15 = "bullish" if e9_15 > e21_15 else "bearish"
        t1h = "bullish" if e9_1h > e21_1h else "bearish"
        return {
            "15m": t15, "1h": t1h,
            "rsi_15m": round(rsi_15, 1),
            "rsi_1h":  round(rsi_1h, 1),
            "aligned": t15 == t1h,
            "bias":    t1h,
        }
    except Exception as e:
        log.warning("HTF failed: " + str(e))
        return None


# ── BUILD SIGNAL ───────────────────────────────────────────────────────────
def build_signal(ind, setup, patterns, htf, bt_rate, bt_trades, session):
    price = ind["price"]
    d     = setup["direction"]
    atr   = ind["atr"]

    sl_d  = round(atr * SL_ATR_MULT, 2)
    tp1_d = round(atr * TP1_ATR_MULT, 2)
    tp2_d = round(atr * TP2_ATR_MULT, 2)
    tp3_d = round(atr * TP3_ATR_MULT, 2)

    if d == "BUY":
        sl  = round(price - sl_d, 2)
        tp1 = round(price + tp1_d, 2)
        tp2 = round(price + tp2_d, 2)
        tp3 = round(price + tp3_d, 2)
        inv = round(price - sl_d * 1.3, 2)
    else:
        sl  = round(price + sl_d, 2)
        tp1 = round(price - tp1_d, 2)
        tp2 = round(price - tp2_d, 2)
        tp3 = round(price - tp3_d, 2)
        inv = round(price + sl_d * 1.3, 2)

    rr = "1:" + str(round(tp2_d / sl_d, 1))

    pat_names = [name for _, name in patterns] if patterns else ["No pattern"]

    try:
        htf_str = ""
        if htf:
            htf_str = "15m: " + htf["15m"] + " RSI " + str(htf["rsi_15m"]) + " | 1H: " + htf["1h"] + " RSI " + str(htf["rsi_1h"])
        prompt = (
            "Professional XAU/USD scalp analyst. 2 sentence technical summary + 1 sentence sentiment.\n"
            "Price: $" + str(price) + " | RSI: " + str(ind["rsi"]) + " | ADX: " + str(ind["adx"]) + "\n"
            "EMA trend: " + ind["ema_trend"] + " | VWAP: $" + str(ind["vwap"]) + "\n"
            "BB: lo=$" + str(ind["bb_lo"]) + " hi=$" + str(ind["bb_hi"]) + "\n"
            "Patterns: " + ", ".join(pat_names) + "\n"
            + htf_str + "\n"
            "Direction: " + d + " | Score: " + str(setup["score"]) + "\n"
            "Return ONLY JSON: {\"summary\":\"...\",\"sentiment\":\"...\"}"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200, temperature=0.3,
        )
        raw  = resp.choices[0].message.content.replace("```json","").replace("```","").strip()
        comm = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception:
        comm = {"summary": "Scalp setup confirmed by RSI, EMA and volume.", "sentiment": "Gold showing clear directional bias."}

    bt_str = str(bt_rate) + "% (" + str(bt_trades) + " trades)" if bt_rate else "Insufficient history"

    return {
        "direction": d, "price": price,
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "inv": inv, "rr": rr, "atr": atr,
        "confidence": setup["confidence"], "score": setup["score"],
        "rsi": ind["rsi"], "adx": ind["adx"],
        "ema_trend": ind["ema_trend"],
        "vwap": ind["vwap"], "above_vwap": ind["above_vwap"],
        "bb_lo": ind["bb_lo"], "bb_hi": ind["bb_hi"],
        "supports": ind["supports"], "resistances": ind["resistances"],
        "patterns": pat_names,
        "reasons": setup["reasons"],
        "htf_15m": htf["15m"] if htf else "N/A",
        "htf_1h":  htf["1h"]  if htf else "N/A",
        "session": session,
        "summary": comm.get("summary", ""),
        "sentiment": comm.get("sentiment", ""),
        "backtest": bt_str,
        "time": datetime.now(timezone.utc),
    }


def fmt_signal(s):
    d   = s["direction"]
    c   = s["confidence"]
    bar = "#" * (c // 10) + "-" * (10 - c // 10)
    sup = " | ".join("$" + str(v) for v in s["supports"] if v > 0)
    res = " | ".join("$" + str(v) for v in s["resistances"] if v > 0)
    pat = " | ".join(s["patterns"])
    reasons = "\n".join("  + " + r for r in s["reasons"])
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    return "\n".join([
        "SCALP SIGNAL: " + SYMBOL_NAME + " " + d,
        "Session: " + s["session"] + " | TF: 5m",
        "HTF: 15m=" + s["htf_15m"] + " | 1H=" + s["htf_1h"],
        "====================",
        "Entry:        $" + str(s["price"]),
        "Stop Loss:    $" + str(s["sl"]) + "  (-$" + str(round(s["price"] - s["sl"] if d == "BUY" else s["sl"] - s["price"], 2)) + ")",
        "TP1:          $" + str(s["tp1"]) + "  (quick scalp)",
        "TP2:          $" + str(s["tp2"]) + "  (main target)",
        "TP3:          $" + str(s["tp3"]) + "  (full run)",
        "Invalidation: $" + str(s["inv"]),
        "R:R:          " + s["rr"],
        "====================",
        "RSI: " + str(s["rsi"]) + " | ADX: " + str(s["adx"]) + " | ATR: $" + str(s["atr"]),
        "EMA: " + s["ema_trend"] + " | VWAP: $" + str(s["vwap"]) + " (" + ("above" if s["above_vwap"] else "below") + ")",
        "BB: $" + str(s["bb_lo"]) + " -- $" + str(s["bb_hi"]),
        "Support:    " + (sup if sup else "none"),
        "Resistance: " + (res if res else "none"),
        "====================",
        "Patterns: " + pat,
        "====================",
        "Confluence:",
        reasons,
        "====================",
        "Backtest (similar setups): " + s["backtest"],
        "====================",
        s["summary"],
        s["sentiment"],
        "====================",
        "Confidence: [" + bar + "] " + str(c) + "%",
        "Score: " + str(s["score"]) + "/" + str(SCORE_NEEDED) + " needed",
        "Time: " + now,
        "For educational purposes only."
    ])


# ── WIN/LOSS TRACKING ──────────────────────────────────────────────────────
async def check_outcomes(bot):
    now = datetime.now(timezone.utc)
    remaining = []
    for t in pending_checks:
        age = (now - t["time"]).total_seconds() / 60
        if age < 20:
            remaining.append(t); continue
        try:
            price = get_live_price()
            if not price:
                remaining.append(t); continue
            d = t["direction"]
            hit_tp = (d=="BUY" and price>=t["tp1"]) or (d=="SELL" and price<=t["tp1"])
            hit_sl = (d=="BUY" and price<=t["sl"])  or (d=="SELL" and price>=t["sl"])
            result = "WIN" if hit_tp else "LOSS" if hit_sl else "OPEN"
            t["result"] = result
            trade_log.append(t)
            if result != "OPEN":
                pnl = abs(t["tp1"] - t["entry"]) if result == "WIN" else abs(t["sl"] - t["entry"])
                msg = (
                    "SCALP RESULT: " + result + "\n"
                    "XAU/USD " + d + " @ $" + str(t["entry"]) + "\n"
                    "Current: $" + str(round(price, 2)) + "\n"
                    "P&L per oz: " + ("+" if result=="WIN" else "-") + "$" + str(round(pnl, 2)) + "\n"
                    "TP1: $" + str(t["tp1"]) + " | SL: $" + str(t["sl"])
                )
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            else:
                remaining.append(t)
        except Exception as e:
            log.warning("Outcome check error: " + str(e))
            remaining.append(t)
    pending_checks.clear()
    pending_checks.extend(remaining)


async def send_report(bot):
    today  = datetime.now(timezone.utc).date()
    trades = [t for t in trade_log if t["time"].date() == today]
    wins   = sum(1 for t in trades if t.get("result") == "WIN")
    losses = sum(1 for t in trades if t.get("result") == "LOSS")
    total  = wins + losses
    rate   = round(wins / total * 100, 1) if total > 0 else 0
    msg = (
        "DAILY SCALP REPORT - XAU/USD\n"
        + datetime.now(timezone.utc).strftime("%Y-%m-%d") + "\n"
        "====================\n"
        "Signals: " + str(total) + " | Wins: " + str(wins) + " | Losses: " + str(losses) + "\n"
        "Win Rate: " + str(rate) + "%\n"
        "====================\n"
        + ("No completed trades today." if total == 0 else "Keep reviewing your execution!")
    )
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)


# ── TELEGRAM COMMANDS ──────────────────────────────────────────────────────
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    try:
        price = get_live_price()
        df    = fetch_candles()
        if price and len(df) >= 60:
            ind      = get_indicators(df, price)
            patterns = detect_candle_patterns(df)
            session, session_ok = get_session()
            pat_str  = ", ".join(n for _, n in patterns) if patterns else "None"
            msg = (
                "XAU/USD SCALP STATUS\n"
                + datetime.now(timezone.utc).strftime("%H:%M UTC") + "\n"
                "====================\n"
                "Price:   $" + str(ind["price"]) + "\n"
                "RSI:     " + str(ind["rsi"]) + "\n"
                "ADX:     " + str(ind["adx"]) + "\n"
                "EMA:     " + ind["ema_trend"] + "\n"
                "VWAP:    $" + str(ind["vwap"]) + " (" + ("above" if ind["above_vwap"] else "below") + ")\n"
                "BB:      $" + str(ind["bb_lo"]) + " -- $" + str(ind["bb_hi"]) + "\n"
                "Vol:     " + str(ind["vol_ratio"]) + "x\n"
                "Session: " + session + "\n"
                "Patterns: " + pat_str
            )
        else:
            msg = "Could not fetch live data right now."
    except Exception as e:
        msg = "Status error: " + str(e)
    await update.message.reply_text(msg)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_paused
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    is_paused = True
    await update.message.reply_text("Bot paused. /resume to restart.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_paused
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    is_paused = False
    await update.message.reply_text("Resumed. Scanning for scalp setups...")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    await send_report(Bot(token=TELEGRAM_TOKEN))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    await update.message.reply_text(
        "XAU/USD SCALP BOT COMMANDS\n"
        "====================\n"
        "/status  - Live price + all indicators\n"
        "/pause   - Stop signals\n"
        "/resume  - Resume signals\n"
        "/report  - Today's win/loss report\n"
        "/help    - This message"
    )


# ── MAIN SCAN ──────────────────────────────────────────────────────────────
async def scan():
    global is_paused, last_signal_time
    if is_paused:
        log.info("Paused"); return

    blocked, reason = is_news_time()
    if blocked:
        log.info("News filter: " + reason); return

    session, session_ok = get_session()

    bot = Bot(token=TELEGRAM_TOKEN)
    await check_outcomes(bot)

    try:
        price = get_live_price()
        if not price:
            log.warning("Could not get live price"); return

        df = fetch_candles()
        if len(df) < 60:
            log.warning("Not enough candles"); return

        ind      = get_indicators(df, price)
        patterns = detect_candle_patterns(df)
        htf      = get_htf_bias()
        setup    = score_scalp_setup(ind, patterns, session_ok)

        log.info(
            "XAU/USD | $" + str(ind["price"]) +
            " | RSI:" + str(ind["rsi"]) +
            " | ADX:" + str(ind["adx"]) +
            " | EMA:" + ind["ema_trend"] +
            " | Score:" + str(setup["score"]) +
            " | Session:" + session +
            " | Patterns:" + str(len(patterns))
        )

        if not setup["is_setup"]:
            return

        # HTF filter: block signals that go against higher timeframe trend
        if htf and htf["aligned"]:
            if setup["direction"] == "BUY" and htf["bias"] == "bearish":
                log.info("HTF bearish - blocking BUY signal"); return
            if setup["direction"] == "SELL" and htf["bias"] == "bullish":
                log.info("HTF bullish - blocking SELL signal"); return

        # Min gap between signals
        now = datetime.now(timezone.utc)
        if last_signal_time:
            gap = (now - last_signal_time).total_seconds() / 60
            if gap < MIN_GAP_MIN:
                log.info("Too soon: " + str(round(gap)) + "min since last signal"); return

        # Backtest check
        bt_rate, bt_trades = backtest(df, setup["direction"])
        if bt_rate is not None and bt_rate < MIN_BACKTEST:
            log.info("Backtest too low: " + str(bt_rate) + "% - skipping"); return

        signal = build_signal(ind, setup, patterns, htf, bt_rate, bt_trades, session)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=fmt_signal(signal))
        last_signal_time = now

        pending_checks.append({
            "direction": signal["direction"],
            "entry": signal["price"],
            "sl": signal["sl"],
            "tp1": signal["tp1"],
            "time": now, "result": None
        })

        log.info("Signal sent: " + signal["direction"] + " @ $" + str(signal["price"]))

    except Exception as e:
        log.error("Scan error: " + str(e))


# ── MAIN ───────────────────────────────────────────────────────────────────
async def main():
    log.info("XAU/USD Scalp Bot starting - scan every " + str(CHECK_INTERVAL) + " min")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pause",  cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("help",   cmd_help))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(scan, "interval", minutes=CHECK_INTERVAL)
    scheduler.add_job(
        lambda: asyncio.create_task(send_report(Bot(token=TELEGRAM_TOKEN))),
        "cron", hour=21, minute=0
    )
    scheduler.start()

    await scan()
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    log.info("Running. Commands: /status /pause /resume /report /help")

    while True:
        await asyncio.sleep(60)


asyncio.run(main())