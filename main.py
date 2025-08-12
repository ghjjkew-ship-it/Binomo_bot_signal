#!/usr/bin/env python3
# coding: utf-8
"""
Телеграм-бот сигналов (AUD/CAD OTC)
Команды:
  /start - приветствие
  /новый  - запрос нового сигнала
  /статистика - посмотреть статистику по пометках
Под кнопками сигнала: "Верный" / "Неверный" — сохраняется фидбек и модель обучается online (SGD).
Использует TwelveData (основной) и yfinance (фоллбек).
Данные и модель сохраняются в sqlite и на диск.
"""

import os
import json
import sqlite3
from datetime import datetime
import aiohttp
import pandas as pd
import numpy as np
import joblib

from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils import executor
from sklearn.linear_model import SGDClassifier

# ------------------ CONFIG ------------------
TG_TOKEN = os.getenv("TG_TOKEN", "7718464528:AAEM9KvxtxpvNg6234PNNeBGoWUCz5y8W-s")
CHAT_ID = int(os.getenv("CHAT_ID", "7830712705"))
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "74c58d1151144bb990851b622ba809b2")

PAIR_TWELVE = "AUD/CAD"    # TwelveData format
PAIR_YFIN = "AUDCAD=X"     # yfinance fallback

DB_FILE = "bot_data.sqlite"
MODEL_FILE = "model.joblib"

# ------------------ Bot init ------------------
bot = Bot(token=TG_TOKEN)
dp = Dispatcher(bot)

# ------------------ DB ------------------
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    chat_id INTEGER,
    message_id INTEGER,
    pair TEXT,
    features TEXT,
    predicted INTEGER,
    label INTEGER
)
""")
conn.commit()

# ------------------ Model init ------------------
# model expects features shape (1,5)
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
    except Exception:
        model = SGDClassifier(loss="log_loss")
        model.partial_fit(np.zeros((1,5)), [0], classes=[0,1])
else:
    model = SGDClassifier(loss="log_loss")
    model.partial_fit(np.zeros((1,5)), [0], classes=[0,1])

# ------------------ Features / Indicators ------------------
def compute_features_from_closes(closes: pd.Series):
    # Input: pd.Series older->newer
    # Ensure enough length
    if len(closes) < 16:
        closes = pd.Series(list(closes) + [closes.iloc[-1]] * (16 - len(closes)))

    last3 = closes.values[-3:]
    sma = closes.rolling(14).mean().iloc[-1]
    sma_prev = closes.rolling(14).mean().iloc[-2]
    sma_diff = 0.0 if pd.isna(sma) or pd.isna(sma_prev) else (sma - sma_prev)

    delta = closes.diff().fillna(0)
    up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    down = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
    rsi = 50.0
    if (up + down) != 0:
        rs = up / (down if down != 0 else 1e-9)
        rsi = 100 - (100 / (1 + rs))

    # normalize last3 by last close to reduce scale effect
    denom = last3[-1] if last3[-1] != 0 else 1
    norm_last3 = last3 / denom

    feats = np.concatenate([norm_last3, [sma_diff, rsi]])
    return feats.reshape(1, -1)

# ------------------ Data fetchers ------------------
async def fetch_twelvedata(interval="1min", outputsize=100):
    url = ("https://api.twelvedata.com/time_series"
           f"?symbol={PAIR_TWELVE}"
           f"&interval={interval}"
           f"&outputsize={outputsize}"
           f"&format=JSON"
           f"&apikey={TWELVEDATA_API_KEY}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            data = await resp.json()
    if "values" in data:
        df = pd.DataFrame(data["values"])
        df = df.sort_values("datetime")
        df["close"] = df["close"].astype(float)
        return df["close"].reset_index(drop=True)
    else:
        raise RuntimeError(f"TwelveData error: {data}")

def fetch_yfinance_closes(limit=100):
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not available") from e
    t = yf.Ticker(PAIR_YFIN)
    hist = t.history(period="1d", interval="1m")
    if hist is None or hist.empty:
        raise RuntimeError("yfinance returned empty")
    closes = hist["Close"].tail(limit)
    closes = closes.reset_index(drop=True)
    return closes

async def get_closes():
    # try TwelveData, else fallback
    try:
        return await fetch_twelvedata()
    except Exception:
        try:
            return fetch_yfinance_closes()
        except Exception as e:
            raise RuntimeError("Data sources failed: " + str(e))

# ------------------ Signal logic ------------------
def decide_from_features(feats):
    rsi = feats[0, -1]
    # simple thresholds
    if rsi < 30:
        return 1  # BUY
    if rsi > 70:
        return 0  # SELL
    # else use model
    pred = int(model.predict(feats)[0])
    return pred

# ------------------ Telegram helpers ------------------
def signal_keyboard():
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton("Верный ✅", callback_data="fb|1"),
        InlineKeyboardButton("Неверный ❌", callback_data="fb|0")
    )
    return kb

async def send_signal(chat_id, dir_text, feats):
    text = (f"📢 <b>Сигнал: AUD/CAD (OTC)</b>\n"
            f"Направление: <b>{dir_text}</b>\n"
            f"Время (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Нажми: Верный / Неверный")
    msg = await bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=signal_keyboard())
    # save to DB
    feats_json = json.dumps(feats.tolist())
    predicted = 1 if dir_text == "BUY" else 0
    cur.execute("INSERT INTO signals (ts, chat_id, message_id, pair, features, predicted, label) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), chat_id, msg.message_id, "AUD/CAD", feats_json, predicted, None))
    conn.commit()
    return msg.message_id

# ------------------ Handlers ------------------
@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    txt = ("Привет! Я бот сигналов.\n\n"
           "Используй команду /новый чтобы получить новый сигнал по AUD/CAD (OTC).\n"
           "Под каждым сигналом есть кнопки: Верный / Неверный — отмечай, чтобы бот учился.\n"
           "Команда /статистика показывает точность по отмеченным сигналам.")
    await message.reply(txt)

@dp.message_handler(commands=["новый"])
async def cmd_new(message: types.Message):
    await message.reply("Генерирую сигнал... ⏳")
    try:
        closes = await get_closes()
    except Exception as e:
        await message.reply(f"Ошибка получения данных: {e}")
        return
    feats = compute_features_from_closes(closes)
    pred = decide_from_features(feats)
    dir_text = "BUY" if pred == 1 else "SELL"
    await send_signal(message.chat.id, dir_text, feats)

@dp.message_handler(commands=["статистика"])
async def cmd_stats(message: types.Message):
    cur.execute("SELECT COUNT(*) FROM signals")
    total = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM signals WHERE label IS NOT NULL")
    labeled = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM signals WHERE label = predicted AND label IS NOT NULL")
    correct = cur.fetchone()[0] or 0
    acc = (correct / labeled * 100) if labeled > 0 else 0.0
    txt = (f"📊 Статистика\n\nВсего сигналов: {total}\nОтмечено: {labeled}\n"
           f"Корректных по пометкам: {correct}\nТочность: {acc:.2f}%")
    await message.reply(txt)

@dp.callback_query_handler(lambda c: c.data and c.data.startswith("fb|"))
async def cb_feedback(callback: types.CallbackQuery):
    try:
        label = int(callback.data.split("|")[1])
    except:
        await callback.answer("Ошибка")
        return
    message_id = callback.message.message_id
    cur.execute("SELECT id, features FROM signals WHERE message_id = ? ORDER BY id DESC LIMIT 1", (message_id,))
    row = cur.fetchone()
    if not row:
        await callback.answer("Сигнал не найден в БД")
        return
    sig_id, feats_json = row
    feats = np.array(json.loads(feats_json))
    # update label
    cur.execute("UPDATE signals SET label = ? WHERE id = ?", (label, sig_id))
    conn.commit()
    # train online
    try:
        model.partial_fit(feats, [label])
        joblib.dump(model, MODEL_FILE)
    except Exception:
        # ignore training errors
        pass
    await callback.answer("Фидбек сохранён ✅")

# ------------------ Startup ------------------
if __name__ == "__main__":
    print("Bot starting...")
    executor.start_polling(dp, skip_updates=True)
