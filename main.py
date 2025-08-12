#!/usr/bin/env python3
# coding: utf-8
"""
Телеграм-бот сигналов (AUD/CAD OTC)
Команды:
  /start - приветствие + главное меню с кнопками Старт/Стоп
  /новый  - запрос нового сигнала (только если подписан)
  /статистика - посмотреть общую статистику
  /история - последние 5 сигналов
Под кнопками сигнала: "Верный" / "Неверный" — сохраняется фидбек и модель обучается online (SGD).
Данные и модель сохраняются в sqlite и на диск.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import numpy as np
import joblib
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor
from sklearn.linear_model import SGDClassifier

# ------------- Логирование -------------
logging.basicConfig(filename='bot.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# ------------------ CONFIG ------------------
TG_TOKEN = os.getenv("TG_TOKEN", "7718464528:AAEM9KvxtxpvNg6234PNNeBGoWUCz5y8W-s")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "74c58d1151144bb990851b622ba809b2")

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
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    chat_id INTEGER PRIMARY KEY,
    subscribed INTEGER DEFAULT 0,
    signals_received INTEGER DEFAULT 0,
    correct_feedbacks INTEGER DEFAULT 0
)
""")
conn.commit()

# ------------------ Model init ------------------
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

    denom = last3[-1] if last3[-1] != 0 else 1
    norm_last3 = last3 / denom

    feats = np.concatenate([norm_last3, [sma_diff, rsi]])
    return feats.reshape(1, -1)

# ------------------ Data fetchers ------------------
PAIR_TWELVE = "AUD/CAD"
PAIR_YFIN = "AUDCAD=X"

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
    import yfinance as yf
    t = yf.Ticker(PAIR_YFIN)
    hist = t.history(period="1d", interval="1m")
    if hist is None or hist.empty:
        raise RuntimeError("yfinance returned empty")
    closes = hist["Close"].tail(limit)
    closes = closes.reset_index(drop=True)
    return closes

async def get_closes():
    try:
        return await fetch_twelvedata()
    except Exception as e:
        logging.error(f"TwelveData failed: {e}")
        try:
            return fetch_yfinance_closes()
        except Exception as e2:
            logging.error(f"yfinance failed: {e2}")
            raise RuntimeError("Data sources failed: " + str(e2))

# ------------------ Signal logic ------------------
def decide_from_features(feats):
    rsi = feats[0, -1]
    if rsi < 30:
        return 1  # BUY
    if rsi > 70:
        return 0  # SELL
    pred = int(model.predict(feats)[0])
    return pred

# ------------------ Keyboards ------------------
signal_keyboard_markup = InlineKeyboardMarkup(row_width=2)
signal_keyboard_markup.add(
    InlineKeyboardButton("Верный ✅", callback_data="fb|1"),
    InlineKeyboardButton("Неверный ❌", callback_data="fb|0")
)

main_kb = ReplyKeyboardMarkup(resize_keyboard=True)
main_kb.add(KeyboardButton("Старт"), KeyboardButton("Стоп"))

# ------------------ Handlers ------------------

@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    txt = ("👋 Привет!\n"
           "Я бот сигналов AUD/CAD OTC.\n\n"
           "Создатель: Твой_ник_или_имя\n"
           "Бот учится на твоих пометках (Верный/Неверный).\n\n"
           "Нажми кнопку Старт, чтобы получать сигналы.")
    await message.reply(txt, reply_markup=main_kb)

@dp.message_handler(lambda m: m.text == "Старт")
async def btn_start(message: types.Message):
    chat_id = message.chat.id
    cur.execute("INSERT OR IGNORE INTO users(chat_id) VALUES (?)", (chat_id,))
    cur.execute("UPDATE users SET subscribed = 1 WHERE chat_id = ?", (chat_id,))
    conn.commit()
    await message.reply("✅ Вы подписались на сигналы.\nИспользуйте команду /новый или ждите сигналов.", reply_markup=main_kb)

@dp.message_handler(lambda m: m.text == "Стоп")
async def btn_stop(message: types.Message):
    chat_id = message.chat.id
    cur.execute("UPDATE users SET subscribed = 0 WHERE chat_id = ?", (chat_id,))
    conn.commit()

    cur.execute("SELECT signals_received, correct_feedbacks FROM users WHERE chat_id = ?", (chat_id,))
    row = cur.fetchone()
    signals = row[0] if row else 0
    correct = row[1] if row else 0
    accuracy = (correct / signals * 100) if signals > 0 else 0

    if correct < 10:
        level = "Новичок"
    elif correct < 30:
        level = "Продвинутый"
    else:
        level = "Эксперт"

    earned = correct * 100

    txt = (f"👋 Вы отписались от сигналов.\n\n"
           f"Статистика:\n"
           f"Получено сигналов: {signals}\n"
           f"Правильных пометок: {correct}\n"
           f"Точность обучения: {accuracy:.2f}%\n"
           f"Уровень: {level}\n"
           f"Примерный заработок: {earned} у.е.\n\n"
           "Спасибо, что были с нами!")

    await message.reply(txt, reply_markup=types.ReplyKeyboardRemove())

@dp.message_handler(commands=["новый"])
async def cmd_new(message: types.Message):
    chat_id = message.chat.id
    cur.execute("SELECT subscribed FROM users WHERE chat_id = ?", (chat_id,))
    row = cur.fetchone()
    if not row or row[0] == 0:
        await message.reply("Вы не подписаны. Нажмите кнопку Старт, чтобы подписаться.")
        return

    await message.reply("Генерирую сигнал... ⏳")
    try:
        closes = await get_closes()
    except Exception as e:
        await message.reply(f"Ошибка получения данных: {e}")
        return
    feats = compute_features_from_closes(closes)
    pred = decide_from_features(feats)
    dir_text = "BUY" if pred == 1 else "SELL"
    alert = feats[0, -1] < 20 or feats[0, -1] > 80
    await send_signal(chat_id, dir_text, feats, alert=alert)

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

@dp.message_handler(commands=["история"])
async def cmd_history(message: types.Message):
    cur.execute("SELECT ts, pair, predicted, label FROM signals ORDER BY ts DESC LIMIT 5")
    rows = cur.fetchall()
    if not rows:
        await message.reply("История сигналов пока пуста.")
        return
    texts = []
    for ts, pair, pred, label in rows:
        pred_text = "BUY" if pred == 1 else "SELL"
        label_text = "❓" if label is None else ("✅" if label == pred else "❌")
        texts.append(f"{ts[:19]} | {pair} | {pred_text} | Отметка: {label_text}")
    await message.reply("Последние сигналы:\n" + "\n".join(texts))

@dp.callback_query_handler(lambda c: c.data and c.data.startswith("fb|"))
async def cb_feedback(callback: types.CallbackQuery):
    try:
        label = int(callback.data.split("|")[1])
    except:
        await callback.answer("Ошибка")
        return
    message_id = callback.message.message_id
    cur.execute("SELECT id, features, predicted, chat_id FROM signals WHERE message_id = ? ORDER BY id DESC LIMIT 1", (message_id,))
    row = cur.fetchone()
    if not row:
        await callback.answer("Сигнал не найден в БД")
        return
    sig_id, feats_json, predicted, chat_id = row
    feats = np.array(json.loads(feats_json))
    cur.execute("UPDATE signals SET label = ? WHERE id = ?", (label, sig_id))

    if label == predicted:
        cur.execute("UPDATE users SET correct_feedbacks = correct_feedbacks + 1 WHERE chat_id = ?", (chat_id,))

    conn.commit()

    try:
        model.partial_fit(feats, [label])
        joblib.dump(model, MODEL_FILE)
    except Exception:
        pass
    await callback.answer("Фидбек сохранён ✅")

# ------------------ Signal sender ------------------
async def send_signal(chat_id, dir_text, feats, alert=False):
    cur.execute("SELECT subscribed FROM users WHERE chat_id = ?", (chat_id,))
    row = cur.fetchone()
    if not row or row[0] == 0:
        return

    cur.execute("UPDATE users SET signals_received = signals_received + 1 WHERE chat_id = ?", (chat_id,))

    alert_text = "⚠️ RSI экстремум! " if alert else ""
    text = (f"📢 <b>{alert_text}Сигнал: AUD/CAD (OTC)</b>\n"
            f"Направление: <b>{dir_text}</b>\n"
            f"Время (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Нажми: Верный / Неверный")
    msg = await bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=signal_keyboard_markup)

    feats_json = json.dumps(feats.tolist())
    predicted = 1 if dir_text == "BUY" else 0
    cur.execute("INSERT INTO signals (ts, chat_id, message_id, pair, features, predicted, label) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), chat_id, msg.message_id, "AUD/CAD", feats_json, predicted, None))
    conn.commit()
    logging.info(f"Sent signal {dir_text} to {chat_id}")
    return msg.message_id

# ------------------ Startup ------------------
if __name__ == "__main__":
    print("Bot starting...")
    executor.start_polling(dp, skip_updates=True)
