import time
import asyncio
import json
from websockets import connect
import numpy as np
import pandas as pd
import logging
import os
import cv2
import dateutil
import gymnasium
from gymnasium import Env
import gymnasium as gym 
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
from gymnasium import spaces
import  gymnasium as Discrete 
import yfinance as yf
import threading
from collections import deque
import ssl
import certifi


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=250):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance = self.initial_balance
        self.trades = []
        self.entry_price = 0
        self.tp = 0
        self.sl = 0
        observation = self.data.iloc[self.current_step].values.astype(np.float32)
        logging.info(f"Observation shape from reset: {observation.shape}, dtype: {observation.dtype}")
        return observation, {}

    def step(self, action):
        price = self.data.iloc[self.current_step]['Close']
        reward = 0
        done = False

        if action == 0 and self.position == 0:  # Buy
            self.entry_price = price
            self.position = 1
            self.tp = price * 1.01  # Take profit à +1%
            self.sl = price * 0.99  # Stop loss à -1%
            logging.info(f"Buy at {price}, TP: {self.tp}, SL: {self.sl}")

        elif action == 1 and self.position == 1:  # Sell
            profit = price - self.entry_price
            self.balance += profit
            reward = profit
            self.trades.append({
                'action': 'Buy -> Sell',
                'entry_price': self.entry_price,
                'exit_price': price,
                'tp': self.tp,
                'sl': self.sl,
                'profit_loss': profit
            })
            self.position = 0
            logging.info(f"Sell at {price}, Profit: {profit}")

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        observation = self.data.iloc[self.current_step].values.astype(np.float32)
        return observation, reward, done, False, {}

    def get_trades(self):
        return self.trades

def compute_fibonacci_levels(data):
    if data.empty:
        logging.error("Les données sont vides. Impossible de calculer les niveaux de Fibonacci.")
        return {}

    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low

    levels = {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "1.0": low
    }
    return levels

def detect_patterns(data, distance=10):
    prices = data['Close'].values
    peaks, _ = find_peaks(prices, distance=distance)
    troughs, _ = find_peaks(-prices, distance=distance)

    patterns = []
    
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            if abs(prices[peaks[i]] - prices[peaks[i + 1]]) < 0.01 * prices[peaks[i]]:
                patterns.append(("Double Top", peaks[i], peaks[i + 1]))
    
    if len(troughs) >= 2:
        for i in range(len(troughs) - 1):
            if abs(prices[troughs[i]] - prices[troughs[i + 1]]) < 0.01 * prices[troughs[i]]:
                patterns.append(("Double Bottom", troughs[i], troughs[i + 1]))

    return patterns

def check_doji(open, high, low, close):
    body = abs(open - close)
    total_range = high - low
    return body <= 0.1 * total_range

def check_marubozu(open, high, low, close):
    body = abs(open - close)
    upper_wick = high - max(open, close)
    lower_wick = min(open, close) - low
    return (upper_wick < 0.05 * body) and (lower_wick < 0.05 * body)

def check_bearish_engulfing(prev_open, prev_close, open, close):
    return (prev_open < prev_close and open > prev_close and 
            close < prev_open and close < open)

def check_bullish_engulfing(prev_open, prev_close, open, close):
    return (prev_open > prev_close and open < prev_close and 
            close > prev_open and close > open)
            
def detect_candlestick_patterns(data):
    pattern_features = pd.DataFrame(index=data.index, columns=['doji', 'marubozu', 'bearish_engulfing', 'bullish_engulfing'])
    
    for i in range(len(data)):
        if i > 0:  # Pour les patterns qui nécessitent plus d'une bougie
            pattern_features.loc[data.index[i], 'doji'] = check_doji(data['Open'].iloc[i], data['High'].iloc[i], data['Low'].iloc[i], data['Close'].iloc[i])
            pattern_features.loc[data.index[i], 'marubozu'] = check_marubozu(data['Open'].iloc[i], data['High'].iloc[i], data['Low'].iloc[i], data['Close'].iloc[i])
            pattern_features.loc[data.index[i], 'bearish_engulfing'] = check_bearish_engulfing(
                data['Open'].iloc[i-1], data['Close'].iloc[i-1], 
                data['Open'].iloc[i], data['Close'].iloc[i])
            pattern_features.loc[data.index[i], 'bullish_engulfing'] = check_bullish_engulfing(
                data['Open'].iloc[i-1], data['Close'].iloc[i-1], 
                data['Open'].iloc[i], data['Close'].iloc[i])
        else:
            pattern_features.loc[data.index[i]] = 0

    return pattern_features

def analyze_market_sentiment(data):
    """Analyse le sentiment du marché basé sur les indicateurs et la volatilité."""
    rsi = RSIIndicator(close=data['Close'], window=14).rsi().iloc[-1]
    macd = MACD(close=data['Close']).macd_diff().iloc[-1]
    bb = BollingerBands(close=data['Close'])
    bb_width = (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / data['Close'].iloc[-1]
    
    # Sentiment basé sur RSI, MACD et volatilité (largeur des bandes de Bollinger)
    sentiment_score = 0
    if rsi > 70:  # Surachat
        sentiment_score -= 0.3
    elif rsi < 30:  # Survente
        sentiment_score += 0.3
    if macd > 0:  # Tendance haussière
        sentiment_score += 0.2
    elif macd < 0:  # Tendance baissière
        sentiment_score -= 0.2
    sentiment_score += bb_width * 0.5  # Volatilité élevée = opportunité
    
    return sentiment_score

def calculate_tp_sl(price, sentiment_score, volatility):
    """Calcule TP et SL basés sur le sentiment et la volatilité."""
    base_tp = 0.015  # 1.5% de base
    base_sl = 0.01   # 1% de base
    
    # Ajuster TP/SL en fonction du sentiment et de la volatilité
    tp_factor = base_tp + (sentiment_score * 0.005) + (volatility * 0.01)
    sl_factor = base_sl - (sentiment_score * 0.005) + (volatility * 0.005)
    
    tp = price * (1 + tp_factor)
    sl = price * (1 - sl_factor)
    return tp, sl

def prepare_data_for_model(historical_data):
    data = historical_data.copy()
    logging.info(f"Shape initial des données : {data.shape}")

    # Indicateurs techniques
    data['rsi'] = RSIIndicator(close=data['Close'], window=14).rsi()
    macd = MACD(close=data['Close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(close=data['Close'])
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['stoch'] = stoch.stoch()

    # Calcul des niveaux de Fibonacci
    fib_levels = compute_fibonacci_levels(data)
    for level, value in fib_levels.items():
        data[f'fib_{level}'] = value

    # Détection des patterns chartistes
    patterns = detect_patterns(data)
    pattern_features = pd.DataFrame(0, index=data.index, columns=['double_top', 'double_bottom'])
    for pattern, idx1, idx2 in patterns:
        if pattern == "Double Top":
            pattern_features.loc[data.index[idx1], 'double_top'] = 1
            pattern_features.loc[data.index[idx2], 'double_top'] = 1
        elif pattern == "Double Bottom":
            pattern_features.loc[data.index[idx1], 'double_bottom'] = 1
            pattern_features.loc[data.index[idx2], 'double_bottom'] = 1
    data = pd.concat([data, pattern_features], axis=1)

    # Détection des patterns candlestick
    candlestick_patterns = detect_candlestick_patterns(data)
    data = pd.concat([data, candlestick_patterns], axis=1)

    # Vérification des NaN
    nan_columns = data.columns[data.isna().any()].tolist()
    if nan_columns:
        logging.warning(f"Colonnes contenant des NaN : {nan_columns}")
        data = data.fillna(0)

    # Normalisation tout en conservant les noms de colonnes
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    logging.info(f"Features finales : shape={features.shape}, min={np.min(features)}, max={np.max(features)}")
    return features, scaler

# Entraînement du modèle pour le trading
def train_trading_ai(data, initial_balance=250):
    env = TradingEnv(pd.DataFrame(data), initial_balance=initial_balance)
    check_env(env)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

# Fonction pour effectuer le trading avec l'IA
def trade_with_ai(model, data):
    env = TradingEnv(pd.DataFrame(data))
    obs = env.reset()
    for _ in range(len(data)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    return env.balance  # Final balance after trading

async def monitor_solana_market():
    websocket_url = "wss://stream.binance.com:9443/ws/solusdt@trade"
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    initial_balance = 250.0
    balance = initial_balance
    position = 0
    entry_price = 0.0
    tp = 0.0
    sl = 0.0
    trade_executed = False
    market_data = deque(maxlen=50)  # Collecter 50 points pour l'analyse

    try:
        async with connect(websocket_url, ssl=ssl_context) as websocket:
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                data = json.loads(message)
                price = float(data['p'])
                quantity = float(data['q'])

                market_data.append({
                    'Time': int(data['E']),
                    'Close': price,
                    'Volume': quantity
                })

                if len(market_data) >= 50 and not trade_executed:
                    df = pd.DataFrame(market_data)
                    sentiment_score = analyze_market_sentiment(df)
                    volatility = df['Close'].pct_change().std()  # Volatilité récente

                    if position == 0:
                        # Décider d'acheter si le sentiment est positif ou neutre
                        if sentiment_score >= 0:
                            entry_price = price
                            amount_invested = initial_balance  # Investir tout le portefeuille
                            sol_bought = amount_invested / entry_price
                            tp, sl = calculate_tp_sl(entry_price, sentiment_score, volatility)
                            position = 1
                            logging.info(f"Position ouverte (Achat) : {sol_bought:.4f} SOL à {entry_price:.2f} $ | Montant investi : {amount_invested:.2f} $ | TP : {tp:.2f} $ | SL : {sl:.2f} $")

                    elif position == 1:
                        if price >= tp:
                            profit = (tp - entry_price) * sol_bought
                            balance = profit  # Balance finale après vente
                            logging.info(f"Position fermée (TP atteint) : Vente à {tp:.2f} $, Gains : {profit:.2f} $")
                            trade_executed = True
                        elif price <= sl:
                            profit = (sl - entry_price) * sol_bought
                            balance = profit  # Balance finale après vente
                            logging.info(f"Position fermée (SL atteint) : Vente à {sl:.2f} $, Pertes : {profit:.2f} $")
                            trade_executed = True

                if trade_executed:
                    break

    except Exception as e:
        logging.error(f"Erreur WebSocket : {e}")

def train_model_thread(market_data, model_ready_event, model_container):
    while len(market_data) < 1000:
        logging.info("Attente de 1000 données pour l'entraînement initial...")
        time.sleep(5)
    
    logging.info("Début de l'entraînement initial")
    df = pd.DataFrame(market_data)
    features, scaler = prepare_data_for_model(df)
    if os.path.exists("ai_trading_model.zip"):
        logging.info("Chargement du modèle existant")
        model_container[0] = DQN.load("ai_trading_model.zip")
    else:
        logging.info("Entraînement d'un nouveau modèle")
        model_container[0] = train_trading_ai(features, initial_balance=250)
        model_container[0].save("ai_trading_model.zip")
    model_ready_event.set()
    logging.info("Modèle prêt pour la simulation")

def main():
    asyncio.run(monitor_solana_market())

if __name__ == '__main__':
    main()
