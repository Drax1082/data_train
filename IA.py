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

class TradingEnv(gymnasium.Env):
    metadata = {'render.modes': []}

    def __init__(self, data, initial_balance=100000):
        super().__init__()
        self.data = data
        self.action_space = gymnasium.spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.balance = initial_balance
        self.trades = []
        self.entry_price = 0
        self.tp = 0
        self.sl = 0

    def step(self, action):
        price = self.data.iloc[self.current_step]['Close']
        reward = 0
        
        if action == 0 and self.position == 1:  # Close long position
            reward = self.balance - price
            profit_loss = self.balance - self.entry_price
            self.trades.append({
                "action": "Sell",
                "entry_price": self.entry_price,
                "exit_price": price,
                "tp": self.tp,
                "sl": self.sl,
                "profit_loss": profit_loss
            })
            self.balance = price
            self.position = 0
        elif action == 2 and self.position == -1:  # Close short position
            reward = price - self.balance
            profit_loss = self.entry_price - price
            self.trades.append({
                "action": "Buy",
                "entry_price": self.entry_price,
                "exit_price": price,
                "tp": self.tp,
                "sl": self.sl,
                "profit_loss": profit_loss
            })
            self.balance = price
            self.position = 0
        elif action == 1:  # Hold
            pass
        elif action == 0 and self.position == 0:  # Open short position
            self.position = -1
            self.entry_price = price
            self.tp = price * 1.03
            self.sl = price * 0.97
        elif action == 2 and self.position == 0:  # Open long position
            self.position = 1
            self.entry_price = price
            self.tp = price * 1.03
            self.sl = price * 0.97

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Convertir les observations en float32
        observation = self.data.iloc[self.current_step].values.astype(np.float32) if not done else self.data.iloc[-1].values.astype(np.float32)
        return (observation, reward, done, False, {})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance = 100000
        self.trades = []
        self.entry_price = 0
        self.tp = 0
        self.sl = 0
        # Convertir l'observation initiale en float32
        observation = self.data.iloc[self.current_step].values.astype(np.float32)
        return observation, {}  # Retourne aussi un dictionnaire 'info'

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

def prepare_data_for_model(historical_data):
    data = historical_data.copy()
    
    # RSI
    data['rsi'] = RSIIndicator(close=data['Close'], window=14).rsi()
    
    # MACD
    macd = MACD(close=data['Close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = BollingerBands(close=data['Close'])
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['stoch'] = stoch.stoch()
    
    fib_levels = compute_fibonacci_levels(data)
    for level, value in fib_levels.items():
        data[f'fib_{level}'] = value

    patterns = detect_patterns(data)
    pattern_features = pd.DataFrame(0, index=data.index, columns=['double_top', 'double_bottom'])
    
    # Dans la fonction prepare_data_for_model
    for pattern, idx1, idx2 in patterns:
        if pattern == "Double Top":
            pattern_features.loc[data.index[idx1], 'double_top'] = 1
            pattern_features.loc[data.index[idx2], 'double_top'] = 1
        elif pattern == "Double Bottom":
            pattern_features.loc[data.index[idx1], 'double_bottom'] = 1
            pattern_features.loc[data.index[idx2], 'double_bottom'] = 1

    data = pd.concat([data, pattern_features], axis=1)

    candlestick_patterns = detect_candlestick_patterns(data)
    data = pd.concat([data, candlestick_patterns], axis=1)

    # Scaling
    scaler = StandardScaler()
    features = scaler.fit_transform(data)
    return features, scaler

# Entraînement du modèle pour le trading
def train_trading_ai(data, initial_balance=100000):
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

# Main function to orchestrate the trading process
def main():
    market_data = deque(maxlen=1000)  # Stocke les 1000 dernières entrées de données
    
    # Configuration du contexte SSL pour ignorer la vérification de certificat (non recommandé en production)
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async def handle_market_data():
        try:
            async with connect('wss://stream.binance.com:9443/ws/btcusdt@trade', ssl=ssl_context) as websocket:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    market_data.append({
                        'Time': int(data['E']),  # Timestamp
                        'Open': float(data['p']),  # Price as 'Open' for simplicity
                        'High': float(data['p']),  # Assuming trade price is both high and low
                        'Low': float(data['p']),
                        'Close': float(data['p']),
                        'Volume': float(data['q'])  # Quantity as volume
                    })
                    if len(market_data) >= 1000:  # Si on a assez de données pour un backtest
                        df = pd.DataFrame(market_data)
                        features, scaler = prepare_data_for_model(df)
                        
                        if os.path.exists("ai_trading_model.zip"):
                            model = DQN.load("ai_trading_model.zip")
                        else:
                            model = train_trading_ai(features, initial_balance=250)  # Utilisation de initial_balance
                            model.save("ai_trading_model.zip")
                        
                        # Simuler le trading avec les données en temps réel
                        env = TradingEnv(pd.DataFrame(features), initial_balance=250)
                        obs = env.reset()
                        for _ in range(len(features)):
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, done, _ = env.step(action)
                            if done:
                                break
                        
                        final_balance = env.balance
                        logging.info(f"Trading terminé. Balance finale : {final_balance}")

                        # Afficher chaque trade
                        for trade in env.get_trades():
                            logging.info(f"Trade: {trade['action']}, Entry: {trade['entry_price']}, Exit: {trade['exit_price']}, TP: {trade['tp']}, SL: {trade['sl']}, P/L: {trade['profit_loss']:.2f}")
                    
        except Exception as e:
            logging.error(f"WebSocket error: {e}")

    # Lancer la réception de données dans un thread séparé
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    thread = threading.Thread(target=loop.run_until_complete, args=(handle_market_data(),))
    thread.start()

    # Votre logique principale ici pourrait continuer à s'exécuter
    while True:
        try:
            time.sleep(60)  # Attendre une minute avant de vérifier à nouveau
        except Exception as e:
            logging.error(f"{time.time()} - Error in main loop: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
