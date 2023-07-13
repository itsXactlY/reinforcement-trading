import ccxt.pro
import pandas as pd
import numpy as np
import threading
import time
import random
import tensorflow as tf

from ccxt.base.errors import BaseError

# API keys
api_key = ''
secret_key = ''

# Create exchange
exchange = ccxt.pro.phemex({
  'enableRateLimit': False,
  'apiKey': api_key,
  'secret': secret_key,
  'options': {
    'defaultType': 'swap'
  }
})

exchange.set_sandbox_mode(True)

symbol = 'BTC/USD'
coin_symbol = 'BTC'
timeframe = '1m'

# Global snapshots
orderbook = None
ticker = None
ohlcv = None
balance = None

orders = {}
positions = {}

# ccxt.pro handlers
async def on_orderbook(book):
    global orderbook
    orderbook = book

async def on_ticker(tick):
    global ticker
    ticker = tick

async def on_ohlcv(data):
    global ohlcv
    ohlcv = data

async def on_balance(bal):
    global balance
    balance = bal

async def on_order(order):
    orders[order['id']] = order

async def on_position(position):
    positions[position['info']['symbol']] = position

# Subscribe with ccxt.pro
exchange.spawn(exchange.watch_order_book, symbol, on_orderbook)
exchange.spawn(exchange.watch_ticker, symbol, on_ticker)
exchange.spawn(exchange.watch_ohlcv, symbol, timeframe, on_ohlcv)
exchange.spawn(exchange.watch_balance, on_balance)

# Parameters 

transaction_fee = 0.001
max_risk = 0.05
min_order_size = 1
min_notional = 7 

# Helper functions

def get_bid_ask(book):
  return book['bids'][0], book['asks'][0]  

def get_ticker_price(symbol):
  global ticker
  return ticker[symbol]['last']

def should_cancel(order):

  age = time.time() - order['timestamp']  
  if age > 120:
    return True
  
  if order['status'] == 'closed':
    return True
  
  return False 

async def get_order_update(order):

  try:
      
    updated = await exchange.fetch_order(order['id'], order['symbol']) #ws.fetch_order(order['id'], order['symbol']) 
    return updated
  except BaseError as e:
    print(f"Error fetching update for order {order['id']}: {e}")
  return None
  

def calculate_total_balance():
  total = 0
  for coin in balance: 
    if coin == coin_symbol:
      total += balance[coin]
    else:
      price = get_ticker_price(coin + '/' + coin_symbol)
      total += balance[coin] * price
  return total

# Strategy
class MarketMakingStrategy:

    def __init__(self, num_states, num_actions, alpha, gamma, epsilon, model):

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = model

        self.placed_orders = []
        self.total_balance = 0

    def get_state(self):

        global orderbook, ohlcv
        bids = orderbook['bids']
        asks = orderbook['asks']
        return np.concatenate((np.array(bids[:6]).flatten(), np.array(asks[:6]).flatten(), ohlcv))

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = self.model(state[np.newaxis])
            return np.argmax(q_values[0])

    async def execute_action(self, action):

        bid, ask = get_bid_ask(orderbook)
        balance = await exchange.fetch_balance()

        if action == 0:  # Buy
            price = ask * (1 - transaction_fee)
            amount = min(balance * max_risk / price, min_order_size)
            amount = max(amount, min_notional / price)
            amount = round(amount, 5)
            order = await exchange.create_order(symbol, 'limit', 'buy', amount, price)
            self.placed_orders.append(order)

        elif action == 1:  # Sell
            price = bid * (1 + transaction_fee)
            # Calculate sell order size
            order = await exchange.create_order(symbol, 'limit', 'sell', amount, price)
            self.placed_orders.append(order)

    async def cancel_order(self, id):
        await exchange.cancel_order(id)

    async def update_order_statuses(self):

        updated_orders = []

        for order in self.placed_orders:
            updated = await exchange.fetch_order(order['id'], order['symbol'])
            if updated:
                updated_orders.append(updated)

        self.placed_orders = updated_orders

    async def cancel_old_orders(self, max_age):
        for order in self.placed_orders:
            if should_cancel(order):
                await self.cancel_order(order['id'])

    async def run(self):

        while True:

            state = self.get_state()
            action = self.choose_action(state)
            await self.execute_action(action)

            await self.update_order_statuses()
            await self.cancel_old_orders(max_age=120)

            # Update epsilon

            time.sleep(1)

# Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse')

# Create and run
strategy = MarketMakingStrategy(30, 3, 0.1, 0.99, 0.99, model)
exchange.spawn(strategy.run)
exchange.run_forever()
