from pprint import pprint
import ccxt
import pandas as pd
import numpy as np
import threading
import time
import random
import tensorflow as tf

api_key = ''
secret_key = ''

sandbox = ccxt.phemex({
    'enableRateLimit': True,
    'timeout': 30000,
    'apiKey' : api_key,
    'secret' : secret_key,
    'options': {
        'defaultType': 'swap',
    },
})
sandbox.set_sandbox_mode(True)
sandbox.verbose = False  # uncomment for debugging purposes if necessary


symbol = 'BTCUSD'
full_symbol = 'BTC/USD:BTC'
collateral_symbol = "BTC"
timeframe = '1m'
transaction_fee = 0.001
max_risk = 0.05

parameter = {'type':'swap', 'code':'BTC'}

def getprice(exchange, curr: str) -> float:
    """ 
    Takes 'BTC' and returns the ticker price for 'BTC/USDT', if 'USDT' is passed in it
    returns 1.0. In this example pass the phemex exchange object in
    """
    if curr == 'USDT':
        return 1.0
    else:
        tick = exchange.fetchTicker(curr+'/USDT')
        mid_point = tick['bid']

    return mid_point

class MarketMakingStrategy:
    def __init__(self, num_states, num_actions, alpha, gamma, epsilon, model):
        self.sandbox = sandbox
        self.starting_balance = self.get_total_balance(self.sandbox)
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.total_balance = 0
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995       
        self.model = model
        self.placed_orders = []
        self.starting_balance = self.get_total_balance()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def update_total_balance(self):
        while True:
            self.total_balance = self.get_phemex_balances().balance.sum()#self.get_total_balance()
            print("Total balance is: ", self.total_balance)
            time.sleep(10)

    def start_balance_updater(self):
        balance_updater_thread = threading.Thread(target=self.update_total_balance)
        balance_updater_thread.daemon = True
        balance_updater_thread.start()


    def get_phemex_balances(self):
        phemexBalance = sandbox.fetch_balance(params=parameter)
        
        balances = []
        for symbol, value in phemexBalance['total'].items():
            if value > 0.0:
                bid_price = getprice(sandbox, symbol)
                datum = {}
                datum['asset'] = symbol
                datum['free'] = value
                datum['locked'] = 0.0
                datum['total'] = value
                datum['price'] = bid_price 
                datum['balance'] = round(bid_price * datum['total'],2)
                datum['platform'] = 'Phemex'
                balances.append(datum)
        
        df = pd.DataFrame(balances)
        return df

    def get_total_balance(self, base_currency=collateral_symbol):
        response = sandbox.fetch_balance(params=parameter)
        balances = response['info']['data']
        total_balance = 0
        
        tickers = sandbox.fetch_balance(params=parameter)   
        available_symbols = set(tickers.keys())

        for balance in balances:
            asset = self.get_phemex_balances().balance.sum()
            free_balance = self.get_phemex_balances().free.sum()
            locked_balance = self.get_phemex_balances().locked.sum()
            total_asset_balance = free_balance + locked_balance

            if asset == base_currency:
                total_balance += total_asset_balance
            else:
                try:
                    ticker = f'{asset}/{base_currency}'
                    if ticker in available_symbols:
                        asset_price = tickers[ticker]['last']
                        total_balance += total_asset_balance * asset_price
                except ccxt.BaseError as e:
                    print(f"Error fetching ticker for {ticker}: {e}")   
        return total_balance

    def get_state(self, data):
        order_book, ohlcv = data
        bids, asks = order_book['bids'], order_book['asks']
        state = np.concatenate((np.array(bids[:6]).flatten(), np.array(asks[:6]).flatten(), ohlcv))
        return state

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = self.model(state[np.newaxis])
            return np.argmax(q_values.numpy())

    def update_q_network(self, state, action, reward, next_state):
        next_q_values = self.model(next_state[np.newaxis]).numpy()
        target = reward + self.gamma * np.max(next_q_values)
        target_q_values = self.model(state[np.newaxis]).numpy()
        target_q_values[0, action] = target
        
        self.model.fit(state[np.newaxis], target_q_values, epochs=1, verbose=0)

    def get_balance(self, currency):  
        balances = sandbox.fetch_balance()  
        for balance in balances['info']['balances']:
            if balance['asset'] == currency:
                return float(balance['free'])  
        return 0

    def execute_action(self, action):
        order_book = sandbox.fetch_order_book(symbol)
        bids, asks = order_book['bids'], order_book['asks']

        sandbox.load_markets()
        min_trade_amount = 1

        min_notional = 1
        min_usd_amount = 6
        buffer = 1.05  # 1% buffer

        if action == 0:  # Buy
            usdt_balance = self.get_phemex_balances().total.sum()
            price = asks[0][0] * (1 - transaction_fee)
            if price:  
                amount = max(max_risk / price, min_trade_amount)
            else:
                amount = 1
            print(price)
            if usdt_balance >= min_notional:
                amount = max(max_risk / price, min_trade_amount)
                amount = max(amount, max(min_notional * buffer / price, min_usd_amount * buffer / price))
                amount = round(amount, 5)  # Round amount after updating

                print("Price:", price)
                print("Amount:", amount)
                print("Notional:", price * amount)
                print("Minimum notional:", min_notional)

                try:
                    order = sandbox.create_limit_buy_order(symbol, amount, price, params=parameter)
                    self.placed_orders.append((time.time(), order))
                    print(order['id'])
                except Exception as e:
                    print("Error creating buy order:", e)

        elif action == 1:  # Sell
            btc_balance = self.get_phemex_balances()['total'].sum()
            if btc_balance * bids[0][0] >= min_notional:
                price = bids[0][0] * (1 + transaction_fee)
                print(price)
                if price:
                    amount = max(max_risk / price, min_trade_amount)
                else:
                    amount = 1
                amount = max(max_risk / price, min_trade_amount)
                amount = max(amount, max(min_notional * buffer / price, min_usd_amount * buffer / price))
                amount = round(amount, 5)  # Round amount after updating

                print("Price:", price)
                print("Amount:", amount)
                print("Notional:", price * amount)
                print("Minimum notional:", min_notional)

                try:
                    order = sandbox.create_limit_sell_order(symbol, amount, price, params=parameter)
                    print(order)
                    self.placed_orders.append((time.time(), order))
                except Exception as e:
                    print("Error creating sell order:", e)

        else:  # Hold
            pass
        
    def cancelorder(self, order):
        # symbol = order['symbol']
        
        print(f"Attempting to cancel order {order['id']}")

        # Fetch open orders to get latest data
        open_orders = sandbox.fetch_open_orders(symbol, params=parameter)

        # Try to cancel order
        try:
            sandbox.cancel_order(order['id'], symbol)
            # sandbox.cancel_all_orders(symbol)
        except ccxt.OrderNotFound:
            print(f"Order {order['id']} not found when trying to cancel")
        except ccxt.NetworkError as e:
            print(f"Error canceling order due to network: {e}")
        except ccxt.ExchangeError as e:
            print(f"Error canceling order: {e}")

        print(f"Order {order['id']} canceled")

        # Update internal order tracking
        for open_order in open_orders:
            if open_order['id'] == order['id']:
                # Order still exists, cancel failed
                print(f"Order {order['id']} failed to cancel, still open")
            return 
        print(f"Order {order['id']} removed from tracking")

        # Add short delay to avoid race conditions
        time.sleep(1)


    def cancel_old_orders(self, max_age_seconds=180):
        current_time = time.time()
        orders_to_remove = []

        for i, (order_time, order) in enumerate(self.placed_orders):
            if current_time - order_time > max_age_seconds:
                try:
                    sandbox.cancel_order(order['id'], symbol)
                    print(f"Order {order['id']} canceled.")
                    orders_to_remove.append(i)
                except ccxt.OrderNotFound as e:
                    print(f"Order {order['id']} not found or already canceled/filled.")
                    orders_to_remove.append(i)
                except Exception as e:
                    print(f"Error canceling order {order['id']}:", e)

        for index in sorted(orders_to_remove, reverse=True):
            del self.placed_orders[index]

    def update_order_statuses(self):
        updated_orders = []
        for _, order in self.placed_orders:
            try:
                order_info = sandbox.fetch_order(order['id'], symbol)
                updated_orders.append((_, order_info))
            except Exception as e:
                print(f"Error fetching order {order['id']} status:", e)
        self.placed_orders = updated_orders

    def get_reward(self, action):
        starting_balance = self.get_phemex_balances().balance.sum()
        self.execute_action(action)
        time.sleep(5)
        new_balance = self.total_balance

        reward = new_balance - starting_balance

        return reward

    def update_order_statuses_and_remove_filled(self):
        orders_to_remove = []

        print(f"Updating order statuses for {len(self.placed_orders)} orders")

        for i, (_, order) in enumerate(self.placed_orders):
            print(self.placed_orders)
            try:
                sandbox.cancel_order(order['id'], symbol)
                orders_to_remove.append(i)
            except ccxt.NetworkError as e:
                print(f"Error updating order status: {e}")

        print(f"Removing {len(orders_to_remove)} orders")

        for index in sorted(orders_to_remove, reverse=True):
            try:
                del self.placed_orders[index]
            except IndexError:
                print(f"Error removing order from tracking")

        print(f"{len(self.placed_orders)} orders left after update")

    def run(self):
        self.starting_balance = self.get_total_balance()
        self.start_balance_updater()
        while True:
            data = (sandbox.fetch_order_book(symbol), sandbox.fetch_ohlcv(symbol, timeframe)[-1])
            print("data")
            state = self.get_state(data)
            print("state")
            action = self.choose_action(state)
            print("action")
            # self.update_order_statuses()
            reward = self.get_reward(action)
            self.update_order_statuses_and_remove_filled()
            print("reward")
            next_data = (sandbox.fetch_order_book(symbol), sandbox.fetch_ohlcv(symbol, timeframe)[-1])
            print("next_data")
            next_state = self.get_state(next_data)
            print("next_state")

            self.update_q_network(state, action, reward, next_state)
            self.update_epsilon() 
            print("epsilon: ", self.epsilon)
            self.cancel_old_orders()

            time.sleep(5)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse')

market_maker = MarketMakingStrategy(30, 3, 0.1, 0.99, 0.99, model)
market_maker.run()
