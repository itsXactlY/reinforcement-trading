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
    'enableRateLimit': False,
    "rateLimit": 1500,
    'timeout': 30000,
    'apiKey': api_key,
    'secret': secret_key,
    'options': {
        'defaultType': 'swap',
    },
})
sandbox.set_sandbox_mode(True)
sandbox.verbose = False  # uncomment for debugging purposes if necessary

phemex = ccxt.phemex()

symbol = 'BTCUSD'
coin_symbol = "BTC"
timeframe = '1m'
transaction_fee = 0.001
max_risk = 0.05

parameter = {'type': 'swap', 'code': 'BTC'}


def getprice(curr: str) -> float:
    if curr == 'USDT':
        return 1.0
    else:
        tick = sandbox.fetch_ticker(curr+'USD')
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
            self.total_balance = self.get_phemex_balances().balance.sum()
            print(f"Total balance is: {self.total_balance}")
            time.sleep(60)

    def start_balance_updater(self):
        balance_updater_thread = threading.Thread(
            target=self.update_total_balance)
        balance_updater_thread.daemon = True
        balance_updater_thread.start()

    def get_phemex_balances(self):
        phemexBalance = sandbox.fetch_balance(params=parameter)

        balances = []
        for symbol, value in phemexBalance['total'].items():
            if value > 0.0:
                bid_price = getprice(coin_symbol)
                datum = {}
                datum['asset'] = symbol
                datum['free'] = value
                datum['locked'] = 0.0
                datum['total'] = value
                datum['price'] = bid_price
                datum['balance'] = round(bid_price * datum['total'], 2)
                datum['platform'] = 'Phemex'
                balances.append(datum)

        df = pd.DataFrame(balances)
        return df

    def get_total_balance(self, base_currency=coin_symbol):
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
        state = np.concatenate(
            (np.array(bids[:6]).flatten(), np.array(asks[:6]).flatten(), ohlcv))
        return state

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = self.model(state[np.newaxis])
            return np.argmax(q_values.numpy())

    def update_q_network(self, states, actions, rewards, next_states):
        next_q_values = self.model(next_states).numpy()
        max_next_q_values = np.max(next_q_values, axis=1)

        target_q_values = rewards + (self.gamma * max_next_q_values)

        masks = tf.one_hot(actions, self.num_actions)

        with tf.GradientTape() as tape:
            all_q_values = self.model(states)
            q_action = tf.reduce_sum(all_q_values * masks, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_action))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

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
        ''' 
        <--- WIP part starts here --->
        '''
        min_trade_amount = 100

        min_notional = 1
        min_usd_amount = 1
        buffer = 1.05  # 1% buffer

        if action == 0:  # Buy
            usdt_balance = self.get_phemex_balances().total.sum()
            price = asks[0][0] * (1 - transaction_fee)
            if price:
                amount = max(max_risk / price, min_trade_amount)
            else:
                amount = 1111
            print(price)
            if usdt_balance >= min_notional:
                amount = max(max_risk / price, min_trade_amount)
                amount = max(amount, max(min_notional * buffer /
                             price, min_usd_amount * buffer / price))
                amount = round(amount, 5)  # Round amount after updating

                print("Price:", price)
                print("Amount:", amount)
                print("Notional:", price * amount)
                print("Minimum notional:", min_notional)

                try:
                    order = sandbox.create_limit_buy_order(
                        symbol, amount, price, params=parameter)
                    self.placed_orders.append((time.time(), order))
                except Exception as e:
                    print("Error creating buy order:", e)

        elif action == 1:  # Sell
            btc_balance = self.get_phemex_balances()['total'].sum()
            if btc_balance * bids[0][0] >= min_notional:
                price = bids[0][0] * (1 + transaction_fee)
                if price:
                    amount = max(max_risk / price, min_trade_amount)
                else:
                    amount = 1111
                amount = max(max_risk / price, min_trade_amount)
                amount = max(amount, max(min_notional * buffer /
                             price, min_usd_amount * buffer / price))
                amount = round(amount, 5)  # Round amount after updating
                '''
                <--- WIP part ends here --->
                '''
                print("Price:", price)
                print("Amount:", amount)
                print("Notional:", price * amount)
                print("Minimum notional:", min_notional)

                try:
                    order = sandbox.create_limit_sell_order(
                        symbol, amount, price, params=parameter)
                    # print(order)
                    # time.sleep(.250)
                    self.placed_orders.append((time.time(), order))
                except Exception as e:
                    print("Error creating sell order:", e)

        else:  # Hold
            pass

    def cancel_old_orders(self, max_age_seconds=120): # from 180
        time.sleep(1)
        current_time = time.time()
        orders_to_remove = []

        for i, (order_time, order) in enumerate(self.placed_orders):
            if current_time - order_time > max_age_seconds:
                try:
                    self.sandbox.cancel_order(order['id'], order['symbol'], params=parameter)
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
        time.sleep(1)
        for order_time, order in self.placed_orders:
            try:
                order_info = self.sandbox.fetch_order(order['id'], order['symbol'], params=parameter)
                if order_info['status'] == 'closed':
                    # Handle closed orders
                    print(f"Order {order_info['id']} is closed.")
                else:
                    updated_orders.append((order_time, order_info))
            except ccxt.OrderNotFound as e:
                try:
                    open_orders = self.sandbox.fetch_open_orders(order['symbol'], params=parameter)
                    order_info = next(filter(lambda x: x['id'] == order['id'], open_orders), None)
                    if order_info is not None:
                        updated_orders.append((order_time, order_info))
                    else:
                        print(f"Order {order['id']} not found or already canceled/filled.")
                except Exception as e:
                    print(f"Error fetching order status for order {order['id']}: {e}")
            except Exception as e:
                print(f"Error fetching order status for order {order['id']}: {e}")
        self.placed_orders = updated_orders

    
    def get_reward(self, action):
        starting_balance = self.get_phemex_balances().balance.sum()

        self.execute_action(action)

        # Wait for order to execute
        time.sleep(5)

        new_balance = self.total_balance

        reward = new_balance - starting_balance

        return reward

    def run(self):

        self.starting_balance = self.get_total_balance()
        self.start_balance_updater()

        states = []
        actions = []
        rewards = []
        next_states = []

        while True:
            # Get current state
            data = (sandbox.fetch_order_book(symbol),
                    phemex.fetch_ohlcv(symbol, timeframe)[-1])
            state = self.get_state(data)
            states.append(state)
            time.sleep(1)
            
            # Choose action
            action = self.choose_action(state)
            actions.append(action)
            time.sleep(1)
            
            # Execute action and get reward
            reward = self.get_reward(action)
            rewards.append(reward)
            time.sleep(1)
            
            # Get next state
            next_data = (sandbox.fetch_order_book(symbol),
                            phemex.fetch_ohlcv(symbol, timeframe)[-1])
            next_state = self.get_state(next_data)
            next_states.append(next_state)
            time.sleep(1)
            
            # Update model
            if len(states) % 10 == 0:
                self.update_q_network(np.array(states), np.array(
                actions), np.array(rewards), np.array(next_states))
                states.clear()
                actions.clear()
                rewards.clear()
                next_states.clear()

            # Update order statuses
            self.update_order_statuses()
            time.sleep(1)
            
            # Cancel old orders
            self.cancel_old_orders()
            time.sleep(1)
            
            # Update epsilon
            self.update_epsilon()

            time.sleep(4)

# Model definition


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse')

market_maker = MarketMakingStrategy(30, 3, 0.1, 0.99, 0.99, model)
market_maker.run()
