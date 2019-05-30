from orderbookrl.preprocessing.ewma import EWMA
from collections import deque
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models.catalog import ModelCatalog
from orderbookmdp.rl.market_order_envs import MarketOrderEnv
from orderbookmdp.order_book.constants import Q_ASK, Q_BID
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet, SGDClassifier
import time
import numpy as np
import pandas as pd


def ofi_et(b_t, b_t_1, v_b_t, v_b_t_1, a_t, a_t_1, v_a_t, v_a_t_1):
    et = int(b_t >= b_t_1) * v_b_t - int(b_t <= b_t_1) * v_b_t_1 - int(a_t <= a_t_1) * v_a_t + int(
        a_t >= a_t_1) * v_a_t_1
    return et


class Zeros(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (1,)

    def transform(self, observation):
        return (0,)


class MarketVariables(Preprocessor):

    shape = (6,)

    def _init_shape(self, obs_space, options):
        obs_shape = obs_space.shape
        n_private_variables = obs_shape[0] - 4

        custom_options = options['custom_options']
        self.k = 0
        macd_fast_l = custom_options.get('fast_macd_l') or 1200
        macd_slow_l = custom_options.get('slow_macd_l') or 2400
        self.macd_fast = EWMA(age=macd_fast_l)
        self.macd_slow = EWMA(age=macd_slow_l)
        self.macd_slow_n = macd_slow_l
        ofi_l = custom_options.get('ofi_l') or 1000
        self.ofi = EWMA(age=ofi_l)
        mid_l = custom_options.get('mid_l') or 100
        self.mid_q = deque(maxlen=mid_l)
        self.shape = (4 + n_private_variables,)

        return self.shape

    def transform(self, observation):

        quotes, private_variables = observation
        ask, ask_vol, bid, bid_vol = quotes

        mid = (ask+bid)/2
        self.macd_fast.add(mid)
        self.macd_slow.add(mid)
        self.mid_q.append(mid)

        if self.k == 0:
            self.k += 1
            self.ofi.add(0)
        else:
            ofi_t = ofi_et(bid, self.prev_bid, bid_vol, self.prev_bid_vol,
                           ask, self.prev_ask, ask_vol, self.prev_ask_vol)
            self.ofi.add(ofi_t)

        # Warmup
        if self.k < self.macd_slow_n:
            self.k += 1
            macd = 0
            mid_std = 0
        else:
            macd = self.macd_fast.value - self.macd_slow.value
            mid_std = np.std(self.mid_q)  # Time consuming

        ofi_ = self.ofi.value
        vol_mb = bid_vol - ask_vol

        self.prev_bid = bid
        self.prev_bid_vol = bid_vol
        self.prev_ask = ask
        self.prev_ask_vol = ask_vol

        return (ofi_, vol_mb, macd, mid_std) + private_variables


class MarketVariablesSingleL(MarketVariables):
    def _init_shape(self, obs_space, options):
        obs_shape = obs_space.shape
        n_private_variables = obs_shape[0] - 4

        custom_options = options['custom_options']
        self.k = 0
        macd_fast_l = custom_options.get('l') * 10
        macd_slow_l = custom_options.get('l')
        self.macd_fast = EWMA(age=macd_fast_l)
        self.macd_slow = EWMA(age=macd_slow_l)
        self.macd_slow_n = macd_slow_l
        ofi_l = custom_options.get('l')
        self.ofi = EWMA(age=ofi_l)
        mid_l = custom_options.get('l')
        self.mid_q = deque(maxlen=mid_l)
        self.shape = (4 + n_private_variables,)

        return self.shape


class PredictiveMarketVariables(MarketVariables):
    def __init__(self, shape, options):
        super(PredictiveMarketVariables, self).__init__(shape, options)
        self.shape = (self.shape[0] + 3,)
        self.observations = deque(maxlen=15000)
        self.mids = deque(maxlen=15000)
        self.regressor = ElasticNet(warm_start=True, alpha=20, l1_ratio=0)
        self.classifier = SGDClassifier(warm_start=True, max_iter=100, alpha=0.001, l1_ratio=0.5, penalty='elasticnet')
        self.is_fitted = False
        self.train_k = 1

    def transform(self, observation):
        if self.train_k == 1:
            prev_mid = (observation[0][Q_BID] + observation[0][Q_ASK]) / 2
        else:
            prev_mid =  (self.prev_ask + self.prev_bid)/2

        obs = MarketVariables.transform(self, observation)
        mid = (observation[0][Q_BID] + observation[0][Q_ASK])/2

        self.mids.append(mid)
        self.observations.append((obs[0:4]))

        self.train_k += 1
        if not self.is_fitted:
            if self.train_k == 1000:
                self.train()
        elif self.train_k % 10000 == 0:
            self.train()

        if self.is_fitted:
            mid_diff = (mid - prev_mid)/mid
            return obs + self.predict(obs[0:4] + (mid_diff,)) + (mid_diff,)
        else:
            return obs + (0, 0, 0)  # Zero percentage change and zero class (no change)

    def train(self):
        y = pd.Series(list(self.mids))
        y_pct = y.pct_change().fillna(0)
        y_sign = np.sign(y_pct)

        X = np.array(self.observations)
        X = np.concatenate([X, y_pct.shift(-1).fillna(0).values.reshape(-1, 1)], axis=1)
        X_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        #t = time.time()
        self.regressor.fit(X_scaled, y_pct)
        #print('Reg fit time:{:.2f}'.format(time.time()-t))

        #t = time.time()
        self.classifier.fit(X_scaled, y_sign)
        #print('Class fit time:{:.2f}'.format(time.time()-t))

        self.is_fitted = True

    def predict(self, obs):
        obs = np.array(obs).reshape(1, -1)
        return self.regressor.predict(obs)[0], self.classifier.predict(obs)[0]


if __name__ == '__main__':
    ModelCatalog.register_custom_preprocessor('mv_pred', PredictiveMarketVariables)
    env = MarketOrderEnv(order_paths='../../data/feather/', snapshot_paths='../../data/snap_json/',
                         max_sequence_skip=10000, max_episode_time='20hours', random_start=False)

    options = {'custom_preprocessor': 'mv_pred',
               'custom_options': {
                   'fast_macd_l': 1200,
                   'slow_macd_l': 2400,
                   'ofi_l': 1000,
                   'mid_l': 1000}
               }

    phi = MarketVariables(env.observation_space, options)

    t = time.time()
    for i in range(2):
        k = 0
        obs = env.reset()
        done = False
        while not done:
            action = 0
            obs, reward, done, info = env.step(action)

            obs_ = phi.transform(obs)
            k += 1
            if k % 100 == 0:
                print(obs_, env.market.time, reward)
        print('stops', env.market.time)

    print(time.time() - t)
