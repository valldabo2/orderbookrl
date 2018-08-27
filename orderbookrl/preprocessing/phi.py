from orderbookrl.preprocessing.ewma import EWMA
from collections import deque
import numpy as np
from ray.rllib.models.preprocessors import Preprocessor


def ofi_et(b_t, b_t_1, v_b_t, v_b_t_1, a_t, a_t_1, v_a_t, v_a_t_1):
    et = int(b_t >= b_t_1) * v_b_t - int(b_t <= b_t_1) * v_b_t_1 - int(a_t <= a_t_1) * v_a_t + int(
        a_t >= a_t_1) * v_a_t_1
    return et


class MarketVariables(Preprocessor):
    def __init__(self, shape, options):
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

        self.shape = shape.shape

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
