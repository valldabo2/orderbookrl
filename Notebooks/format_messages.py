import gc
import os

import feather
import numpy as np
import pandas as pd
import ujson
from sortedcontainers.sortedlist import SortedList
from tqdm import tqdm
import json

data_dir = '../data'
print('Reformats data from:', data_dir)

files = os.listdir(data_dir + '/json/')  # noqa
mess_files = SortedList([filename for filename in files if 'mess' in filename],
                        key=lambda fn: pd.to_datetime(fn[:-10], format='%d_%m_%Y_%H_%M_%S'))

keys = {'order_type', 'reason', 'sequence', 'side', 'size', 'type', 'price', 'funds', 'order_id', 'time'}
price_tick = 0.01
price_dec = int(np.log10(1 / price_tick))

try:
    os.makedirs(data_dir + '/feather/')
except FileExistsError:
    pass
for k, messfile in tqdm(enumerate(list(mess_files))):
    messages = []
    with open(data_dir + '/json/' + messfile, 'r') as f:
        mess = f.readlines()
        for i, m in enumerate(mess):
            try:
                ms = ujson.loads(m)
                ms = {k: v for k, v in ms.items() if k in keys}
                messages.append(ms)
            except Exception as e:
                #print(e)
                pass
    df = pd.DataFrame(messages)
    del messages
    try:
        df['funds'] = pd.to_numeric(df['funds'], errors="coerce")
    except KeyError:
        pass
    try:
        df['price'] = pd.to_numeric(df['price'], errors="coerce").round(price_dec)
    except KeyError:
        pass
    try:
        df['size'] = pd.to_numeric(df['size'], errors="coerce")
    except KeyError:
        pass

    df.replace('sell', 1, inplace=True)
    df.replace('buy', 0, inplace=True)
    df.side = df.side.fillna(-1)
    df.side = pd.to_numeric(df['side'], errors="coerce")
    df['trader_id'] = -1
    # df.time = pd.to_datetime(df.time)
    df.loc[df['size'].isnull(), 'size'] = -1

    start_seq = df['sequence'].values[0]
    end_seq = df['sequence'].values[-1]
    save_str = str(k) + '_' + str(int(start_seq)) + '_' + str(int(end_seq)) + '.feather'
    feather.write_dataframe(df, data_dir + '/feather/' + save_str)
    del df
    gc.collect()