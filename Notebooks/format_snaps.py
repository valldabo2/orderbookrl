import gc
import os

import feather
import numpy as np
import pandas as pd
import ujson
from sortedcontainers.sortedlist import SortedList
from tqdm import tqdm

data_dir = '../data'
print('Reformats data from:', data_dir)
files = os.listdir(data_dir + '/json/')  # noqa
snap_files = SortedList([filename for filename in files if 'snaps' in filename],
                        key=lambda fn: pd.to_datetime(fn[:-11], format='%d_%m_%Y_%H_%M_%S'))

try:
    os.makedirs(data_dir + '/snap_json/')
except FileExistsError:
    pass
for snapfile in tqdm(snap_files):
    with open(data_dir + '/json/' + snapfile, 'r') as f:
        snaps = f.readlines()
        for snap in snaps:

            try:
                snap = ujson.loads(snap)
                try:
                    seq = snap['sequence']
                    with open(data_dir + '/snap_json/snap_' + str(seq) + '.json', 'w') as snapf:
                        ujson.dump(snap, snapf)
                except:  # noqa
                    pass
            except Exception as e:
                print(e)

