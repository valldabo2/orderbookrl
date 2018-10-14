import os
import json
import pandas as pd


def get_data(logs_dir, cols):
    log_dirs = os.listdir(logs_dir)
    data_dict = {}
    for log_dir in log_dirs:
        try:
            with open(logs_dir + log_dir + '/params.json') as f:
                params = json.load(f)
            data = pd.read_csv(logs_dir + log_dir + '/progress.csv', index_col='time_total_s')
            data = data[cols]

            data_dict[log_dir] = {'data': data, 'params': params}
        except:
            pass
    return data_dict


def get_dataframe(data_dict, hpcols):
    temp = {}
    for k, v in data_dict.items():
        temp[k] = v['data']['cap_mean']
    temp = pd.DataFrame(temp)

    for item in data_dict.values():
        params = item['params']
        temp_ = pd.io.json.json_normalize(params).to_dict(orient='records')[0]
        temp_ = {k: v for k, v in temp_.items() if k in hpcols}

    tuples = []
    for item in data_dict.values():
        params = item['params']
        temp_ = pd.io.json.json_normalize(params).to_dict(orient='records')[0]
        temp_ = {k:v for k,v in temp_.items() if k in hpcols}

        tuples.append(tuple(str(value) for value in temp_.values()))
    names = tuple(temp_.keys())
    index = pd.MultiIndex.from_tuples(tuples, names=names)
    temp.columns = index

    return temp
