import os
from orderbookmdp.rl.dist_envs import DistEnv, SpreadEnv
from orderbookmdp.rl.market_order_envs import MarketOrderEnv, MarketOrderEnvCumReturn, MarketOrderEnvAdjustment
from orderbookmdp.rl.market_order_envs import MarketOrderEnvBuyingPower, MarketOrderEnvEndReward
from orderbookmdp.rl.market_order_envs import MarketOrderEnvFunds, MarketOrderEnvCritic, MarketOrderEnvOpt
from orderbookmdp.rl.market_order_envs import MarketOrderEnvBuySell

from orderbookrl.preprocessing.phi import MarketVariables, PredictiveMarketVariables, MarketVariablesSingleL, Zeros
from ray.rllib.models import ModelCatalog
from ray.tune import register_env, register_trainable
#from orderbookrl.tests.ppo_adv import PPOAdv
#from orderbookrl.tests.ppo_cumret import PPOCumRet


def get_default_path():
    path = os.path.dirname(os.path.realpath(__file__))
    path = path[:-11]
    return path + '/data/'


def env_creator_distenv(path, env_config):
    env = DistEnv(order_paths=path + 'data/feather/',
                  snapshot_paths=path + 'data/snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "DistEnv-v0"
register_env(env_creator_name, env_creator_distenv)


def env_creator_spreadenv(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = SpreadEnv(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "SpreadEnv-v0"
register_env(env_creator_name, env_creator_spreadenv)


def env_creator_marketorderenv(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnv(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnv-v0"
register_env(env_creator_name, env_creator_marketorderenv)


def env_creator_marketorderenvcumreturn(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnvCumReturn(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvCumReturn-v0"
register_env(env_creator_name, env_creator_marketorderenvcumreturn)

def env_creator_marketorderenvadjusted(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnvAdjustment(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvAdjustment-v0"
register_env(env_creator_name, env_creator_marketorderenvadjusted)


def env_creator_marketorderenvbp(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnvBuyingPower(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvBP-v0"
register_env(env_creator_name, env_creator_marketorderenvbp)


def env_creator_marketorderenvendreward(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnvEndReward(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvEndReward-v0"
register_env(env_creator_name, env_creator_marketorderenvendreward)

def env_creator_marketorderenvfunds(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnvFunds(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvFunds-v0"
register_env(env_creator_name, env_creator_marketorderenvfunds)


def env_creator_marketorderenvcritic(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnvCritic(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvCritic-v0"
register_env(env_creator_name, env_creator_marketorderenvcritic)

def env_creator_marketorderenvopt(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnvOpt(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvOpt-v0"
register_env(env_creator_name, env_creator_marketorderenvopt)


def env_creator_marketorderenvbuysell(env_config):
    if not 'data_path' in env_config:
        path = get_default_path()
    else:
        path = env_config['data_path']
    env = MarketOrderEnvBuySell(order_paths=path + 'feather/',
                  snapshot_paths=path + 'snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvBuySell-v0"
register_env(env_creator_name, env_creator_marketorderenvbuysell)

ModelCatalog.register_custom_preprocessor('mv', MarketVariables)

ModelCatalog.register_custom_preprocessor('mv_l', MarketVariablesSingleL)

ModelCatalog.register_custom_preprocessor('mv_pred', PredictiveMarketVariables)

ModelCatalog.register_custom_preprocessor('zeros', Zeros)

#register_trainable('PPOADV', PPOAdv)
#
#register_trainable('PPOCUMRET', PPOCumRet)


def get_env(env_id, env_config):
    if env_id == "DistEnv-v0":
        return env_creator_distenv(env_config)
    elif env_id == 'SpreadEnv-v0':
        return env_creator_spreadenv(env_config)
    elif env_id == 'MarketOrderEnv-v0':
        return env_creator_marketorderenv(env_config)
    elif env_id == 'MarketOrderEnvCumReturn-v0':
        return env_creator_marketorderenvcumreturn(env_config)
    elif env_id == 'MarketOrderEnvAdjustment-v0':
        return env_creator_marketorderenvadjusted(env_config)
    elif env_id == 'MarketOrderEnvBP-v0':
        return env_creator_marketorderenvbp(env_config)
    elif env_id == 'MarketOrderEnvEndReward-v0':
        return env_creator_marketorderenvendreward(env_config)
    elif env_id == 'MarketOrderEnvFunds-v0':
        return env_creator_marketorderenvfunds(env_config)
    elif env_id == 'MarketOrderEnvCritic-v0':
        return env_creator_marketorderenvcritic(env_config)
    elif env_id == 'MarketOrderEnvOpt-v0':
        return env_creator_marketorderenvopt(env_config)
    elif env_id == 'MarketOrderEnvBuySell-v0':
        return env_creator_marketorderenvbuysell(env_config)
    else:
        raise NotImplementedError(env_id)
