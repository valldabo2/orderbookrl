import os
from orderbookmdp.rl.dist_envs import DistEnv, SpreadEnv
from orderbookmdp.rl.market_order_envs import MarketOrderEnv, MarketOrderEnvCumReturn, MarketOrderEnvAdjustment
from orderbookrl.preprocessing.phi import MarketVariables
from ray.rllib.models import ModelCatalog
from ray.tune import register_env, register_trainable
from orderbookrl.tests.test_adv import PPOAdv

path = os.path.dirname(os.path.realpath(__file__))
path = path[:-11]


def env_creator_distenv(env_config):
    env = DistEnv(order_paths=path + 'data/feather/',
                  snapshot_paths=path + 'data/snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "DistEnv-v0"
register_env(env_creator_name, env_creator_distenv)


def env_creator_spreadenv(env_config):
    env = SpreadEnv(order_paths=path + 'data/feather/',
                  snapshot_paths=path + 'data/snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "SpreadEnv-v0"
register_env(env_creator_name, env_creator_spreadenv)


def env_creator_marketorderenv(env_config):
    env = MarketOrderEnv(order_paths=path + 'data/feather/',
                  snapshot_paths=path + 'data/snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnv-v0"
register_env(env_creator_name, env_creator_marketorderenv)


def env_creator_marketorderenvcumreturn(env_config):
    env = MarketOrderEnvCumReturn(order_paths=path + 'data/feather/',
                  snapshot_paths=path + 'data/snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvCumReturn-v0"
register_env(env_creator_name, env_creator_marketorderenvcumreturn)

def env_creator_marketorderenvadjusted(env_config):
    env = MarketOrderEnvAdjustment(order_paths=path + 'data/feather/',
                  snapshot_paths=path + 'data/snap_json/', **env_config)
    return env  # or return your own custom env


env_creator_name = "MarketOrderEnvAdjustment-v0"
register_env(env_creator_name, env_creator_marketorderenvadjusted)

ModelCatalog.register_custom_preprocessor('mv', MarketVariables)

register_trainable('PPOADV', PPOAdv)


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
    else:
        raise NotImplementedError(env_id)
