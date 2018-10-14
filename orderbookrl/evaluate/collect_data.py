import ray
from ray.rllib.agents.agent import get_agent_class
from ray.rllib.models import ModelCatalog
from orderbookrl.utils import get_env
import json
import pandas as pd


def load_agent(agent_id, env_id, checkpoint, config):
    ray.init()
    cls = get_agent_class(agent_id)
    if config.get('num_workers'):
        config['num_workers'] = 1
    agent = cls(env=env_id, config=config, logger_creator=None)
    agent.restore(checkpoint)
    return agent


def load_env(env_id, env_config, model_config):
    env_config['max_sequence_skip'] = int(10e10)  # TODO Fix
    env_config['random_start'] = False
    env_config['max_episode_time'] = '100 days'
    env_config['taker_fee'] = 0
    env = get_env(env_id, env_config)
    env = ModelCatalog.get_preprocessor_as_wrapper(env, options=model_config)
    return env


def run_through_all_data(env, agent):
    k = 0
    capitals = []
    possessions = []
    funds = []
    times = []

    rewards = []
    states = []
    actions = []
    quotes = []

    state = env.reset()
    done = False

    while not done:
        action = agent.compute_action(state)
        next_state, reward, done, info = env.step(action)

        k += 1
        time = env.env.market.time
        if k % 10000 == 0:
            print(time, env.env.capital/env.env.initial_funds)

        times.append(time)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        capitals.append(env.env.capital)
        possessions.append(env.env.possession)
        funds.append(env.env.funds)
        quotes.append(env.env.quotes)

        state = next_state

    trades = env.env.trades_list
    times = pd.Index(pd.to_datetime(times), name='time')
    result = pd.DataFrame({'capital': capitals, 'possession': possessions, 'funds': funds}, index=times)

    trades = pd.DataFrame(trades, columns=['time', 'size', 'price', 'buy_sell'])
    trades = trades.set_index('time')

    states = pd.DataFrame(states, index=times)
    actions = pd.DataFrame(actions, index=times)
    rewards = pd.DataFrame(rewards, index=times)

    quotes = pd.DataFrame(quotes, columns=['ask', 'ask_vol', 'bid', 'bid_vol'], index=times)

    return result, trades, states, actions, rewards, quotes


def load_env_agent(agent_id, path, checkpoint, data_path=None):
    checkpoint_path = path + 'checkpoint-' + str(checkpoint)
    params_path = path + 'params.json'
    result_path = path + 'result.json'

    with open(params_path) as f:
        config = json.load(f)
        model_config = config['model']

    with open(result_path) as f:
        result_ = json.loads(f.readline())
        env_id = result_['config']['env']
        env_config = result_['config']['env_config']

    if data_path != None:
        env_config['data_path'] = data_path

    env = load_env(env_id, env_config, model_config)
    config['num_workers'] = 1
    agent = load_agent(agent_id, env_id, checkpoint_path, config)

    return env, agent


if __name__ == '__main__':
    ray.init()
    agent_id = 'PPO'
    path = '../../logs/marketorderenv/ppo/PPO_MarketOrderEnv-v0_0_2018-08-21_22-57-42w6gr_35y/'
    checkpoint = 190

    env, agent = load_env_agent(agent_id, path, checkpoint)
    result, trades, states, actions, rewards, quotes = run_through_all_data(env, agent)

    print(result.head())


