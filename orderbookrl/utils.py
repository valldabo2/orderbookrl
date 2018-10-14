import os
from orderbookmdp.rl.dist_envs import DistEnv, SpreadEnv
from orderbookmdp.rl.market_order_envs import MarketOrderEnv, MarketOrderEnvCumReturn, MarketOrderEnvAdjustment
from orderbookmdp.rl.market_order_envs import MarketOrderEnvBuyingPower, MarketOrderEnvEndReward
from orderbookmdp.rl.market_order_envs import MarketOrderEnvFunds, MarketOrderEnvCritic, MarketOrderEnvOpt
from orderbookmdp.rl.market_order_envs import MarketOrderEnvBuySell

from orderbookrl.preprocessing.phi import MarketVariables, PredictiveMarketVariables
from ray.rllib.models import ModelCatalog
from ray.tune import register_env, register_trainable
from orderbookrl.tests.ppo_adv import PPOAdv
from orderbookrl.tests.ppo_cumret import PPOCumRet


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

ModelCatalog.register_custom_preprocessor('mv_pred', PredictiveMarketVariables)

register_trainable('PPOADV', PPOAdv)

register_trainable('PPOCUMRET', PPOCumRet)


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


# Extra ray stuff
from ray.rllib.evaluation.metrics import *

def summarize_episodes(episodes, new_episodes):
    """Summarizes a set of episode metrics tuples.

    Arguments:
        episodes: smoothed set of episodes including historical ones
        new_episodes: just the new episodes in this iteration
    """

    episode_rewards = []
    episode_lengths = []
    policy_rewards = collections.defaultdict(list)
    episode_caps = []
    for episode in episodes:
        episode_lengths.append(episode.episode_length)
        episode_rewards.append(episode.episode_reward)
        episode_caps.append(episode.episode_cap_init_cap)

        for (_, policy_id), reward in episode.agent_rewards.items():
            if policy_id != DEFAULT_POLICY_ID:
                policy_rewards[policy_id].append(reward)
    if episode_rewards:
        min_reward = min(episode_rewards)
        max_reward = max(episode_rewards)
        max_cap = max(episode_caps)
        min_cap = min(episode_caps)
    else:
        min_reward = float('nan')
        max_reward = float('nan')
        max_cap = float('nan')
        min_cap = float('nan')

    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_cap = np.mean(episode_caps)

    for policy_id, rewards in policy_rewards.copy().items():
        policy_rewards[policy_id] = np.mean(rewards)

    return dict(
        cap_max=max_cap,
        cap_min=min_cap,
        cap_mean=avg_cap,
        episode_reward_max=max_reward,
        episode_reward_min=min_reward,
        episode_reward_mean=avg_reward,
        episode_len_mean=avg_length,
        episodes=len(new_episodes),
        policy_reward_mean=dict(policy_rewards))

from ray.rllib.evaluation.sampler import *

RolloutMetrics = namedtuple(
    "RolloutMetrics", ["episode_length", "episode_reward", "agent_rewards", "cap"])


def _env_runner(async_vector_env,
                extra_batch_callback,
                policies,
                policy_mapping_fn,
                num_local_steps,
                horizon,
                obs_filters,
                clip_rewards,
                pack,
                tf_sess=None):
    """This implements the common experience collection logic.

    Args:
        async_vector_env (AsyncVectorEnv): env implementing AsyncVectorEnv.
        extra_batch_callback (fn): function to send extra batch data to.
        policies (dict): Map of policy ids to PolicyGraph instances.
        policy_mapping_fn (func): Function that maps agent ids to policy ids.
            This is called when an agent first enters the environment. The
            agent is then "bound" to the returned policy for the episode.
        num_local_steps (int): Number of episode steps before `SampleBatch` is
            yielded. Set to infinity to yield complete episodes.
        horizon (int): Horizon of the episode.
        obs_filters (dict): Map of policy id to filter used to process
            observations for the policy.
        clip_rewards (bool): Whether to clip rewards before postprocessing.
        pack (bool): Whether to pack multiple episodes into each batch. This
            guarantees batches will be exactly `num_local_steps` in size.
        tf_sess (Session|None): Optional tensorflow session to use for batching
            TF policy evaluations.

    Yields:
        rollout (SampleBatch): Object containing state, action, reward,
            terminal condition, and other fields as dictated by `policy`.
    """

    try:
        if not horizon:
            horizon = (
                async_vector_env.get_unwrapped()[0].spec.max_episode_steps)
    except Exception:
        print("Warning, no horizon specified, assuming infinite")
    if not horizon:
        horizon = float("inf")

    # Pool of batch builders, which can be shared across episodes to pack
    # trajectory data.
    batch_builder_pool = []

    def get_batch_builder():
        if batch_builder_pool:
            return batch_builder_pool.pop()
        else:
            return MultiAgentSampleBatchBuilder(policies, clip_rewards)

    def new_episode():
        return MultiAgentEpisode(policies, policy_mapping_fn,
                                 get_batch_builder, extra_batch_callback)

    active_episodes = defaultdict(new_episode)

    while True:
        # Get observations from all ready agents
        unfiltered_obs, rewards, dones, infos, off_policy_actions = \
            async_vector_env.poll()

        # Map of policy_id to list of PolicyEvalData
        to_eval = defaultdict(list)

        # Map of env_id -> agent_id -> action replies
        actions_to_send = defaultdict(dict)

        # For each environment
        for env_id, agent_obs in unfiltered_obs.items():
            new_episode = env_id not in active_episodes
            episode = active_episodes[env_id]
            if not new_episode:
                episode.length += 1
                episode.batch_builder.count += 1
                episode._add_agent_rewards(rewards[env_id])

            # Check episode termination conditions
            if dones[env_id]["__all__"] or episode.length >= horizon:
                all_done = True
                atari_metrics = sampler._fetch_atari_metrics(async_vector_env)
                if atari_metrics is not None:
                    for m in atari_metrics:
                        yield m
                else:
                    try:
                        cap = infos[0]['single_agent']['cap_init_cap']
                    except:
                        cap = 0
                    yield RolloutMetrics(episode.length, episode.total_reward,
                                         dict(episode.agent_rewards), 0)
            else:
                all_done = False
                # At least send an empty dict if not done
                actions_to_send[env_id] = {}

            # For each agent in the environment
            for agent_id, raw_obs in agent_obs.items():
                policy_id = episode.policy_for(agent_id)
                filtered_obs = sampler._get_or_raise(obs_filters, policy_id)(raw_obs)
                agent_done = bool(all_done or dones[env_id].get(agent_id))
                if not agent_done:
                    to_eval[policy_id].append(
                        PolicyEvalData(env_id, agent_id, filtered_obs,
                                       episode.rnn_state_for(agent_id)))

                last_observation = episode.last_observation_for(agent_id)
                episode._set_last_observation(agent_id, filtered_obs)

                # Record transition info if applicable
                if last_observation is not None and \
                        infos[env_id][agent_id].get("training_enabled", True):
                    episode.batch_builder.add_values(
                        agent_id,
                        policy_id,
                        t=episode.length - 1,
                        eps_id=episode.episode_id,
                        obs=last_observation,
                        actions=episode.last_action_for(agent_id),
                        rewards=rewards[env_id][agent_id],
                        dones=agent_done,
                        infos=infos[env_id][agent_id],
                        new_obs=filtered_obs,
                        **episode.last_pi_info_for(agent_id))

            # Cut the batch if we're not packing multiple episodes into one,
            # or if we've exceeded the requested batch size.
            if episode.batch_builder.has_pending_data():
                if (all_done and not pack) or \
                        episode.batch_builder.count >= num_local_steps:
                    yield episode.batch_builder.build_and_reset()
                elif all_done:
                    # Make sure postprocessor stays within one episode
                    episode.batch_builder.postprocess_batch_so_far()

            if all_done:
                # Handle episode termination
                batch_builder_pool.append(episode.batch_builder)
                del active_episodes[env_id]
                resetted_obs = async_vector_env.try_reset(env_id)
                if resetted_obs is None:
                    # Reset not supported, drop this env from the ready list
                    assert horizon == float("inf"), \
                        "Setting episode horizon requires reset() support."
                else:
                    # Creates a new episode
                    episode = active_episodes[env_id]
                    for agent_id, raw_obs in resetted_obs.items():
                        policy_id = episode.policy_for(agent_id)
                        filtered_obs = sampler._get_or_raise(obs_filters,
                                                             policy_id)(raw_obs)
                        episode._set_last_observation(agent_id, filtered_obs)
                        to_eval[policy_id].append(
                            PolicyEvalData(env_id, agent_id, filtered_obs,
                                           episode.rnn_state_for(agent_id)))

        # Batch eval policy actions if possible
        if tf_sess:
            builder = TFRunBuilder(tf_sess, "policy_eval")
            pending_fetches = {}
        else:
            builder = None
        eval_results = {}
        rnn_in_cols = {}
        for policy_id, eval_data in to_eval.items():
            rnn_in = sampler._to_column_format([t.rnn_state for t in eval_data])
            rnn_in_cols[policy_id] = rnn_in
            policy = sampler._get_or_raise(policies, policy_id)
            if builder and (policy.compute_actions.__code__ is
                            TFPolicyGraph.compute_actions.__code__):
                pending_fetches[policy_id] = policy.build_compute_actions(
                    builder, [t.obs for t in eval_data],
                    rnn_in,
                    is_training=True)
            else:
                eval_results[policy_id] = policy.compute_actions(
                    [t.obs for t in eval_data],
                    rnn_in,
                    is_training=True,
                    episodes=[active_episodes[t.env_id] for t in eval_data])
        if builder:
            for k, v in pending_fetches.items():
                eval_results[k] = builder.get(v)

        # Record the policy eval results
        for policy_id, eval_data in to_eval.items():
            actions, rnn_out_cols, pi_info_cols = eval_results[policy_id]
            # Add RNN state info
            for f_i, column in enumerate(rnn_in_cols[policy_id]):
                pi_info_cols["state_in_{}".format(f_i)] = column
            for f_i, column in enumerate(rnn_out_cols):
                pi_info_cols["state_out_{}".format(f_i)] = column
            # Save output rows
            for i, action in enumerate(actions):
                env_id = eval_data[i].env_id
                agent_id = eval_data[i].agent_id
                actions_to_send[env_id][agent_id] = action
                episode = active_episodes[env_id]
                episode._set_rnn_state(agent_id, [c[i] for c in rnn_out_cols])
                episode._set_last_pi_info(
                    agent_id, {k: v[i]
                               for k, v in pi_info_cols.items()})
                if env_id in off_policy_actions and \
                        agent_id in off_policy_actions[env_id]:
                    episode._set_last_action(
                        agent_id, off_policy_actions[env_id][agent_id])
                else:
                    episode._set_last_action(agent_id, action)

        # Return computed actions to ready envs. We also send to envs that have
        # taken off-policy actions; those envs are free to ignore the action.
        async_vector_env.send_actions(dict(actions_to_send))

