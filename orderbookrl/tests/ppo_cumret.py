from ray.rllib.agents.ppo import PPOAgent
#from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
import numpy as np
np.set_printoptions(precision=4)
from ray.rllib.evaluation.sample_batch import SampleBatch
import time


def compute_advantages(rollout, gamma=1, modify=False):
    """Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estamation

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}

    trajsize = len(rollout["actions"])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    rewards = traj['rewards']

    gammas = np.power(gamma, np.arange(trajsize))
    cum_ret_t = np.zeros(trajsize)
    for t in range(trajsize):
        if t == 0:
            cum_ret_t[t] = np.cumprod(1 + rewards*gammas)[-1]
        else:
            cum_ret_t[t] = np.cumprod(1 + rewards[t:] * gammas[:-t])[-1]


    cum_ret_t -= 1
    if modify:
        cum_ret_t[(-0.01 < cum_ret_t) & (cum_ret_t <= 0)] = -0.01
        cum_ret_t *= 1000

    if 'vf_preds' in traj:
        traj["advantages"] = cum_ret_t - traj['vf_preds']
        traj["value_targets"] = (traj["advantages"] + traj["vf_preds"]).copy().astype(np.float32)
    else:
        traj["advantages"] = cum_ret_t
        traj["value_targets"] = traj["value_targets"] = np.zeros_like(traj["advantages"])

    traj["advantages"] = traj["advantages"].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


class CumRetPPOGraph(PPOPolicyGraph):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None):
        batch = compute_advantages(sample_batch, self.config["gamma"], self.config["modify_cumret"])
        return batch


class PPOCumRet(PPOAgent):
    _policy_graph = CumRetPPOGraph
    _allow_unknown_configs = True

