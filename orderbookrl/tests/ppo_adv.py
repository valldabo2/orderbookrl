from ray.rllib.agents.ppo import PPOAgent
#from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
import numpy as np
from ray.rllib.evaluation.postprocessing import discount
from ray.rllib.evaluation.sample_batch import SampleBatch


def compute_advantages(rollout, last_r, gamma=0.9, lambda_=1.0, use_gae=True):
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

    if use_gae:
        assert "vf_preds" in rollout, "Values not found!"
        vpred_t = np.concatenate([rollout["vf_preds"], np.array([last_r])])
        # delta_t = traj["rewards"] + gamma * vpred_t[1:] - vpred_t[:-1]
        delta_t = (1 + traj['rewards'])*(1 + gamma * vpred_t[1:]) - 1 - vpred_t[:-1]
        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj["advantages"] = discount(delta_t, gamma * lambda_)
        traj["value_targets"] = (traj["advantages"] + traj["vf_preds"]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout["rewards"], np.array([last_r])])
        traj["advantages"] = discount(rewards_plus_v, gamma)[:-1]
        # TODO(ekl): support using a critic without GAE
        traj["value_targets"] = np.zeros_like(traj["advantages"])

    traj["advantages"] = traj["advantages"].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


class AdvPPOGraph(PPOPolicyGraph):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):

        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(len(self.model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            last_r = self._value(sample_batch["new_obs"][-1], *next_state)

        batch = compute_advantages(
            sample_batch,
            last_r,
            self.config["gamma"],
            self.config["lambda"],
            use_gae=self.config["use_gae"])
        return batch


class PPOAdv(PPOAgent):
    _policy_graph = AdvPPOGraph
