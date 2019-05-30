from ray.tune.schedulers import PopulationBasedTraining
from orderbookrl.utils import get_env
from ray import tune
import ray


def on_episode_end(info):
    episode = info["episode"]
    env = info['env'].get_unwrapped()[0]
    if hasattr(env, 'capital'):
        capital_return = (env.capital - env.initial_funds) / env.initial_funds
        episode.custom_metrics['capital_return'] = capital_return


scheduler = PopulationBasedTraining(
    time_attr='training_iteration',
    reward_attr='episode_reward_mean',
    perturbation_interval=2,
    hyperparam_mutations=dict(
        lr=[5e-3, 5e-4],
        entropy_coeff=[0, 0.01, 0.001],
        train_batch_size=[2000, 5000, 15000],
        n_sgd_iter=[20, 30, 40],
        sgd_minibatch_size=[1024, 4096, 10240]
    )
)

config = {
    'env': 'MarketOrderEnv-v0',
    'env_config': {
        'max_sequence_skip': 1500,
        'random_start': True,
        'max_episode_time': '2hours'
    },
    'use_gae': True,
    'lambda': 0.8,
    'kl_coeff': 0.2,
    'model': {
        'fcnet_hiddens': [32, 32],
        'free_log_std': False,
        'squash_to_range': True,
        'custom_preprocessor': 'mv',
        'custom_options': {
            'fast_macd_l': 120,
            'slow_macd_l': 240,
            'ofi_l': 100,
            'mid_l': 100
        }
    },
    'observation_filter': 'MeanStdFilter',
    'clip_rewards': True,
    'train_batch_size': 1000,
    'sgd_minibatch_size': 124,
    'sample_batch_size': 500,
    'batch_mode': 'truncate_episodes',
    'num_sgd_iter': 30,
    'lr': 0.005,
    'vf_loss_coeff': 1.0,
    'entropy_coeff': 0.01,
    'clip_param': 0.3,
    'kl_target': 0.01,
    'simple_optimizer': True,
    'num_workers': 1,
    'num_gpus': 0,
    'num_gpus_per_worker': 0,
    'num_cpus_per_worker': 1,
    'grad_clip': 40,
    'vf_share_layers': True,
    'callbacks': {
        'on_episode_end': tune.function(on_episode_end)
    }
}

ray.init()
tune.run('PPO', name='PBT_PPO',
         stop={'training_iteration': 7},
         num_samples=1,
         config=config,
         local_dir='~/Documents/Hobby/PythonProjects/orderbookrl/logs/marketorderenv/pbt_ppo',
         checkpoint_at_end=True,
         checkpoint_freq=1,
         scheduler=scheduler
         )