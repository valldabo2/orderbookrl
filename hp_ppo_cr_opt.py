from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#!/usr/bin/env python


import yaml
from ray.tune.config_parser import make_parser
from orderbookrl.utils import get_env

import argparse
import random

import ray
from ray.tune import run_experiments
from ray.tune.schedulers import PopulationBasedTraining
EXAMPLE_USAGE = """
Training example via executable:
    python train.py -f rl_setups/marketorderenv/ppo.yaml

Note that -f overrides all other trial-specific command-line options.
"""


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--redis-address",
        default=None,
        type=str,
        help="The Redis address of the cluster.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to pass to Ray."
        " This only has an affect in local mode.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to pass to Ray."
        " This only has an affect in local mode.")
    parser.add_argument(
        "--queue-trials",
        default=False,
        type=bool,
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    return parser


def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.load(f)

        if hasattr(args, 'restore'):
            key = list(experiments.keys())[0]
            experiments[key]['restore'] = args.restore

    ray.init(
        redis_address=args.redis_address,
        num_cpus=args.ray_num_cpus,
        num_gpus=args.ray_num_gpus)

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="cap_mean",
        perturbation_interval=5,
        hyperparam_mutations={
            'gamma': lambda: random.uniform(0.9999, 0.90),
            'num_sgd_iter': [10, 20, 30],
            'lr': lambda: random.uniform(0.00001, 1),
            'sgd_minibatch_size': [2048, 40960, 10240, 20480],
            "entropy_coeff": lambda: random.uniform(0, 0.1),
            "clip_param": lambda: random.uniform(0.0, 0.3),
            # Allow perturbations within this set of categorical values.
        })

    run_experiments(
        experiments,
        scheduler=pbt,
        queue_trials=args.queue_trials)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
