#!/usr/ort logging

import ray
from ray.tune.config_parser import make_parser
from ray.tune.tune import _make_scheduler, run_experiments
import argparse
import yaml
from orderbookrl.utils import get_env
from ray import tune

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


def on_episode_end(info):
    episode = info["episode"]
    env = info['env'].get_unwrapped()[0]
    if hasattr(env, 'capital'):
        capital_return = (env.capital - env.initial_funds)/env.initial_funds
        episode.custom_metrics['capital_return'] = capital_return


def run(args, parser):

    with open(args.config_file) as f:
        experiments = yaml.load(f)

    if hasattr(args, 'restore'):
        key = list(experiments.keys())[0]
        experiments[key]['restore'] = args.restore


    key = list(experiments.keys())[0]
    experiments[key]["config"]["callbacks"] = {
                    "on_episode_end": tune.function(on_episode_end)
                }

    for exp in experiments.values():
        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")

    ray.init(
        redis_address=args.redis_address,
        num_cpus=args.ray_num_cpus,
        num_gpus=args.ray_num_gpus,
        #object_store_memory=20000 * (2**20), # 100000 #int(25*10**9) #30gb
        #redis_max_memory=5000 * (2**20)
    )
    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
