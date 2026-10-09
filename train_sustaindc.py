"""
Train the environment using the selected algorithm.

The original code is from https://github.com/PKU-MARL/HARL
Several modifications are made to adapt to the SustainDC environment.
"""

import os
import sys
import warnings
import argparse
import json

warnings.filterwarnings('ignore')
# sys.path.insert(0, os.getcwd())

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils.tracking import normalize_wandb_config


def apply_wandb_overrides(algo_args, parsed_args):
    """Merge explicit W&B flags into resolved logger configuration."""
    logger_config = algo_args.setdefault("logger", {})
    if not isinstance(logger_config, dict):
        raise ValueError("algo_args.logger must be a mapping")
    raw_config = dict(logger_config.get("wandb", {}))
    flag_values = dict(parsed_args)

    for flag, key in (
        ("wandb_project", "project"),
        ("wandb_entity", "entity"),
        ("wandb_group", "group"),
        ("wandb_tags", "tags"),
    ):
        if flag_values.get(flag) is not None:
            raw_config[key] = flag_values[flag]
    if flag_values.get("wandb_artifacts"):
        raw_config["log_artifacts"] = True

    mode = flag_values.get("wandb_mode")
    if mode is not None:
        raw_config["mode"] = mode
        raw_config["enabled"] = mode != "disabled"
    elif flag_values.get("wandb"):
        raw_config["enabled"] = True
        if raw_config.get("mode", "disabled") == "disabled":
            raw_config["mode"] = "online"

    logger_config["wandb"] = normalize_wandb_config({"wandb": raw_config})


def run_and_close(runner):
    """Run a runner and always close its local and optional tracking output."""
    try:
        runner.run()
    finally:
        runner.close()

def main():
    """Main function to train the environment using the selected algorithm."""
    # Create an argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add command-line arguments
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="sustaindc",
        choices=["sustaindc"],
        help="Environment name. Choose from: sustaindc."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="installtest",
        help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file."
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking.")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"),
        help="W&B tracking mode.",
    )
    parser.add_argument("--wandb-project", help="W&B project name.")
    parser.add_argument("--wandb-entity", help="Optional W&B entity.")
    parser.add_argument("--wandb-group", help="Optional W&B run group.")
    parser.add_argument("--wandb-tags", nargs="+", help="W&B run tags.")
    parser.add_argument(
        "--wandb-artifacts", action="store_true", help="Upload opt-in W&B artifacts."
    )

    # Parse known arguments and process unknown arguments
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        """Evaluate the argument if possible, otherwise return the argument as is."""
        try:
            return eval(arg)
        except:
            return arg

    # Process unparsed arguments to a dictionary
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    # Convert args to dictionary
    args = vars(args)

    # Load configuration
    if args["load_config"] != "":
        # Load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:
        # Load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    # Update args from command line
    update_args(unparsed_dict, algo_args, env_args)
    apply_wandb_overrides(algo_args, args)

    # Start training
    from harl.runners import RUNNER_REGISTRY

    # Initialize and run the selected algorithm
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    run_and_close(runner)


if __name__ == "__main__":
    main()
