"""Tools for loading and updating configs."""
import time
import os
import json
import yaml
from uu import Error


def get_defaults_yaml_args(algo, env):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args


def _leaf_paths(node, prefix=()):
    """Yield ``(path, parent)`` for every non-dict entry under ``node``."""
    for name, value in node.items():
        if type(value) is dict:
            yield from _leaf_paths(value, prefix + (name,))
        else:
            yield prefix + (name,), node


def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.

    Each key names exactly one setting: a bare name when only one setting has
    it, otherwise a dotted path whose trailing components are unique (for
    example ``seed.seed`` or ``fixed_month_validation.seed``). A key that
    matches no setting, or several, raises ``ValueError`` before anything is
    written, so a typo or a shared name never changes the wrong setting.

    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """
    leaves = [(path, parent) for args_dict in args for path, parent in _leaf_paths(args_dict)]
    targets = []
    for key, value in unparsed_dict.items():
        parts = tuple(key.split("."))
        matches = [(path, parent) for path, parent in leaves if path[-len(parts):] == parts]
        if not matches:
            raise ValueError(f"--{key} matches no setting in the loaded configuration")
        if len(matches) > 1:
            names = ", ".join("--" + ".".join(path) for path, _ in matches)
            raise ValueError(f"--{key} matches several settings ({names}); pass one of these paths instead")
        path, parent = matches[0]
        targets.append((parent, path[-1], value))
    for parent, name, value in targets:
        parent[name] = value


def get_task_name(env, env_args):
    """Get task name."""
    if env == 'sustaindc':
        task = env_args["location"]
    else:
        raise ValueError('Environment not defined')
    return task


def init_dir(
    env,
    env_args,
    algo,
    exp_name,
    seed,
    logger_path,
    *,
    main_args=None,
    algo_args=None,
    job_type="train",
):
    """Init directory for saving results and optional tracking state."""
    task = get_task_name(env, env_args)
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_path = os.path.join(
        logger_path,
        env,
        task,
        algo,
        exp_name,
        "-".join(["seed-{:0>5}".format(seed), hms_time]),
    )
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    from tensorboardX import SummaryWriter

    writter = SummaryWriter(log_path)
    if main_args is not None and algo_args is not None:
        from harl.utils.tracking import create_tracking_writer

        writter = create_tracking_writer(
            tensorboard_writer=writter,
            run_dir=results_path,
            main_args=main_args,
            algo_args=algo_args,
            env_args=env_args,
            job_type=job_type,
        )
    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, models_path, writter


def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)
