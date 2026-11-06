"""Sequential battery observation helpers for SustainDC runners."""
from sustaindc_env import EnvConfig


def sequential_battery_index(env_args):
    """Return agent_bat's index when it decides after workload/cooling, else None.

    With battery.observation_timing 'current', runners call begin_step with the
    earlier agents' actions and give agent_bat the returned decision observation.
    """
    config = EnvConfig(env_args)
    if config['battery']['observation_timing'] != 'current':
        return None
    return list(config['agents']).index('agent_bat')
