"""Base-agent fallbacks for absent agents must reproduce the noop baseline.

An environment configured with ``agents: [agent_bat]`` relies on
``sustaindc_env`` falling back to the base agents for the workload and
cooling actions each step. The protocol test for that fallback: a
battery-only environment steps a full calendar-month episode, and with the
battery held idle its bills and service/thermal gates equal the three-agent
noop baseline exactly.
"""
import contextlib
import copy
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import run_module_tests
from harl.envs.sustaindc.metrics import FacilityMetricAccumulator
from sustaindc_env import EnvConfig, SustainDC
from utils.base_agents import (
    BaseBatteryAgent,
    BaseHVACAgent,
    BaseLoadShiftingAgent,
)

# The 10 MW Montreal Rate L scenario used for training and evaluation,
# minus the ``agents`` key (parameterized per environment below).
_SCENARIO = {
    'battery': {
        'degradation_cost_c_per_kwh': 5.78,
        'discharge_mode': 'peak_limit',
        'discharge_peak_floor_kw': 6310.0,
        'duration_hours': 2.0,
        'initial_soc': 0.5,
        'observation_timing': 'current',
        'peak_charge_headroom': True,
        'peak_charge_headroom_basis': 'tariff',
        'power_fraction': 1.0,
        'round_trip_efficiency': 0.9,
    },
    'cintensity_file': 'QC_NG_&_avgCI.csv',
    'console_verbosity': 'summary',
    'datacenter_capacity_mw': 10,
    'days_per_episode': 30,
    'dc_config_file': 'dc_config.json',
    'episode_start_policy': 'calendar_month',
    'facility_sizing': {'mode': 'scaling'},
    'flexible_load': 0.6,
    'location': 'montreal',
    'max_bat_cap_Mw': 1.0,
    'month': 6,
    'nonoverlapping_shared_obs_space': True,
    'partial_obs': True,
    'pricing': {
        'config_file': 'data/Pricing/hydro_quebec_2026.yaml',
        'model': 'hydro_quebec',
        'options': {'demand_floor_kw': 0.0, 'tariff': 'auto'},
    },
    'pv': {
        'capacity_fraction_of_datacenter': 0.05,
        'dc_ac_ratio': 1.2,
        'enabled': True,
        'export_policy': 'curtail',
        'gamma_pdc_per_deg_c': -0.0037,
        'inverter_efficiency': 0.96,
        'surface_azimuth_deg': 180.0,
        'surface_tilt_deg': 35.0,
        'system_losses_fraction': 0.1408,
        'temperature_model': 'close_mount_glass_glass',
    },
    # Reward mode 'legacy', not the training scenario's team mode: the team
    # mode's config validator requires the exact three-agent list, which a
    # battery-only environment cannot satisfy. Bills and gates are metering
    # (price manager + sub-environment counters), independent of reward mode,
    # so the fallback comparison is unaffected.
    'reward': {'mode': 'legacy'},
    'sample_whole_year': True,
    'state_type': 'FP',
    'weather': {'randomize': False},
    'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
}

_FEBRUARY_FIRST_DAY = 31      # zero-based day-of-year index of February 1
_FEBRUARY_STEPS = 28 * 96     # shortest calendar month at 15-minute steps

_GATE_KEYS = (
    'episode/workload/queue_overflow_forced_tasks',
    'episode/sla/overdue_task_steps',
    'episode/thermal/crac_clamp_steps',
    'episode/thermal/cooling_capacity_exceeded_steps',
)
_BILL_KEY = 'episode/tariff/cost/total_c'


def test_every_base_agent_exposes_act_returning_do_nothing():
    for cls in (BaseLoadShiftingAgent, BaseHVACAgent, BaseBatteryAgent):
        agent = cls()
        assert agent.act() == agent.do_nothing_action(), cls.__name__


def _run_february(agents):
    config = copy.deepcopy(_SCENARIO)
    config['agents'] = list(agents)
    env = SustainDC(EnvConfig(config))
    ls_dc = {name: 1 for name in agents if name != 'agent_bat'}
    actions = dict(ls_dc, agent_bat=2)
    accumulator = FacilityMetricAccumulator('episode')
    env.reset(seed=102, options={'init_day': _FEBRUARY_FIRST_DAY, 'init_hour': 0})
    steps = 0
    truncated = {'__all__': False}
    while not truncated['__all__']:
        assert steps <= _FEBRUARY_STEPS, 'episode failed to truncate'
        env.begin_step(ls_dc)
        _, _, _, truncated, info = env.step(actions)
        accumulator.add(info['__common__'])
        steps += 1
    return steps, accumulator.snapshot()


def _values_equal(a, b):
    if (isinstance(a, float) and isinstance(b, float)
            and math.isnan(a) and math.isnan(b)):
        return True
    return a == b


def test_battery_only_idle_equals_three_agent_noop_exactly():
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        bat_steps, bat_snap = _run_february(['agent_bat'])
        ref_steps, ref_snap = _run_february(['agent_ls', 'agent_dc', 'agent_bat'])
    assert bat_steps == _FEBRUARY_STEPS, bat_steps
    assert ref_steps == _FEBRUARY_STEPS, ref_steps
    for key in _GATE_KEYS + (_BILL_KEY,):
        assert key in ref_snap, key
    assert set(bat_snap) == set(ref_snap), sorted(set(bat_snap) ^ set(ref_snap))[:10]
    diffs = {key: (bat_snap[key], ref_snap[key]) for key in ref_snap
             if not _values_equal(bat_snap[key], ref_snap[key])}
    assert not diffs, sorted(diffs.items())[:10]


if __name__ == '__main__':
    sys.exit(run_module_tests(globals()))
