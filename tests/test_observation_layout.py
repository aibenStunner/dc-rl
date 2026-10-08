"""Locks the observation layout that several files depend on positionally.

Three separate files hardcode widths or indices derived from the state
builders in sustaindc_env.py:
  - the three sub-env Box declarations (envs/carbon_ls.py,
    utils/make_envs_pyenv.py, envs/bat_env_fwd_view.py)
  - harl/envs/sustaindc/harlsustaindc_env.py, which picks shared-observation
    features out of dc_state/bat_state BY POSITION
  - harl/envs/sustaindc/sustaindc_ptzoo.py, which sizes the critic's shared
    observation space

A mismatch in the first group raises loudly. A mismatch in the second group
does NOT -- the critic simply trains on the wrong feature. This module exists
to turn that silent failure into a loud one.

Run:  python3 tests/test_observation_layout.py
"""
import sys
import warnings
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

warnings.filterwarnings("ignore")

import numpy as np

from tests._harness import run_module_tests
from sustaindc_env import SustainDC

_ENV_CONFIG = {
    "agents": ["agent_ls", "agent_dc", "agent_bat"],
    "location": "ca",
    "cintensity_file": "CA_NG_&_avgCI.csv",
    "workload_file": "Alibaba_CPU_Data_Hourly_1.csv",
    "datacenter_capacity_mw": 5,
    "month": 6,
    "days_per_episode": 30,
    "bat_reward": "default_price_reward",
    "pricing": {
        "model": "hydro_quebec",
        "config_file": "data/Pricing/hydro_quebec_2026.yaml",
        "options": {"tariff": "auto", "demand_floor_kw": 0.0},
    },
}

_env = None


def _get_env():
    """One env for the whole module -- construction sizes a chiller and is slow."""
    global _env
    if _env is None:
        _env = SustainDC(dict(_ENV_CONFIG))
    return _env


def test_declared_box_widths_match_built_vectors():
    """The hardcoded Box shapes must equal what the builders actually emit."""
    env = _get_env()
    obs = env.reset()
    declared = [s.shape[0] for s in env.observation_space]
    actual = [obs["agent_ls"].shape[0], obs["agent_dc"].shape[0], obs["agent_bat"].shape[0]]
    assert declared == actual, f"declared {declared} != produced {actual}"


def test_widths_are_the_expected_values():
    """Pin the numbers so a width change has to be deliberate."""
    env = _get_env()
    obs = env.reset()
    assert obs["agent_ls"].shape[0] == 40, obs["agent_ls"].shape
    assert obs["agent_dc"].shape[0] == 28, obs["agent_dc"].shape
    assert obs["agent_bat"].shape[0] == 27, obs["agent_bat"].shape


def test_shared_prefix_is_identical_across_agents():
    """All three states open with the same time + CI prefix."""
    env = _get_env()
    obs = env.reset()
    n = SustainDC.OBS_PREFIX_LEN
    assert np.allclose(obs["agent_ls"][:n], obs["agent_dc"][:n])
    assert np.allclose(obs["agent_ls"][:n], obs["agent_bat"][:n])


def test_season_features_are_present_and_in_range():
    """cos_day/sin_day must be in the observation (they were sliced off
    before) and, like the hour pair, rescaled to [0, 1]."""
    env = _get_env()
    obs = env.reset()
    for agent in ("agent_ls", "agent_dc", "agent_bat"):
        cos_day, sin_day = obs[agent][2], obs[agent][3]
        assert 0.0 <= cos_day <= 1.0, (agent, cos_day)
        assert 0.0 <= sin_day <= 1.0, (agent, sin_day)
    # and they must actually vary with the calendar, not be constant
    seen = set()
    for _ in range(25):
        o = env.reset()
        seen.add(round(float(o["agent_bat"][2]), 4))
    assert len(seen) > 1, "cos_day never changes across resets"


def test_layout_indices_point_at_the_features_they_claim():
    """SustainDC.DC_IDX / BAT_IDX are consumed positionally by
    harlsustaindc_env.py to build the critic's shared observation. Verify
    each index against the value it is supposed to be naming."""
    env = _get_env()
    obs = env.reset()

    soc_slot = float(obs["agent_bat"][SustainDC.BAT_IDX["battery_soc"]])
    assert abs(soc_slot - env.bat_env.get_battery_soc()) < 1e-6, (
        f"BAT_IDX['battery_soc']={SustainDC.BAT_IDX['battery_soc']} is not the SoC")

    temp_slot = float(obs["agent_bat"][SustainDC.BAT_IDX["current_temperature"]])
    assert abs(temp_slot - env.weather_m.get_current_temperature()) < 1e-6

    nw_slot = float(obs["agent_dc"][SustainDC.DC_IDX["next_workload"]])
    assert abs(nw_slot - env.workload_m.get_next_workload()) < 1e-6, (
        f"DC_IDX['next_workload']={SustainDC.DC_IDX['next_workload']} is wrong")

    nt_slot = float(obs["agent_dc"][SustainDC.DC_IDX["next_out_temp"]])
    assert abs(nt_slot - env.weather_m.get_next_temperature()) < 1e-6


def test_pv_block_precedes_and_does_not_displace_trailing_price_block():
    env = _get_env()
    obs = env.reset()
    expected_pv = env._pv_feature_block()
    expected_price = env._price_feature_block()
    for agent in ("agent_ls", "agent_dc", "agent_bat"):
        assert np.allclose(obs[agent][-12:-3], expected_pv, atol=1e-6), agent
        assert np.allclose(obs[agent][-3:], expected_price, atol=1e-6), agent


def test_pv_observation_widths_and_shared_critic_width_are_pinned():
    env = _get_env()
    obs = env.reset()
    assert [obs[a].shape[0] for a in env.agents] == [40, 28, 27]
    # The non-overlapping critic receives the entire widest (LS) vector,
    # then DC next workload/temperature and battery SoC.
    assert obs["agent_ls"].shape[0] + 2 + 1 == 43


def test_price_block_is_the_last_three_slots_of_every_agent():
    """harlsustaindc_env.py once used states[2][-1] for SoC; the price block
    moved in behind it and the critic silently trained on the wrong feature.
    Assert the block really is trailing, so [-1] stays a known-bad idiom."""
    env = _get_env()
    obs = env.reset()
    expected = env._price_feature_block()
    for agent in ("agent_ls", "agent_dc", "agent_bat"):
        assert np.allclose(obs[agent][-3:], expected, atol=1e-6), agent


def test_shared_observation_width_matches_what_is_produced():
    """The exact mismatch that crashed training: sustaindc_ptzoo.py sizes the
    critic's Box, harlsustaindc_env.py fills it, and the two are computed in
    different files. Width = whole ls_state + 2 dc features + 1 bat feature."""
    env = _get_env()
    obs = env.reset()
    produced = obs["agent_ls"].shape[0] + 2 + 1
    # mirror sustaindc_ptzoo.py's formula
    declared = env.observation_space[0].shape[0] + 2 + 1
    assert produced == declared, f"produced {produced} != declared {declared}"


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
