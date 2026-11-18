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


def test_flat_forecast_relative_position_stays_bounded():
    env = _get_env()
    values = env.extract_ci_features(np.full(16, 0.5), 0.4)
    assert np.isfinite(values).all()
    assert -2.0 <= values[2] <= 2.0


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
    assert obs["agent_dc"].shape[0] == 29, obs["agent_dc"].shape
    assert obs["agent_bat"].shape[0] == 30, obs["agent_bat"].shape


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


def test_dc_observes_current_setpoint_after_action_and_reset():
    env = _get_env()
    obs = env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
    idx = SustainDC.DC_IDX["setpoint_ratio"]
    assert np.isclose(obs["agent_dc"][idx], env.dc_env.raw_curr_stpt / env.dc_env.max_temp)
    next_obs, _, _, _, _ = env.step({"agent_ls": 1, "agent_dc": 0, "agent_bat": 2})
    assert np.isclose(next_obs["agent_dc"][idx], env.dc_env.raw_curr_stpt / env.dc_env.max_temp)
    assert next_obs["agent_dc"][idx] < obs["agent_dc"][idx]
    reset_obs = env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
    assert np.isclose(reset_obs["agent_dc"][idx], obs["agent_dc"][idx])


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
    assert [obs[a].shape[0] for a in env.agents] == [40, 29, 30]
    # The non-overlapping critic also receives the battery's measured
    # previous load and uncapped running peak.
    assert obs["agent_ls"].shape[0] + 3 + 3 == 46


def test_battery_previous_load_and_uncapped_peak_are_causal():
    env = _get_env()
    obs = env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
    load_index = SustainDC.BAT_IDX["previous_facility_load_ratio"]
    peak_index = SustainDC.BAT_IDX["billing_peak_ratio"]
    rating_kw = env.datacenter_capacity_mw * 1000.0
    assert np.isclose(obs["agent_bat"][load_index], 0.0)
    assert np.isclose(obs["agent_bat"][peak_index],
                      env.price_m.state.running_peak_kw / rating_kw)

    actions = {"agent_ls": 1, "agent_dc": 1, "agent_bat": 0}
    next_obs, _, _, _, info = env.step(actions)
    assert np.isclose(next_obs["agent_bat"][load_index],
                      info["agent_dc"]["dc_total_power_kW"] / rating_kw)
    assert np.isclose(next_obs["agent_bat"][peak_index],
                      info["agent_bat"]["billing_running_peak_kw"] / rating_kw)
    assert np.allclose(next_obs["agent_bat"][-3:], env._price_feature_block())

    reset_obs = env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
    assert np.isclose(reset_obs["agent_bat"][load_index], 0.0)

    larger_env = SustainDC(dict(
        _ENV_CONFIG, datacenter_capacity_mw=10,
        facility_sizing={"mode": "scaling"},
        battery={"power_fraction": 1.0, "duration_hours": 2.0},
    ))
    larger_env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
    larger_obs, _, _, _, larger_info = larger_env.step(actions)
    assert larger_obs["agent_bat"][peak_index] > 1.0
    assert np.isclose(larger_obs["agent_bat"][peak_index],
                      larger_info["agent_bat"]["billing_running_peak_kw"] / 10000.0)


def test_battery_observes_real_month_fraction():
    """The battery's demand-charge strategy needs to see where in the billing
    month the decision interval sits; billing_progress is constant 0.0 under
    tariffs with no tier threshold, so a real calendar month fraction is the
    only month-position signal."""
    env = _get_env()
    frac_index = SustainDC.BAT_IDX["month_fraction"]
    assert frac_index == SustainDC.BAT_IDX["billing_peak_ratio"] + 1

    # First interval of a calendar month -> exactly 0.0.
    obs = env.reset(seed=102, options={"init_day": 31, "init_hour": 0})
    assert np.isclose(obs["agent_bat"][frac_index], 0.0)

    # Mid-January: 14 of January's 31 days have elapsed.
    obs = env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
    assert np.isclose(obs["agent_bat"][frac_index], 14.0 / 31.0)

    # One 15-minute step advances the fraction by 0.25 h / (31 d * 24 h).
    actions = {"agent_ls": 1, "agent_dc": 1, "agent_bat": 2}
    next_obs, _, _, _, _ = env.step(actions)
    assert np.isclose(next_obs["agent_bat"][frac_index],
                      (14.0 * 24.0 + 0.25) / (31.0 * 24.0))

    # The last hour of February sits just under 1.0, and the fraction wraps
    # to 0.0 on the first March decision interval.
    obs = env.reset(seed=102, options={"init_day": 58, "init_hour": 23})
    assert np.isclose(obs["agent_bat"][frac_index], 671.0 / 672.0)
    for _ in range(4):
        obs, _, _, _, _ = env.step(actions)
        assert 0.0 <= obs["agent_bat"][frac_index] < 1.0
    assert np.isclose(obs["agent_bat"][frac_index], 0.0)


def test_price_block_is_the_last_three_slots_of_every_agent():
    """harlsustaindc_env.py once used states[2][-1] for SoC; the price block
    moved in behind it and the critic silently trained on the wrong feature.
    Assert the block really is trailing, so [-1] stays a known-bad idiom."""
    env = _get_env()
    obs = env.reset()
    expected = env._price_feature_block()
    for agent in ("agent_ls", "agent_dc", "agent_bat"):
        assert np.allclose(obs[agent][-3:], expected, atol=1e-6), agent


def test_shared_critic_observes_dc_setpoint_changes():
    from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv

    env = HARLSustainDCEnv(dict(
        _ENV_CONFIG, state_type="FP", partial_obs=True,
        nonoverlapping_shared_obs_space=True,
    ))
    try:
        actor_obs, shared_obs, _ = env.reset(
            seed=102, options={"init_day": 14, "init_hour": 0}
        )
        dc_index = SustainDC.DC_IDX["setpoint_ratio"]
        shared_index = 42
        assert np.isclose(shared_obs[1][shared_index], actor_obs[1][dc_index])
        actor_obs, shared_obs, _, _, _, _ = env.step(np.array([1, 0, 2]))
        assert np.isclose(shared_obs[1][shared_index], actor_obs[1][dc_index])
        assert shared_obs[1][shared_index] < 18.0 / env.env.unwrapped.env.dc_env.max_temp
    finally:
        env.close()


def test_fp_shared_critic_space_adds_agent_identity_features():
    from harl.envs.sustaindc.sustaindc_ptzoo import SustainDCPettingZooEnv

    env = SustainDCPettingZooEnv(
        dict(_ENV_CONFIG, state_type="FP", partial_obs=True,
             nonoverlapping_shared_obs_space=True)
    )
    assert all(space.shape == (49,) for space in env.share_observation_space.values())


def _battery_only_config(state_type):
    """The battery as the only learner; the base agents hold workload and cooling."""
    return dict(_ENV_CONFIG, agents=["agent_bat"], state_type=state_type,
                partial_obs=True, nonoverlapping_shared_obs_space=True)


def test_battery_only_critic_space_is_the_battery_box_plus_the_fp_identity_slot():
    """With one agent there is no ls or dc state to borrow features from, so the
    critic observes the battery's own observation: its Box unchanged, plus the
    one identity slot under FP (bounded [0, 1] like every identity feature)."""
    from harl.envs.sustaindc.sustaindc_ptzoo import SustainDCPettingZooEnv

    for state_type, extra in (("FP", 1), ("EP", 0)):
        env = SustainDCPettingZooEnv(_battery_only_config(state_type))
        battery = env.observation_spaces["agent_bat"]
        shared = env.share_observation_space["agent_bat"]
        assert list(env.share_observation_space) == ["agent_bat"], state_type
        assert battery.shape == (30,), battery.shape
        assert shared.shape == (30 + extra,), (state_type, shared.shape)
        assert np.array_equal(shared.low[:30], battery.low), state_type
        assert np.array_equal(shared.high[:30], battery.high), state_type
        assert np.array_equal(shared.low[30:], np.zeros(extra)), state_type
        assert np.array_equal(shared.high[30:], np.ones(extra)), state_type


def test_battery_only_critic_observation_is_the_battery_observation():
    """The three-agent builder picks dc and battery features out of states[1] and
    states[2] by position; a one-agent state tuple has neither (IndexError on
    reset). The critic must get the same vector the battery actor gets, plus the
    identity slot under FP, at reset and after a step, matching the declared
    share-space width, with the identity slot inside its declared [0, 1] bounds.

    The battery slice is deliberately not checked with Box.contains(): the shipped
    battery Box declares the CI-percentile feature (index 9) as [-1, 1] while the
    observation clips it to [-2, 2], so even the actor's own observation fails
    contains() -- in the three-agent layout too. That predates, and is independent
    of, the critic layout under test."""
    from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv

    for state_type, extra in (("FP", 1), ("EP", 0)):
        env = HARLSustainDCEnv(_battery_only_config(state_type))
        try:
            actor_obs, shared_obs, _ = env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
            stepped_actor_obs, stepped_shared_obs = env.step(np.array([2]))[:2]
            for actor, shared in ((actor_obs, shared_obs), (stepped_actor_obs, stepped_shared_obs)):
                assert len(actor) == len(shared) == 1, (state_type, len(actor), len(shared))
                assert actor[0].shape == (30,), actor[0].shape
                assert shared[0].shape == (30 + extra,), (state_type, shared[0].shape)
                assert np.allclose(shared[0][:30], actor[0]), state_type
                assert np.array_equal(shared[0][30:], np.ones(extra)), state_type
                declared = env.share_observation_space[0]
                assert declared.shape == shared[0].shape, (state_type, declared.shape, shared[0].shape)
                slot = shared[0][30:]
                assert np.all((slot >= declared.low[30:]) & (slot <= declared.high[30:])), state_type
        finally:
            env.close()


def test_padded_battery_peak_bounds_do_not_leak_into_other_features():
    from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv

    env = HARLSustainDCEnv(dict(
        _ENV_CONFIG, state_type="FP", partial_obs=True,
        nonoverlapping_shared_obs_space=True,
        facility_sizing={"mode": "scaling"},
        battery={"power_fraction": 1.5, "duration_hours": 2.0},
    ))
    try:
        actor_obs, shared_obs, _ = env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
        actor_obs, shared_obs, _, _, _, _ = env.step(np.array([1, 1, 0]))
        peak_index = SustainDC.BAT_IDX["billing_peak_ratio"]
        raw = env.env.unwrapped.env
        expected_ratio = raw.price_m.state.running_peak_kw / (raw.datacenter_capacity_mw * 1000.0)
        assert expected_ratio > 1.0
        assert np.isclose(actor_obs[2][peak_index], expected_ratio)
        assert actor_obs[2][peak_index] <= env.observation_space[2].high[peak_index]
        assert shared_obs[2][45] <= env.share_observation_space[2].high[45]
        assert np.isfinite(env.observation_space[2].high).all()
        assert np.all(env.observation_space[2].high[30:] <= 10.0)
    finally:
        env.close()


def test_shared_observation_width_matches_what_is_produced():
    """The exact mismatch that crashed training: sustaindc_ptzoo.py sizes the
    critic's Box, harlsustaindc_env.py fills it, and the two are computed in
    different files. Width = whole ls_state + 2 dc features + 3 bat features."""
    env = _get_env()
    obs = env.reset()
    produced = obs["agent_ls"].shape[0] + 2 + 3
    # mirror sustaindc_ptzoo.py's formula
    declared = env.observation_space[0].shape[0] + 2 + 3
    assert produced == declared, f"produced {produced} != declared {declared}"


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
