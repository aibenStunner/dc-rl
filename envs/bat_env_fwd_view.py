import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import envs.battery_model as batt


class BatteryEnvFwd(gym.Env):
    """Battery controller environment with explicit bus/cell energy flows."""

    STEP_HOURS = 0.25

    def __init__(self, env_config) -> None:
        super().__init__()
        n_fwd_steps = env_config['n_fwd_steps']
        max_bat_cap = env_config['max_bat_cap']
        self.round_trip_efficiency = _validate_rte(env_config['round_trip_efficiency'])
        self.degradation_cost_c_per_kwh = _validate_degradation_cost(
            env_config['degradation_cost_c_per_kwh']
        )
        self.observation_space = spaces.Box(low=np.float32(-1.0 * np.ones(18)),
                                            high=np.float32(1.0 * np.ones(18)))
        self.max_dc_pw_MW = env_config['max_dc_pw_MW']
        self.action_space = spaces.Discrete(3)
        self._action_to_direction = {0: 'charge', 1: 'discharge', 2: 'idle'}
        self.observation_max = np.array([self.max_dc_pw_MW, max_bat_cap])
        self.observation_min = np.array([0.0, 0.0])
        self.delta = self.observation_max - self.observation_min
        self.battery = batt.Battery2(
            capacity=max_bat_cap,
            current_load=0.0,
            round_trip_efficiency=self.round_trip_efficiency,
        )
        self.n_fwd_steps = n_fwd_steps
        self.charging_rate = env_config['charging_rate']
        self.spot_CI = None
        self.ma_CI = None
        self.dataset_end = True
        self.dcload = 0.0
        self.temp_state = None
        self.max_bat_cap = max_bat_cap
        self.total_energy_with_battery = 0.0
        self.ci = 0.0
        self.ci_n = []
        self.dcload_max = env_config['dcload_max']
        self.dcload_min = env_config['dcload_min']
        self._reset_episode_metrics()

    def _reset_episode_metrics(self):
        self.last_flow = batt.BatteryFlow()
        self.total_energy_with_battery = 0.0
        self.CO2_total = 0.0
        self.cumulative_cell_throughput_kwh = 0.0
        self.cumulative_degradation_cost_c = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.battery.reset()
        self._reset_episode_metrics()
        self.dcload = self.dcload_min
        self.raw_obs = self._hist_data_collector()
        self.temp_state = self._process_obs(self.raw_obs)
        return self.temp_state, self._info(action_id=-1, action_name='idle')

    def step(self, action_id):
        action_name = self._action_to_direction[action_id]
        self.last_flow = self._simulate_battery_operation(self.battery, action_name)
        self.CO2_total = self._update_grid_meter_and_co2(self.last_flow)
        self._record_degradation(self.last_flow)
        self.raw_obs = self._hist_data_collector()
        self.temp_state = self._process_obs(self.raw_obs)
        return self.temp_state, 0.0, False, False, self._info(action_id, action_name)

    def _info(self, action_id, action_name):
        base_kwh = self.dcload * 1000.0 * self.STEP_HOURS
        flow = self.last_flow
        return {
            'bat_action': action_id,
            'bat_SOC': self.get_battery_soc(),
            'bat_CO2_footprint': self.CO2_total,
            'bat_avg_CI': self.ci,
            'bat_total_energy_without_battery_KWh': base_kwh,
            'bat_total_energy_with_battery_KWh': self.total_energy_with_battery,
            'bat_grid_import_kwh': self.total_energy_with_battery,
            'bat_dc_load_kwh': base_kwh,
            'bat_charge_bus_KWh': flow.charge_bus_mwh * 1000.0,
            'bat_discharge_bus_KWh': flow.discharge_bus_mwh * 1000.0,
            'bat_charge_cell_KWh': flow.charge_cell_mwh * 1000.0,
            'bat_discharge_cell_KWh': flow.discharge_cell_mwh * 1000.0,
            'bat_cell_throughput_KWh': self._cell_throughput_kwh(flow),
            'bat_cumulative_cell_throughput_KWh': self.cumulative_cell_throughput_kwh,
            'bat_degradation_cost_c': self._degradation_cost_c(flow),
            'bat_cumulative_degradation_cost_c': self.cumulative_degradation_cost_c,
            'bat_round_trip_efficiency': self.round_trip_efficiency,
            'bat_charge_efficiency': self.battery.eff_c,
            'bat_discharge_efficiency': self.battery.eff_d,
            'bat_max_bat_cap': self.max_bat_cap,
            'bat_a_t': action_name,
            'bat_dcload_min': self.dcload_min,
            'bat_dcload_max': self.dcload_max,
        }

    def update_ci(self, ci, ci_n):
        self.ci = ci
        self.ci_n = ci_n

    def _process_obs(self, state):
        return np.float32((state - self.observation_min) / self.delta)

    def set_dcload(self, dc_load):
        self.dcload = dc_load

    def get_battery_soc(self):
        return self.battery.get_battery_soc()

    def _hist_data_collector(self):
        return np.array([self.dcload, self.battery.current_load])

    def _simulate_battery_operation(self, battery, action_name):
        if action_name == 'charge':
            requested_power = self.charging_rate_modifier(battery)
            return battery.charge(requested_power, self.STEP_HOURS)
        if action_name == 'discharge':
            requested_power = self.discharging_rate_modifier(battery)
            return battery.discharge(requested_power, self.STEP_HOURS, self.dcload)
        return batt.BatteryFlow()

    def _update_grid_meter_and_co2(self, flow):
        base_kwh = self.dcload * 1000.0 * self.STEP_HOURS
        grid_import_kwh = max(
            0.0,
            base_kwh + flow.charge_bus_mwh * 1000.0 - flow.discharge_bus_mwh * 1000.0,
        )
        self.total_energy_with_battery = grid_import_kwh
        self.energy_added_removed = getattr(self, 'energy_added_removed', [])
        self.energy_added_removed.append(grid_import_kwh - base_kwh)
        return grid_import_kwh * self.ci

    def _cell_throughput_kwh(self, flow):
        return (flow.charge_cell_mwh + flow.discharge_cell_mwh) * 1000.0

    def _degradation_cost_c(self, flow):
        return self._cell_throughput_kwh(flow) * self.degradation_cost_c_per_kwh

    def _record_degradation(self, flow):
        throughput = self._cell_throughput_kwh(flow)
        cost = self._degradation_cost_c(flow)
        self.cumulative_cell_throughput_kwh += throughput
        self.cumulative_degradation_cost_c += cost

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def charging_rate_modifier(self, battery):
        scaled_soc = battery.current_load / battery.capacity
        return np.round(0.5 * (1 - self.sigmoid(10 * (scaled_soc - 0.5))), 4)

    def discharging_rate_modifier(self, battery):
        scaled_soc = battery.current_load / battery.capacity
        return max(0.5, 4 * self.sigmoid(10 * (scaled_soc - 0.25)))


def _validate_rte(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
        raise ValueError('round_trip_efficiency must be finite in (0, 1]')
    return float(value)


def _validate_degradation_cost(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError('degradation_cost_c_per_kwh must be finite and non-negative')
    return float(value)
