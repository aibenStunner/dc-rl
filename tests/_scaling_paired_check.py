"""Paired capacity-scaling verification for the scaled SustainDC scenario.

Runs ten matched 30-day episodes at 1 MW and 10 MW with identical seeds,
start days and fixed actions, and reports facility/energy/battery outcomes.
Read-only: it writes nothing into the repository.
"""
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harl.utils.configs_tools import get_defaults_yaml_args
from sustaindc_env import EnvConfig, SustainDC

START_DAYS = [68, 28, 121, 120, 318, 293, 165, 116, 237, 292]
ACTIONS = {"agent_ls": 1, "agent_dc": 1, "agent_bat": 2}   # no-op, hold setpoint, idle


def dispatch_actions(step):
    """A deterministic charge/discharge pattern over the 96 daily intervals.

    Charges in the first third of the day, discharges in the evening peak,
    and idles otherwise. This exercises the rated power and cell energy
    limits, which an all-idle battery never touches.
    """
    hour = (step % 96) * 0.25
    if 2.0 <= hour < 8.0:
        battery = 0        # charge
    elif 16.0 <= hour < 20.0:
        battery = 1        # discharge
    else:
        battery = 2        # idle
    return {"agent_ls": 1, "agent_dc": 1, "agent_bat": battery}


def build(capacity_mw):
    """Build the environment from the real active config, changing only capacity."""
    _algo, env_args = get_defaults_yaml_args("happo", "sustaindc")
    config = dict(env_args)
    config["datacenter_capacity_mw"] = capacity_mw
    config["days_per_episode"] = 30
    config["sample_whole_year"] = False
    config["month"] = 6
    return SustainDC(EnvConfig(config))


def run(capacity_mw, dispatch=False):
    random.seed(2026)
    np.random.seed(2026)
    env = build(capacity_mw)
    meta = env.dc_env.sizing_metadata
    rows = []
    for index, day in enumerate(START_DAYS):
        random.seed(2026 + index)
        np.random.seed(2026 + index)
        env.reset(seed=2026 + index, options={'init_day': day, 'init_hour': 0})
        totals = {"facility_kwh": 0.0, "it_kwh": 0.0, "hvac_kwh": 0.0,
                  "grid_kwh": 0.0, "peak_kw": 0.0, "cost_c": 0.0,
                  "charge_kwh": 0.0, "discharge_kwh": 0.0, "pv_kwh": 0.0,
                  "max_charge_kw": 0.0, "max_discharge_kw": 0.0,
                  "min_soc": 1.0, "max_soc": 0.0, "curtailed_kwh": 0.0}
        for step in range(2880):
            action = dispatch_actions(step) if dispatch else ACTIONS
            _, _, _, _, info = env.step(action)
            common = info["__common__"]
            dc = info["agent_dc"]
            bat = info["agent_bat"]
            totals["facility_kwh"] += dc["dc_total_power_kW"] * 0.25
            totals["it_kwh"] += dc["dc_ITE_total_power_kW"] * 0.25
            totals["hvac_kwh"] += dc["dc_HVAC_total_power_kW"] * 0.25
            totals["grid_kwh"] += bat["bat_grid_import_kwh"]
            totals["peak_kw"] = max(totals["peak_kw"], dc["dc_total_power_kW"])
            totals["cost_c"] += common["total_price_cost_this_step_c"]
            totals["charge_kwh"] += bat["bat_charge_bus_KWh"]
            totals["discharge_kwh"] += bat["bat_discharge_bus_KWh"]
            totals["pv_kwh"] += bat["pv_ac_kwh"]
            totals["curtailed_kwh"] += bat["pv_curtailed_kwh"]
            totals["max_charge_kw"] = max(
                totals["max_charge_kw"], bat["bat_charge_bus_KWh"] / 0.25)
            totals["max_discharge_kw"] = max(
                totals["max_discharge_kw"], bat["bat_discharge_bus_KWh"] / 0.25)
            totals["min_soc"] = min(totals["min_soc"], bat["bat_SOC"])
            totals["max_soc"] = max(totals["max_soc"], bat["bat_SOC"])
        rows.append(totals)
    return meta, rows


def summarize(name, meta, rows, dispatch=False):
    label = "with battery dispatch" if dispatch else "battery idle"
    print(f"\n=== {name} ({label}) ===")
    print(f"design: target={meta['target_mw']} MW achieved={meta['achieved_mw']:.4f} MW "
          f"({meta['relative_error']*100:+.3f}%) racks={meta['rack_count']} "
          f"servers/rack={meta['servers_per_rack']} total_servers={meta['server_count']} "
          f"ambient={meta['ambient_temperature_c']}C")
    for key in ("facility_kwh", "it_kwh", "hvac_kwh", "grid_kwh", "peak_kw",
                "cost_c", "charge_kwh", "discharge_kwh", "pv_kwh", "curtailed_kwh",
                "max_charge_kw", "max_discharge_kw", "min_soc", "max_soc"):
        values = [r[key] for r in rows]
        print(f"  {key:<16} mean={np.mean(values):>14,.2f}  min={min(values):>14,.2f}  "
              f"max={max(values):>14,.2f}")
    return rows


if __name__ == "__main__":
    for dispatch in (False, True):
        results = {}
        for name, capacity in (("1 MW", 1.0), ("10 MW", 10.0)):
            started = time.time()
            meta, rows = run(capacity, dispatch=dispatch)
            results[name] = summarize(name, meta, rows, dispatch)
            print(f"  wall time: {time.time() - started:.1f} s")
        one, ten = results["1 MW"], results["10 MW"]
        print("\n=== paired ratios (10 MW / 1 MW) ===")
        for key in ("facility_kwh", "it_kwh", "hvac_kwh", "grid_kwh", "cost_c", "pv_kwh"):
            a = np.mean([r[key] for r in one])
            b = np.mean([r[key] for r in ten])
            print(f"  {key:<14} {b / a:>7.2f}x   ({a:,.1f} -> {b:,.1f})")
        print(f"  peak_kw        {np.mean([r['peak_kw'] for r in ten]) / np.mean([r['peak_kw'] for r in one]):>7.2f}x")
        print()
