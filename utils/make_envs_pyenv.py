import os

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd

from envs.bat_env_fwd_view import BatteryEnvFwd as battery_env_fwd
from envs.carbon_ls import CarbonLoadEnv
from envs.dc_gym import dc_gymenv
from utils.utils_cf import get_init_day

import envs.datacenter as DataCenter
from utils.dc_config_reader import DC_Config

import itertools

def make_ls_env(month,
                n_vars_ci: int = 4,
                n_vars_energy : int = 4,
                n_vars_battery : int = 1,
                queue_max_len: int = 500,
                flexible_workload_ratio: float = 0.2,
                test_mode = False):
    """Method to build the Load shifting environment

    Args:
        month (int): Month of the year in which the agent is training.
        n_vars_energy (int, optional): Number of variables from the Energy environment. Defaults to 4.
        n_vars_battery (int, optional): Number of variables from the Battery environment. Defaults to 1.
        queue_max_len (int, optional): The size of the queue where the tasks are stored to be processed latter. Default to 500.

    Returns:
        CarbonLoadEnv: Load Shifting environment
    """
    
    return CarbonLoadEnv(n_vars_ci=n_vars_ci,
                         n_vars_energy=n_vars_energy,
                         n_vars_battery=n_vars_battery,
                         queue_max_len=queue_max_len,
                         flexible_workload_ratio=flexible_workload_ratio,
                         test_mode=test_mode)
    
    

def make_bat_fwd_env(month,
                    max_bat_cap_Mwh : float = 2.0,
                    charging_rate : float = 0.5,
                    max_dc_pw_MW : float = 7.23,
                    dcload_max : float = 2.5,
                    dcload_min : float = 0.1,
                    n_fwd_steps : int = 4,
                    round_trip_efficiency : float = 0.90,
                    degradation_cost_c_per_kwh : float = 5.78,
                    initial_soc: float = 0.0,
                    rated_power_mw: float = None
                    ):
    """Method to build the Battery environment.

    Args:
        month (int): Month of the year in which the agent is training.
        max_bat_cap_Mwh (float, optional): Max battery capacity. Defaults to 2.0.
        charging_rate (float, optional): Charging rate of the battery. Defaults to 0.5.
        reward_method (str, optional): Method used to calculate the rewards. Defaults to 'default_bat_reward'.

    Returns:
        battery_env_fwd: Batery environment.
    """
    init_day = get_init_day(month)
    env_config= {'n_fwd_steps':n_fwd_steps,
                 'max_dc_pw_MW':max_dc_pw_MW,
                 'max_bat_cap':max_bat_cap_Mwh,
                 'charging_rate':charging_rate,
                 'start_point':init_day,
                 'dcload_max':dcload_max, 
                 'dcload_min':dcload_min,
                 'round_trip_efficiency': round_trip_efficiency,
                 'degradation_cost_c_per_kwh': degradation_cost_c_per_kwh,
                 'initial_soc': initial_soc,
                 'rated_power_mw': rated_power_mw}
    bat_env = battery_env_fwd(env_config)
    return bat_env

def _size_scaling(config, target_mw, min_temp, max_temp, ambient):
    """Calibrate a deterministic cyclic layout using the runtime meter boundary."""
    config.sizing_mode = 'scaling'
    # Preserve the original template rack density and characteristics.
    # servers, preserving each source rack's CPU and thermal characteristics.
    source_supply = config.RACK_SUPPLY_APPROACH_TEMP_LIST[:]
    source_return = config.RACK_RETURN_APPROACH_TEMP_LIST[:]
    source_power = config.DEFAULT_SERVER_POWER_CHARACTERISTICS[:]
    templates = [(source_supply[i], source_return[i], source_power[i], rack)
                 for i, rack in enumerate(config.RACK_CPU_CONFIG)]
    template_supply, template_return, template_power, template_cpus = map(list, zip(*templates))
    template_count = len(templates)
    # Keep original rack density; prevent the legacy per-rack power budget from
    # rejecting any template server. It is not a physical rack power limit.
    config.MAX_W_PER_RACK = max(sum(cpu['full_load_pwr'] for cpu in rack)
                                for rack in template_cpus) + 1
    measured = {}

    def evaluate(count, occupancy=None):
        """Evaluate one candidate layout: `count` racks at `occupancy` servers.

        `occupancy` of None means every rack carries the full template density.
        A lower occupancy models a partially populated rack, which gives the
        calibration finer granularity than whole 200-server racks.
        """
        key = (count, occupancy)
        if key in measured:
            return measured[key]
        config.NUM_RACKS = count
        config.NUM_ROWS = 1
        config.NUM_RACKS_PER_ROW = count
        config.RACK_SUPPLY_APPROACH_TEMP_LIST = [template_supply[i % template_count] for i in range(count)]
        config.RACK_RETURN_APPROACH_TEMP_LIST = [template_return[i % template_count] for i in range(count)]
        if occupancy is None:
            config.RACK_CPU_CONFIG = [template_cpus[i % template_count] for i in range(count)]
        else:
            config.RACK_CPU_CONFIG = [
                list(template_cpus[i % template_count][:occupancy])
                for i in range(count)
            ]
        config.DEFAULT_SERVER_POWER_CHARACTERISTICS = [
            template_power[i % template_count]
            for i in range(count)]
        flow, rated = DataCenter.chiller_sizing(config, min_CRAC_setpoint=min_temp,
                                                max_CRAC_setpoint=max_temp, max_ambient_temp=ambient)
        config.CT_REFRENCE_AIR_FLOW_RATE, config.CT_FAN_REF_P = flow, rated
        dc = DataCenter.DataCenter_ITModel(count, config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                                           config.RACK_CPU_CONFIG, config.MAX_W_PER_RACK, config)
        cpu, fan, outlet = dc.compute_datacenter_IT_load_outlet_temp([100.0] * count, max_temp)
        it = sum(cpu) + sum(fan)
        return_temp = DataCenter.calculate_avg_CRAC_return_temp(config.RACK_RETURN_APPROACH_TEMP_LIST, outlet)
        crac, tower, _, compressor, cw, ct = DataCenter.calculate_HVAC_power(
            max_temp, return_temp, ambient, it, config)
        result = (it + tower + compressor, it, tower, compressor, crac, cw, ct)
        if not all(np.isfinite(result)) or result[0] <= 0:
            raise ValueError('Facility sizing produced nonfinite or nonpositive power')
        measured[key] = result
        return result

    target = target_mw * 1e6
    low, high = 1, 1
    # Bounded exponential search; a non-monotone sampled model is rejected,
    # rather than relying on a binary search with an invalid assumption.
    while evaluate(high)[0] < target and high < 4096:
        high = min(high * 2, 4096)
    while low < high:
        mid = (low + high) // 2
        if evaluate(mid)[0] < target:
            low = mid + 1
        else:
            high = mid
    for count in range(max(1, low - 2), min(4096, low + 2) + 1):
        evaluate(count)
    ordered = sorted(k for k in measured if k[1] is None)
    if any(measured[a][0] >= measured[b][0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Facility design power is not monotone over sampled layouts')
    best = min(ordered, key=lambda n: (abs(measured[n][0] - target), n[0]))

    # Refine with a partially populated final rack. The template density is
    # quantized to whole racks, which can leave a small target outside
    # tolerance; a partial rack gives finer control of the achieved rating.
    # A rack must stay above the model's minimum viable occupancy, which the
    # thermal model rejects below with an outlet-rise error.
    full_density = len(template_cpus[0])
    minimum_occupancy = 45
    if abs((measured[best][0] - target) / target) > 0.02:
        for count in (best[0], best[0] + 1):
            for occupancy in range(minimum_occupancy, full_density):
                try:
                    evaluate(count, occupancy)
                except ValueError:
                    continue
    refined = [k for k in measured if k[1] is not None]
    if refined:
        candidate = min(refined, key=lambda n: (abs(measured[n][0] - target), n[0], -n[1]))
        if abs((measured[candidate][0] - target) / target) < abs((measured[best][0] - target) / target):
            best = candidate

    total, it, tower, compressor, crac, cw, ct = measured[best]
    error = (total - target) / target
    if abs(error) > 0.02:
        raise ValueError(f'Facility target {target_mw:g} MW cannot be achieved within 2%; '
                         f'nearest sampled design {total / 1e6:.6g} MW with {best[0]} racks')
    # Restore the winning layout even when its cached evaluation was not last.
    del measured[best]
    evaluate(*best)
    return {'mode': 'scaling', 'target_mw': target_mw, 'achieved_mw': total / 1e6,
            'relative_error': error, 'rack_count': best[0],
            'servers_per_rack': best[1] if best[1] is not None else full_density,
            'server_count': sum(map(len, config.RACK_CPU_CONFIG)),
            'utilization': 1.0, 'crac_setpoint_c': max_temp,
            'min_crac_setpoint_c': min_temp, 'ambient_temperature_c': ambient,
            'it_mw': it / 1e6, 'tower_fan_mw': tower / 1e6,
            'compressor_mw': compressor / 1e6, 'hvac_mw': (tower + compressor) / 1e6,
            'excluded_crac_fan_mw': crac / 1e6, 'excluded_cw_pump_mw': cw / 1e6,
            'excluded_ct_pump_mw': ct / 1e6,
            'caveat': 'Modeled design rating, not a cap or independently validated site demand; CRAC fan and pumps excluded from meter.'}


def make_dc_pyeplus_env(month : int = 1,
                        location : str = 'NYIS',
                        dc_config_file: str = 'dc_config_file.json',
                        datacenter_capacity_mw: int = 1,
                        max_bat_cap_Mw : float = 2.0,
                        add_cpu_usage : bool = True,
                        add_CI : bool = True,
                        episode_length_in_time : pd.Timedelta = None,
                        use_ls_cpu_load : bool = False,
                        num_sin_cos_vars : int = 4,
                        facility_sizing_mode: str = 'legacy',
                        ):
    """Method that creates the data center environment with the timeline, location, proper data files, gym specifications and auxiliary methods

    Args:
        month (int, optional): The month of the year for which the Environment uses the weather and Carbon Intensity data. Defaults to 1.
        location (str, optional): The geographical location in a standard format for which Carbon Intensity files are accessed. Supported options are 
                                'NYIS', 'AZPS', 'BPAT'. Defaults to 'NYIS'.
        datacenter_capacity_mw (int, optional): Maximum capacity (MW) of the data center. This value will scale the number of servers installed in the data center.
        max_bat_cap_Mw (float, optional): The battery capacity in Megawatts for the installed battery. Defaults to 2.0.
        add_cpu_usage (bool, optional): Boolean Flag to indicate whether cpu usage is part of the environment statespace. Defaults to True.
        add_CI (bool, optional): Boolean Flag to indicate whether Carbon Intensity is part of the environment statespace. Defaults to True.
        episode_length_in_time (pd.Timedelta, optional): Length of an episode in terms of pandas time-delta object. Defaults to None.
        use_ls_cpu_load (bool, optional): Use the cpu workload value from a separate Load Shifting agent. This turns of reading default cpu data. Defaults to False.
        num_sin_cos_vars (int, optional): Number of sin and cosine variable that will be added externally from the centralized data source
        facility_sizing_mode: 'legacy' retains historical sizing; 'scaling'
            sizes whole racks to total IT plus tower fan and compressor at full
            utilization, maximum setpoint, and location design ambient. This is
            a modeled design rating, not a power cap. CRAC fan and pumps are
            calculated by the model but excluded from the runtime meter.
    Returns:
        envs.dc_gym.dc_gymenv: The environment instantiated with the particular month.
    """
    if facility_sizing_mode not in ('legacy', 'scaling'):
        raise ValueError(f'Unknown facility_sizing_mode: {facility_sizing_mode!r}')
    observation_variables = []
    ############################################################################
    ######################### Standard Variables included as default ###########
    ############################################################################
    observation_variables += [
        'Site Outdoor Air Drybulb Temperature(Environment)',
        'Zone Thermostat Cooling Setpoint Temperature(West Zone)',
        'Zone Air Temperature(West Zone)',
        'Facility Total HVAC Electricity Demand Rate(Whole Building)',  # 'HVAC POWER'
        # TODO: Will add sum of IT POWER  and HVAC Power Here if AGP wants it
        'Facility Total Building Electricity Demand Rate(Whole Building)'  #  'IT POWER'
    ]
    # 14 + 3 price features (_price_feature_block) + 2 season features
    # (cos_day/sin_day) -- see sustaindc_env.py's state builders.
    observation_space = spaces.Box(low=np.float32(-1.0*np.ones(28)),
                                    high=np.float32(1.0*np.ones(28)),
                                    )

    
    ################################################################################
    ########################## Action Variables ####################################
    ################################################################################
    
    action_variables = ['Cooling_Setpoint_RL']
    action_definition = {'cooling setpoints': {'name': 'Cooling_Setpoint_RL', 'initial_value': 18}}
    min_temp = 15.0
    max_temp = 21.6
    action_mapping = {
        0: (-1),
        1: (0),
        2: (1),
    }
    action_space = spaces.Discrete(len(action_mapping))
    
    
    ################################################################################
    ##########################  System Sizing  #####################################
    ################################################################################
    
    # from DC_Config, scale the variable number of CPUs to have a similar value to "datacenter_capacity_mw"
    dc_config = DC_Config(dc_config_file=dc_config_file, datacenter_capacity_mw=datacenter_capacity_mw)  # Specify the relative or absolute path

    # Perform Cooling Tower Sizing
    # This step determines the potential maximum loading of the CT
    # setting a higher ambient temp here will cause the CT to consume less power for cooling water under normal ambient temperature. Lower amb temp -> higher HVAC power
    # setting a lower value of min_CRAC_setpoint will cause the CT to consume more power for higher crac setpoints during normal use. Lower min_CRAC_set -> higher HVAC power
    
    # dictionary with locations and min_CRAC_setpoint/max_amb_temp
    #
    # The design ambient is the temperature the cooling plant is sized for.
    # It must match the scenario's weather file: an oversized design ambient
    # under-sizes the chiller and cooling tower, so the modeled facility can
    # no longer meet its electrical rating at hot weather.

    if 'QC'.lower() in location.lower() or 'MONTREAL'.lower() in location.lower():
        # Montreal-Trudeau TMYx: maximum dry-bulb 33.0 C, 99th percentile 29.0 C.
        max_amb_temperature = 33.0
    elif 'NY'.lower() in location.lower():
        max_amb_temperature = 30.0
    elif 'AZ'.lower() in location.lower():
        max_amb_temperature = 50.0
    elif 'WA'.lower() in location.lower():
        max_amb_temperature = 20.0
    else:
        print('WARNING, using default values for chiller sizing...')
        max_amb_temperature = 50.0

    if facility_sizing_mode == 'scaling':
        sizing_metadata = _size_scaling(dc_config, datacenter_capacity_mw, min_temp,
                                           max_temp, max_amb_temperature)
    else:
        sizing_metadata = {'mode': 'legacy'}
    dc_config.sizing_metadata = sizing_metadata
    ctafr, ct_rated_load = DataCenter.chiller_sizing(dc_config, min_CRAC_setpoint=min_temp, max_CRAC_setpoint=max_temp, max_ambient_temp=max_amb_temperature)
    dc_config.CT_REFRENCE_AIR_FLOW_RATE = ctafr
    dc_config.CT_FAN_REF_P = ct_rated_load
    
    
    # Perform sizing of ITE power and ambient temperature
    # Find highest and lowest values of ITE power, rackwise outlet temperature
    dc = DataCenter.DataCenter_ITModel(num_racks=dc_config.NUM_RACKS, rack_supply_approach_temp_list=dc_config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                                    rack_CPU_config=dc_config.RACK_CPU_CONFIG, max_W_per_rack=dc_config.MAX_W_PER_RACK, DC_ITModel_config=dc_config)
    raw_curr_stpt_list = range(15,23)
    cpu_load_list = range(0,110,10) # We assume same data center load for all servers; Here it will be max
    p = itertools.product(raw_curr_stpt_list,cpu_load_list)
    dc_ambient_temp_list = []
    total_ite_pwr = []
    for raw_curr_stpt, cpu_load in p:
        ITE_load_pct_list = [cpu_load for i in range(dc_config.NUM_RACKS)] 
        rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp = \
            dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=raw_curr_stpt)
        total_ite_pwr.append(sum(rackwise_cpu_pwr) + sum(rackwise_itfan_pwr))
        dc_ambient_temp_list.append(sum(rackwise_outlet_temp)/len(rackwise_outlet_temp))   
    
    
    # Calculate the maximum power consumption of the chiller
    # Assuming the worst-case scenario with the highest load and ambient temperature
    max_cooling_cap = ct_rated_load  # Maximum cooling capacity of the chiller
    highest_ambient_temp = max_amb_temperature  # Highest ambient temperature
    max_load = max(total_ite_pwr)  # Maximum load from your previous calculations

    chiller_max_load = DataCenter.calculate_chiller_power(max_cooling_cap, max_load, highest_ambient_temp)

    # This will be battery sizing
    max_dc_power_w =  1.1*max(total_ite_pwr) + 1.1*ct_rated_load + 1.1*chiller_max_load # Watts
    
    # By carbon explorer paper, we need a battery to feed the DC by at least 1 hours, so we need the energy metric
    num_hours_battery = 1
    num_timestep_hour = 4
    
    max_dc_energy = (max_dc_power_w / num_timestep_hour) * (num_timestep_hour * num_hours_battery) # (energy_per_timestep) * (timestep * num_hours) Wh
    max_dc_energy = max_dc_energy / 1e6 # Mhw
    
    ranges = {
        'sinhour': [-1.0, 1.0], #0
        'coshour': [-1.0, 1.0], #1
        'sindayOTY':[-1.0, 1.0], #2
        'cosdayOTY':[-1.0, 1.0], #3
        'hour':[0.0, 23.0], #4
        'dayOTY':[1.0, 366.0], #5 
        
        'Site Outdoor Air Drybulb Temperature(Environment)': [-10.0, 40.0], #6
        'Zone Thermostat Cooling Setpoint Temperature(West Zone)': [15.0, 30.0],  # reasonable range for setpoint; can be updated based on need #7
        'Zone Air Temperature(West Zone)':[0.9*min(dc_ambient_temp_list), 1.1*max(dc_ambient_temp_list)],
        'Facility Total HVAC Electricity Demand Rate(Whole Building)':  [0.0, 1.1*ct_rated_load + 1.1*chiller_max_load],  # this is cooling tower power and chiller power
        'Facility Total Electricity Demand Rate(Whole Building)': [0.9*min(total_ite_pwr), 1.1*max(total_ite_pwr) +  1.1*ct_rated_load + 1.1*chiller_max_load],  # TODO: This is not a part of the observation variables right now
        'Facility Total Building Electricity Demand Rate(Whole Building)':[0.9*min(total_ite_pwr), 1.1*max(total_ite_pwr)],  # this is it power
        
        'cpuUsage':[0.0, 1.0],
        'carbonIntensity':[0.0, 1000.0],
        'batterySoC': [0.0, max_bat_cap_Mw*1e6],
        'max_battery_energy_Mwh' : max_dc_energy
    }
    
    ################################################################################
    ############################## Create the Environment ##########################
    ################################################################################
        
    dc_env = dc_gymenv(observation_variables=observation_variables,
                    observation_space=observation_space,
                    action_variables=action_variables,
                    action_space=action_space,
                    action_mapping=action_mapping,
                    ranges=ranges,
                    add_cpu_usage=add_cpu_usage,
                    min_temp=min_temp,
                    max_temp=max_temp,
                    action_definition=action_definition,
                    DC_Config=dc_config,
                    episode_length_in_time=episode_length_in_time
                    )
    
    dc_env.sizing_metadata = sizing_metadata
    dc_env.NormalizeObservation()
    max_dc_pw = ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][1]+\
        ranges['Facility Total Building Electricity Demand Rate(Whole Building)'][1]
    
    return dc_env, max_dc_pw
    
    
    