import os
import numpy as np

class CPU():

    def __init__(self, full_load_pwr = None, idle_pwr = None, cpu_config = None): 
        """CPU class in charge of the energy consumption and termal calculations of the individuals CPUs
            in a Rack.

        Args:
            full_load_pwr (float, optional): Power at full capacity.
            idle_pwr (float, optional): Power while idle.
            cpu_config (config): Configuration for the DC.
        """
        self.cpu_config = cpu_config
        
        self.full_load_pwr = full_load_pwr  if not None else self.cpu_config.HP_PROLIANT[0]
        self.idle_pwr = idle_pwr if not None else self.cpu_config.HP_PROLIANT[1]

        self.m_cpu = None
        self.c_cpu = None
        self.m_itfan = None
        self.c_itfan = None
        self.cpu_curve1()
        self.itfan_curve2()
        
        self.v_fan = None  # needed later for calculating outlet temperature
        self.itfan_v_ratio_at_inlet_temp = None
        self.total_DC_full_load = None

    def cpu_curve1(self,):
        """
        initialize the  cpu power ratio curve at different IT workload ratios as a function of inlet temperatures [3]
        """
        # curve parameters at lowest ITE utilization 0%
        self.m_cpu  = (self.cpu_config.CPU_POWER_RATIO_UB[0]-self.cpu_config.CPU_POWER_RATIO_LB[0])/(self.cpu_config.INLET_TEMP_RANGE[1]-self.cpu_config.INLET_TEMP_RANGE[0])
        self.c_cpu  = self.cpu_config.CPU_POWER_RATIO_UB[0] - self.m_cpu*self.cpu_config.INLET_TEMP_RANGE[1]
        # max vertical shift in power ratio curve at a given point in inlet temperature for 100% change in ITE input load pct
        self.ratio_shift_max_cpu = self.cpu_config.CPU_POWER_RATIO_LB[1] - self.cpu_config.CPU_POWER_RATIO_LB[0]

    def itfan_curve2(self,):
        """
        initialize the itfan velocity ratio curve at different IT workload ratios as a function of inlet temperatures [3]
        """
        # curve parameters at ITE utilization 25%
        self.m_itfan = (self.cpu_config.IT_FAN_AIRFLOW_RATIO_UB[0]-self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[0])/(self.cpu_config.INLET_TEMP_RANGE[1]-self.cpu_config.INLET_TEMP_RANGE[0])
        self.c_itfan =  self.cpu_config.IT_FAN_AIRFLOW_RATIO_UB[0] - self.m_itfan*self.cpu_config.INLET_TEMP_RANGE[1]
        # max vertical shift in fan flow ratio curve at a given point for 75% change in ITE input load pct
        self.ratio_shift_max_itfan = self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[1] - self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[0]


class Rack():

    def __init__(self, CPU_config_list, max_W_per_rack = 10000, rack_config = None):  # [3] Table 2 Mid-tier data center
        """Defines the rack as a collection of CPUs

        Args:
            CPU_config_list (config): CPU configuration
            max_W_per_rack (int): Maximun power allowed for a whole rack. Defaults to 10000.
            rack_config (config): Rack configuration. Defaults to None.
        """
        
        self.rack_config = rack_config
        
        self.CPU_list = []
        self.current_rack_load = 0
        for CPU_config in CPU_config_list:
            self.CPU_list.append(CPU(full_load_pwr = CPU_config['full_load_pwr'], idle_pwr = CPU_config['idle_pwr'], cpu_config = self.rack_config))
            self.current_rack_load += self.CPU_list[-1].full_load_pwr
            if self.current_rack_load>= max_W_per_rack:
                rejected = self.CPU_list.pop()
                if getattr(self.rack_config, 'sizing_mode', 'legacy') == 'scaling':
                    self.current_rack_load -= rejected.full_load_pwr
                break
            
        self.num_CPUs = len(self.CPU_list)
        self.cpu_and_fan_init()
        self.v_fan_rack = None  # will be later pointing to a 1d np array whose shape is the number of cpus in the rack class
  
    def cpu_and_fan_init(self,):
        """
        Initialize the CPU and Fan parameters for the servers in each rack with the specified data center configurations
        """
        
        #common to both cpu and fan 
        inlet_temp_lb,inlet_temp_ub =[],[]
        
        # only for cpu
        m_cpu,c_cpu,ratio_shift_max_cpu,\
            idle_pwr, full_load_pwr = [],[],[],[],[]
            
        # only for it fan
        m_itfan,c_itfan,ratio_shift_max_itfan,\
        ITFAN_REF_P,ITFAN_REF_V_RATIO,IT_FAN_FULL_LOAD_V = \
            [],[],[],[],[],[]
            
        for CPU_item in self.CPU_list:
            
            #common to both cpu and fan
            inlet_temp_lb.append(CPU_item.cpu_config.INLET_TEMP_RANGE[0])
            inlet_temp_ub.append(CPU_item.cpu_config.INLET_TEMP_RANGE[1])
            
            # only for cpu
            m_cpu.append(CPU_item.m_cpu)
            c_cpu.append(CPU_item.c_cpu)
            ratio_shift_max_cpu.append(CPU_item.ratio_shift_max_cpu)
            idle_pwr.append(CPU_item.idle_pwr)
            full_load_pwr.append(CPU_item.full_load_pwr)
            
            # only for itfan
            m_itfan.append(CPU_item.m_itfan)
            c_itfan.append(CPU_item.c_itfan)
            ratio_shift_max_itfan.append(CPU_item.ratio_shift_max_itfan)
            ITFAN_REF_P.append(CPU_item.cpu_config.ITFAN_REF_P)
            ITFAN_REF_V_RATIO.append(CPU_item.cpu_config.ITFAN_REF_V_RATIO)
            IT_FAN_FULL_LOAD_V.append(CPU_item.cpu_config.IT_FAN_FULL_LOAD_V)
            
            
        # commont to both cpu and itfan
        self.inlet_temp_lb,self.inlet_temp_ub =np.array(inlet_temp_lb),\
                                    np.array(inlet_temp_ub)
        
        # only for cpu
        self.m_cpu,self.c_cpu,self.ratio_shift_max_cpu,\
            self.idle_pwr, self.full_load_pwr = \
                                    np.array(m_cpu),np.array(c_cpu),\
                                    np.array(ratio_shift_max_cpu),\
                                    np.array(idle_pwr), np.array(full_load_pwr)
        
        # only for itfan
        self.m_itfan,self.c_itfan,self.ratio_shift_max_itfan,\
        self.ITFAN_REF_P,self.ITFAN_REF_V_RATIO,self.IT_FAN_FULL_LOAD_V = \
            np.array(m_itfan),np.array(c_itfan),np.array(ratio_shift_max_itfan),\
            np.array(ITFAN_REF_P),np.array(ITFAN_REF_V_RATIO),np.array(IT_FAN_FULL_LOAD_V) 
            
    def compute_instantaneous_pwr(self,inlet_temp, ITE_load_pct):
        """Calculate the power consumption of the whole rack at the current step

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage

        Returns:
            cpu_power (float): Current CPU power usage
        """
        # Assuming that all CPUs have the same parameters
        cpu = self.CPU_list[0]
        tot_cpu_pwr = cpu.compute_instantaneous_cpu_pwr(inlet_temp, ITE_load_pct)*self.num_CPUs

        tot_itfan_pwr = []
        for CPU_item in self.CPU_list:
            tot_itfan_pwr.append(CPU_item.compute_instantaneous_fan_pwr(inlet_temp, ITE_load_pct))

        return tot_cpu_pwr, np.array(tot_itfan_pwr).sum()

    def compute_instantaneous_pwr_vecd(self, inlet_temp, ITE_load_pct):
        """Calculate the power consumption of the whole rack at the current step in a vectorized manner

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage
        Returns:
            cpu_power (float): Current CPU power usage
        """
        
        # CPU
        base_cpu_power_ratio = self.m_cpu * inlet_temp + self.c_cpu
        cpu_power_ratio_at_inlet_temp = base_cpu_power_ratio + self.ratio_shift_max_cpu*(ITE_load_pct/100)
        temp_arr = np.concatenate((self.idle_pwr.reshape(1,-1),
                                   (self.full_load_pwr*cpu_power_ratio_at_inlet_temp).reshape(1,-1)),
                                  axis=0)
        cpu_power = np.max(temp_arr,axis=0)
        
        # itfan
        low_utilization_ratio = self.m_itfan * inlet_temp + self.c_itfan
        inlet_fraction = (inlet_temp - self.inlet_temp_lb) / (self.inlet_temp_ub - self.inlet_temp_lb)
        high_utilization_ratio = (
            self.rack_config.IT_FAN_AIRFLOW_RATIO_LB[1]
            + (self.rack_config.IT_FAN_AIRFLOW_RATIO_UB[1] - self.rack_config.IT_FAN_AIRFLOW_RATIO_LB[1]) * inlet_fraction
        )
        utilization_fraction = np.clip((ITE_load_pct - 25.0) / 75.0, 0.0, 1.0)
        itfan_v_ratio_at_inlet_temp = np.clip(
            low_utilization_ratio + (high_utilization_ratio - low_utilization_ratio) * utilization_fraction,
            0.0, 1.0,
        )
        itfan_pwr = self.ITFAN_REF_P * (itfan_v_ratio_at_inlet_temp/self.ITFAN_REF_V_RATIO)  # [4] Eqn (3)
        self.v_fan_rack = self.IT_FAN_FULL_LOAD_V*itfan_v_ratio_at_inlet_temp
        
        return np.sum(cpu_power), np.sum(itfan_pwr)

    def get_average_rack_fan_v(self,):
        """Calculate the average fan velocity for each rack

        Returns:
            (float): Average fan flow rate for the rack
        """
            
        return self.v_fan_rack[0]
    
    def get_total_rack_fan_v(self,):
        """Calculate the total fan velocity for each rack

        Returns:
            (float): Total fan flow rate for the rack
        """
        return np.sum(self.v_fan_rack)
    
    
    def get_current_rack_load(self,):
        """Returns the total power consumption of the rack

        Returns:
            float: Total power consumption of the rack
        """
        return self.current_rack_load

    def clamp_supply_approach_temp(self,supply_approach_temperature):
        """Returns the clamped delta/ supply approach temperature between the range [3.8, 5.3]

        Returns:
            float: Supply approach temperature
        """
        return max(3.8, min(supply_approach_temperature, 5.3))
    
class DataCenter_ITModel():

    def __init__(self, num_racks, rack_supply_approach_temp_list, rack_CPU_config, max_W_per_rack = 10000, DC_ITModel_config = None, chiller_sizing = False) -> None:
        """Creates the DC from a giving DC configuration

        Args:
            num_racks (int): Number of racks in the DC
            rack_supply_approach_temp_list (list[float]): models the supply approach temperature for each rack based on geometry and estimated from CFD
            rack_CPU_config (list[list[dict]]): A list of lists where each list is associated with a rack. 
            It is a list of dictionaries with their full load and idle load values in W
            chiller_sizing (bool): Whether to perform Chiller Power Sizing
        """
        self.DC_ITModel_config = DC_ITModel_config
        if len(rack_CPU_config) != num_racks or len(rack_supply_approach_temp_list) != num_racks:
            raise ValueError('Rack configurations and supply temperatures must match num_racks')
        self.racks_list = []
        self.rack_supply_approach_temp_list = rack_supply_approach_temp_list
        self.rack_CPU_config = rack_CPU_config
        
        self.rackwise_inlet_temp = []
        
        for _, CPU_config_list in zip(range(num_racks),self.rack_CPU_config):
            self.racks_list.append(Rack(CPU_config_list, max_W_per_rack = max_W_per_rack, rack_config=self.DC_ITModel_config))
        
        self.total_datacenter_full_load()
        
        self.tower_flow_rate = np.round(DC_ITModel_config.CT_WATER_FLOW_RATE * 3600, 4)  # m³/hr
        self.hot_water_temp = None  # °C
        self.cold_water_temp = None  # °C
        self.wet_bulb_temp = None  # °C
        self.cycles_of_concentration = 5
        self.drift_rate = 0.01
        
        
        
    def compute_datacenter_IT_load_outlet_temp(self,ITE_load_pct_list, CRAC_setpoint):
        
        """Calculate the average outlet temperature of all the racks

        Args:
            ITE_load_pct_list (List[float]): CPU load for each rack
            CRAC_setpoint (float): CRAC setpoint

        Returns:
            rackwise_cpu_pwr (List[float]): Rackwise CPU power usage
            rackwise_itfan_pwr (List[float]):  Rackwise IT fan power usage
            rackwise_outlet_temp (List[float]): Rackwise outlet temperature
        """

        rackwise_cpu_pwr = [] 
        rackwise_itfan_pwr = []
        rackwise_outlet_temp = []
        rackwise_inlet_temp = []
                
        for rack, rack_supply_approach_temp, ITE_load_pct \
                                in zip(self.racks_list, self.rack_supply_approach_temp_list, ITE_load_pct_list):
            #clamp supply approach temperatures
            rack_supply_approach_temp = rack.clamp_supply_approach_temp(rack_supply_approach_temp)
            rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint 
            rackwise_inlet_temp.append(rack_inlet_temp)
            rack_cpu_power, rack_itfan_power = rack.compute_instantaneous_pwr_vecd(rack_inlet_temp, ITE_load_pct)
            rackwise_cpu_pwr.append(rack_cpu_power)
            rackwise_itfan_pwr.append(rack_itfan_power)

            airflow_m3_s = rack.get_total_rack_fan_v()
            if airflow_m3_s <= 0:
                raise ValueError('rack airflow must be positive')
            rack_heat_w = rack_cpu_power + rack_itfan_power
            outlet_temp = rack_inlet_temp + rack_heat_w / (
                self.DC_ITModel_config.RHO_AIR * self.DC_ITModel_config.C_AIR * airflow_m3_s
            )
            rackwise_outlet_temp.append(outlet_temp)
        self.rackwise_inlet_temp = rackwise_inlet_temp
            
        return rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp
    
    def total_datacenter_full_load(self,):
        """Calculate the total DC IT(IT CPU POWER + IT FAN POWER) power consumption
        """
        x = [rack_item.get_current_rack_load() for rack_item in self.racks_list]
        self.total_DC_full_load = sum(x)

    def calculate_cooling_tower_water_usage(self):
        """
        Calculate the estimated water usage of the cooling tower.

        This function uses the attributes set in the class to estimate the water usage based 
        [Sharma, R.K., Shah, A., Bash, C.E., Christian, T., & Patel, C.D. (2009). Water efficiency management in datacenters: Metrics and methodology. 2009 IEEE International Symposium on Sustainable Systems and Technology, 1-6.]
        [Mohammed Shublaq, Ahmad K. Sleiti., (2020).  Experimental analysis of water evaporation losses in cooling towers using filters]
        https://spxcooling.com/water-calculator/
        """
        # We're assuming m³/hr, which is standard

        # Calculate the range (difference between hot and cold water temperature)
        range_temp = self.hot_water_temp - self.cold_water_temp
        
        y_intercept = 0.3528 * range_temp + 0.101
        
        # The water usage estimation formula would need to be derived from the graph you provided.
        
        norm_water_usage = 0.044 * self.wet_bulb_temp + y_intercept
        
        water_usage = np.clip(norm_water_usage, 0, None)
        
        water_usage += water_usage * self.drift_rate  # adjust for drift

        # Convert m³/hr to the desired unit (e.g., liters per 15 minutes) if necessary
        # There are 1000 liters in a cubic meter. There are 4 15-minute intervals in an hour.
        water_usage_liters_per_15min = np.round((water_usage * 1000) / 4, 4)

        return water_usage_liters_per_15min


def chiller_available_cooling_capacity(max_cooling_cap, ambient_temp):
    """Return the chiller's temperature-adjusted thermal capacity in watts."""
    # Air-cooled Chiller:Electric example at fixed 6.67 C chilled-water outlet.
    # Ambient is the condenser inlet temperature for that example.
    delta_temp = (ambient_temp - 35.0) / 2.778
    capacity_ratio = 0.9949 - 0.045954 * delta_temp - 0.0013543 * delta_temp**2
    return max(0.0, max_cooling_cap * capacity_ratio)


def calculate_chiller_power(max_cooling_cap, load, ambient_temp):
    """
    Calculate the chiller power consumption based on load and operating conditions.
    
    Obtained from:
        1) https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/testfiles/AirCooledElectricChiller.idf
        2) https://github.com/NREL/EnergyPlus/issues/763
        3) https://dmey.github.io/EnergyPlusFortran-Reference/proc/calcelectricchillermodel.html
        4) https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/tst/EnergyPlus/unit/ChillerElectric.unit.cc#L95

    Args:
        max_cooling_cap (float): Maximum cooling capacity of the chiller (Watts).
        load (float): The heat load to be removed by the chiller (Watts).
        ambient_temp (float): Current ambient temperature (Celsius).
        
    Returns:
        float: Estimated power consumption of the chiller (Watts).
    """
    
    # Coefficients from https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/tst/EnergyPlus/unit/ChillerElectric.unit.cc#L95
    power_coefficients = [2.333, -1.975, 0.6121]
    full_load_factor = [0.03303, 0.6852, 0.2818]

    # Chiller design specifications
    min_plr = 0.05
    max_plr = 1.0
    rated_cop = 2.58  # AirCooledElectricChiller.idf nominal COP

    available_capacity = chiller_available_cooling_capacity(max_cooling_cap, ambient_temp)
    avail_nom_cap_rat = available_capacity / max_cooling_cap if max_cooling_cap > 0 else 0.0

    # Calculate power ratio
    full_power_ratio = power_coefficients[0] + power_coefficients[1] * avail_nom_cap_rat + power_coefficients[2] * avail_nom_cap_rat**2
    
    # Determine part load ratio (PLR)
    part_load_ratio = max(min_plr, min(load / available_capacity, max_plr)) if available_capacity > 0 else 0
    
    # Calculate fractional full load power
    frac_full_load_power = full_load_factor[0] + full_load_factor[1] * part_load_ratio + full_load_factor[2] * part_load_ratio**2
    
    # Determine operational part load ratio (OperPartLoadRat)
    # If the PLR is less than Min PLR calculate the actual PLR for calculations. The power will then adjust for the cycling.
    if available_capacity > 0:
        if load / available_capacity < min_plr:
            oper_part_load_rat = load / available_capacity
        else:
            oper_part_load_rat = part_load_ratio
    else:
        oper_part_load_rat = 0.0
    
    # Operational PLR for actual conditions
    if oper_part_load_rat < min_plr:
        frac = min(1.0, oper_part_load_rat / min_plr)
    else:
        frac = 1.0

    # Calculate the chiller compressor power
    power = frac_full_load_power * full_power_ratio * available_capacity / rated_cop * frac

    #  Total heat rejection is the sum of the cooling capacity and the power input
    total_heat_rejection = load + power

    return power if oper_part_load_rat > 0 else 0


def calculate_HVAC_power(CRAC_setpoint, avg_CRAC_return_temp, ambient_temp, data_center_full_load, DC_Config, ctafr=None):
    """Calculate the HVAV power attributes

        Args:
            CRAC_Setpoint (float): The control action
            avg_CRAC_return_temp (float): The average of the temperatures from all the Racks + their corresponding return approach temperature (Delta)
            ambient_temp (float): outside air temperature
            data_center_full_load (float): total data center capacity

        Returns:
            CRAC_Fan_load (float): CRAC fan power
            CT_Fan_pwr (float):  Cooling tower fan power
            CRAC_cooling_load (float): CRAC cooling load
            Compressor_load (float): Chiller compressor load
        """
    # Air system calculations
    # In steady state the cooling plant removes the electrical heat dissipated
    # by the modeled servers and their fans. The approach temperatures do not
    # define an independent CRAC air flow or a heat-recovery fraction.
    CRAC_cooling_load = data_center_full_load
    CRAC_Fan_load = DC_Config.CRAC_FAN_REF_P * (DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu / DC_Config.CRAC_REFRENCE_AIR_FLOW_RATE_pu)**3
    
    chiller_power = calculate_chiller_power(DC_Config.CHILLER_NOMINAL_COOLING_CAPACITY_W, CRAC_cooling_load, ambient_temp)

    # The evaporator still circulates chilled water, while the air-cooled
    # condenser needs neither a cooling tower nor a condenser-water pump.
    chilled_water_pump_w = (DC_Config.CW_PRESSURE_DROP * DC_Config.CW_WATER_FLOW_RATE
                            / DC_Config.CW_PUMP_EFFICIENCY)
    return CRAC_Fan_load, 0.0, CRAC_cooling_load, chiller_power, chilled_water_pump_w, 0.0

def chiller_sizing(DC_Config, min_CRAC_setpoint=16, max_CRAC_setpoint=22, max_ambient_temp=40.0):
    '''
    Calculates the chiller sizing for a data center based on the given configuration and parameters.
    
    Parameters:
        DC_Config (object): The data center configuration object.
        min_CRAC_setpoint (float): The minimum CRAC setpoint temperature in degrees Celsius. Default is 16.
        max_CRAC_setpoint (float): The maximum CRAC setpoint temperature in degrees Celsius. Default is 22.
        max_ambient_temp (float): The maximum ambient temperature in degrees Celsius. Default is 40.0.
    
    Returns:
        tuple: Cooling tower reference airflow (m³/s) and nominal chiller cooling capacity (thermal W).
    '''
    dc = DataCenter_ITModel(num_racks=DC_Config.NUM_RACKS,
                            rack_supply_approach_temp_list=DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                            rack_CPU_config=DC_Config.RACK_CPU_CONFIG,
                            max_W_per_rack=DC_Config.MAX_W_PER_RACK,
                            DC_ITModel_config=DC_Config)
    
    cpu_load = 100.0
    ITE_load_pct_list = [cpu_load for i in range(DC_Config.NUM_RACKS)]
        
    rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp = \
                        dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=max_CRAC_setpoint)
                        
    avg_CRAC_return_temp = calculate_avg_CRAC_return_temp(rack_return_approach_temp_list=DC_Config.RACK_RETURN_APPROACH_TEMP_LIST,
                                                                         rackwise_outlet_temp=rackwise_outlet_temp)
    
    data_center_total_ITE_Load = sum(rackwise_cpu_pwr) + sum(rackwise_itfan_pwr)
    
    CRAC_cooling_load = data_center_total_ITE_Load
    Cooling_tower_air_delta = max(50 - (max_ambient_temp-min_CRAC_setpoint), 1)  
    
    m_air = CRAC_cooling_load/(DC_Config.C_AIR*Cooling_tower_air_delta) 
    
    # Cooling Tower Reference Air FlowRate
    ctafr = m_air/DC_Config.RHO_AIR
    
    '''
    # Assuming that the HVAC consumes 43% of energy on a datacenter, and the rest is IT
    # We assume that on months where temperature is lower than 15C, the HVAC energy consumption is almost 0. (> 60% of time)
    # Also, the HVAC energy consumption follow the external temperature trend (almost a sin signal).
    # The average area for a sin period of a sin is 2*pi
    # The average power consumption of a day where the maximum temperature is raised and the lowest setpoint is used (MaxPower) is 2*pi*MaxPower [W]
    # The average energy for that day is (2*pi*MaxPower)[W]/24[hours] -> 0.26 * MAxPower [W/h]
    # So, to obtain an average HVAC energy consumption of 43%, we need to scale the total maximum energy consumption with a factor of 10
    # This value is obtained after a methodic literature search.
    '''
    
    capacity_ratio = chiller_available_cooling_capacity(1.0, max_ambient_temp)
    if capacity_ratio <= 0:
        raise ValueError('Chiller capacity curve has no available cooling at design ambient')
    nominal_chiller_cooling_capacity_w = CRAC_cooling_load / capacity_ratio

    return ctafr, nominal_chiller_cooling_capacity_w
    
def calculate_avg_CRAC_return_temp(rack_return_approach_temp_list,rackwise_outlet_temp):   
    """Calculate the CRAC return air temperature

        Args:
            rack_return_approach_temp_list (List[float]): The delta change in temperature from each rack to the CRAC unit
            rackwise_outlet_temp (float): The outlet temperature of each rack
        Returns:
            (float): CRAC return air temperature
        """
    n = len(rack_return_approach_temp_list)
    return sum([i + j for i,j in zip(rack_return_approach_temp_list,rackwise_outlet_temp)])/n  # CRAC return is averaged across racks


"""
References:
[1]: Postema, Björn Frits. "Energy-efficient data centres: model-based analysis of power-performance trade-offs." (2018).
[2]: Raghunathan, S., & Vk, M. (2014). Power management using dynamic power state transitions and dynamic voltage frequency
     scaling controls in virtualized server clusters. Turkish Journal of Electrical Engineering and Computer Sciences, 24(4). doi: 10.3906/elk-1403-264
[3]: Sun, Kaiyu, et al. "Prototype energy models for data centers." Energy and Buildings 231 (2021): 110603.
[4]: Breen, Thomas J., et al. "From chip to cooling tower data center modeling: Part I influence of server inlet temperature and temperature 
     rise across cabinet." 2010 12th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems. IEEE, 2010.
[5]: https://h2ocooling.com/blog/look-cooling-tower-fan-efficiences/#:~:text=The%20tower%20has%20been%20designed,of%200.42%20inches%20of%20water.
"""