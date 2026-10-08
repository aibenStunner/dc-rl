"""Carbon-intensity trace manager."""
import numpy as np
import pandas as pd

from .common import CoherentNoise, REPO_ROOT, _cyc, _cyc_at

PATH = str(REPO_ROOT)


class CI_Manager():
    """Manager of the carbon intensity data.

    Args:
        filename (str, optional): Filename of the carbon intensity data. Defaults to ''.
        location (str, optional): Location identifier. Defaults to 'NYIS'.
        init_day (int, optional): Initial day of the episode. Defaults to 0.
        future_steps (int, optional): Number of steps of the CI forecast. Defaults to 4.
        weight (float, optional): Weight value for coherent noise. Defaults to 0.1.
        desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 5.
        timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
    """
    def __init__(self, filename='', location='NYIS', init_day=0, future_steps=4, weight=0.1, desired_std_dev=5, timezone_shift=0, debug=False):
        """Initialize the CI_Manager class.

        Args:
            filename (str, optional): Filename of the carbon intensity data. Defaults to ''.
            location (str, optional): Location identifier. Defaults to 'NYIS'.
            init_day (int, optional): Initial day of the episode. Defaults to 0.
            future_steps (int, optional): Number of steps of the CI forecast. Defaults to 4.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.1.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 5.
            timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
        """
        # Load carbon intensity data from a CSV file
        # One year data=24*365=8760
        if not location == '':
            carbon_data_list = pd.read_csv(PATH+f"/data/CarbonIntensity/{location}_NG_&_avgCI.csv")['avg_CI'].values[:8760]
        else:
            carbon_data_list = pd.read_csv(PATH+f"/data/CarbonIntensity/{filename}")['avg_CI'].values[:8760]

        assert len(carbon_data_list) == 8760, "The number of data points in the carbon intensity data is not one year data=24*365=8760."
        self.debug = debug
        carbon_data_list = carbon_data_list.astype(float)
        self.init_day = init_day
        self.timezone_shift = timezone_shift

        self.timestep_per_hour = 4
        self.time_steps_day = self.timestep_per_hour*24
        
        # Handle nan values just in case. Replace with average value
        if np.isnan(carbon_data_list).any():
            avg_value = np.nanmean(carbon_data_list)
            carbon_data_list = np.nan_to_num(carbon_data_list, nan=avg_value)
        
        # If self.debug is True, replace the carbon intensity data with a sine wave for testing of 24 timesteps of period but with the same length of the carbon intensity data
        # if self.debug:
            # Create a sine wave with 24 timesteps of period (one day cycle) and the same length as carbon_data_list
            # t = np.arange(len(carbon_data_list)*4)  # Create an array of timesteps equal to the length of the carbon data
            # carbon_data_list = np.sin(2 * np.pi * t / self.time_steps_day) * 0.5 + 0.5  # Create sine wave with period of 24 timesteps (one day)
    
            
        x = range(0, len(carbon_data_list))
        xcarbon_new = np.linspace(0, len(carbon_data_list), len(carbon_data_list)*self.timestep_per_hour)
        
        # Interpolate the carbon data to increase the number of data points
        self.carbon_smooth = np.interp(xcarbon_new, x, carbon_data_list)
        
        # Shift the data to match the timezone shift
        self.carbon_smooth =  np.roll(self.carbon_smooth, -1*self.timezone_shift*self.timestep_per_hour)

        # Save a copy of the original data
        self.original_data = self.carbon_smooth.copy()
        
        self.time_step = 0

        # Initialize CoherentNoise process
        self.coherent_noise = CoherentNoise(base=0, weight=weight, desired_std_dev=desired_std_dev)
        
        self.future_steps = future_steps
        
        
        

    # Function to return all carbon intensity data
    def get_total_ci(self):
        """Function to obtain the total carbon intensity

        Returns:
            List[float]: Total carbon intesity
        """
        return self.carbon_smooth[self.time_step:]

    def reset(self, init_day=None, init_hour=None):
        """Reset CI_Manager to a specific initial day and hour.

        Args:
            init_day (int, optional): Day to start from. If None, defaults to the initial day set during initialization.
            init_hour (int, optional): Hour to start from. If None, defaults to 0.

        Returns:
            float: Carbon intensity at current time step.
            float: Normalized carbon intensity at current time step and its forecast.
        """
        self.time_step = (init_day if init_day is not None else self.init_day) * self.time_steps_day + (init_hour if init_hour is not None else 0) * self.timestep_per_hour

        # Add noise to the carbon data using the CoherentNoise
        self.carbon_smooth = self.original_data# + self.coherent_noise.generate(len(self.original_data))
        
        self.carbon_smooth = np.clip(self.carbon_smooth, 0, None)
        
        # num_roll_days = np.random.randint(0, 14) # Random roll the workload some days. I can roll the carbon intensity up to 14 days.
        # self.carbon_smooth =  np.roll(self.carbon_smooth, num_roll_days*self.timestep_per_hour*24)

        # if self.debug:
            # Expand the range of the carbon intensity data x10
            # self.carbon_smooth = (self.carbon_smooth - min(self.carbon_smooth)*0.3) * 10
        self.min_ci = min(self.carbon_smooth)
        self.max_ci = max(self.carbon_smooth)
        # self.norm_carbon = normalize(self.carbon_smooth, self.min_ci, self.max_ci)
        # self.norm_carbon = (self.carbon_smooth - np.mean(self.carbon_smooth)) / np.std(self.carbon_smooth)
        
        # Normalize the carbon intensity data using the next 30 days of data
        # mean_30_days = np.mean(self.carbon_smooth[self.time_step:30*self.time_steps_day + self.time_step])
        # std_30_days = np.std(self.carbon_smooth[self.time_step:30*self.time_steps_day + self.time_step])
        # self.norm_carbon = (self.carbon_smooth - mean_30_days) / std_30_days
        # Scale the carbon intensity data using the min max calues of the next 30 days
        max_30_days = np.max(self.carbon_smooth[self.time_step:30*self.time_steps_day + self.time_step])
        min_30_days = np.min(self.carbon_smooth[self.time_step:30*self.time_steps_day + self.time_step])
        self.norm_carbon = (self.carbon_smooth - min_30_days) / (max_30_days - min_30_days)
        # self.norm_carbon = (np.clip(self.norm_carbon, -1, 1) + 1) * 0.5
        # if self.debug:
            # self.norm_carbon = self.carbon_smooth
        
        self._current_carbon_smooth = self.carbon_smooth[self.time_step]
        self._next_carbon_smooth = _cyc_at(self.carbon_smooth, self.time_step + 1)
        
        self._current_norm_carbon = self.norm_carbon[self.time_step]
        self._next_norm_carbon = _cyc_at(self.norm_carbon, self.time_step + 1)
        self._forecast_norm_carbon = _cyc(self.norm_carbon, self.time_step + 1, self.future_steps)

        return self._current_norm_carbon, self._forecast_norm_carbon, self._current_carbon_smooth
    
    # Function to advance the time step and return the carbon intensity at the new time step
    def step(self):
        """Step CI_Manager

        Returns:
            float: Carbon intensity at current time step
            float: Normalized carbon intensity at current time step and it's forecast
        """
        self.time_step +=1
        
        # If it tries to read further, restart from the initial index
        if self.time_step >= len(self.carbon_smooth):
            self.time_step = self.init_day*self.time_steps_day
            
        # Renormalize every 24 hours (self.time_steps_day)
        # if self.time_step % self.time_steps_day == 0:
        #     self.renormalize_current_day()

        self._current_carbon_smooth = self.carbon_smooth[self.time_step]
        self._current_norm_carbon = self.norm_carbon[self.time_step]
        self._next_norm_carbon = _cyc_at(self.norm_carbon, self.time_step + 1)
        self._forecast_norm_carbon = _cyc(self.norm_carbon, self.time_step + 1, self.future_steps)

        return self._current_norm_carbon, self._forecast_norm_carbon, self._current_carbon_smooth
    
    def get_current_ci(self):
        return self._current_norm_carbon
    
    def get_forecast_ci(self):
        return self._forecast_norm_carbon
    
    def get_n_past_ci(self, n):
        return _cyc(self.norm_carbon, self.time_step - n, n)

# Class to manage weather data
# Where to obtain other weather files:
# https://climate.onebuilding.org/
