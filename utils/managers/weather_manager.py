"""Weather and wet-bulb trace manager."""
import numpy as np
import pandas as pd
import psychrolib as psy

from .common import CoherentNoise, REPO_ROOT, _cyc, _cyc_at, normalize

PATH = str(REPO_ROOT)
# Preserve the previous module-import side effect before wet-bulb calculation.
psy.SetUnitSystem(psy.SI)


class Weather_Manager():
    """Manager of the weather data.
       Where to obtain other weather files:
       https://climate.onebuilding.org/

    Args:
        filename (str, optional): Filename of the weather data. Defaults to ''.
        location (str, optional): Location identifier. Defaults to 'NY'.
        init_day (int, optional): Initial day of the year. Defaults to 0.
        weight (float, optional): Weight value for coherent noise. Defaults to 0.02.
        desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.75.
        temp_column (int, optional): Column that contains the temperature data. Defaults to 6.
        rh_column (int, optional): Column that contains the relative humidity data. Defaults to 8.
        pres_column (int, optional): Column that contains the pressure data. Defaults to 9.
        timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
    """
    def __init__(self, filename='', location='NY', init_day=0, weight=0.02, desired_std_dev=0.75, temp_column=6, rh_column=8, pres_column=9, timezone_shift=0, debug=False, randomize=True):
        """Initialize the Weather_Manager class.

        Args:
            filename (str, optional): Filename of the weather data. Defaults to ''.
            location (str, optional): Location identifier. Defaults to 'NY'.
            init_day (int, optional): Initial day of the year. Defaults to 0.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.02.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.75.
            temp_column (int, optional): Column that contains the temperature data. Defaults to 6.
            rh_column (int, optional): Column that contains the relative humidity data. Defaults to 8.
            pres_column (int, optional): Column that contains the pressure data. Defaults to 9.
            timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
        """
        # Load weather data from a CSV file

        if not location == '':
            weather_data = pd.read_csv(PATH+f'/data/Weather/{location}', skiprows=8, header=None).values
        else:
            weather_data = pd.read_csv(PATH+f'/data/Weather/{filename}', skiprows=8, header=None).values
        
        # The weather data has a granularity of 1 hour, so we need to interpolate it to have a granularity of 15 minutes
        temperature_data = weather_data[:,temp_column].astype(float)
        relative_humidity_data = weather_data[:,rh_column].astype(float)  # Added for relative humidity
        pressure_data = weather_data[:,pres_column].astype(float)  # Added for atmospheric pressure
                
        self.wet_bulb_data = [psy.GetTWetBulbFromRelHum(t, rh / 100, p) for t, rh, p in zip(temperature_data, relative_humidity_data, pressure_data)]

        # Normalize wet bulb temperature data
        self.min_wb_temp = 0
        self.max_wb_temp = 45

        self.init_day = init_day
        # One year data=24*365=8760
        x = range(0, len(temperature_data))
        self.timestep_per_hour = 4

        xtemperature_new = np.linspace(0, len(temperature_data), len(temperature_data)*self.timestep_per_hour )
        
        self.min_temp = 0
        self.max_temp = 45
        
        # Interpolate the data to increase the number of data points
        self.wet_bulb_data = np.interp(xtemperature_new, x, self.wet_bulb_data)
        self.norm_wet_bulb_data = normalize(self.wet_bulb_data, self.min_wb_temp, self.max_wb_temp)

        self.temperature_data = np.interp(xtemperature_new, x, temperature_data)
        self.norm_temp_data = normalize(self.temperature_data, self.min_temp, self.max_temp)

        self.time_step = 0
        self.timezone_shift = timezone_shift
        
        # Shift the data to match the timezone shift
        self.temperature_data =  np.roll(self.temperature_data, -1*self.timezone_shift*self.timestep_per_hour)
        self.wet_bulb_data =  np.roll(self.wet_bulb_data, -1*self.timezone_shift*self.timestep_per_hour)

        # Save a copy of the original data
        self.original_temp_data = self.temperature_data.copy()
        self.original_wb_data = self.wet_bulb_data.copy()

        # Initialize CoherentNoise process
        self.coherent_noise = CoherentNoise(base=0, weight=weight, desired_std_dev=desired_std_dev)
                
        self.time_steps_day = self.timestep_per_hour*24
        
        self.debug = debug
        if type(randomize) is not bool:
            raise ValueError("weather.randomize must be a boolean")
        self.randomize = randomize

    # Function to return all weather data
    def get_total_weather(self):
        """Obtain the weather data in a List form

        Returns:
            List[form]: Total temperature data
        """
        return self.temperature_data[self.time_step:]

    # Function to reset the time step and return the weather at the first time step
    def reset(self, init_day=None, init_hour=None):
        """Reset Weather_Manager to a specific initial day and hour.

        Args:
            init_day (int, optional): Day to start from. If None, defaults to the initial day set during initialization.
            init_hour (int, optional): Hour to start from. If None, defaults to 0.

        Returns:
            tuple: Temperature at current step, normalized temperature at current step, wet bulb temperature at current step, normalized wet bulb temperature at current step.
        """

        self.time_step = (init_day if init_day is not None else self.init_day) * self.time_steps_day + (init_hour if init_hour is not None else 0) * self.timestep_per_hour
        
        if not self.debug and self.randomize:
            # Add noise to the temperature data using the CoherentNoise
            coh_noise = self.coherent_noise.generate(len(self.original_temp_data))
            # print(f'TODO: check the generated coherent noise: {coh_noise[:3]} and the original temperature data: {self.original_temp_data[:3]} and the wet bulb data: {self.original_wb_data[:3]}' )
            self.temperature_data = self.original_temp_data + coh_noise
            self.wet_bulb_data = self.original_wb_data + coh_noise
            
            num_roll_days = np.random.randint(0, 14) # Random roll the temperature some days.
            self.temperature_data =  np.roll(self.temperature_data, num_roll_days*self.timestep_per_hour*24)
            self.wet_bulb_data =  np.roll(self.wet_bulb_data, num_roll_days*self.timestep_per_hour*24)

            self.temperature_data = np.clip(self.temperature_data, self.min_temp, self.max_temp)
            max_30_days = np.max(self.temperature_data[self.time_step:30*self.time_steps_day + self.time_step])
            min_30_days = np.min(self.temperature_data[self.time_step:30*self.time_steps_day + self.time_step])
            self.norm_temp_data = (self.temperature_data - min_30_days) / (max_30_days - min_30_days)
            
            self.wet_bulb_data = np.clip(self.wet_bulb_data, self.min_wb_temp, self.max_wb_temp)
            max_30_days = np.max(self.wet_bulb_data[self.time_step:30*self.time_steps_day + self.time_step])
            min_30_days = np.min(self.wet_bulb_data[self.time_step:30*self.time_steps_day + self.time_step])
            self.norm_wet_bulb_data = (self.wet_bulb_data - min_30_days) / (max_30_days - min_30_days)
            
        elif self.debug:
            # Use a fixed temperature for debugging and wet bulb temperature
            self.temperature_data = np.ones_like(self.temperature_data) * 30
            self.norm_temp_data = np.ones_like(self.norm_temp_data) * 0.5
            self.wet_bulb_data = np.ones_like(self.wet_bulb_data) * 25
            
        self._current_temp = self.temperature_data[self.time_step]
        self._next_temp = _cyc_at(self.temperature_data, self.time_step + 1)
        self._current_norm_temp = self.norm_temp_data[self.time_step]
        self._next_norm_temp = _cyc_at(self.norm_temp_data, self.time_step + 1)
        self._current_wet_bulb = self.wet_bulb_data[self.time_step]
        self._current_norm_wet_bulb = self.norm_wet_bulb_data[self.time_step]
        
        return self._current_temp, self._current_norm_temp, self._current_wet_bulb, self._current_norm_wet_bulb

    
    
    # Function to advance the time step and return the weather at the new time step
    def step(self):
        """Step on the Weather_Manager

        Returns:
            float: Temperature a current step
            float: Normalized temperature a current step
        """
                
        self.time_step += 1
        
        # If it tries to read further, restart from the initial index
        if self.time_step >= len(self.temperature_data):
            self.time_step = self.init_day*self.time_steps_day
            
        self._current_temp = self.temperature_data[self.time_step]
        self._next_temp = _cyc_at(self.temperature_data, self.time_step + 1)
        self._current_norm_temp = self.norm_temp_data[self.time_step]
        self._next_norm_temp = _cyc_at(self.norm_temp_data, self.time_step + 1)
        self._current_wet_bulb = self.wet_bulb_data[self.time_step]
        self._current_norm_wet_bulb = self.norm_wet_bulb_data[self.time_step]
            
        return self._current_temp, self._current_norm_temp, self._current_wet_bulb, self._current_norm_wet_bulb
    
    def get_current_temperature(self):
        return self._current_norm_temp
    
    def get_next_temperature(self):
        return self._next_norm_temp
    
    def get_n_next_temperature(self, n):
        return _cyc(self.norm_temp_data, self.time_step + 1, n)
    
    def get_current_wet_bulb(self):
        return self._current_wet_bulb