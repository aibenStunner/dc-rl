"""CPU workload trace manager."""
import numpy as np
import pandas as pd

from .common import CoherentNoise, REPO_ROOT, _cyc, _cyc_at

PATH = str(REPO_ROOT)


class Workload_Manager():
    def __init__(self, workload_filename='', init_day=0, future_steps=4, weight=0.005, desired_std_dev=0.01, timezone_shift=0, debug=False):
        """Manager of the DC workload.

        Args:
            workload_filename (str, optional): Filename of the CPU data. Defaults to ''. Should be a .csv file containing the CPU hourly normalized workload data between 0 and 1. Should contain 'cpu_load' column.
            init_day (int, optional): Initial day of the episode. Defaults to 0.
            future_steps (int, optional): Number of steps of the workload forecast. Defaults to 4.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.01.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.025.
            timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
        """
        
        # Load CPU data from a CSV file
        # One year data=24*365=8760
        if workload_filename == '':
            cpu_data_list = pd.read_csv(PATH+'/data/Workload/Alibaba_CPU_Data_Hourly_1.csv')['cpu_load'].values[:8760]
        else:
            cpu_data_list = pd.read_csv(PATH+f'/data/Workload/{workload_filename}')['cpu_load'].values[:8760]

        assert len(cpu_data_list) == 8760, "The number of data points in the workload data is not one year data=24*365=8760."

        cpu_data_list = cpu_data_list.astype(float)
        self.time_step = 0
        self.future_steps = future_steps
        self.timestep_per_hour = 4
        self.time_steps_day = self.timestep_per_hour*24
        self.init_day = init_day
        self.timezone_shift = timezone_shift

        # Interpolate the CPU data to increase the number of data points
        x = range(0, len(cpu_data_list))
        xcpu_new = np.linspace(0, len(cpu_data_list), len(cpu_data_list)*self.timestep_per_hour)  
        self.cpu_smooth = np.interp(xcpu_new, x, cpu_data_list)
        
        # Shift the data to match the timezone shift
        self.cpu_smooth =  np.roll(self.cpu_smooth, -1*self.timezone_shift*self.timestep_per_hour)
        
        # Save a copy of the original data
        self.original_data = self.cpu_smooth.copy()
                
        # Initialize CoherentNoise process
        self.coherent_noise = CoherentNoise(base=0, weight=weight, desired_std_dev=desired_std_dev)
        
        # Debug mode
        self.debug = debug

    def smooth_workload(self, window_size=3):
        """Apply moving average to smooth out workload changes.

        Args:
            window_size (int): The size of the moving window. Defaults to 3.

        Returns:
            np.array: Smoothed workload data.
        """
        return np.convolve(self.cpu_smooth, np.ones(window_size) / window_size, mode='same')


    # Function to return all workload data
    def get_total_wkl(self):
        """Get current workload

        Returns:
            List[float]: CPU data
        """
        return np.array(self.cpu_smooth[self.time_step:])

    def scale_array(self, arr):
        """
        Scales the input array so that approximately 90% of its values
        fall within the range of 0.2 to 0.8, based on the 5th and 95th percentiles.
        
        Parameters:
        arr (np.array): The input numpy array to be scaled.
        
        Returns:
        np.array: The scaled numpy array.
        """
        
        # Calculate the 5th and 95th percentiles of the array
        p5 = np.percentile(arr, 5)
        p95 = np.percentile(arr, 95)
        
        # Scale the array based on the percentiles, without clipping
        # This ensures values outside the 5th to 95th percentile range naturally
        # fall outside the 0.2 to 0.8 range.
        scaled_arr = 0.2 + ((arr - p5) * (0.8 - 0.2) / (p95 - p5))
        
        # Clip values to be within 0 to 1
        scaled_arr = np.clip(scaled_arr, 0, 1)
        
        return scaled_arr

    # Function to reset the time step and return the workload at the first time step
    def reset(self, init_day=None, init_hour=None):
        """Reset Workload_Manager to a specific initial day and hour.

        Args:
            init_day (int, optional): Day to start from. If None, defaults to the initial day set during initialization.
            init_hour (int, optional): Hour to start from. If None, defaults to 0.

        Returns:
            float: CPU workload at current time step.
        """
        self.time_step = (init_day if init_day is not None else self.init_day) * self.time_steps_day + (init_hour if init_hour is not None else 0) * self.timestep_per_hour
        self.init_time_step = self.time_step
        
        baseline = np.random.random()*0.5 - 0.25
        
        # if not self.debug:
        # Add noise to the workload data using the CoherentNoise 
        cpu_data = self.original_data# * np.random.uniform(0.95, 1.05, len(self.original_data))
        # print(f'Check the original cpu data: {self.original_data} and the coherent noise: {self.coherent_noise.generate(len(cpu_data))}')
        cpu_smooth = cpu_data# * 0.7 + self.coherent_noise.generate(len(cpu_data)) * 0.3 + baseline
        
        self.cpu_smooth = self.scale_array(cpu_smooth)
        
        # Apply smoothing method
        self.cpu_smooth = self.smooth_workload(window_size=16)
        
        # num_roll_weeks = np.random.randint(0, 52) # Random roll the workload because is independed on the month, so I am rolling across weeks (52 weeks in a year)
        # self.cpu_smooth =  np.roll(self.cpu_smooth, num_roll_weeks*self.timestep_per_hour*24*7)
        
        # else:
            # Fixed utilization for debugging using 60% of the CPU
            # self.cpu_smooth = np.ones_like(self.cpu_smooth) * 0.6

        self._current_workload = self.cpu_smooth[self.time_step]
        self._next_workload = _cyc_at(self.cpu_smooth, self.time_step + 1)
        return self._current_workload
        
    # Function to advance the time step and return the workload at the new time step
    def step(self):
        """Step function for the Workload_Manager

        Returns:
            float: CPU workload at current time step
            float: Amount of daily flexible workload
        """
        self.time_step += 1
        
        # If it tries to read further, restart from the inital day
        if self.time_step >= len(self.cpu_smooth):
            self.time_step = self.init_time_step
        
        self._current_workload = self.cpu_smooth[self.time_step]
        self._next_workload = _cyc_at(self.cpu_smooth, self.time_step + 1)
        
        # assert self.time_step < len(self.cpu_smooth), f'Episode length: {self.time_step} is longer than the provide cpu_smooth: {len(self.cpu_smooth)}'
        return self._current_workload  # to avoid logical error
    
    def get_current_workload(self):
        return self._current_workload

    def get_next_workload(self):
        return self._next_workload
    
    def set_current_workload(self, workload):         
        self.cpu_smooth[self.time_step] = workload
        
    def get_n_next_workloads(self, n):
        return _cyc(self.cpu_smooth, self.time_step + 1, n)


# Class to manage carbon intensity data
