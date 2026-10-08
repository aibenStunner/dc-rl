"""Simulation clock manager and cyclic time encoding."""
import numpy as np

def sc_obs(current_hour, current_day):
    """Generate sine and cosine of the hour and day

    Args:
        current_hour (int): Current hour of the day
        current_day (int): Current day of the year

    Returns:
        List[float]: Sine and cosine of the hour and day
    """
    # Normalize and round the current hour and day
    two_pi = np.pi * 2

    norm_hour = round(current_hour/24, 3) * two_pi
    norm_day = round((current_day)/365, 3) * two_pi
    
    # Calculate cosine and sine values for the current hour and day
    cos_hour = np.cos(norm_hour)*0.5 + 0.5
    sin_hour = np.sin(norm_hour)*0.5 + 0.5
    cos_day = np.cos(norm_day)*0.5 + 0.5
    sin_day = np.sin(norm_day)*0.5 + 0.5
    
    return [cos_hour, sin_hour, cos_day, sin_day]


class Time_Manager():
    """Class to manage the time dimenssion over an episode

        Args:
            init_day (int, optional): Day to start from. Defaults to 0.
            days_per_episode (int, optional): Number of days that an episode would last. Defaults to 30.
            timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
    """
    def __init__(self, init_day=0, days_per_episode=30, timezone_shift=0):
        """Initialize the Time_Manager class.

        Args:
            init_day (int, optional): Day to start from. Defaults to 0.
            days_per_episode (int, optional): Number of days that an episode would last. Defaults to 30.
            timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
        """
        self.init_day = init_day
        self.timestep_per_hour = 4
        self.days_per_episode = days_per_episode
        self.timezone_shift = timezone_shift
        
        # Calculate the total timesteps for the episode based on init_day
        self.simulated_total_timesteps = (self.days_per_episode * 24 * self.timestep_per_hour)
        self.current_timestep = 0

    def reset(self, init_day=None, init_hour=None):
        """Reset the time manager to a specific initial day and hour."""
        self.day = init_day if init_day is not None else self.init_day
        self.hour = init_hour if init_hour is not None else self.timezone_shift

        # Recalculate the current timestep based on day/hour
        self.current_timestep = int(self.day * 24 * self.timestep_per_hour + self.hour * self.timestep_per_hour)
        self.total_timesteps = self.current_timestep + self.simulated_total_timesteps
        return sc_obs(self.hour, self.day)

        
    def step(self):
        """Step function for the time maneger

        Returns:
            List[float]: Current hour and day in sine and cosine form.
            bool: Signal if the episode has reach the end.
        """
        self.current_timestep += 1
        self.hour += 1 / self.timestep_per_hour
        if self.hour >= 24:
            self.hour = 0
            self.day += 1
        return self.day, self.hour, sc_obs(self.hour, self.day), self.isterminal()
    
    def isterminal(self):
        """Function to identify terminal state

        Returns:
            bool: Signals if a state is terminal or not
        """
        return self.current_timestep >= self.total_timesteps



# Class to manage CPU workload data
