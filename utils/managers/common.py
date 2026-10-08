"""Shared utilities for SustainDC data and time managers."""
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
HOURS_PER_YEAR = 365 * 24


class CoherentNoise:
    """Class to add coherent noise to the data.

        Args:
            base (List[float]): Base data
            weight (float): Weight of the noise to be added
            desired_std_dev (float, optional): Desired standard deviation. Defaults to 0.1.
            scale (int, optional): Scale. Defaults to 1.
    """
    def __init__(self, base, weight, desired_std_dev=0.1, scale=1):
        """Initialize CoherentNoise class

        Args:
            base (List[float]): Base data
            weight (float): Weight of the noise to be added
            desired_std_dev (float, optional): Desired standard deviation. Defaults to 0.1.
            scale (int, optional): Scale. Defaults to 1.
        """
        self.base = base
        self.weight = weight
        self.desired_std_dev = desired_std_dev
        self.scale = scale

    def generate(self, n_steps):
        """
        Generate coherent noise 

        Args:
            n_steps (int): Length of the data to generate.

        Returns:
            numpy.ndarray: Array of generated coherent noise.
        """
        steps = np.random.normal(loc=0, scale=self.scale, size=n_steps)
        random_walk = np.cumsum(self.weight * steps)
        normalized_noise = (random_walk / np.std(random_walk)) * self.desired_std_dev
        return self.base + normalized_noise


# Function to normalize a value v given a minimum and a maximum
def normalize(v, min_v, max_v):
    """Function to normalize values

    Args:
        v (float): Value to be normalized
        min_v (float): Lower limit
        max_v (float): Upper limit

    Returns:
        float: Normalized value
    """
    return (v - min_v)/(max_v - min_v)

def _cyc(array, start, n):
    """`n` consecutive values from `array` starting at `start`, wrapping at the
    year boundary.

    These arrays hold exactly one annual cycle (365*96 = 35040 points) and each
    manager's step() already wraps its own pointer when it reaches the end, so
    the lookahead/lookback WINDOWS must wrap too. Without this an episode
    starting near 31 December gets a SHORT forward window -- and once it is
    empty, the np.polyfit in sustaindc_env._create_ls_state raises
    "SVD did not converge in Linear Least Squares" -- while one starting on
    1 January gets an EMPTY backward window (the slice [-16:0]). Both were
    latent while episodes were confined to a +/-7 day window around July.
    """
    return np.take(array, np.arange(start, start + n), mode="wrap")


def _cyc_at(array, idx):
    """Single value at `idx`, wrapping at the year boundary (see _cyc)."""
    return array[idx % len(array)]

