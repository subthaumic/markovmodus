"""
markovmodus
~~~~~~~~~~~

Synthetic single-cell splicing datasets generated from Markov-modulated ODE models.
"""

from .api import simulate_dataset
from .config import SimulationParameters

__all__ = ["simulate_dataset", "SimulationParameters"]
__version__ = "0.1.0"
