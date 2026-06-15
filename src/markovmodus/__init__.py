"""
markovmodus
~~~~~~~~~~~

Synthetic single-cell splicing datasets generated from Markov-modulated ODE models.
"""

from .api import simulate_dataset
from .config import SimulationParameters
from .simulation import Simulation, SimulationState

__all__ = ["simulate_dataset", "Simulation", "SimulationParameters", "SimulationState"]
__version__ = "0.3.0"
