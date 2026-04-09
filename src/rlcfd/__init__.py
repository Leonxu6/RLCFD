"""
RLCFD: Scalable CFD Development Framework for Reinforcement Learning

A framework for integrating computational fluid dynamics (CFD) simulations
with reinforcement learning training pipelines.
"""

__version__ = "0.1.0"

from rlcfd.mesh.grid import CartGrid, CurvilinearGrid
from rlcfd.mesh.field import ScalarField, VectorField
from rlcfd.envs.base import CFDEnv

__all__ = [
    "CartGrid",
    "CurvilinearGrid",
    "ScalarField",
    "VectorField",
    "CFDEnv",
    "__version__",
]
