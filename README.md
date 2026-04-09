# RLCFD — Scalable CFD Development Framework for Reinforcement Learning

<p align="center">
  <img width="500" src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
  <img width="120" src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

**RLCFD** is a framework for integrating computational fluid dynamics (CFD) simulations with reinforcement learning training pipelines. It provides structured grids, field representations, differential operators, and Gymnasium-compatible RL environments for flow control and optimization tasks.

---

## Features

- **Structured grids**: Uniform Cartesian and curvilinear grids with coordinate mappings
- **Field representations**: Scalar and vector fields on grids with NumPy backends
- **Differential operators**: Gradient, divergence, curl, Laplacian — both 2D and 3D
- **RL environments**: Gymnasium-compatible interfaces for CFD-based RL (e.g. Lid-Driven Cavity)
- **Extensible**: Subclass `CFDEnv` to wrap any CFD solver as an RL environment

---

## Installation

```bash
pip install -e .
```

Optional dependencies:
```bash
pip install -e ".[rl]"      # Gymnasium + stable-baselines3 for RL training
pip install -e ".[viz]"      # Matplotlib + PyVista for visualization
pip install -e ".[dev]"      # Pytest, Black, Ruff for development
```

---

## Quick Start

### Basic grid and field operations

```python
from rlcfd.mesh.grid import CartGrid
from rlcfd.mesh.field import ScalarField, VectorField

# Create a 2D Cartesian grid (20x15 cells, 1m x 0.75m)
grid = CartGrid(nx=20, ny=15, Lx=1.0, Ly=0.75)

# Define a scalar field (e.g., pressure)
pressure = ScalarField(grid, init_value=101325.0)

# Define a vector field (e.g., velocity)
velocity = VectorField(grid)
velocity.view()[:, :, 0] = 1.0   # u = 1 m/s in x
velocity.view()[:, :, 1] = 0.5   # v = 0.5 m/s in y

# Differential operators
grad_p = pressure.gradient()
div_v = velocity.divergence()
```

### RL environment (Lid-Driven Cavity)

```python
import gymnasium as gym
from rlcfd.envs.ldc import LDCEnv

env = LDCEnv(nx=41, ny=41, Re=100)

obs, info = env.reset(seed=42)
for step in range(1000):
    action = env.action_space.sample()  # or your RL policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

---

## Project Structure

```
src/rlcfd/
├── __init__.py           # Package root
├── mesh/
│   ├── __init__.py
│   ├── grid.py           # CartGrid, CurvilinearGrid
│   └── field.py          # ScalarField, VectorField
├── envs/
│   ├── __init__.py
│   ├── base.py           # CFDEnv base class (Gymnasium)
│   └── ldc.py            # LDCEnv — Lid-Driven Cavity
└── utils/
    ├── __init__.py
    └── io.py             # Field I/O, running stats, convergence
examples/
└── example_ldc.py        # Usage demonstrations
```

---

## Grid and Field API

### `CartGrid`

```python
grid = CartGrid(nx=100, ny=50, nz=0, Lx=1.0, Ly=0.5)
# Properties: nx, ny, nz, dx, dy, dz, ncells, ndims, cell_centers, cell_volumes
```

### `CurvilinearGrid`

Extends `CartGrid` with arbitrary coordinate mappings:

```python
def channel_map(xi, eta, _zeta):
    x = xi * 2.0
    y = np.tanh(eta * 2 - 1) / np.tanh(1.0)
    return x, y

grid = CurvilinearGrid(nx=50, ny=40, mapping=channel_map)
```

### `ScalarField`

```python
T = ScalarField(grid, init_value=300.0)  # temperature
lap_T = T.laplacian()                     # discrete Laplacian
grad_T = T.gradient()                      # gradient
```

### `VectorField`

```python
u = VectorField(grid)                     # velocity
div_u = u.divergence()                   # divergence
mag_u = u.magnitude()                    # speed magnitude
dot_u = u.dot(u)                         # dot product with itself
```

---

## Building Custom Environments

Subclass `CFDEnv` and implement the required methods:

```python
from rlcfd.envs.base import CFDEnv

class MyCFDEnv(CFDEnv):
    def __init__(self):
        super().__init__(grid=CartGrid(nx=41, ny=41), max_steps=2000)
        self.observation_space = spaces.Box(-10, 10, shape=(3 * grid.ncells,))
        self.action_space = spaces.Box(-1, 1, shape=(1,))

    def _reset_flow_state(self):
        self.velocity.fill(0.0)
        self.pressure.fill(0.0)

    def _step_flow_state(self):
        # advance your CFD solver by one step
        ...

    def _get_obs(self):
        return np.concatenate([self.u.data, self.v.data, self.p.data])

    def _compute_reward(self):
        return -float(np.sum(self.vorticity.data**2))
```

---

## Citation

If you use RLCFD in your research, please cite:

```bibtex
@software{rlcfd2024,
  title = {RLCFD: Scalable CFD Development Framework for Reinforcement Learning},
  author = {Leon},
  url = {https://github.com/xuliyang/RLCFD},
  year = {2024},
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
