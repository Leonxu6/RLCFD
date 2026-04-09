"""
2D Lid-Driven Cavity (LDC) RL environment.

The LDC problem is a standard benchmark in CFD: a square cavity
with a moving top wall (lid) and stationary side/bottom walls.
The RL agent controls the lid velocity to achieve a target flow state.

This environment demonstrates the integration of a simple CFD solver
with a Gymnasium-compatible RL interface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None

from rlcfd.mesh.grid import CartGrid
from rlcfd.mesh.field import ScalarField, VectorField
from rlcfd.envs.base import CFDEnv


class LDCEnv(CFDEnv):
    """
    Lid-Driven Cavity environment for RL-based flow control.

    State (observation):
        - u-velocity field (nx*ny array, normalized)
        - v-velocity field (nx*ny array, normalized)
        - pressure field (nx*ny array, normalized)

    Action:
        - lid velocity U_lid (clamped to [-1, 1], scaled to [0, 1] m/s internally)

    Reward:
        - Negative squared vorticity deviation from a target (encourages desired flow)
        - Small penalty on control effort

    Episode terminates when max_steps reached or if simulation diverges.

    Example:
        >>> env = LDCEnv(nx=41, ny=41, Re=100)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, term, trunc, info = env.step(action)
    """

    def __init__(
        self,
        nx: int = 41,
        ny: int = 41,
        Re: float = 100.0,
        L: float = 1.0,
        max_steps: int = 2000,
        target_vorticity: float | None = None,
        lid_velocity_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """
        Args:
            nx: Number of cells in x-direction.
            ny: Number of cells in y-direction.
            Re: Reynolds number based on cavity size and reference velocity.
            L: Physical size of the cavity (meters).
            max_steps: Maximum simulation steps per episode.
            target_vorticity: Target mean vorticity. If None, use steady-state LDC value.
            lid_velocity_range: (min, max) lid velocity in m/s.
        """
        if gym is None:
            raise ImportError("gymnasium is required for LDCEnv. Install with: pip install rlcfd[rl]")

        grid = CartGrid(nx=nx, ny=ny, Lx=L, Ly=L)
        super().__init__(grid=grid, max_steps=max_steps, dt=0.001)

        self.Re = Re
        self.L = L
        self.lid_vel_min, self.lid_vel_max = lid_velocity_range
        self.nx = nx
        self.ny = ny

        # Physical parameters
        self.nu = 1.0 / Re  # kinematic viscosity (L=1, U_ref=1 => nu=1/Re)
        self.dx = grid.dx
        self.dy = grid.dy
        self.dt_cfd = min(0.0005, 0.2 * min(self.dx, self.dy) ** 2 / self.nu)  # CFL-limited

        # Flow fields
        self.u = ScalarField(grid)  # x-velocity
        self.v = ScalarField(grid)  # y-velocity
        self.p = ScalarField(grid)  # pressure
        self.p_old = ScalarField(grid)  # pressure from previous iteration
        self.vort = ScalarField(grid)  # vorticity

        # Target vorticity (steady-state for Re=100 LDC, approximately -2.0)
        self.target_vorticity = target_vorticity if target_vorticity is not None else -2.0

        # RL agent sets lid velocity (scaled by action)
        self.lid_velocity = 1.0  # default: unit lid velocity

        # Observation and action spaces
        n_obs = 3 * grid.ncells  # u, v, p fields
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _reset_flow_state(self) -> None:
        """Initialize flow fields to zero."""
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.p.fill(0.0)
        self.p_old.fill(0.0)
        self._apply_bc()
        self._compute_vorticity()

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        """Converts RL action to lid velocity."""
        # action in [-1, 1] -> lid_velocity in [lid_vel_min, lid_vel_max]
        self.lid_velocity = self.lid_vel_min + (action[0] + 1.0) * 0.5 * (
            self.lid_vel_max - self.lid_vel_min
        )
        self.lid_velocity = float(np.clip(self.lid_velocity, self.lid_vel_min, self.lid_vel_max))

    def _step_flow_state(self) -> None:
        """Advances the LDC simulation using a simple projection method."""
        # Run a few sub-iterations per RL step for faster convergence
        for _ in range(10):
            self._solve_velocity()
            self._solve_pressure()
            self._apply_bc()

        self._compute_vorticity()

    def _solve_velocity(self) -> None:
        """Solves the momentum equation using explicit Euler (simplified)."""
        u_view = self.u.view()
        v_view = self.v.view()

        u_new = ScalarField(self.grid)
        v_new = ScalarField(self.grid)
        u_new_view = u_new.view()
        v_new_view = v_new.view()

        dt = self.dt_cfd
        dx = self.dx
        dy = self.dy
        nu = self.nu

        # u-momentum
        u_new_view[1:-1, 1:-1] = (
            u_view[1:-1, 1:-1]
            + dt
            * (
                nu * ((u_view[2:, 1:-1] - 2 * u_view[1:-1, 1:-1] + u_view[:-2, 1:-1]) / dx**2
                + (u_view[1:-1, 2:] - 2 * u_view[1:-1, 1:-1] + u_view[1:-1, :-2]) / dy**2)
                - u_view[1:-1, 1:-1] * (u_view[2:, 1:-1] - u_view[:-2, 1:-1]) / (2 * dx)
                - v_view[1:-1, 1:-1] * (u_view[1:-1, 2:] - u_view[1:-1, :-2]) / (2 * dy)
                + (self.p.view()[1:-1, 1:-1] - self.p.view()[:-2, 1:-1]) / dx
            )
        )

        # v-momentum
        v_new_view[1:-1, 1:-1] = (
            v_view[1:-1, 1:-1]
            + dt
            * (
                nu * ((v_view[2:, 1:-1] - 2 * v_view[1:-1, 1:-1] + v_view[:-2, 1:-1]) / dx**2
                + (v_view[1:-1, 2:] - 2 * v_view[1:-1, 1:-1] + v_view[1:-1, :-2]) / dy**2)
                - u_view[1:-1, 1:-1] * (v_view[2:, 1:-1] - v_view[:-2, 1:-1]) / (2 * dx)
                - v_view[1:-1, 1:-1] * (v_view[1:-1, 2:] - v_view[1:-1, :-2]) / (2 * dy)
                + (self.p.view()[1:-1, 1:-1] - self.p.view()[1:-1, :-2]) / dy
            )
        )

        self.u.data = u_new.data
        self.v.data = v_new.data

    def _solve_pressure(self) -> None:
        """Solves the pressure Poisson equation using Jacobi iteration."""
        p_view = self.p.view()
        p_new = ScalarField(self.grid)
        p_new_view = p_new.view()

        dx2 = self.dx**2
        dy2 = self.dy**2
        dt = self.dt_cfd

        for _ in range(50):  # Jacobi iterations
            p_new_view[1:-1, 1:-1] = (
                (
                    (p_view[2:, 1:-1] + p_view[:-2, 1:-1]) * dy2
                    + (p_view[1:-1, 2:] + p_view[1:-1, :-2]) * dx2
                )
                / (2 * (dx2 + dy2))
                - (dx2 * dy2)
                / (2 * (dx2 + dy2))
                * (
                    (self.u.view()[2:, 1:-1] - self.u.view()[:-2, 1:-1]) / (2 * dt * self.dx)
                    + (self.v.view()[1:-1, 2:] - self.v.view()[1:-1, :-2]) / (2 * dt * self.dy)
                )
            )
            p_view[:] = p_new_view[:]

        self.p.data = p_view[:]

    def _apply_bc(self) -> None:
        """Applies no-slip boundary conditions and lid velocity."""
        u_view = self.u.view()
        v_view = self.v.view()

        # All walls: no-slip (u=0, v=0)
        u_view[0, :] = 0.0
        u_view[-1, :] = 0.0
        u_view[:, 0] = 0.0
        u_view[:, -1] = 0.0
        v_view[0, :] = 0.0
        v_view[-1, :] = 0.0
        v_view[:, 0] = 0.0
        v_view[:, -1] = 0.0

        # Lid (top wall): moving at lid_velocity in x-direction
        u_view[:, -1] = self.lid_velocity
        v_view[:, -1] = 0.0

    def _compute_vorticity(self) -> None:
        """Computes the z-vorticity (curl of velocity) at cell centers."""
        u_view = self.u.view()
        v_view = self.v.view()
        vort_view = self.vort.view()

        dv_dx = (v_view[1:, :-1] - v_view[:-1, :-1]) / self.dx
        du_dy = (u_view[:-1, 1:] - u_view[:-1, :-1]) / self.dy

        vort_view[:-1, :-1] = dv_dx - du_dy

    def _get_obs(self) -> NDArray[np.float32]:
        """Returns the flattened velocity and pressure fields as observation."""
        u_flat = self.u.data.ravel()
        v_flat = self.v.data.ravel()
        p_flat = self.p.data.ravel()

        u_norm = u_flat.astype(np.float32) / (self.lid_velocity_max + 1e-6)
        v_norm = v_flat.astype(np.float32) / (self.lid_velocity_max + 1e-6)
        p_norm = p_flat.astype(np.float32) / (abs(self.lid_velocity_max) * self.Re + 1e-6)

        return np.concatenate([u_norm, v_norm, p_norm])

    def _compute_reward(self) -> float:
        """Reward = negative deviation from target vorticity + control penalty."""
        interior = self.grid.interior_mask().ravel()
        vort_dev = self.vort.data[interior] - self.target_vorticity
        vort_penalty = float(np.mean(vort_dev**2))

        effort_penalty = 0.01 * (self.lid_velocity / self.lid_vel_max) ** 2

        return -(vort_penalty + effort_penalty)

    def _is_terminated(self) -> bool:
        """Check for simulation divergence."""
        u_max = self.u.max()
        return bool(np.isnan(u_max) or u_max > 100.0)

    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "lid_velocity": self.lid_velocity,
            "u_max": self.u.max(),
            "v_max": self.v.max(),
            "mean_vorticity": self.vort.interior_mean(),
        }

    @property
    def lid_velocity_max(self) -> float:
        return max(abs(self.lid_vel_min), abs(self.lid_vel_max))

    def __repr__(self) -> str:
        return f"LDCEnv(nx={self.nx}, ny={self.ny}, Re={self.Re})"
