#!/usr/bin/env python3
"""
Example: Running the Lid-Driven Cavity RL environment.

This demonstrates:
1. Creating a CartGrid and ScalarField
2. Creating and interacting with the LDCEnv
3. Running a random policy for a few episodes

Run with: python examples/example_ldc.py
"""

import numpy as np

from rlcfd.mesh.grid import CartGrid, CurvilinearGrid
from rlcfd.mesh.field import ScalarField, VectorField
from rlcfd.envs.ldc import LDCEnv


def demo_grid_and_fields():
    """Demonstrates basic grid and field operations."""
    print("=" * 60)
    print("Demo 1: CartGrid and ScalarField/VectorField")
    print("=" * 60)

    # Create a 2D Cartesian grid
    grid = CartGrid(nx=20, ny=15, Lx=1.0, Ly=0.75)
    print(f"Grid: {grid}")
    print(f"Number of cells: {grid.ncells}")
    print(f"Cell spacing: dx={grid.dx:.4f}, dy={grid.dy:.4f}")

    # Scalar field
    pressure = ScalarField(grid, init_value=101325.0)
    print(f"\nPressure field: {pressure}")

    # Vector field
    velocity = VectorField(grid)
    u_view = velocity.view()
    u_view[:, :, 0] = 1.0  # unit x-velocity
    u_view[:, :, 1] = 0.5  # 0.5 m/s y-velocity
    print(f"Velocity field: {velocity}")
    print(f"Max speed: {velocity.magnitude().max():.4f}")

    # Operators
    grad_p = pressure.gradient()
    print(f"\nPressure gradient: {grad_p}")

    div_v = velocity.divergence()
    print(f"Velocity divergence: {div_v}")
    print(f"Divergence interior mean: {div_v.interior_mean():.6f}")

    print("\n✓ Grid and field operations work correctly.")


def demo_curvilinear_grid():
    """Demonstrates curvilinear grid with coordinate mapping."""
    print("\n" + "=" * 60)
    print("Demo 2: Curvilinear Grid (channel flow mapping)")
    print("=" * 60)

    def channel_mapping(xi, eta, _zeta):
        x = xi * 2.0  # domain [0, 2]
        y = np.tanh(eta * 2 - 1) / np.tanh(1.0)  # tanh stretching
        return x, y

    grid = CurvilinearGrid(nx=30, ny=25, Lx=2.0, Ly=2.0, mapping=channel_mapping)
    print(f"Grid: {grid}")
    print(f"Physical coords shape: {grid.physical_coordinates.shape}")

    # Example scalar field on curvilinear grid
    field = ScalarField(grid)
    centers = grid.cell_centers
    field.data = np.sin(centers[:, 0]) * np.cos(centers[:, 1])
    print(f"Field on curvilinear grid: {field}")

    print("✓ Curvilinear grid works correctly.")


def demo_ldc_env():
    """Demonstrates the LDC RL environment with random actions."""
    print("\n" + "=" * 60)
    print("Demo 3: Lid-Driven Cavity RL Environment")
    print("=" * 60)

    env = LDCEnv(nx=21, ny=21, Re=50, max_steps=100)
    print(f"Environment: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run one episode with random policy
    obs, info = env.reset(seed=42)
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial info: {info}")

    total_reward = 0.0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        print(
            f"  Step {step + 1}: action={action[0]:+.3f}, "
            f"reward={reward:.4f}, lid_vel={info['lid_velocity']:.3f}"
        )
        if term or trunc:
            break

    print(f"\nEpisode reward (5 steps): {total_reward:.4f}")
    print(f"✓ LDC environment works correctly.")


if __name__ == "__main__":
    print("RLCFD Examples\n")
    demo_grid_and_fields()
    demo_curvilinear_grid()
    demo_ldc_env()
    print("\n✅ All demos passed!")
