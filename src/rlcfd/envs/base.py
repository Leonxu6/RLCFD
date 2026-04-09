"""
Base RL environment for CFD simulations.

Provides a Gymnasium-compatible interface for CFD-based
reinforcement learning environments.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class CFDEnv:
    """
    Base class for CFD-based RL environments.

    Subclass this to create environment wrappers for specific
    CFD simulation tasks (e.g., flow control, shape optimization).

    The environment follows the Gymnasium interface:
    - observation: numpy array of current flow state
    - action: numpy array of control inputs
    - reward: float scalar reward signal
    - terminated: whether episode has ended
    - truncated: whether episode was truncated (time limit)

    Subclasses must implement:
        - _get_obs()
        - _compute_reward()
        - _reset_flow_state()
        - _step_flow_state()

    Attributes:
        grid: The underlying CFD grid.
        max_steps: Maximum number of steps per episode.

    Example:
        >>> class LDCEnv(CFDEnv):
        ...     def __init__(self):
        ...         super().__init__(grid=CartGrid(nx=41, ny=41), max_steps=1000)
        ...         self.observation_space = spaces.Box(-1, 1, shape=(41*41,))
        ...         self.action_space = spaces.Box(-1, 1, shape=(1,))
        ...
        ...     def _get_obs(self):
        ...         return self.velocity.data[:, 0].flatten()  # u-velocity
        ...
        ...     def _compute_reward(self):
        ...         return -float(np.sum(self.vorticity.data**2))
        ...
        ...     def _reset_flow_state(self):
        ...         self.velocity.fill(0.0)
        ...         self.pressure.fill(0.0)
    """

    def __init__(
        self,
        grid: Any,  # CartGrid | CurvilinearGrid — avoid circular import
        max_steps: int = 1000,
        dt: float = 0.001,
    ) -> None:
        if gym is None:
            raise ImportError(
                "gymnasium is required for CFDEnv. "
                "Install with: pip install rlcfd[rl]"
            )

        self.grid = grid
        self.max_steps = max_steps
        self.dt = dt

        self.current_step = 0

        # Default spaces — subclasses should override
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(grid.ncells,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._state: dict[str, Any] = {}

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """
        Resets the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            observation: Initial observation array.
            info: Additional information dictionary.
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self._reset_flow_state()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(
        self, action: NDArray[np.floating] | list[float]
    ) -> tuple[
        NDArray[np.floating],
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        """
        Executes one environment step.

        Args:
            action: Control input from the RL agent.

        Returns:
            observation: Next observation.
            reward: Scalar reward.
            terminated: Whether episode ended (terminal state).
            truncated: Whether episode ended due to time limit.
            info: Additional information.
        """
        self._apply_action(np.asarray(action))
        self._step_flow_state()
        self.current_step += 1

        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Any:
        """
        Renders the current flow state.

        Args:
            mode: Rendering mode. 'human' for display, 'rgb_array' for pixels.
        """
        # Base implementation does nothing — subclasses can override
        return None

    def close(self) -> None:
        """Cleans up environment resources."""
        pass

    # --- Abstract methods (must be overridden) ---

    def _get_obs(self) -> NDArray[np.floating]:
        """
        Returns the current observation array.

        This is what the RL agent sees as input.
        Must be implemented by subclass.
        """
        raise NotImplementedError

    def _compute_reward(self) -> SupportsFloat:
        """
        Computes the reward for the current state.

        Must be implemented by subclass.
        """
        raise NotImplementedError

    def _reset_flow_state(self) -> None:
        """
        Resets the internal CFD state variables.

        Must be implemented by subclass.
        """
        raise NotImplementedError

    def _step_flow_state(self) -> None:
        """
        Advances the CFD simulation by one time step (self.dt).

        Must be implemented by subclass.
        """
        raise NotImplementedError

    # --- Optional override methods ---

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        """
        Applies the RL agent's action to the flow state.

        Default implementation does nothing. Override to add controls.
        """
        pass

    def _is_terminated(self) -> bool:
        """
        Returns whether the episode has reached a terminal state.

        Default always returns False. Override for early termination logic.
        """
        return False

    def _get_info(self) -> dict[str, Any]:
        """
        Returns auxiliary information dict for debugging/analysis.
        """
        return {"step": self.current_step}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grid={self.grid}, max_steps={self.max_steps})"
