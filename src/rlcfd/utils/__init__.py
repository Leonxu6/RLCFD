"""
Utility functions for RLCFD.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def save_field(
    field: Any,  # ScalarField | VectorField — avoid circular import
    path: str | Path,
    format: str = "pickle",
) -> None:
    """
    Saves a ScalarField or VectorField to disk.

    Args:
        field: The field to save.
        path: Output file path.
        format: 'pickle' (default) or 'numpy'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pickle":
        with open(path, "wb") as f:
            pickle.dump({"grid_nx": field.grid.nx, "grid_ny": field.grid.ny,
                         "grid_nz": field.grid.nz, "data": field.data}, f)
    elif format == "numpy":
        np.savez(
            path,
            data=field.data,
            grid_nx=field.grid.nx,
            grid_ny=field.grid.ny,
            grid_nz=field.grid.nz,
        )
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'pickle' or 'numpy'.")


def load_field(
    path: str | Path,
    field_class: type,  # ScalarField or VectorField
) -> Any:
    """
    Loads a ScalarField or VectorField from disk.

    Args:
        path: Input file path.
        field_class: The field class to instantiate (ScalarField or VectorField).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        data = np.load(path)
        # Reconstruct grid and field — simplified
        raise NotImplementedError("load_field from .npz needs grid reconstruction. Use pickle format.")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    from rlcfd.mesh.grid import CartGrid
    grid = CartGrid(nx=obj["grid_nx"], ny=obj["grid_ny"], nz=obj["grid_nz"])
    field = field_class(grid, init_value=0.0)
    field.data = obj["data"]
    return field


def running_mean(values: NDArray[np.floating], window: int = 100) -> NDArray[np.floating]:
    """
    Computes a running mean over a window.

    Args:
        values: Input array of values.
        window: Size of the averaging window.

    Returns:
        Array of running means with same length as input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    result = np.zeros_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = float(np.mean(values[start : i + 1]))
    return result


def write_json_log(path: str | Path, data: dict[str, Any]) -> None:
    """Appends a JSON record to a log file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def convergence_history(
    residual: NDArray[np.floating],
    window: int = 10,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    Analyzes convergence of an iterative solver.

    Args:
        residual: Array of residual values over iterations.
        window: Window size for checking rate of decrease.
        tolerance: Convergence tolerance.

    Returns:
        Dict with converged (bool), rate (float), iterations (int).
    """
    converged = bool(np.min(residual) < tolerance)
    n = len(residual)
    rate = 0.0
    if n > window:
        recent = residual[-window:]
        older = residual[-2 * window : -window]
        if np.all(older > 0) and np.all(recent > 0):
            rate = float(np.mean(np.log(recent / older)))

    return {"converged": converged, "rate": rate, "iterations": n}
