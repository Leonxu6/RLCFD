"""
Field representations for CFD data on structured grids.

Provides ScalarField and VectorField classes that store and operate
on data defined at cell centers of a CartGrid or CurvilinearGrid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rlcfd.mesh.grid import CartGrid, CurvilinearGrid


class ScalarField:
    """
    Scalar field defined on a structured grid.

    Stores a scalar quantity (e.g., pressure, temperature, density)
    at each cell center of a grid.

    Attributes:
        grid: The underlying grid structure.
        data: Flat array of shape (ncells,) containing field values.

    Example:
        >>> grid = CartGrid(nx=10, ny=10)
        >>> pressure = ScalarField(grid)
        >>> pressure.data[:] = 101325.0  # ambient pressure everywhere
        >>> print(f"Mean pressure: {pressure.mean():.2f} Pa")
    """

    def __init__(self, grid: CartGrid | CurvilinearGrid, init_value: float = 0.0) -> None:
        self.grid = grid
        self.data: NDArray[np.floating] = np.full(grid.ncells, init_value, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the logical shape of the field data."""
        if self.grid.nz > 0:
            return (self.grid.nx, self.grid.ny, self.grid.nz)
        return (self.grid.nx, self.grid.ny)

    def view(self) -> NDArray[np.floating]:
        """Returns a reshaped view of the data for logical indexing."""
        return self.data.reshape(self.shape)

    def fill(self, value: float) -> None:
        """Fills the entire field with a constant value."""
        self.data.fill(value)

    def copy(self) -> ScalarField:
        """Returns a deep copy of this field."""
        other = ScalarField(self.grid, init_value=0.0)
        other.data = self.data.copy()
        return other

    def mean(self) -> float:
        return float(np.mean(self.data))

    def max(self) -> float:
        return float(np.max(self.data))

    def min(self) -> float:
        return float(np.min(self.data))

    def norm(self, p: float = 2.0) -> float:
        """
        Computes the L-p norm of the field.

        Args:
            p: Norm order. p=2 is Euclidean norm, p=1 is Manhattan norm.
        """
        return float(np.linalg.norm(self.data, ord=p))

    def interior_mean(self) -> float:
        """Computes mean over interior cells only (excludes boundaries)."""
        mask = self.grid.interior_mask().ravel()
        return float(np.mean(self.data[mask]))

    def laplacian(self) -> ScalarField:
        """
        Computes the discrete Laplacian operator on this field.

        Uses a standard second-order central difference scheme.
        For interior cells only; boundary cells are set to zero.

        Returns:
            A new ScalarField containing the Laplacian at each cell.
        """
        lap = ScalarField(self.grid)
        view = self.view()
        lap_view = lap.view()

        dx2 = self.grid.dx ** 2
        dy2 = self.grid.dy ** 2

        if self.grid.nz > 0:
            dz2 = self.grid.dz ** 2
            lap_view[1:-1, 1:-1, 1:-1] = (
                (view[2:, 1:-1, 1:-1] - 2 * view[1:-1, 1:-1, 1:-1] + view[:-2, 1:-1, 1:-1]) / dx2
                + (view[1:-1, 2:, 1:-1] - 2 * view[1:-1, 1:-1, 1:-1] + view[1:-1, :-2, 1:-1]) / dy2
                + (view[1:-1, 1:-1, 2:] - 2 * view[1:-1, 1:-1, 1:-1] + view[1:-1, 1:-1, :-2]) / dz2
            )
        else:
            lap_view[1:-1, 1:-1] = (
                (view[2:, 1:-1] - 2 * view[1:-1, 1:-1] + view[:-2, 1:-1]) / dx2
                + (view[1:-1, 2:] - 2 * view[1:-1, 1:-1] + view[1:-1, :-2]) / dy2
            )

        return lap

    def gradient(self) -> VectorField:
        """
        Computes the gradient of this scalar field.

        Uses second-order central differences for interior cells
        and first-order differences at boundaries.

        Returns:
            A VectorField containing the gradient vector at each cell.
        """
        grad = VectorField(self.grid)
        view = self.view()
        grad_view = grad.view()

        dx = self.grid.dx
        dy = self.grid.dy

        if self.grid.nz > 0:
            dz = self.grid.dz
            # interior: central difference
            grad_view[1:-1, :, :, 0] = (view[2:, 1:-1, 1:-1] - view[:-2, 1:-1, 1:-1]) / (2 * dx)
            grad_view[:, 1:-1, :, 1] = (view[1:-1, 2:, 1:-1] - view[1:-1, :-2, 1:-1]) / (2 * dy)
            grad_view[:, :, 1:-1, 2] = (view[1:-1, 1:-1, 2:] - view[1:-1, 1:-1, :-2]) / (2 * dz)
            # boundaries: forward/backward difference
            grad_view[0, :, :, 0] = (view[1, :, :] - view[0, :, :]) / dx
            grad_view[-1, :, :, 0] = (view[-1, :, :] - view[-2, :, :]) / dx
            grad_view[:, 0, :, 1] = (view[:, 1, :] - view[:, 0, :]) / dy
            grad_view[:, -1, :, 1] = (view[:, -1, :] - view[:, -2, :]) / dy
            grad_view[:, :, 0, 2] = (view[:, :, 1] - view[:, :, 0]) / dz
            grad_view[:, :, -1, 2] = (view[:, :, -1] - view[:, :, -2]) / dz
        else:
            grad_view[1:-1, :, 0] = (view[2:, :] - view[:-2, :]) / (2 * dx)
            grad_view[:, 1:-1, 1] = (view[:, 2:] - view[:, :-2]) / (2 * dy)
            grad_view[0, :, 0] = (view[1, :] - view[0, :]) / dx
            grad_view[-1, :, 0] = (view[-1, :] - view[-2, :]) / dx
            grad_view[:, 0, 1] = (view[:, 1] - view[:, 0]) / dy
            grad_view[:, -1, 1] = (view[:, -1] - view[:, -2]) / dy

        return grad

    def __repr__(self) -> str:
        return f"ScalarField({self.grid}, min={self.min():.4g}, max={self.max():.4g}, mean={self.mean():.4g})"


class VectorField:
    """
    Vector field defined on a structured grid.

    Stores a vector quantity (e.g., velocity, momentum, force)
    at each cell center of a grid. Vector dimension matches grid dimensions.

    Attributes:
        grid: The underlying grid structure.
        data: Flat array of shape (ncells, ndims) containing field values.

    Example:
        >>> grid = CartGrid(nx=20, ny=20)
        >>> velocity = VectorField(grid)
        >>> v_view = velocity.view()
        >>> v_view[:, :, 0] = 1.0  # unit x-velocity everywhere
        >>> print(f"Max speed: {velocity.magnitude().max():.4f}")
    """

    def __init__(self, grid: CartGrid | CurvilinearGrid, init_value: float = 0.0) -> None:
        self.grid = grid
        self.data: NDArray[np.floating] = np.full(
            (grid.ncells, grid.ndims), init_value, dtype=np.float64
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the logical shape of the field data."""
        if self.grid.nz > 0:
            return (self.grid.nx, self.grid.ny, self.grid.nz, self.grid.ndims)
        return (self.grid.nx, self.grid.ny, self.grid.ndims)

    def view(self) -> NDArray[np.floating]:
        """Returns a reshaped view of the data for logical indexing."""
        return self.data.reshape(self.shape)

    def fill(self, value: float | NDArray[np.floating]) -> None:
        """Fills the entire field with a constant scalar or vector value."""
        if isinstance(value, np.ndarray):
            self.data[:] = value
        else:
            self.data.fill(value)

    def copy(self) -> VectorField:
        """Returns a deep copy of this field."""
        other = VectorField(self.grid, init_value=0.0)
        other.data = self.data.copy()
        return other

    def magnitude(self) -> ScalarField:
        """Returns the magnitude (Euclidean norm) of the vector field as a ScalarField."""
        mag = ScalarField(self.grid)
        if self.grid.nz > 0:
            mag.data = np.sqrt(
                self.data[:, 0] ** 2 + self.data[:, 1] ** 2 + self.data[:, 2] ** 2
            )
        else:
            mag.data = np.sqrt(self.data[:, 0] ** 2 + self.data[:, 1] ** 2)
        return mag

    def dot(self, other: VectorField) -> ScalarField:
        """
        Computes the dot product of this vector field with another.

        Args:
            other: Another VectorField defined on the same grid.

        Returns:
            A ScalarField containing the dot product at each cell.
        """
        if self.grid.ndims != other.grid.ndims:
            raise ValueError("Vector fields must have the same dimensionality")
        result = ScalarField(self.grid)
        result.data = np.sum(self.data * other.data, axis=1)
        return result

    def divergence(self) -> ScalarField:
        """
        Computes the divergence of this vector field.

        Uses second-order central differences for interior cells.

        Returns:
            A ScalarField containing the divergence at each cell.
        """
        div = ScalarField(self.grid)
        view = self.view()
        div_view = div.view()

        dx = self.grid.dx
        dy = self.grid.dy

        if self.grid.nz > 0:
            dz = self.grid.dz
            div_view[1:-1, 1:-1, 1:-1] = (
                (view[2:, 1:-1, 1:-1, 0] - view[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
                + (view[1:-1, 2:, 1:-1, 1] - view[1:-1, :-2, 1:-1, 1]) / (2 * dy)
                + (view[1:-1, 1:-1, 2:, 2] - view[1:-1, 1:-1, :-2, 2]) / (2 * dz)
            )
        else:
            div_view[1:-1, 1:-1] = (
                (view[2:, 1:-1, 0] - view[:-2, 1:-1, 0]) / (2 * dx)
                + (view[1:-1, 2:, 1] - view[1:-1, :-2, 1]) / (2 * dy)
            )

        return div

    def curl(self) -> VectorField:
        """
        Computes the curl of this vector field (3D only).

        Returns:
            A VectorField containing the curl at each cell.
        """
        if self.grid.nz == 0:
            raise RuntimeError("Curl is only defined for 3D vector fields")

        curl = VectorField(self.grid)
        view = self.view()
        curl_view = curl.view()

        dx = self.grid.dx
        dy = self.grid.dy
        dz = self.grid.dz

        curl_view[1:-1, 1:-1, 1:-1, 0] = (
            (view[1:-1, 2:, 1:-1, 2] - view[1:-1, :-2, 1:-1, 2]) / dy
            - (view[1:-1, 1:-1, 2:, 1] - view[1:-1, 1:-1, :-2, 1]) / dz
        )
        curl_view[1:-1, 1:-1, 1:-1, 1] = (
            (view[1:-1, 1:-1, 2:, 0] - view[1:-1, 1:-1, :-2, 0]) / dz
            - (view[2:, 1:-1, 1:-1, 2] - view[:-2, 1:-1, 1:-1, 2]) / dx
        )
        curl_view[1:-1, 1:-1, 1:-1, 2] = (
            (view[2:, 1:-1, 1:-1, 1] - view[:-2, 1:-1, 1:-1, 1]) / dx
            - (view[1:-1, 2:, 1:-1, 0] - view[1:-1, :-2, 1:-1, 0]) / dy
        )

        return curl

    def __repr__(self) -> str:
        mag = self.magnitude()
        return f"VectorField({self.grid}, |v| min={mag.min():.4g}, max={mag.max():.4g})"
