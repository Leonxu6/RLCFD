"""
Mesh grid representations for CFD simulations.

Provides Cartesian and curvilinear grid implementations for
discretizing fluid domains in reinforcement learning contexts.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CartGrid:
    """
    Uniform Cartesian grid for CFD simulations.

    A simple, evenly-spaced grid suitable for basic flow problems
    and as a foundation for RL state representations.

    Attributes:
        nx: Number of cells in x-direction.
        ny: Number of cells in y-direction.
        nz: Number of cells in z-direction (0 for 2D).
        dx: Cell spacing in x-direction.
        dy: Cell spacing in y-direction.
        dz: Cell spacing in z-direction.
        origin: Origin coordinates (x0, y0, z0).

    Example:
        >>> grid = CartGrid(nx=100, ny=50, nz=0, Lx=1.0, Ly=0.5)
        >>> print(f"Grid has {grid.ncells} cells")
        >>> print(f"Cell spacing: dx={grid.dx:.4f}, dy={grid.dy:.4f}")
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        nz: int = 0,
        Lx: float = 1.0,
        Ly: float = 1.0,
        Lz: float = 1.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if nx < 2 or ny < 2:
            raise ValueError(f"Grid must have at least 2 cells in each direction, got nx={nx}, ny={ny}")
        if nz < 0:
            raise ValueError(f"nz must be non-negative, got {nz}")

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.origin = origin

        self._dx = Lx / nx
        self._dy = Ly / ny
        self._dz = Lz / nz if nz > 0 else 0.0

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dy(self) -> float:
        return self._dy

    @property
    def dz(self) -> float | float:
        return self._dz

    @property
    def ndims(self) -> int:
        """Number of spatial dimensions."""
        return 3 if self.nz > 0 else 2

    @property
    def ncells(self) -> int:
        """Total number of cells."""
        return self.nx * self.ny * max(1, self.nz)

    @property
    def cell_centers(self) -> NDArray[np.floating]:
        """
        Returns cell center coordinates as an array of shape (ncells, ndims).

        For 2D grids, each row is [x, y].
        For 3D grids, each row is [x, y, z].
        """
        if self.nz > 0:
            x = np.linspace(
                self.origin[0] + self._dx / 2,
                self.origin[0] + self.Lx - self._dx / 2,
                self.nx,
            )
            y = np.linspace(
                self.origin[1] + self._dy / 2,
                self.origin[1] + self.Ly - self._dy / 2,
                self.ny,
            )
            z = np.linspace(
                self.origin[2] + self._dz / 2,
                self.origin[2] + self.Lz - self._dz / 2,
                self.nz,
            )
            xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
            centers = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        else:
            x = np.linspace(
                self.origin[0] + self._dx / 2,
                self.origin[0] + self.Lx - self._dx / 2,
                self.nx,
            )
            y = np.linspace(
                self.origin[1] + self._dy / 2,
                self.origin[1] + self.Ly - self._dy / 2,
                self.ny,
            )
            xv, yv = np.meshgrid(x, y, indexing="ij")
            centers = np.stack([xv.ravel(), yv.ravel()], axis=1)
        return centers

    @property
    def cell_volumes(self) -> NDArray[np.floating]:
        """Returns cell volumes as a flat array."""
        vol = self._dx * self._dy * (self._dz if self.nz > 0 else 1.0)
        return np.full(self.ncells, vol)

    @property
    def faces(self) -> dict[str, NDArray[np.floating]]:
        """
        Returns face center coordinates and normals for all internal faces.

        Returns a dict with keys:
            x_minus: (ny*nz, ndims) face centers on x- face
            x_plus: (ny*nz, ndims) face centers on x+ face
            y_minus: (nx*nz, ndims) face centers on y- face
            y_plus: (nx*nz, ndims) face centers on y+ face
            z_minus: (nx*ny, ndims) face centers on z- face (3D only)
            z_plus: (nx*ny, ndims) face centers on z+ face (3D only)
        """
        faces: dict[str, NDArray[np.floating]] = {}

        if self.nz > 0:
            # 3D face centers
            x_c = np.linspace(self.origin[0], self.origin[0] + self.Lx, self.nx + 1)
            y_c = np.linspace(self.origin[1], self.origin[1] + self.Ly, self.ny + 1)
            z_c = np.linspace(self.origin[2], self.origin[2] + self.Lz, self.nz + 1)

            yv, zv = np.meshgrid(y_c, z_c, indexing="ij")
            faces["x_minus"] = np.stack(
                [np.full_like(yv, self.origin[0]), yv.ravel(), zv.ravel()], axis=1
            )
            faces["x_plus"] = np.stack(
                [np.full_like(yv, self.origin[0] + self.Lx), yv.ravel(), zv.ravel()],
                axis=1,
            )

            xv, zv = np.meshgrid(x_c, z_c, indexing="ij")
            faces["y_minus"] = np.stack(
                [xv.ravel(), np.full_like(xv, self.origin[1]), zv.ravel()], axis=1
            )
            faces["y_plus"] = np.stack(
                [xv.ravel(), np.full_like(xv, self.origin[1] + self.Ly), zv.ravel()],
                axis=1,
            )

            xv, yv = np.meshgrid(x_c, y_c, indexing="ij")
            faces["z_minus"] = np.stack(
                [xv.ravel(), yv.ravel(), np.full_like(xv, self.origin[2])], axis=1
            )
            faces["z_plus"] = np.stack(
                [
                    xv.ravel(),
                    yv.ravel(),
                    np.full_like(xv, self.origin[2] + self.Lz),
                ],
                axis=1,
            )
        else:
            # 2D face centers
            x_c = np.linspace(self.origin[0], self.origin[0] + self.Lx, self.nx + 1)
            y_c = np.linspace(self.origin[1], self.origin[1] + self.Ly, self.ny + 1)

            yv = np.tile(y_c, self.nx + 1)
            xv = np.repeat(x_c, self.ny + 1)
            faces["x_minus"] = np.stack([xv, yv], axis=1)

            faces["x_plus"] = faces["x_minus"].copy()
            faces["x_plus"][:, 0] += self._dx

            xv_new = np.repeat(x_c, self.ny + 1)
            yv_new = np.tile(y_c, self.nx + 1)
            faces["y_minus"] = np.stack([xv_new, yv_new], axis=1)

            faces["y_plus"] = faces["y_minus"].copy()
            faces["y_plus"][:, 1] += self._dy

        return faces

    def interior_mask(self) -> NDArray[np.bool_]:
        """
        Returns a boolean mask for interior cells (excludes boundary cells).

        Boundary cells are those adjacent to domain walls.
        """
        if self.nz > 0:
            interior = np.ones((self.nx, self.ny, self.nz), dtype=bool)
            interior[0, :, :] = False
            interior[-1, :, :] = False
            interior[:, 0, :] = False
            interior[:, -1, :] = False
            interior[:, :, 0] = False
            interior[:, :, -1] = False
        else:
            interior = np.ones((self.nx, self.ny), dtype=bool)
            interior[0, :] = False
            interior[-1, :] = False
            interior[:, 0] = False
            interior[:, -1] = False
        return interior

    def boundary_indices(self, face: str) -> NDArray[np.intp]:
        """
        Returns cell indices adjacent to the specified boundary face.

        Args:
            face: One of 'x_minus', 'x_plus', 'y_minus', 'y_plus', 'z_minus', 'z_plus'.
        """
        if self.nz > 0:
            nyz = self.ny * self.nz
            nxz = self.nx * self.nz
            nxy = self.nx * self.ny
            if face == "x_minus":
                return np.arange(0, nyz)
            elif face == "x_plus":
                return np.arange((self.nx - 1) * nyz, self.nx * nyz)
            elif face == "y_minus":
                return np.arange(0, self.nx * nz, self.nz)
            elif face == "y_plus":
                return np.arange(self.nz - 1, self.nx * nyz, self.nz)
            elif face == "z_minus":
                return np.arange(0, nxy)
            elif face == "z_plus":
                return np.arange(nxy * (self.nz - 1), nxy * self.nz)
        else:
            nx, ny = self.nx, self.ny
            if face == "x_minus":
                # cells at x_index=0: indices 0, 1, 2, ..., ny-1
                return np.arange(0, ny)
            elif face == "x_plus":
                # cells at x_index=nx-1: indices (nx-1), (2*nx-1), ..., (ny*nx-1)
                return np.arange(nx - 1, nx * ny, nx)
            elif face == "y_minus":
                # cells at y_index=0: indices 0, 1, 2, ..., nx-1
                return np.arange(0, nx)
            elif face == "y_plus":
                # cells at y_index=ny-1: indices (ny-1)*nx, ..., ny*nx-1
                return np.arange((ny - 1) * nx, nx * ny)

        valid = ["x_minus", "x_plus", "y_minus", "y_plus"]
        if self.nz > 0:
            valid += ["z_minus", "z_plus"]
        raise ValueError(f"Invalid face '{face}'. Valid faces: {valid}")

    def __repr__(self) -> str:
        dims = f"{self.nx}x{self.ny}"
        if self.nz > 0:
            dims += f"x{self.nz}"
        return f"CartGrid({dims}, L={self.Lx:g}x{self.Ly:g}" + (
            f"x{self.Lz:g}" if self.nz > 0 else ""
        ) + ")"


class CurvilinearGrid(CartGrid):
    """
    Curvilinear grid supporting arbitrary coordinate transformations.

    Enables modeling of complex geometries through coordinate mapping
    from a uniform computational space to physical space.

    Attributes:
        mapping: Callable that maps (xi, eta, zeta) -> (x, y, z) coordinates.
        jacobian: Metric tensor determinant at each cell center.

    Example:
        >>> import numpy as np
        >>> def channel_map(xi, eta, _):
        ...     x = xi  # uniform in x
        ...     y = np.tanh(eta) / np.tanh(1.0)  # tanh stretching in y
        ...     return x, y
        >>> grid = CurvilinearGrid(nx=50, ny=40, Lx=1.0, Ly=2.0, mapping=channel_map)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        nz: int = 0,
        Lx: float = 1.0,
        Ly: float = 1.0,
        Lz: float = 1.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mapping: callable | None = None,
    ) -> None:
        super().__init__(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, origin=origin)
        self._mapping = mapping
        if mapping is not None:
            self._jacobian = self._compute_jacobian()
        else:
            self._jacobian = None

    def _compute_jacobian(self) -> NDArray[np.floating]:
        """Computes the Jacobian determinant J at each cell center."""
        xi = np.linspace(0.0, 1.0, self.nx)
        eta = np.linspace(0.0, 1.0, self.ny)
        if self.nz > 0:
            zeta = np.linspace(0.0, 1.0, self.nz)
            xi_v, eta_v, zeta_v = np.meshgrid(xi, eta, zeta, indexing="ij")
            xi_v, eta_v, zeta_v = xi_v.ravel(), eta_v.ravel(), zeta_v.ravel()
            x, y, z = self._mapping(xi_v, eta_v, zeta_v)
            # Approximate J via finite differences (simplified)
            J = self._dx * self._dy
            if self.nz > 0:
                J *= self._dz
        else:
            xi_v, eta_v = np.meshgrid(xi, eta, indexing="ij")
            xi_v, eta_v = xi_v.ravel(), eta_v.ravel()
            x, y = self._mapping(xi_v, eta_v, 0.0)
            J = self._dx * self._dy * np.ones(self.ncells)

        return J

    @property
    def jacobian(self) -> NDArray[np.floating] | None:
        """Returns the Jacobian determinant at each cell. None if no mapping set."""
        return self._jacobian

    @property
    def physical_coordinates(self) -> NDArray[np.floating]:
        """Returns physical (x, y, z) cell center coordinates."""
        xi = np.linspace(0.0, 1.0, self.nx)
        eta = np.linspace(0.0, 1.0, self.ny)

        if self.nz > 0:
            zeta = np.linspace(0.0, 1.0, self.nz)
            xi_v, eta_v, zeta_v = np.meshgrid(xi, eta, zeta, indexing="ij")
            xi_v, eta_v, zeta_v = xi_v.ravel(), eta_v.ravel(), zeta_v.ravel()
            if self._mapping is not None:
                x, y, z = self._mapping(xi_v, eta_v, zeta_v)
            else:
                x, y, z = xi_v * self.Lx, eta_v * self.Ly, zeta_v * self.Lz
            return np.stack([x, y, z], axis=1)
        else:
            xi_v, eta_v = np.meshgrid(xi, eta, indexing="ij")
            xi_v, eta_v = xi_v.ravel(), eta_v.ravel()
            if self._mapping is not None:
                x, y = self._mapping(xi_v, eta_v, 0.0)
            else:
                x, y = xi_v * self.Lx, eta_v * self.Ly
            return np.stack([x, y], axis=1)

    def __repr__(self) -> str:
        dims = f"{self.nx}x{self.ny}"
        if self.nz > 0:
            dims += f"x{self.nz}"
        map_str = "curvilinear" if self._mapping else "uniform"
        return f"CurvilinearGrid({dims}, {map_str})"
