"""
Tests for rlcfd.mesh.grid module.
"""

import numpy as np
import pytest

from rlcfd.mesh.grid import CartGrid, CurvilinearGrid


class TestCartGrid:
    def test_basic_properties(self):
        grid = CartGrid(nx=10, ny=5, Lx=1.0, Ly=0.5)
        assert grid.nx == 10
        assert grid.ny == 5
        assert grid.nz == 0
        assert grid.ndims == 2
        assert grid.ncells == 50
        assert grid.dx == pytest.approx(0.1)
        assert grid.dy == pytest.approx(0.1)

    def test_3d_grid(self):
        grid = CartGrid(nx=5, ny=4, nz=3, Lx=1.0, Ly=1.0, Lz=1.0)
        assert grid.ndims == 3
        assert grid.ncells == 60
        assert grid.dx == pytest.approx(0.2)
        assert grid.dy == pytest.approx(0.25)
        assert grid.dz == pytest.approx(1.0 / 3.0)

    def test_cell_centers_2d(self):
        grid = CartGrid(nx=3, ny=2, Lx=3.0, Ly=2.0)
        centers = grid.cell_centers
        assert centers.shape == (6, 2)
        # Check x range
        assert centers[:, 0].min() > 0
        assert centers[:, 0].max() < 3.0
        # Check y range
        assert centers[:, 1].min() > 0
        assert centers[:, 1].max() < 2.0

    def test_interior_mask(self):
        grid = CartGrid(nx=5, ny=5)
        mask = grid.interior_mask()
        assert mask.shape == (5, 5)
        assert not mask[0, :].any()  # boundary
        assert not mask[-1, :].any()  # boundary
        assert mask[1:-1, 1:-1].all()  # interior

    def test_boundary_indices(self):
        grid = CartGrid(nx=5, ny=4)
        # x_minus: left face, spans y-direction, has ny=4 cells
        idx = grid.boundary_indices("x_minus")
        assert len(idx) == 4
        # x_plus: right face, spans y-direction, has ny=4 cells
        idx = grid.boundary_indices("x_plus")
        assert len(idx) == 4
        # y_minus: bottom face, spans x-direction, has nx=5 cells
        idx = grid.boundary_indices("y_minus")
        assert len(idx) == 5
        # y_plus: top face, spans x-direction, has nx=5 cells
        idx = grid.boundary_indices("y_plus")
        assert len(idx) == 5

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            CartGrid(nx=1, ny=5)  # nx < 2
        with pytest.raises(ValueError):
            CartGrid(nx=5, ny=5, nz=-1)  # nz < 0

    def test_repr(self):
        grid = CartGrid(nx=10, ny=5)
        r = repr(grid)
        assert "CartGrid" in r
        assert "10x5" in r


class TestCurvilinearGrid:
    def test_default_is_uniform(self):
        grid = CurvilinearGrid(nx=5, ny=4, Lx=1.0, Ly=1.0)
        assert grid._mapping is None
        assert grid.jacobian is None
        centers = grid.physical_coordinates
        assert centers.shape == (20, 2)

    def test_with_mapping(self):
        def simple_map(xi, eta, _zeta):
            x = xi * 2.0
            y = eta * 3.0
            return x, y

        grid = CurvilinearGrid(nx=3, ny=2, Lx=2.0, Ly=3.0, mapping=simple_map)
        assert grid._mapping is not None
        assert grid.jacobian is not None
        assert grid.jacobian.shape == (6,)

    def test_repr(self):
        grid = CurvilinearGrid(nx=5, ny=4)
        assert "CurvilinearGrid" in repr(grid)
        grid_mapped = CurvilinearGrid(nx=5, ny=4, mapping=lambda x, y, z: (x, y))
        assert "curvilinear" in repr(grid_mapped)
