"""
Tests for rlcfd.mesh.field module.
"""

import numpy as np
import pytest

from rlcfd.mesh.grid import CartGrid
from rlcfd.mesh.field import ScalarField, VectorField


class TestScalarField:
    def test_creation_and_properties(self):
        grid = CartGrid(nx=10, ny=10)
        sf = ScalarField(grid, init_value=5.0)
        assert sf.data.shape == (100,)
        assert sf.mean() == 5.0
        assert sf.min() == 5.0
        assert sf.max() == 5.0

    def test_fill(self):
        grid = CartGrid(nx=5, ny=5)
        sf = ScalarField(grid, init_value=1.0)
        sf.fill(3.0)
        assert sf.mean() == 3.0

    def test_copy(self):
        grid = CartGrid(nx=5, ny=5)
        sf = ScalarField(grid, init_value=2.0)
        sf_copy = sf.copy()
        sf.fill(0.0)
        assert sf_copy.mean() == 2.0  # original unchanged

    def test_norm(self):
        grid = CartGrid(nx=3, ny=3)
        sf = ScalarField(grid)
        sf.data[:] = 3.0
        assert sf.norm(p=2) == pytest.approx(3.0 * np.sqrt(9))
        assert sf.norm(p=1) == pytest.approx(27.0)

    def test_interior_mean(self):
        grid = CartGrid(nx=5, ny=5)
        sf = ScalarField(grid)
        sf.data[:] = 1.0
        interior_mean = sf.interior_mean()
        assert interior_mean == 1.0

    def test_laplacian_2d(self):
        grid = CartGrid(nx=5, ny=5, Lx=1.0, Ly=1.0)
        sf = ScalarField(grid)
        sf.view()[2, 2] = 1.0  # delta function at center
        lap = sf.laplacian()
        # Laplacian at interior cells should be bounded
        assert np.isfinite(lap.data).all()

    def test_gradient(self):
        grid = CartGrid(nx=5, ny=5, Lx=1.0, Ly=1.0)
        sf = ScalarField(grid)
        sf.view()[:, :] = sf.grid.cell_centers[:, 0].reshape(5, 5)  # T = x
        grad = sf.gradient()
        grad_view = grad.view()
        # dT/dx should be ~1 at interior
        assert grad_view[2, 2, 0] == pytest.approx(1.0, abs=0.01)


class TestVectorField:
    def test_creation(self):
        grid = CartGrid(nx=5, ny=5)
        vf = VectorField(grid, init_value=1.0)
        assert vf.data.shape == (25, 2)
        assert vf.data[:, 0].all() == 1.0

    def test_fill_scalar(self):
        grid = CartGrid(nx=3, ny=3)
        vf = VectorField(grid)
        vf.fill(2.5)
        assert np.allclose(vf.data, 2.5)

    def test_magnitude(self):
        grid = CartGrid(nx=5, ny=5)
        vf = VectorField(grid)
        vf.data[:, 0] = 3.0  # u = 3
        vf.data[:, 1] = 4.0  # v = 4
        mag = vf.magnitude()
        assert mag.data[0] == pytest.approx(5.0)

    def test_dot_product(self):
        grid = CartGrid(nx=5, ny=5)
        vf1 = VectorField(grid)
        vf2 = VectorField(grid)
        vf1.data[:, 0] = 1.0
        vf1.data[:, 1] = 0.0
        vf2.data[:, 0] = 1.0
        vf2.data[:, 1] = 0.0
        dot = vf1.dot(vf2)
        assert dot.data[0] == pytest.approx(1.0)

    def test_divergence_2d(self):
        grid = CartGrid(nx=5, ny=5, Lx=1.0, Ly=1.0)
        vf = VectorField(grid)
        vf.data[:, 0] = vf.grid.cell_centers[:, 0]  # u = x
        vf.data[:, 1] = 0.0
        div = vf.divergence()
        # du/dx = 1 at interior
        assert div.view()[2, 2] == pytest.approx(1.0, abs=0.01)
