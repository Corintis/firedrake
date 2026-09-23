"""Tests of submeshes of unstructured prism meshes.

A codim-1 submesh of a prism mesh is a triangle mesh or a quadrilateral mesh.
It does not inherit the closure of the prism cells: the prism closure keeps
the cone order, so the two cells of a submesh facet would see the points of
that facet in different orders. The triangle cells take the sorted closure of
a simplex, the quadrilateral cells take the oriented closure of a
quadrilateral mesh.

The meshes of prism_interior_marked.msh are the box [-0.5, 0.5]^2 x [0, 1].
Marker 10 is the plane z = 0.5 (triangles only), marker 20 is the plane
x = 0 (quadrilaterals only). The *_scrambled.msh copy permutes the node list
of each prism, so the cells present their facets in many orientations.
"""
from pathlib import Path

import numpy as np
import pytest

from firedrake import (Constant, Function, FunctionSpace, Mesh,
                       RelabeledMesh, SpatialCoordinate, Submesh,
                       VectorFunctionSpace, assemble, conditional, dS, ds, dx,
                       inner, jump)

MESHDIR = Path(__file__).parent.parent / "meshes" / "prism"

INTERIOR_MESHNAMES = ("prism_interior_marked.msh", "prism_interior_marked_scrambled.msh")

# The cell of the submesh on each marker, and the corners of the unit square
# that the marked surface is, in the order of its boundary.
SURFACES = {
    10: ("triangle", ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5))),
    20: ("quadrilateral", ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0), (0.0, 0.5, 1.0), (0.0, -0.5, 1.0))),
}


def _poly(x):
    """A cubic with no symmetry in the plane of either marked surface."""
    return x[0]**3 + x[0] * x[1] + 2 * x[1]**2 * x[2] + x[2]**3 - x[1] + 0.25 * x[0] * x[2]


def _boundary_integral(f, corners):
    """The integral of f over the boundary of a polygon, with Gauss points.

    :arg f: A function of a point (a numpy array of length 3).
    :arg corners: The corners of the polygon, in the order of its boundary.
    """
    points, weights = np.polynomial.legendre.leggauss(6)
    corners = [np.asarray(c, dtype=float) for c in corners]
    total = 0.0
    for a, b in zip(corners, corners[1:] + corners[:1]):
        half = 0.5 * np.linalg.norm(b - a)
        for t, w in zip(points, weights):
            total += w * half * f(a + 0.5 * (t + 1) * (b - a))
    return total


@pytest.fixture(scope="module", params=INTERIOR_MESHNAMES)
def interior_mesh(request):
    return Mesh(str(MESHDIR / request.param))


@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_cell_shape(interior_mesh, marker):
    sub = Submesh(interior_mesh, 2, marker)
    assert sub.ufl_cell().cellname == SURFACES[marker][0]
    assert np.isclose(assemble(Constant(1.0) * dx(domain=sub)), 1.0, rtol=1e-14)


@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_dS_point_order(interior_mesh, marker):
    """The two cells of a submesh facet see its points in one order.

    |x+ - x-|^2 is zero only if the points of the two sides agree. Before the
    fix it was 8.4e-02 on marker 10 and 4.7e-02 on marker 20 of the
    scrambled mesh.
    """
    sub = Submesh(interior_mesh, 2, marker)
    x = SpatialCoordinate(sub)
    assert assemble(inner(x('+') - x('-'), x('+') - x('-')) * dS(domain=sub)) < 1e-24


@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_dS_jump_of_continuous_field(interior_mesh, marker):
    """A DG3 field that interpolates a cubic has no jump on the submesh."""
    sub = Submesh(interior_mesh, 2, marker)
    u = Function(FunctionSpace(sub, "DG", 3)).interpolate(_poly(SpatialCoordinate(sub)))
    assert assemble(u('+') * u('-') * dS(domain=sub)) > 1e-3
    assert assemble(jump(u)**2 * dS(domain=sub)) < 1e-24


@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_ds(interior_mesh, marker):
    """The boundary of the submesh is the boundary of the unit square."""
    sub = Submesh(interior_mesh, 2, marker)
    corners = SURFACES[marker][1]
    assert np.isclose(assemble(Constant(1.0) * ds(domain=sub)), 4.0, rtol=1e-14)
    exact = _boundary_integral(_poly, corners)
    assert np.isclose(assemble(_poly(SpatialCoordinate(sub)) * ds(domain=sub)), exact, rtol=1e-13)


@pytest.mark.parametrize("family", ["CG", "DG"])
@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_interpolate_parent_to_submesh(interior_mesh, marker, family):
    """The closure of the submesh does not change the interpolation from the parent."""
    sub = Submesh(interior_mesh, 2, marker)
    parent = Function(FunctionSpace(interior_mesh, family, 3)).interpolate(_poly(SpatialCoordinate(interior_mesh)))
    Vs = FunctionSpace(sub, family, 3)
    got = Function(Vs).interpolate(parent)
    exact = Function(Vs).interpolate(_poly(SpatialCoordinate(sub)))
    assert np.abs(got.dat.data_ro - exact.dat.data_ro).max() < 1e-13


@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_interpolate_submesh_to_parent(interior_mesh, marker):
    """The parent dofs on the marked surface get the values of the submesh field.

    The dofs of a CG3 space on the surface are on the vertices, the edges (two
    per edge, so the edge orientation counts) and the faces.
    """
    sub = Submesh(interior_mesh, 2, marker)
    V = FunctionSpace(interior_mesh, "CG", 3)
    source = Function(FunctionSpace(sub, "CG", 3)).interpolate(_poly(SpatialCoordinate(sub)))
    got = Function(V).interpolate(source, allow_missing_dofs=True)
    exact = Function(V).interpolate(_poly(SpatialCoordinate(interior_mesh)))
    X = Function(VectorFunctionSpace(interior_mesh, "CG", 3)).interpolate(SpatialCoordinate(interior_mesh))
    X = X.dat.data_ro
    on_surface = {10: np.abs(X[:, 2] - 0.5) < 1e-12, 20: np.abs(X[:, 0]) < 1e-12}[marker]
    assert on_surface.sum() > 100
    assert np.abs(got.dat.data_ro[on_surface] - exact.dat.data_ro[on_surface]).max() < 1e-13


def _codim0_submesh(mesh):
    """The submesh of the prisms with x < 0, and its parent with that cell marker."""
    x, _, _ = SpatialCoordinate(mesh)
    indicator = Function(FunctionSpace(mesh, "DG", 0)).interpolate(conditional(x < 0, 1.0, 0.0))
    mesh = RelabeledMesh(mesh, [indicator], [100])
    return mesh, Submesh(mesh, 3, 100)


def test_prism_codim0_submesh_facets(interior_mesh):
    """A codim-0 submesh of a prism mesh keeps the prism cells and the facet markers.

    In the half x < 0, marker 10 (z = 0.5) is interior with area 0.5, and
    marker 20 (x = 0) is on the boundary with area 1.
    """
    _, sub = _codim0_submesh(interior_mesh)
    assert sub.ufl_cell().cellname == "prism"
    x = SpatialCoordinate(sub)
    assert np.isclose(assemble(Constant(1.0) * dx(domain=sub)), 0.5, rtol=1e-14)
    assert assemble(inner(x('+') - x('-'), x('+') - x('-')) * dS(domain=sub)) < 1e-24
    assert np.isclose(assemble(Constant(1.0) * dS(10, domain=sub)), 0.5, rtol=1e-14)
    assert np.isclose(assemble(Constant(1.0) * ds(20, domain=sub)), 1.0, rtol=1e-14)
    u = Function(FunctionSpace(sub, "DG", 3)).interpolate(_poly(x))
    assert assemble(jump(u)**2 * dS(domain=sub)) < 1e-24


def test_prism_codim0_submesh_interpolate(interior_mesh):
    mesh, sub = _codim0_submesh(interior_mesh)
    V = FunctionSpace(mesh, "CG", 3)
    Vs = FunctionSpace(sub, "CG", 3)
    parent = Function(V).interpolate(_poly(SpatialCoordinate(mesh)))
    exact = Function(Vs).interpolate(_poly(SpatialCoordinate(sub)))
    got = Function(Vs).interpolate(parent)
    assert np.abs(got.dat.data_ro - exact.dat.data_ro).max() < 1e-13
    back = Function(V).interpolate(exact, allow_missing_dofs=True)
    assert assemble((back - parent)**2 * dx(100)) < 1e-24
