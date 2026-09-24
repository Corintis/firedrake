"""Tests of submeshes of unstructured prism meshes.

A codim-1 submesh of a prism mesh is a triangle mesh or a quadrilateral mesh.
It does not inherit the closure of the prism cells: the prism closure keeps
the cone order, so the two cells of a submesh facet would see the points of
that facet in different orders. The triangle cells take the sorted closure of
a simplex, the quadrilateral cells take the oriented closure of a
quadrilateral mesh.

The second part tests the integrals over more than one mesh: the facets of
a prism mesh coupled with the cells of a facet submesh, as a Lagrange
multiplier method on a surface needs.

The meshes of prism_interior_marked.msh are the box [-0.5, 0.5]^2 x [0, 1].
Marker 10 is the plane z = 0.5 (triangles only), marker 20 is the plane
x = 0 (quadrilaterals only). The *_scrambled.msh copy permutes the node list
of each prism, so the cells present their facets in many orientations.
"""
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from firedrake import (Constant, DirichletBC, DistributedMeshOverlapType,
                       Function, FunctionSpace, Measure, Mesh, RelabeledMesh, SpatialCoordinate, Submesh,
                       TestFunction, TestFunctions, TrialFunction,
                       TrialFunctions, VectorFunctionSpace, assemble,
                       conditional, dS, div, ds, dx, grad, inner, jump, solve)
from firedrake.tsfc_interface import compile_form

MESHDIR = Path(__file__).parent.parent / "meshes" / "prism"

INTERIOR_MESHNAMES = ("prism_interior_marked.msh", "prism_interior_marked_scrambled.msh")

# A quadrilateral submesh needs a symmetric halo in parallel, so the parent
# needs a RIDGE (or VERTEX) overlap. With the default FACET overlap, two
# quadrilaterals on the plane x = 0 can share an edge while their prisms
# share no facet. See test_prism_quad_submesh_default_overlap.
DISTRIBUTION = {"overlap_type": (DistributedMeshOverlapType.RIDGE, 1)}

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
    return Mesh(str(MESHDIR / request.param), distribution_parameters=DISTRIBUTION)


def _max_abs(comm, a):
    """The largest absolute value of a over all ranks (0 for no values)."""
    return comm.allreduce(np.abs(a).max() if len(a) else 0.0, op=MPI.MAX)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_cell_shape(interior_mesh, marker):
    sub = Submesh(interior_mesh, 2, marker)
    assert sub.ufl_cell().cellname == SURFACES[marker][0]
    assert np.isclose(assemble(Constant(1.0) * dx(domain=sub)), 1.0, rtol=1e-14)


@pytest.mark.parallel([1, 2, 3])
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


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_dS_jump_of_continuous_field(interior_mesh, marker):
    """A DG3 field that interpolates a cubic has no jump on the submesh."""
    sub = Submesh(interior_mesh, 2, marker)
    u = Function(FunctionSpace(sub, "DG", 3)).interpolate(_poly(SpatialCoordinate(sub)))
    assert assemble(u('+') * u('-') * dS(domain=sub)) > 1e-3
    assert assemble(jump(u)**2 * dS(domain=sub)) < 1e-24


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_ds(interior_mesh, marker):
    """The boundary of the submesh is the boundary of the unit square."""
    sub = Submesh(interior_mesh, 2, marker)
    corners = SURFACES[marker][1]
    assert np.isclose(assemble(Constant(1.0) * ds(domain=sub)), 4.0, rtol=1e-14)
    exact = _boundary_integral(_poly, corners)
    assert np.isclose(assemble(_poly(SpatialCoordinate(sub)) * ds(domain=sub)), exact, rtol=1e-13)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("family", ["CG", "DG"])
@pytest.mark.parametrize("marker", sorted(SURFACES))
def test_prism_submesh_interpolate_parent_to_submesh(interior_mesh, marker, family):
    """The closure of the submesh does not change the interpolation from the parent."""
    sub = Submesh(interior_mesh, 2, marker)
    parent = Function(FunctionSpace(interior_mesh, family, 3)).interpolate(_poly(SpatialCoordinate(interior_mesh)))
    Vs = FunctionSpace(sub, family, 3)
    got = Function(Vs).interpolate(parent)
    exact = Function(Vs).interpolate(_poly(SpatialCoordinate(sub)))
    assert _max_abs(got.comm, got.dat.data_ro - exact.dat.data_ro) < 1e-13


@pytest.mark.parallel([1, 2, 3])
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
    # A rank can own no dof on the surface, so count and compare over all
    # ranks, before any assert.
    count = interior_mesh.comm.allreduce(int(on_surface.sum()), op=MPI.SUM)
    error = _max_abs(interior_mesh.comm, got.dat.data_ro[on_surface] - exact.dat.data_ro[on_surface])
    assert count > 100
    assert error < 1e-13


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", INTERIOR_MESHNAMES)
def test_prism_quad_submesh_default_overlap(meshname):
    """A quadrilateral submesh of a mesh with the default FACET overlap.

    In parallel the halo of Submesh(mesh, 2, 20) is not symmetric: a rank
    has a halo quadrilateral whose owner does not see the owned
    quadrilateral next to it. The quadrilateral orientation algorithm then
    waited forever. Now every rank raises NotImplementedError, or, if the
    halo happens to be symmetric, the submesh is correct.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    try:
        sub = Submesh(mesh, 2, 20)
        sub.cell_closure
    except NotImplementedError as e:
        assert mesh.comm.size > 1
        assert "RIDGE" in str(e)
        return
    x = SpatialCoordinate(sub)
    assert assemble(inner(x('+') - x('-'), x('+') - x('-')) * dS(domain=sub)) < 1e-24


def _codim0_submesh(mesh):
    """The submesh of the prisms with x < 0, and its parent with that cell marker."""
    x, _, _ = SpatialCoordinate(mesh)
    indicator = Function(FunctionSpace(mesh, "DG", 0)).interpolate(conditional(x < 0, 1.0, 0.0))
    mesh = RelabeledMesh(mesh, [indicator], [100])
    return mesh, Submesh(mesh, 3, 100)


@pytest.mark.parallel([1, 2, 3])
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


# Serial only: in parallel the forward interpolation from a submesh to its
# parent can miss the ghost dofs of a continuous space, Firedrake issue 4483
# (https://github.com/firedrakeproject/firedrake/issues/4483).
def test_prism_codim0_submesh_interpolate(interior_mesh):
    mesh, sub = _codim0_submesh(interior_mesh)
    V = FunctionSpace(mesh, "CG", 3)
    Vs = FunctionSpace(sub, "CG", 3)
    parent = Function(V).interpolate(_poly(SpatialCoordinate(mesh)))
    exact = Function(Vs).interpolate(_poly(SpatialCoordinate(sub)))
    got = Function(Vs).interpolate(parent)
    assert _max_abs(got.comm, got.dat.data_ro - exact.dat.data_ro) < 1e-13
    back = Function(V).interpolate(exact, allow_missing_dofs=True)
    assert assemble((back - parent)**2 * dx(100)) < 1e-24


# Integrals over more than one mesh: the facets of a prism mesh and the cells
# of a facet submesh, for Lagrange multiplier methods on a surface. Each
# case is (the facet kind on the prism mesh, the marker). The "exterior"
# case of marker 20 uses the codim-0 submesh of the prisms with x < 0, on
# which the plane x = 0 is a boundary of quadrilateral facets.
CROSS_CASES = [("interior", 10), ("interior", 20), ("exterior", 1), ("exterior", 20)]


def _surface_integral(f, marker, half=False):
    """The integral of f over a marked surface, with Gauss points.

    :arg f: A function of three numpy arrays (x, y, z).
    :arg marker: 1 (z = 0), 10 (z = 0.5) or 20 (x = 0).
    :arg half: If True, integrate over the part x < 0 of marker 1 or 10.
    """
    t, w = np.polynomial.legendre.leggauss(8)
    if half:
        a, b = np.meshgrid(0.25 * (t - 1), 0.5 * t, indexing="ij")
        ww = 0.125 * np.outer(w, w)
    else:
        a, b = np.meshgrid(0.5 * t, 0.5 * t, indexing="ij")
        ww = 0.25 * np.outer(w, w)
    zero = np.zeros_like(a)
    x, y, z = {1: (a, b, zero), 10: (a, b, zero + 0.5), 20: (zero, a, b + 0.5)}[marker]
    return float(np.sum(ww * f(x, y, z)))


def _cross_meshes(mesh, kind, marker):
    """The prism mesh, its facet submesh, and the two measures of the coupling.

    The first measure integrates over the submesh cells, the second over the
    marked facets of the prism mesh.
    """
    if kind == "exterior" and marker == 20:
        _, mesh = _codim0_submesh(mesh)
    sub = Submesh(mesh, 2, marker)
    facet = {"interior": "dS", "exterior": "ds"}[kind]
    on_sub = Measure("dx", sub, intersect_measures=(Measure(facet, mesh),))
    on_mesh = Measure(facet, mesh, intersect_measures=(Measure("dx", sub),))(marker)
    return mesh, sub, on_sub, on_mesh


def _restrict(u, kind, side):
    return u(side) if kind == "interior" else u


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("kind,marker", CROSS_CASES)
def test_prism_cross_mesh_keeps_one_facet_shape(interior_mesh, kind, marker):
    """A facet integral coupled with a facet submesh gives one kernel.

    The cells of the submesh have one shape, so only the facet type of that
    shape is compiled. The kernel of the other shape would run over an empty
    set.
    """
    mesh, sub, _, on_mesh = _cross_meshes(interior_mesh, kind, marker)
    shape = {"triangle": "tri", "quadrilateral": "quad"}[sub.ufl_cell().cellname]
    expected = [f"{kind}_facet_{shape}"]
    kernels = compile_form(Constant(1.0) * on_mesh, "form")
    assert sorted(k.kinfo.integral_type for k in kernels) == expected


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("kind,marker", CROSS_CASES)
def test_prism_cross_mesh_facet_integral(interior_mesh, kind, marker):
    """The prism facet and the submesh cell see the points in one order.

    u_mesh and u_sub interpolate the same cubic, so their squared difference
    is zero only if the two meshes agree on each point. The product is an
    exact value that also needs the right facet of each prism.
    """
    mesh, sub, on_sub, on_mesh = _cross_meshes(interior_mesh, kind, marker)
    u_mesh = Function(FunctionSpace(mesh, "DG", 3)).interpolate(_poly(SpatialCoordinate(mesh)))
    u_sub = Function(FunctionSpace(sub, "DG", 3)).interpolate(_poly(SpatialCoordinate(sub)))
    exact = _surface_integral(lambda x, y, z: _poly((x, y, z))**2, marker)
    for side in ("+", "-"):
        u = _restrict(u_mesh, kind, side)
        assert assemble((u - u_sub)**2 * on_sub) < 1e-24
        assert assemble((u - u_sub)**2 * on_mesh) < 1e-24
        assert np.isclose(assemble(u * u_sub * on_sub), exact, rtol=1e-13)
        assert np.isclose(assemble(u * u_sub * on_mesh), exact, rtol=1e-13)
    assert np.isclose(assemble(Constant(1.0) * on_sub), 1.0, rtol=1e-14)
    assert np.isclose(assemble(Constant(1.0) * on_mesh), 1.0, rtol=1e-14)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("facet,marker", [("ds", 1), ("dS", 10)])
def test_prism_codim0_cross_mesh_facet_integral(interior_mesh, facet, marker):
    """A facet integral on a codim-0 submesh, coupled with the same facets of the parent.

    Both meshes split the facet integral by shape. Marker 1 and marker 10
    hold triangles only, so the quadrilateral kernel runs over an empty set.
    """
    mesh, sub = _codim0_submesh(interior_mesh)
    on_sub = Measure(facet, sub, intersect_measures=(Measure(facet, mesh),))(marker)
    u_mesh = Function(FunctionSpace(mesh, "DG", 3)).interpolate(_poly(SpatialCoordinate(mesh)))
    u_sub = Function(FunctionSpace(sub, "DG", 3)).interpolate(_poly(SpatialCoordinate(sub)))
    kind = {"ds": "exterior", "dS": "interior"}[facet]
    exact = _surface_integral(lambda x, y, z: _poly((x, y, z))**2, marker, half=True)
    assert np.isclose(assemble(Constant(1.0) * on_sub), 0.5, rtol=1e-14)
    for side in ("+", "-"):
        u, us = _restrict(u_mesh, kind, side), _restrict(u_sub, kind, side)
        assert assemble((u - us)**2 * on_sub) < 1e-24
        assert np.isclose(assemble(u * us * on_sub), exact, rtol=1e-13)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("kind,marker", CROSS_CASES)
def test_prism_cross_mesh_facet_matrix(interior_mesh, kind, marker):
    """The coupling matrices map the dofs of one mesh to the other exactly.

    B[i, j] = (phi_j of the prism mesh, psi_i of the submesh). B times the
    interpolant of a cubic is the load vector of that cubic on the submesh,
    and the transpose times the interpolant on the submesh is the load vector
    on the prism facets.
    """
    mesh, sub, on_sub, on_mesh = _cross_meshes(interior_mesh, kind, marker)
    V = FunctionSpace(mesh, "CG", 3)
    Q = FunctionSpace(sub, "CG", 3)
    p_mesh = Function(V).interpolate(_poly(SpatialCoordinate(mesh)))
    p_sub = Function(Q).interpolate(_poly(SpatialCoordinate(sub)))
    u, v = TrialFunction(V), TestFunction(V)
    lam, mu = TrialFunction(Q), TestFunction(Q)
    load_sub = assemble(_poly(SpatialCoordinate(sub)) * mu * dx(domain=sub)).dat.data_ro
    load_mesh = assemble(_restrict(p_mesh * v, kind, "+") * on_mesh).dat.data_ro
    for measure in (on_sub, on_mesh):
        for side in ("+", "-"):
            B = assemble(_restrict(u, kind, side) * mu * measure).petscmat
            Bt = assemble(lam * _restrict(v, kind, side) * measure).petscmat
            with p_mesh.dat.vec_ro as x:
                got = B.createVecLeft()
                B.mult(x, got)
                assert _max_abs(mesh.comm, got.array_r - load_sub) < 1e-14
            with p_sub.dat.vec_ro as x:
                got = Bt.createVecLeft()
                Bt.mult(x, got)
                assert _max_abs(mesh.comm, got.array_r - load_mesh) < 1e-14
    assert _max_abs(mesh.comm, load_sub) > 1e-3


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("marker", [10, 20])
def test_prism_lagrange_multiplier_interface(interior_mesh, marker):
    """Poisson problem with a Lagrange multiplier on a marked interior surface.

    The exact solution u = q + a|d| has a kink on the surface d = 0, and q is
    a cubic. The multiplier lam on Submesh(mesh, 2, marker) imposes u = g on
    the surface, and lam is the jump of the normal flux, 2a. u is in the CG3
    space and lam is in the CG2 space, so the discrete solution is exact.
    """
    a = 0.75
    X = SpatialCoordinate(interior_mesh)
    d = _plane_distance(X, marker)
    q = _poly(X)
    u_exact = q + a * abs(d)
    sub = Submesh(interior_mesh, 2, marker)
    Xs = SpatialCoordinate(sub)
    V = FunctionSpace(interior_mesh, "CG", 3)
    Q = FunctionSpace(sub, "CG", 2)
    W = V * Q
    u, lam = TrialFunctions(W)
    v, mu = TestFunctions(W)
    gamma = Measure("dS", interior_mesh, intersect_measures=(Measure("dx", sub),))(marker)
    on_sub = Measure("dx", sub, intersect_measures=(Measure("dS", interior_mesh),))
    f = -div(grad(q))
    g = _poly(Xs)
    form = (inner(grad(u), grad(v)) * dx(domain=interior_mesh)
            + lam * v('+') * gamma + mu * u('-') * on_sub)
    rhs = f * v * dx(domain=interior_mesh) + mu * g * on_sub
    bcs = [DirichletBC(W.sub(0), u_exact, "on_boundary"),
           DirichletBC(W.sub(1), 2 * a, "on_boundary")]
    w = Function(W)
    solve(form == rhs, w, bcs=bcs,
          solver_parameters={"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"})
    u_h, lam_h = w.subfunctions
    assert np.sqrt(assemble((u_h - u_exact)**2 * dx)) < 1e-10
    assert np.sqrt(assemble((lam_h - 2 * a)**2 * dx)) < 1e-9


def _plane_distance(X, marker):
    """The signed distance from the plane of marker 10 or 20."""
    return {10: X[2] - 0.5, 20: X[0]}[marker]


@pytest.mark.parallel([1, 2, 3])
def test_prism_cross_mesh_unsupported_coupling(interior_mesh):
    """A prism facet integral coupled with a prism cell integral raises."""
    mesh, sub = _codim0_submesh(interior_mesh)
    measure = Measure("ds", mesh, intersect_measures=(Measure("dx", sub),))(1)
    with pytest.raises(NotImplementedError, match="more than one shape"):
        assemble(Constant(1.0) * measure)
