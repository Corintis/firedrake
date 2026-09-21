"""Unstructured prism meshes read from gmsh files.

The meshes come from ``prism_meshes/`` at the repository root. They hold
``DM_POLYTOPE_TRI_PRISM`` cells, which is the only prism cell type that this
code supports. Facet integrals (``ds``) are not part of these tests.

Two gaps stop a prism mesh above degree 1. Each one has an xfail test below
that turns green when a later task closes the gap.

Gap 1, the global numbering. ``create_section`` gives every plex point of the
same topological dimension the same number of dofs. The dimension 2 points of
a prism mesh are not uniform: the quadrilateral faces and the triangular faces
carry different dof counts. The triangular faces therefore get a dof block
sized for a quadrilateral, and the surplus dofs are never referenced. A
stiffness matrix then has one empty row per triangular face and is singular.

Gap 2, the orientation of a quadrilateral face. ``_compute_orientation``
reports orientation 4 or 6 for every quadrilateral face of a prism, that is
``eo = 1``: PETSc orders the cone of a TRI_PRISM quadrilateral face with its
axes transposed against the UFCQuadrilateral convention. The FInAT prism
element supplies only 4 orientations for such a face, the ``eo = 0`` half,
because a prism quadrilateral is a triangle edge times the interval and those
two axes can not be exchanged. A hexahedron face supplies 8. The orientation
therefore indexes past the permutation table.
"""
import itertools
from pathlib import Path

import numpy as np
import pytest

from firedrake import (Constant, DirichletBC, Function, FunctionSpace, Mesh,
                       SpatialCoordinate, TestFunction, TrialFunction,
                       assemble, dx, grad, inner, solve)
from firedrake.petsc import PETSc


MESHDIR = Path(__file__).parents[3] / "prism_meshes"
MESHNAMES = ("prism_reference.msh", "prism_slab.msh", "prism_warped.msh")

pytestmark = pytest.mark.skipif(
    not MESHDIR.is_dir(),
    reason=f"prism mesh directory {MESHDIR} is missing; "
           "run prism_meshes/make_prism_mesh.py to create it",
)


# ------------------------------------------------------------------ helpers

def _read_gmsh22_prisms(path):
    """Read the nodes and the prism cells of a gmsh 2.2 ASCII file.

    :arg path: The path of the .msh file.
    :returns: A pair (coords, cells). coords is a (nnodes, 3) array. cells is
        an (ncells, 6) array of indices into coords, in gmsh prism order:
        the bottom triangle first, then the top triangle.

    This parser is deliberately independent of Firedrake, so that a volume
    computed from its output is an independent check.
    """
    lines = Path(path).read_text().split("\n")
    i = lines.index("$Nodes")
    nnodes = int(lines[i + 1])
    tag_to_row = {}
    coords = np.empty((nnodes, 3), dtype=float)
    for row in range(nnodes):
        fields = lines[i + 2 + row].split()
        tag_to_row[int(fields[0])] = row
        coords[row] = [float(x) for x in fields[1:4]]
    j = lines.index("$Elements")
    nelem = int(lines[j + 1])
    cells = []
    for row in range(nelem):
        fields = [int(x) for x in lines[j + 2 + row].split()]
        # fields: id, type, ntags, tags..., node tags. Type 6 is the 6-node prism.
        if fields[1] != 6:
            continue
        ntags = fields[2]
        cells.append([tag_to_row[t] for t in fields[3 + ntags:]])
    return coords, np.asarray(cells, dtype=int)


def _gauss(n):
    """Return the n-point Gauss-Legendre rule on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _prism_volume_from_gmsh(path):
    """Compute the total volume of a gmsh prism mesh, independently of Firedrake.

    Each cell carries the degree 1 map of the prism, which is linear on the
    triangle and linear on the interval. The volume is the integral of the
    determinant of its Jacobian. A 4 point Gauss rule on each axis, with the
    Duffy transform on the triangle, integrates that determinant exactly.
    """
    coords, cells = _read_gmsh22_prisms(path)
    # Duffy transform of the reference triangle: l1 = u, l2 = v * (1 - u).
    u, wu = _gauss(4)
    v, wv = _gauss(4)
    s, ws = _gauss(4)
    total = 0.0
    for cell in cells:
        bottom = coords[cell[:3]]   # (3, 3)
        top = coords[cell[3:]]      # (3, 3)
        for iu, iv, isz in itertools.product(range(4), repeat=3):
            l1 = u[iu]
            l2 = v[iv] * (1.0 - u[iu])
            duffy = 1.0 - u[iu]
            lam = np.array([1.0 - l1 - l2, l1, l2])
            # c_t(s) is the point of the axis edge of triangle vertex t.
            c = (1.0 - s[isz]) * bottom + s[isz] * top      # (3, 3)
            dX_dl1 = c[1] - c[0]
            dX_dl2 = c[2] - c[0]
            dX_ds = lam @ (top - bottom)
            jac = np.linalg.det(np.stack([dX_dl1, dX_dl2, dX_ds]))
            total += wu[iu] * wv[iv] * ws[isz] * duffy * abs(jac)
    return total


@pytest.fixture(params=MESHNAMES)
def meshname(request):
    return request.param


# ------------------------------------------------------------- mesh building

def test_prism_mesh_builds(meshname):
    mesh = Mesh(str(MESHDIR / meshname))
    assert mesh.ufl_cell().cellname == "prism"
    assert mesh.topological_dimension == 3
    assert mesh.geometric_dimension == 3


@pytest.mark.parallel([1, 3])
def test_prism_mesh_builds_on_a_process_with_no_cells():
    """prism_reference.msh holds one cell, so two of three processes get none.

    _ufl_cell reduces over the communicator, so every process must agree.
    """
    mesh = Mesh(str(MESHDIR / "prism_reference.msh"))
    assert mesh.ufl_cell().cellname == "prism"
    assert np.isclose(float(assemble(Constant(1.0) * dx(domain=mesh))), 0.5)


def test_prism_dm_cell_type(meshname):
    mesh = Mesh(str(MESHDIR / meshname))
    assert mesh.topology.dm_cell_types == (PETSc.DM.PolytopeType.TRI_PRISM,)


# ----------------------------------------------------------- the cell closure

def test_prism_cell_closure_is_a_permutation_of_the_plex_closure(meshname):
    mesh = Mesh(str(MESHDIR / meshname))
    plex = mesh.topology.topology_dm
    closure = mesh.topology.cell_closure
    cStart, cEnd = plex.getHeightStratum(0)
    assert closure.shape == (cEnd - cStart, 21)
    cell_numbering = mesh.topology._cell_numbering
    for point in range(cStart, cEnd):
        cell = cell_numbering.getOffset(point)
        plex_closure, _ = plex.getTransitiveClosure(point)
        assert sorted(closure[cell]) == sorted(int(p) for p in plex_closure)


def test_prism_cell_closure_entity_types(meshname):
    mesh = Mesh(str(MESHDIR / meshname))
    plex = mesh.topology.topology_dm
    closure = mesh.topology.cell_closure
    expected = ([PETSc.DM.PolytopeType.POINT] * 6
                + [PETSc.DM.PolytopeType.SEGMENT] * 9
                + [PETSc.DM.PolytopeType.QUADRILATERAL] * 3
                + [PETSc.DM.PolytopeType.TRIANGLE] * 2
                + [PETSc.DM.PolytopeType.TRI_PRISM])
    for row in closure:
        got = [plex.getCellType(int(p)) for p in row]
        assert got == expected


def test_prism_cell_closure_matches_fiat_connectivity(meshname):
    """Each FIAT entity of the closure has the plex cone that FIAT prescribes."""
    import FIAT
    mesh = Mesh(str(MESHDIR / meshname))
    plex = mesh.topology.topology_dm
    closure = mesh.topology.cell_closure
    topology = FIAT.ufc_cell("prism").get_topology()
    # Offsets of dimension d entities in the flat FIAT closure.
    offset = {0: 0, 1: 6, 2: 15, 3: 20}
    for row in closure:
        for dim in (1, 2, 3):
            for ent, verts in topology[dim].items():
                point = int(row[offset[dim] + ent])
                plex_verts, _ = plex.getTransitiveClosure(point)
                vStart, vEnd = plex.getDepthStratum(0)
                got = {int(p) for p in plex_verts if vStart <= p < vEnd}
                want = {int(row[v]) for v in verts}
                assert got == want


def test_prism_cell_orientation_is_zero(meshname):
    """The docstring contract of _reorder_plex_closure: cell orientation is 0."""
    mesh = Mesh(str(MESHDIR / meshname))
    eo = mesh.topology.entity_orientations
    assert eo.shape[1] == 21
    assert (eo[:, -1] == 0).all()


# ------------------------------------------------------------------ assembly

def test_prism_mass_matrix_total_is_the_volume():
    mesh = Mesh(str(MESHDIR / "prism_reference.msh"))
    for degree in (1, 2, 3):
        V = FunctionSpace(mesh, "CG", degree)
        M = assemble(inner(TrialFunction(V), TestFunction(V)) * dx).M.values
        assert np.isclose(M.sum(), 0.5, rtol=0, atol=1e-12)


def test_prism_volume(meshname):
    mesh = Mesh(str(MESHDIR / meshname))
    expected = _prism_volume_from_gmsh(MESHDIR / meshname)
    got = float(assemble(Constant(1.0) * dx(domain=mesh)))
    assert np.isclose(got, expected, rtol=1e-12, atol=1e-12)


def test_prism_slab_volume_from_the_gmsh_geometry():
    """The slab is the unit square extruded to height H = 0.6."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    got = float(assemble(Constant(1.0) * dx(domain=mesh)))
    assert np.isclose(got, 1.0 * 0.6, rtol=0, atol=1e-12)


# ------------------------------------------------------------- interpolation

@pytest.mark.parametrize("degree", [
    1,
    2,
    pytest.param(3, marks=pytest.mark.xfail(
        strict=True, reason="gap 2: the quadrilateral face orientation is out "
                            "of the range the FInAT prism element supplies")),
    pytest.param(4, marks=pytest.mark.xfail(
        strict=True, reason="gap 2: the quadrilateral face orientation is out "
                            "of the range the FInAT prism element supplies")),
])
def test_prism_interpolation_is_exact_on_one_cell(degree):
    """A polynomial of P_k(triangle) x P_k(interval) interpolates exactly."""
    mesh = Mesh(str(MESHDIR / "prism_reference.msh"))
    V = FunctionSpace(mesh, "CG", degree)
    x, y, z = SpatialCoordinate(mesh)
    expr = (x + 2.0 * y)**degree * (1.0 + 3.0 * z)**degree + x**degree + z**degree
    u = Function(V).interpolate(expr)
    error = float(np.sqrt(abs(assemble(inner(u - expr, u - expr) * dx))))
    assert error < 1e-12


# ------------------------------------------------------------------- Poisson

@pytest.mark.parametrize("degree", [
    1,
    pytest.param(2, marks=pytest.mark.xfail(
        strict=True, reason="gap 1: the section over-allocates the triangular "
                            "faces, so the stiffness matrix is singular")),
])
def test_prism_poisson_with_strong_dirichlet(degree):
    """A harmonic polynomial of the FE space is reproduced exactly."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degree)
    x, y, z = SpatialCoordinate(mesh)
    if degree == 1:
        exact = 1.0 + x + 2.0 * y + 3.0 * z
    else:
        exact = 1.0 + x + 2.0 * y + x**2 + y**2 - 2.0 * z**2
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, exact, "on_boundary")
    solve(inner(grad(u), grad(v)) * dx == 0, u, bcs=bc,
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    error = float(np.sqrt(abs(assemble(inner(u - exact, u - exact) * dx))))
    assert error < 1e-10


# --------------------------------------------------------- the two known gaps

@pytest.mark.xfail(strict=True, reason="gap 1: create_section gives every "
                                       "dimension 2 point the dof count of a "
                                       "quadrilateral face")
@pytest.mark.parametrize("degree", [2, 3])
def test_prism_function_space_has_no_unreferenced_dofs(degree):
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degree)
    referenced = np.unique(V.cell_node_map().values).size
    assert referenced == V.dim()


@pytest.mark.xfail(strict=True, reason="gap 2: PETSc transposes the axes of a "
                                       "TRI_PRISM quadrilateral face cone")
def test_prism_quad_face_orientations_are_in_range(meshname):
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", 3)
    permutations = V.finat_element.entity_permutations
    eo = mesh.topology.entity_orientations
    for face in range(5):
        column = 15 + face
        assert eo[:, column].max() < len(permutations[2][face])
