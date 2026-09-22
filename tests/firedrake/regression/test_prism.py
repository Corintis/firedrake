"""Unstructured prism meshes read from gmsh files.

The meshes come from ``prism_meshes/`` at the repository root. They hold
``DM_POLYTOPE_TRI_PRISM`` cells, which is the only prism cell type that this
code supports. Facet integrals (``ds``) are not part of these tests.

Two gaps stopped a prism mesh above degree 1. Task 2b closed the first one. The
second one still has an xfail test below that turns green when Task 2c closes it.

Gap 1, Task 2b, the global numbering. CLOSED. ``create_section`` used to give
every plex point of the same topological dimension the same number of dofs. The
dimension 2 points of a prism mesh are not uniform: the quadrilateral faces and
the triangular faces carry different dof counts. The triangular faces therefore
got a dof block sized for a quadrilateral, the surplus dofs were never
referenced, and a stiffness matrix had one empty row per triangular face and was
singular. The numbering now keys the dof count on the DMPlex polytope type of the
point. See ``_numbering_strata`` in ``firedrake/mesh.py``. CG2 works from here.

Gap 2, Task 2c, the orientation of a quadrilateral face. ``_compute_orientation``
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

_GAP2_UNSAFE_HEAD = (
    "Task 2c, gap 2. DO NOT DELETE THIS SKIP AS MERELY UNSUPPORTED. The test "
    "PASSES today, but only because an out-of-bounds read returned 0 in this "
    "build. At CG2 and above a quadrilateral face carries dofs, so "
    "get_cell_nodes indexes the permutation table past its end under "
    "boundscheck(False): at CG2 the read is index 36 of a 36 element array. "
)
_GAP2_UNSAFE_TAIL = " Re-enable this when gap 2 is closed."

CG2_CORRECT_BUT_NOT_YET_SAFE = """\
Task 2c, gap 2. DO NOT READ A GREEN CG2 TEST AS "CG2 IS MEMORY SAFE".

Separate the result from the read.

The RESULT is correct, and for a structural reason, not by luck. At CG2 a
quadrilateral face carries exactly ONE interior dof, and the only permutation of
a one element set is the identity. No orientation, in range or out of range, can
misorder a single dof. So the dof numbering that Task 2b gives is the whole
answer at this degree, and these tests measure it honestly.

The READ is still unsafe. get_cell_nodes indexes the permutation table past its
end under boundscheck(False), because a prism quadrilateral face presents all 8
orientations while the FInAT element supplies only the 4 with extrinsic part 0.
That is undefined behaviour whatever value it returns. Task 2c removes it, and
Task 2c has to re-run these tests once it does.

From CG3 upwards a quadrilateral face carries 4 or more dofs, the identity
argument fails, and the result is wrong as well. The CG3 and CG4 cases below are
strict xfails for exactly that reason.
"""

CG3_RESULT_IS_UNVERIFIED = (
    "Task 2c, gap 2. DO NOT DELETE THIS SKIP AS MERELY UNSUPPORTED. The test "
    "PASSES today, and that is why it is skipped. This is NOT the CG2 case "
    "above. At CG2 a quadrilateral face carries one dof, the only permutation "
    "of a one element set is the identity, and the result is therefore correct "
    "for a structural reason. At CG3 a quadrilateral face carries 4 dofs, the "
    "FInAT element supplies only the 4 orientations with extrinsic part 0, and "
    "the shipped prism meshes present the values 4 and 6 as well. The "
    "permutation then selects between genuinely different answers. So at CG3 "
    "the read in get_cell_nodes is out of bounds AND the result it produces is "
    "unverified: this test passes on a memory layout accident, not for a "
    "reason. Re-enable it when gap 2 is closed."
)

GAP2_UNSAFE_MASS = (
    _GAP2_UNSAFE_HEAD
    + "The assertion can not detect the defect either, because a mass matrix "
      "total equals the cell volume even when dofs land on the wrong entity."
    + _GAP2_UNSAFE_TAIL
)

GAP2_UNSAFE_INTERP = (
    _GAP2_UNSAFE_HEAD
    + "This case asserts an L2 interpolation error, which CAN detect the "
      "defect, and does so from CG3 upwards, where those two cases are strict "
      "xfails. At CG2 a quadrilateral face carries one dof, so the defect "
      "stays invisible to the assertion as long as the read returns 0."
    + _GAP2_UNSAFE_TAIL
)


@pytest.mark.parametrize("degree", [
    1,
    pytest.param(2, marks=pytest.mark.skip(reason=GAP2_UNSAFE_MASS)),
    pytest.param(3, marks=pytest.mark.skip(reason=GAP2_UNSAFE_MASS)),
])
def test_prism_mass_matrix_total_is_the_volume(degree):
    mesh = Mesh(str(MESHDIR / "prism_reference.msh"))
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
    pytest.param(2, marks=pytest.mark.skip(reason=GAP2_UNSAFE_INTERP)),
    pytest.param(3, marks=pytest.mark.xfail(
        strict=True, reason="Task 2c, gap 2: the quadrilateral face orientation "
                            "is out of the range the FInAT prism element supplies")),
    pytest.param(4, marks=pytest.mark.xfail(
        strict=True, reason="Task 2c, gap 2: the quadrilateral face orientation "
                            "is out of the range the FInAT prism element supplies")),
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

# Task 2b turns the degree 2 case on. See CG2_CORRECT_BUT_NOT_YET_SAFE above:
# the result is correct, the permutation read is not yet in bounds.
@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2])
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


# --------------------------------------------------- the dof numbering, gap 1
#
# Task 2b closed gap 1, so the tests below pass. Every one of them that builds a
# space of degree 2 or more also performs the out of bounds permutation read that
# CG2_CORRECT_BUT_NOT_YET_SAFE describes. Read that text before you take a green
# result here as a statement about memory safety.

# The plex entity counts of prism_slab.msh. test_prism_slab_entity_counts below
# checks them against the mesh itself.
SLAB_VERTICES = 60
SLAB_EDGES = 175
SLAB_TRIANGLES = 78
SLAB_QUADRILATERALS = 90
SLAB_CELLS = 52


def _expected_cg_dim(degree):
    """The dimension of the CG space of the given degree on prism_slab.msh.

    The interior dof counts of a degree k prism Lagrange space are k-1 on an
    edge, whether it is an axis edge or a base edge, (k-1)(k-2)/2 on a
    triangular face, (k-1)^2 on a quadrilateral face, and (k-1)(k-2)/2 * (k-1)
    in the cell, which is the triangle interior count times the interval
    interior count.
    """
    edge = degree - 1
    triangle = (degree - 1) * (degree - 2) // 2
    quadrilateral = (degree - 1) ** 2
    cell = triangle * (degree - 1)
    return (SLAB_VERTICES
            + SLAB_EDGES * edge
            + SLAB_TRIANGLES * triangle
            + SLAB_QUADRILATERALS * quadrilateral
            + SLAB_CELLS * cell)


def test_prism_slab_entity_counts():
    """The constants above are the plex entity counts of prism_slab.msh."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    plex = mesh.topology.topology_dm
    counts = {}
    for dim in range(4):
        start, end = plex.getDepthStratum(dim)
        for point in range(start, end):
            cell_type = plex.getCellType(point)
            counts[cell_type] = counts.get(cell_type, 0) + 1
    assert counts == {
        PETSc.DM.PolytopeType.POINT: SLAB_VERTICES,
        PETSc.DM.PolytopeType.SEGMENT: SLAB_EDGES,
        PETSc.DM.PolytopeType.TRIANGLE: SLAB_TRIANGLES,
        PETSc.DM.PolytopeType.QUADRILATERAL: SLAB_QUADRILATERALS,
        PETSc.DM.PolytopeType.TRI_PRISM: SLAB_CELLS,
    }


def test_prism_numbering_strata_split_dimension_two():
    """Dimension 2 supplies one numbering stratum per polytope type."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    topology = mesh.topology
    assert topology._numbering_strata == (
        (0, None),
        (1, None),
        (2, PETSc.DM.PolytopeType.TRIANGLE),
        (2, PETSc.DM.PolytopeType.QUADRILATERAL),
        (3, None),
    )


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_prism_function_space_dimension(degree):
    """V.dim() is the analytic dof count, at every degree."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degree)
    assert V.dim() == _expected_cg_dim(degree)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_prism_dofs_per_plex_entity(degree):
    """One dof count per numbering stratum, in stratum order."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degree)
    got = tuple(mesh.topology.make_dofs_per_plex_entity(
        V.finat_element.entity_dofs()))
    assert got == (1,
                   degree - 1,
                   (degree - 1) * (degree - 2) // 2,
                   (degree - 1) ** 2,
                   (degree - 1) * (degree - 2) // 2 * (degree - 1))


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [
    2,
    pytest.param(3, marks=pytest.mark.skip(reason=CG3_RESULT_IS_UNVERIFIED)),
])
def test_prism_function_space_has_no_unreferenced_dofs(degree):
    """Every node of the space is referenced by the cell node map."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degree)
    local = np.unique(V.cell_node_map().values)
    referenced = V.dof_dset.lgmap.indices[local]
    gathered = np.concatenate(mesh.comm.allgather(referenced))
    assert np.unique(gathered).size == V.dim()


@pytest.mark.parallel([1, 2, 3])
def test_prism_cg2_mass_matrix_has_no_empty_row():
    """A CG2 mass matrix on a prism mesh is not singular.

    One empty row per triangular face was the visible symptom of gap 1.
    """
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", 2)
    matrix = assemble(inner(TrialFunction(V), TestFunction(V)) * dx).M.handle
    start, end = matrix.getOwnershipRange()
    empty = [row for row in range(start, end)
             if matrix.getRow(row)[0].size == 0]
    assert empty == []


# ---------------------------------------------------------- the known gap, 2

@pytest.mark.xfail(strict=True, reason="Task 2c, gap 2: PETSc transposes the "
                                       "axes of a TRI_PRISM quadrilateral face "
                                       "cone")
def test_prism_quad_face_orientations_are_in_range(meshname):
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", 3)
    permutations = V.finat_element.entity_permutations
    eo = mesh.topology.entity_orientations
    for face in range(5):
        column = 15 + face
        assert eo[:, column].max() < len(permutations[2][face])
