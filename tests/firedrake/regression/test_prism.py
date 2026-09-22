"""Unstructured prism meshes read from gmsh files.

The meshes come from ``prism_meshes/`` at the repository root. They hold
``DM_POLYTOPE_TRI_PRISM`` cells, which is the only prism cell type that this
code supports. Facet integrals (``ds``) are not part of these tests.

Two gaps stopped a prism mesh above degree 1. Both are closed.

Gap 1, Task 2b, the global numbering. CLOSED. ``create_section`` used to give
every plex point of the same topological dimension the same number of dofs. The
dimension 2 points of a prism mesh are not uniform: the quadrilateral faces and
the triangular faces carry different dof counts. The triangular faces therefore
got a dof block sized for a quadrilateral, the surplus dofs were never
referenced, and a stiffness matrix had one empty row per triangular face and was
singular. The numbering now keys the dof count on the DMPlex polytope type of the
point. See ``_numbering_strata`` in ``firedrake/mesh.py``. CG2 works from here.

Gap 2, Task 2c, the orientation of a quadrilateral face. CLOSED. A quadrilateral
face of a prism receives all 8 orientations of a quadrilateral, but the FInAT
prism element supplied only the 4 with extrinsic part 0, so ``get_cell_nodes``
indexed past the end of the permutation table. The element now supplies all 8.
See ``_make_axis_perms_tensorproduct`` in ``FIAT/orientation_utils.py``.

The fix belongs in FIAT, not in Firedrake, because the gap is real. The mesh
``prism_two_perpendicular.msh`` holds two prisms whose axes are PERPENDICULAR
and which share a quadrilateral face. The two cells report different extrinsic
parts for that one face, so no convention on the Firedrake side puts both of
them inside a 4-entry table. The test named
``test_prism_perpendicular_cells_disagree_on_the_extrinsic_part`` below pins
that, and it is the case that this whole task exists for.
"""
import itertools
from pathlib import Path

import numpy as np
import pytest

from firedrake import (Constant, DirichletBC, Function, FunctionSpace, Mesh,
                       SpatialCoordinate, TestFunction, TrialFunction,
                       UnitCubeMesh, assemble, dx, grad, inner, solve)
from firedrake.petsc import PETSc


MESHDIR = Path(__file__).parents[3] / "prism_meshes"
MESHNAMES = ("prism_reference.msh", "prism_slab.msh", "prism_warped.msh",
             "prism_two_perpendicular.msh")

# The meshes that hold more than one cell, so they have interior faces.
MULTICELL_MESHNAMES = ("prism_slab.msh", "prism_warped.msh",
                       "prism_two_perpendicular.msh")

# The two cells of prism_two_perpendicular.msh meet on a quadrilateral face
# only, so that mesh has no interior triangular face to check.
TRIANGLE_SHARING_MESHNAMES = ("prism_slab.msh", "prism_warped.msh")

# Closure column of each FIAT face of a prism. The closure holds 6 vertices,
# then 9 edges, then the 3 quadrilateral faces, then the 2 triangular faces,
# then the cell.
QUAD_FACE_COLUMNS = (15, 16, 17)
TRIANGLE_FACE_COLUMNS = (18, 19)

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

@pytest.mark.parametrize("degree", [1, 2, 3, 4])
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

@pytest.mark.parametrize("degree", [1, 2, 3, 4])
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

@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
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
# Task 2b closed gap 1, so the tests below pass.

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
@pytest.mark.parametrize("degree", [2, 3, 4])
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


# ------------------------------------ the quadrilateral face orientation, gap 2
#
# A quadrilateral face of a prism carries all 8 orientations of a
# quadrilateral: 2 axis permutations times 2 reflections per axis. The FInAT
# prism element used to supply only the 4 with extrinsic part 0, so
# get_cell_nodes indexed the permutation table past its end.
#
# The helpers below check the dof numbering against the physical position of
# each dof, which is what "the orientation selects the right permutation"
# means. They do not consult the permutation table at all, so a table that
# merely has the right LENGTH does not satisfy them.


def _reference_dof_points(V):
    """The reference prism coordinates of every local dof of V, in dof order."""
    nodes = V.finat_element.fiat_equivalent.dual.nodes
    points = []
    for node in nodes:
        point, = node.get_point_dict().keys()
        points.append(point)
    return np.asarray(points, dtype=float)


def _physical_dof_points(mesh, V):
    """The physical coordinates of every local dof, per cell.

    :returns: An array of shape (ncells, ndofs, gdim).

    The map of a prism cell of a degree 1 coordinate field is linear on the
    triangle and linear on the interval, so it is written out here directly
    from the six vertex coordinates. This does not go through the element, so
    it is an independent statement of where each dof sits.
    """
    reference = _reference_dof_points(V)
    coordinates = mesh.coordinates
    vertex_map = coordinates.function_space().cell_node_map().values
    data = coordinates.dat.data_ro_with_halos
    x, y, z = reference[:, 0], reference[:, 1], reference[:, 2]
    # Barycentric coordinates on the triangle factor.
    barycentric = np.stack([1.0 - x - y, x, y], axis=1)
    points = np.empty((vertex_map.shape[0], reference.shape[0], data.shape[1]))
    for cell in range(vertex_map.shape[0]):
        vertices = data[vertex_map[cell]]
        # FIAT prism vertex v is triangle vertex v // 2 at interval end v % 2.
        bottom = barycentric @ vertices[0::2]
        top = barycentric @ vertices[1::2]
        points[cell] = (1.0 - z)[:, None] * bottom + z[:, None] * top
    return points


def _interior_faces(mesh, columns):
    """The interior faces of the given closure columns.

    :arg columns: The closure columns to walk, one per FIAT face.
    :returns: A list of pairs of (cell, column), one pair per interior face.
    """
    closure = mesh.topology.cell_closure
    incident = {}
    for cell in range(closure.shape[0]):
        for column in columns:
            incident.setdefault(int(closure[cell, column]), []).append((cell, column))
    return [pair for pair in incident.values() if len(pair) == 2]


def _assert_shared_face_dofs_agree(meshname, degree, columns, dofs_per_face):
    """Both cells of every interior face give its dofs the same global nodes.

    They must also give them in the order that puts each node at one physical
    point. A permutation that merely reaches the right SET of nodes reorders
    the face and puts a node at two different places, which this detects.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", degree)
    nodes = V.cell_node_map().values
    points = _physical_dof_points(mesh, V)
    entity_dofs = V.finat_element.entity_dofs()

    faces = _interior_faces(mesh, columns)
    assert faces, f"{meshname} has no interior face in columns {columns}"
    for (cell_a, column_a), (cell_b, column_b) in faces:
        dofs_a = entity_dofs[2][column_a - 15]
        dofs_b = entity_dofs[2][column_b - 15]
        assert len(dofs_a) == len(dofs_b) == dofs_per_face
        if dofs_per_face == 0:
            continue
        nodes_a = [int(nodes[cell_a, dof]) for dof in dofs_a]
        nodes_b = [int(nodes[cell_b, dof]) for dof in dofs_b]
        assert sorted(nodes_a) == sorted(nodes_b)
        # The node that each cell puts at a given physical point is the same.
        place_a = {node: points[cell_a, dof] for node, dof in zip(nodes_a, dofs_a)}
        place_b = {node: points[cell_b, dof] for node, dof in zip(nodes_b, dofs_b)}
        for node in place_a:
            assert np.allclose(place_a[node], place_b[node], rtol=0, atol=1e-12)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_prism_quad_face_orientations_are_in_range(meshname, degree):
    """Every orientation a face receives indexes inside its permutation table."""
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", degree)
    permutations = V.finat_element.entity_permutations
    orientations = mesh.topology.entity_orientations
    for face in range(5):
        table = permutations[2][face]
        column = 15 + face
        assert sorted(table) == list(range(len(table)))
        assert orientations[:, column].max() < len(table)
        assert orientations[:, column].min() >= 0
    # The quadrilateral faces supply 8 orientations and the triangular faces 6.
    assert [len(permutations[2][face]) for face in range(3)] == [8, 8, 8]
    assert [len(permutations[2][face]) for face in (3, 4)] == [6, 6]


def test_prism_quad_faces_reach_the_second_axis_permutation(meshname):
    """The shipped meshes do present orientations that the old table lacked.

    Without this the test above is satisfied by a mesh that never leaves the
    first four orientations, and the widened table would guard nothing.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    orientations = mesh.topology.entity_orientations
    quad = orientations[:, list(QUAD_FACE_COLUMNS)]
    assert quad.max() >= 4


def test_prism_perpendicular_cells_disagree_on_the_extrinsic_part():
    """Two prisms with perpendicular axes share a quadrilateral face.

    This is the mesh that decides the shape of the fix. The two cells report
    different extrinsic parts for the same face, so there is no single frame
    in which both of them sit in a 4-entry table. Transposing the face cone
    puts one cell right and the other wrong.
    """
    mesh = Mesh(str(MESHDIR / "prism_two_perpendicular.msh"))
    orientations = mesh.topology.entity_orientations
    assert orientations.shape[0] == 2

    faces = _interior_faces(mesh, QUAD_FACE_COLUMNS)
    assert len(faces) == 1
    (cell_a, column_a), (cell_b, column_b) = faces[0]
    orientation_a = int(orientations[cell_a, column_a])
    orientation_b = int(orientations[cell_b, column_b])
    # The orientation is (2 ** 2) * extrinsic + intrinsic for a quadrilateral.
    assert orientation_a // 4 != orientation_b // 4
    assert {orientation_a, orientation_b} == {1, 6}

    # The two cells really do have perpendicular axes. The axis of a prism runs
    # along its edges 0, 1 and 2, which are the edges of the closure columns
    # 6, 7 and 8.
    coordinates = mesh.coordinates.dat.data_ro_with_halos
    closure = mesh.topology.cell_closure
    plex = mesh.topology.topology_dm
    vertex_start, _ = plex.getDepthStratum(0)
    axes = []
    for cell in (cell_a, cell_b):
        cone = plex.getCone(int(closure[cell, 6]))
        tail, head = (coordinates[int(v) - vertex_start] for v in cone)
        direction = head - tail
        axes.append(direction / np.linalg.norm(direction))
    assert np.isclose(abs(float(np.dot(axes[0], axes[1]))), 0.0, atol=1e-12)


@pytest.mark.parametrize("meshname", MULTICELL_MESHNAMES)
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_prism_interior_quad_face_dofs_agree(meshname, degree):
    """Both cells of every interior quadrilateral face agree on its dofs."""
    _assert_shared_face_dofs_agree(meshname, degree, QUAD_FACE_COLUMNS,
                                   (degree - 1) ** 2)


@pytest.mark.parametrize("meshname", TRIANGLE_SHARING_MESHNAMES)
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_prism_interior_triangular_face_dofs_agree(meshname, degree):
    """The triangular faces are a regression guard: they must not move."""
    _assert_shared_face_dofs_agree(meshname, degree, TRIANGLE_FACE_COLUMNS,
                                   (degree - 1) * (degree - 2) // 2)


def test_prism_perpendicular_mesh_shares_no_triangular_face():
    """Say why the triangular test above skips the perpendicular mesh.

    The two prisms meet on a quadrilateral face only. If a later change gives
    them a shared triangular face as well, add the mesh to
    TRIANGLE_SHARING_MESHNAMES rather than deleting this test.
    """
    mesh = Mesh(str(MESHDIR / "prism_two_perpendicular.msh"))
    assert _interior_faces(mesh, TRIANGLE_FACE_COLUMNS) == []


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_prism_every_node_sits_at_one_physical_point(meshname, degree):
    """No global node is claimed by two cells for two different places.

    This is the whole-mesh form of the face tests above, and it also covers
    the edges and the vertices. Before the fix it failed from CG3 upwards, by
    as much as 0.63 of a cell width.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", degree)
    nodes = V.cell_node_map().values
    points = _physical_dof_points(mesh, V)

    place = {}
    for cell in range(nodes.shape[0]):
        for dof in range(nodes.shape[1]):
            node = int(nodes[cell, dof])
            point = points[cell, dof]
            if node in place:
                assert np.allclose(place[node], point, rtol=0, atol=1e-12)
            else:
                place[node] = point
    # Distinct nodes sit at distinct points, so no dof was dropped either.
    stacked = np.array([place[node] for node in sorted(place)])
    assert np.unique(stacked.round(10), axis=0).shape[0] == len(place)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_prism_permutation_reads_stay_inside_the_table(meshname, degree):
    """get_cell_nodes never indexes past the end of the permutation buffer.

    get_cell_nodes reads entity_permutations_c[perm_offset + ndofs * orient + j]
    with 0 <= j < ndofs, under boundscheck(False). This walks the same offsets
    and asserts the largest of them is inside the buffer. At CG2 on
    prism_reference.msh the largest index used to be 36 of a 36 element buffer.
    """
    from firedrake.cython.dmcommon import _make_entity_permutations_c

    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", degree)
    entity_dofs = V.finat_element.entity_dofs()
    buffer, num_orientations = _make_entity_permutations_c(
        entity_dofs, V.finat_element.entity_permutations)
    ndofs = [len(entity_dofs[dim][entity]) for dim in sorted(entity_dofs)
             for entity in range(len(entity_dofs[dim]))]
    orientations = mesh.topology.entity_orientations

    highest = -1
    for cell in range(orientations.shape[0]):
        offset = 0
        for slot, count in enumerate(ndofs):
            if count > 0:
                orient = int(orientations[cell, slot])
                highest = max(highest, offset + count * orient + count - 1)
            offset += count * int(num_orientations[slot])
    assert highest < len(buffer)


def test_prism_cg3_quad_face_nodes_are_four_distinct_nodes():
    """The reference cell at CG3: face 2 must not collapse onto one node.

    Before the fix this face read the wrong block of the permutation buffer
    and returned the same node four times.
    """
    mesh = Mesh(str(MESHDIR / "prism_reference.msh"))
    V = FunctionSpace(mesh, "CG", 3)
    entity_dofs = V.finat_element.entity_dofs()
    nodes = V.cell_node_map().values[0]
    face_nodes = [int(nodes[dof]) for dof in entity_dofs[2][2]]
    assert len(face_nodes) == 4
    assert len(set(face_nodes)) == 4


def test_hexahedron_face_still_supplies_eight_orientations():
    """The regression guard on the cell the fix must not have moved."""
    mesh = UnitCubeMesh(2, 2, 2, hexahedral=True)
    V = FunctionSpace(mesh, "CG", 3)
    permutations = V.finat_element.entity_permutations
    for face in range(6):
        assert sorted(permutations[2][face]) == list(range(8))
    orientations = mesh.topology.entity_orientations
    # A hexahedron closure is 8 vertices, 12 edges, 6 faces, then the cell.
    assert orientations[:, 20:26].max() < 8
