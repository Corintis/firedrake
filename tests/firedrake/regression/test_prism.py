"""Unstructured prism meshes read from gmsh files.

The meshes come from ``tests/firedrake/meshes/prism/``. They hold
``DM_POLYTOPE_TRI_PRISM`` cells, which is the only prism cell type that this
code supports. Exterior facet integrals (``ds``) and interior facet integrals
(``dS``) are tested at the end of the file.

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
from mpi4py import MPI

from firedrake import (CellDiameter, Constant, DirichletBC, ExtrudedMesh,
                       FacetNormal, FiniteElement, Function, FunctionSpace, Mesh,
                       PointNotInDomainError, PointEvaluator,
                       SpatialCoordinate, TestFunction, TestFunctions,
                       TensorProductElement, TrialFunction, TrialFunctions, UnitCubeMesh,
                       UnitSquareMesh, VectorFunctionSpace,
                       VertexOnlyMeshMissingPointsError, VTKFile, as_vector,
                       assemble, avg, conditional, cos, dS, div, dot, ds, ds_b,
                       ds_t, ds_v, dx, errornorm, exp, grad, gt, inner, interval,
                       jump, pi, sin, solve, triangle)
from firedrake.petsc import PETSc


MESHDIR = Path(__file__).parent.parent / "meshes" / "prism"
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


# ---------------------------------------------------------- element families

@pytest.mark.parametrize("family, degree", [("HDiv Trace", 0), ("Bernstein", 1),
                                            ("DG L2", 1)])
def test_prism_unsupported_family_is_rejected(family, degree):
    """A family that the prism tensor product does not build correctly fails.

    Before the check, "HDiv Trace" 0 built a space with dimension 0 and no
    error: the product puts its dofs on the base edges, not on the facets.
    """
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    with pytest.raises(NotImplementedError, match="not supported on an unstructured prism"):
        FunctionSpace(mesh, family, degree)


def test_prism_supported_families_build():
    """CG, DG and Real on prism_slab.msh, which has 52 cells and 60 vertices."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    assert FunctionSpace(mesh, "CG", 1).dim() == 60
    assert FunctionSpace(mesh, "DG", 0).dim() == 52
    assert FunctionSpace(mesh, "DG", 1).dim() == 6 * 52
    R = FunctionSpace(mesh, "R", 0)
    assert R.dim() == 1
    assert np.isclose(assemble(Function(R).assign(2.0) * dx(domain=mesh)),
                      2.0 * assemble(Constant(1.0) * dx(domain=mesh)), rtol=0, atol=1e-12)


def test_prism_dof_count_that_changes_inside_one_polytope_type_is_rejected():
    """The numbering stops on an element whose edges carry different dof counts.

    The edges of a prism mesh have one polytope type, so they are one stratum
    with one dof count. The product of two "HDiv Trace" 0 elements has no dof
    on the base edges and one dof on each other edge. Built here directly, so
    that the FInAT check of the family does not stop it first. Before the
    check, the numbering took the count of edge 0 and dropped all 6 dofs.
    """
    import finat
    import finat.ufl
    import ufl
    from finat.element_factory import create_element, prism_tpc

    element = finat.FlattenedDimensions(create_element(finat.ufl.TensorProductElement(
        finat.ufl.FiniteElement("HDiv Trace", ufl.triangle, 0),
        finat.ufl.FiniteElement("HDiv Trace", ufl.interval, 0), cell=prism_tpc)))
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    with pytest.raises(NotImplementedError, match="different numbers of dofs"):
        mesh.topology.make_dofs_per_plex_entity(element.entity_dofs())


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


def _mixed_degree_prism_element(base_degree, axis_degree):
    """Continuous tensor product with independently selected prism degrees."""
    base = FiniteElement("CG", triangle, base_degree)
    axis = FiniteElement("CG", interval, axis_degree)
    return TensorProductElement(base, axis)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", ["prism_slab.msh",
                                      "prism_interior_marked_scrambled.msh"])
def test_prism_p2_base_p1_axis_interpolation_and_dS(meshname):
    """P2(triangle) x P1(interval) is conforming on axis-consistent prisms."""
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, _mixed_degree_prism_element(2, 1))
    assert V.finat_element.degree == 2
    if meshname == "prism_slab.msh":
        assert V.dim() == 195
    x, y, z = SpatialCoordinate(mesh)
    exact = (1 + x + 2 * y + x * y + x**2 - y**2) * (1 + 0.5 * z)
    u = Function(V).interpolate(exact)
    assert errornorm(exact, u, degree_rise=2) < 1e-11
    assert abs(assemble(jump(u)**2 * dS(domain=mesh))) < 1e-20


def test_prism_mixed_degree_rejects_axis_inconsistent_mesh():
    """A shared edge cannot be a base edge in one prism and an axis in another."""
    mesh = Mesh(str(MESHDIR / "prism_two_perpendicular.msh"))
    with pytest.raises(NotImplementedError, match="consistent base or axis role"):
        FunctionSpace(mesh, _mixed_degree_prism_element(2, 1))


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
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
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
    # The map below is the degree 1 one, so the coordinate field must be too.
    assert vertex_map.shape[1] == 6
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
        # A face that reads the wrong block of the permutation table gives the
        # same node to several of its dofs, so check for that before the sets.
        assert len(set(nodes_a)) == len(nodes_a)
        assert len(set(nodes_b)) == len(nodes_b)
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


# ------------------------------------------------------------------ VTK output

# The VTK cell types that a prism cell takes.
VTK_WEDGE = 13
VTK_LAGRANGE_WEDGE = 73

# The mesh that the exactness checks below run on.
#
# The rule that makes those checks exact is about DEGREE, not about whether a
# cell is affine. A gmsh prism mesh carries a degree 1 coordinate field, so x,
# y and z each lie in P1(triangle) x P1(interval). That space sits inside
# Pk(triangle) x Pk(interval) for every k >= 1, so interpolating the
# coordinates into the degree k output space reproduces them exactly. A total
# degree m monomial is in the degree k space exactly when m <= k. None of that
# asks whether the cell is warped.
#
# This name therefore records a choice, not a necessity. Do not read it as a
# claim that the warped mesh would fail these checks.
EXACTNESS_MESHNAME = "prism_slab.msh"


def _write_pvd(tmp_path, *functions, name="prism"):
    """Write functions to tmp_path/name.pvd.

    :returns: A pair (pvd, vtu) of paths. VTKFile puts the .vtu files in a
        subdirectory that carries the name of the .pvd file.
    """
    pvd = tmp_path / f"{name}.pvd"
    VTKFile(str(pvd)).write(*functions)
    return pvd, tmp_path / name / f"{name}_0.vtu"


def _read_vtu(path):
    """Read a .vtu with VTK itself, so VTK is the reader under test.

    :returns: A tuple (points, cells, types, point_data). points is an
        (npoints, 3) array. cells is an (ncells, nodes_per_cell) array of
        indices into points. types is an (ncells,) array of VTK cell types.
        point_data maps each array name to an (npoints, ...) array.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    points = vtk_to_numpy(grid.GetPoints().GetData())
    # GetCellType per cell rather than GetCellTypesArray, which VTK deprecated
    # at 9.6. These meshes hold tens of cells, so the loop costs nothing.
    types = np.array([grid.GetCellType(i)
                      for i in range(grid.GetNumberOfCells())], dtype="uint8")
    connectivity = vtk_to_numpy(grid.GetCells().GetConnectivityArray())
    offsets = vtk_to_numpy(grid.GetCells().GetOffsetsArray())
    sizes = np.diff(offsets)
    assert sizes.size == types.size
    # Every cell of a prism mesh holds the same number of nodes, so the
    # connectivity reshapes to one row per cell.
    assert np.all(sizes == sizes[0])
    cells = connectivity.reshape(types.size, int(sizes[0]))

    arrays = grid.GetPointData()
    point_data = {arrays.GetArrayName(i): vtk_to_numpy(arrays.GetArray(i))
                  for i in range(arrays.GetNumberOfArrays())}
    return points, cells, types, point_data


def _match_rows(points, table, tol=1e-10):
    """Find the row of table that each row of points sits on.

    :arg points: An (n, 3) array.
    :arg table: An (m, 3) array of points, which must be distinct.
    :returns: An (n,) array of indices into table.

    The rows of table must be distinct, or the nearest row below is not the
    only row and the answer means nothing. That is asserted here rather than
    at each call, so that no caller can leave it out.
    """
    assert np.unique(table.round(10), axis=0).shape[0] == table.shape[0]
    distance = np.linalg.norm(points[:, None, :] - table[None, :, :], axis=2)
    rows = distance.argmin(axis=1)
    assert distance[np.arange(points.shape[0]), rows].max() < tol
    return rows


def _prism_map(vertices, reference):
    """Map reference prism points through the degree 1 prism map.

    :arg vertices: A (6, 3) array of physical vertices in VTK wedge order,
        that is the first triangular face, then the second.
    :arg reference: An (n, 3) array of points of the reference prism.
    :returns: An (n, 3) array of physical points.
    """
    x, y, z = reference[:, 0], reference[:, 1], reference[:, 2]
    barycentric = np.stack([1.0 - x - y, x, y], axis=1)
    first = barycentric @ vertices[:3]
    second = barycentric @ vertices[3:]
    return (1.0 - z)[:, None] * first + z[:, None] * second


def _vtk_wedge_reference_points(degree):
    """The reference prism points of VTK's wedge, in VTK's own order.

    VTK is the only authority on the node layout of a VTK_LAGRANGE_WEDGE, so
    this asks VTK through the wrapper that the output module uses. The degree 1
    corners are asserted against a hand written list, which is independent.
    """
    from firedrake.output.paraview_reordering import vtk_wedge_local_to_cart

    points = np.array([np.asarray(p)
                       for p in vtk_wedge_local_to_cart((degree, degree))])
    # VTK puts the 6 corners first, in VTK_WEDGE order.
    corners = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    assert np.allclose(points[:6], corners)
    return points


def test_prism_vtk_output_writes_a_file(tmp_path):
    """VTKFile accepts a prism mesh and writes both files."""
    mesh = Mesh(str(MESHDIR / EXACTNESS_MESHNAME))
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V, name="f")
    pvd, vtu = _write_pvd(tmp_path, f)
    assert pvd.is_file()
    assert vtu.is_file()


def test_prism_vtk_output_writes_wedge_cells(tmp_path):
    """Every cell is a VTK_WEDGE of 6 nodes."""
    mesh = Mesh(str(MESHDIR / EXACTNESS_MESHNAME))
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V, name="f")
    _, vtu = _write_pvd(tmp_path, f)
    points, cells, types, _ = _read_vtu(vtu)

    _, gmsh_cells = _read_gmsh22_prisms(MESHDIR / EXACTNESS_MESHNAME)
    assert types.size == gmsh_cells.shape[0]
    assert np.all(types == VTK_WEDGE)
    assert cells.shape == (gmsh_cells.shape[0], 6)


def test_prism_vtk_output_node_order_matches_the_mesh(tmp_path):
    """The written node order is the order that VTK_WEDGE asks for.

    This is the test that catches a wrong node permutation. The gmsh file is
    the independent authority: gmsh writes a 6 node prism as one triangular
    face, then the opposite face, with node 4 across the axis from node 1. That
    is also what VTK_WEDGE asks for. A file written with a wrong permutation
    opens without an error and renders tangled cells, so nothing below may rely
    on the writer.
    """
    mesh = Mesh(str(MESHDIR / EXACTNESS_MESHNAME))
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V, name="f")
    _, vtu = _write_pvd(tmp_path, f)
    points, cells, types, _ = _read_vtu(vtu)

    coords, gmsh_cells = _read_gmsh22_prisms(MESHDIR / EXACTNESS_MESHNAME)
    by_vertices = {frozenset(int(v) for v in cell): [int(v) for v in cell]
                   for cell in gmsh_cells}
    assert len(by_vertices) == gmsh_cells.shape[0]

    for cell in cells:
        rows = [int(r) for r in _match_rows(points[cell], coords)]
        assert len(set(rows)) == 6
        gmsh_cell = by_vertices[frozenset(rows)]
        first, second = rows[:3], rows[3:]
        # The two triangular faces are contiguous triples. Which of the two
        # comes first is free, so accept either and name them to suit.
        near, far = gmsh_cell[:3], gmsh_cell[3:]
        if set(first) != set(near):
            near, far = far, near
        assert set(first) == set(near)
        assert set(second) == set(far)
        # Node k of the first triple and node k of the second triple are the
        # two ends of one axis edge.
        for k in range(3):
            assert far[near.index(first[k])] == second[k]


@pytest.mark.parametrize("meshname", MESHNAMES)
def test_prism_vtk_output_preserves_the_cell_handedness(tmp_path, meshname):
    """The written order must not mirror the cell.

    The test above does not pin the handedness, and on its own it is not
    enough. The permutation [0, 4, 2, 1, 5, 3] reverses the winding of both
    triangles while it keeps each triple a whole face and each axis pair
    together, so it passes every other test in this file. It writes a mirrored
    cell, which renders inside out and gives a negative volume.

    The check compares two reference frames that are defined independently of
    each other. The FIAT reference prism sends its x, y and z axes to the
    vertices 2, 4 and 1. VTK's reference wedge sends its r, s and t axes to the
    nodes 1, 2 and 3, which the corner list in _vtk_wedge_reference_points
    states. Both frames map onto the same physical cell, so the determinants of
    the two maps must carry the same sign. A mirrored order flips one and not
    the other.

    This holds on a warped cell too. The determinants are taken at the corner
    the two frames share, so the check is about orientation, not about whether
    the map is affine.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V, name="f")
    _, vtu = _write_pvd(tmp_path, f)
    points, cells, types, _ = _read_vtu(vtu)

    vertex_map = mesh.coordinates.function_space().cell_node_map().values
    coordinates = mesh.coordinates.dat.data_ro_with_halos
    assert vertex_map.shape == (cells.shape[0], 6)

    def sorted_rows(array):
        return np.array(sorted(tuple(row) for row in array))

    for cell in range(cells.shape[0]):
        fiat = coordinates[vertex_map[cell]]
        written = points[cells[cell]]
        # Guard the assumption that row `cell` of the connectivity belongs to
        # cell `cell` of the mesh. If that ever stops holding, fail here rather
        # than compare two different cells below.
        assert np.allclose(sorted_rows(fiat), sorted_rows(written),
                           rtol=0, atol=1e-10)
        det_fiat = np.linalg.det(np.stack([fiat[2] - fiat[0],
                                           fiat[4] - fiat[0],
                                           fiat[1] - fiat[0]]))
        det_vtk = np.linalg.det(np.stack([written[1] - written[0],
                                          written[2] - written[0],
                                          written[3] - written[0]]))
        # A flat cell would make the sign meaningless.
        assert abs(det_fiat) > 1e-12
        assert np.sign(det_vtk) == np.sign(det_fiat)


@pytest.mark.parametrize("degree", [2, 3])
def test_prism_vtk_output_writes_lagrange_wedge_cells(tmp_path, degree):
    """Above degree 1 the cell type is VTK_LAGRANGE_WEDGE."""
    mesh = Mesh(str(MESHDIR / EXACTNESS_MESHNAME))
    V = FunctionSpace(mesh, "CG", degree)
    f = Function(V, name="f")
    _, vtu = _write_pvd(tmp_path, f)
    points, cells, types, _ = _read_vtu(vtu)

    _, gmsh_cells = _read_gmsh22_prisms(MESHDIR / EXACTNESS_MESHNAME)
    assert types.size == gmsh_cells.shape[0]
    assert np.all(types == VTK_LAGRANGE_WEDGE)
    # A prism of degree k holds (k + 1)(k + 2)/2 nodes on the triangle and
    # k + 1 on the axis.
    assert cells.shape[1] == (degree + 1) * (degree + 2) // 2 * (degree + 1)


@pytest.mark.parametrize("degree", [2, 3])
def test_prism_vtk_output_points_sit_where_vtk_expects(tmp_path, degree):
    """Every written node sits at the place its VTK local index names.

    This extends the degree 1 node order test to the whole higher order
    layout. The degree 1 prism map through the 6 corners gives the physical
    place of every node exactly, because the coordinate field of a gmsh prism
    mesh has degree 1 and the output space has degree k >= 1, which contains
    it. That is a statement about degree, not about whether the cell is
    affine.

    Degree 1 is absent on purpose. The check below builds the map FROM the 6
    written corners, so at degree 1, where the corners are the whole cell, it
    reduces to got == got and holds whatever the permutation is. A mutation run
    confirmed that: breaking the linear permutation left the degree 1 case
    green. Degree 1 is covered instead by
    test_prism_vtk_output_node_order_matches_the_mesh, which checks against the
    gmsh file rather than against the written corners.
    """
    mesh = Mesh(str(MESHDIR / EXACTNESS_MESHNAME))
    V = FunctionSpace(mesh, "CG", degree)
    f = Function(V, name="f")
    _, vtu = _write_pvd(tmp_path, f)
    points, cells, types, _ = _read_vtu(vtu)

    reference = _vtk_wedge_reference_points(degree)
    assert cells.shape[1] == reference.shape[0]
    for cell in cells:
        got = points[cell]
        expected = _prism_map(got[:6], reference)
        assert np.allclose(got, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prism_vtk_output_point_data_matches_the_written_points(tmp_path, degree):
    """The value written at a node is the value of the function at that node.

    A linear function is in the space at every degree here, so the written
    values must match it to round off.
    """
    mesh = Mesh(str(MESHDIR / EXACTNESS_MESHNAME))
    V = FunctionSpace(mesh, "CG", degree)
    x, y, z = SpatialCoordinate(mesh)
    f = Function(V, name="linear").interpolate(2.0 * x - 3.0 * y + 5.0 * z + 1.0)
    _, vtu = _write_pvd(tmp_path, f)
    points, cells, types, point_data = _read_vtu(vtu)

    expected = (2.0 * points[:, 0] - 3.0 * points[:, 1]
                + 5.0 * points[:, 2] + 1.0)
    assert np.allclose(point_data["linear"], expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("meshname", MESHNAMES)
def test_prism_vtk_output_round_trips_the_geometry(tmp_path, meshname):
    """Every prism mesh writes, and the written cells hold the mesh vertices.

    This runs on all four meshes and checks the vertex sets only, so it is the
    broad guard that every mesh writes and survives the round trip. The node
    order is pinned by the two tests above.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V, name="f")
    _, vtu = _write_pvd(tmp_path, f)
    points, cells, types, _ = _read_vtu(vtu)

    coords, gmsh_cells = _read_gmsh22_prisms(MESHDIR / meshname)
    assert np.all(types == VTK_WEDGE)
    assert cells.shape == (gmsh_cells.shape[0], 6)
    written = sorted(sorted(_match_rows(points[cell], coords).tolist())
                     for cell in cells)
    expected = sorted(sorted(int(v) for v in cell) for cell in gmsh_cells)
    assert written == expected


def test_prism_vtk_output_takes_a_discontinuous_function(tmp_path):
    """A discontinuous function reaches get_sup_element, which must not ask
    for a "DQ" element on a prism."""
    mesh = Mesh(str(MESHDIR / EXACTNESS_MESHNAME))
    V = FunctionSpace(mesh, "DG", 1)
    f = Function(V, name="f")
    _, vtu = _write_pvd(tmp_path, f)
    points, cells, types, _ = _read_vtu(vtu)
    assert np.all(types == VTK_WEDGE)
    # A discontinuous output does not share a node between two cells.
    assert points.shape[0] == cells.shape[0] * 6


def test_prism_sup_element_is_discontinuous_lagrange():
    """get_sup_element must not ask for "DQ" on a prism.

    "DQ" is registered on the hypercubes only, so it raises on a prism. The
    discontinuous prism space is P_k(triangle) x P_k(interval), which is what
    "DG" builds.
    """
    import finat.ufl
    import ufl
    from firedrake.output.vtk_output import get_sup_element

    element = finat.ufl.FiniteElement("CG", cell=ufl.Cell("prism"), degree=2)
    sup = get_sup_element(element, continuous=False)
    assert sup.family() == "Discontinuous Lagrange"
    assert sup.cell.cellname == "prism"
    with pytest.raises(ValueError):
        finat.ufl.FiniteElement("DQ", cell=ufl.Cell("prism"), degree=2)


# The extruded wedge is the cell that the two shared functions this task edits
# must not move. get_sup_element picks the family of every cell type, and
# vtk_lagrange_wedge_reorder permutes the nodes of both wedge flavours.
#
def test_extruded_wedge_sup_element_takes_neither_new_branch():
    """get_sup_element must not send the extruded wedge down the prism branch.

    This asserts the branch condition, not the element it returns, and the
    reason is worth stating. No element level guard can do this job. For any
    ufl.TensorProductCell, canonical_element_description rewrites the family
    "Discontinuous Lagrange" to "DQ", so "DG" and "DQ" build the SAME element
    on a wedge. A get_sup_element that wrongly sent the wedge to "DG" would
    return an element equal to the right one, and a guard that compared
    elements or families would pass. The only observable difference is a
    warnings.warn at order >= 1, and pinning a warning string owned by FInAT
    would test a diagnostic rather than a behaviour.

    So the real invariant is the one below: the cellname of a wedge is
    "triangle * interval", which cannot equal "prism" or any other name in the
    set. The change to get_sup_element is therefore unreachable from a wedge.
    The measurement that the extruded wedge output does not move is the
    byte-identical comparison of its .vtu files, not this test.
    """
    import ufl

    wedge = ufl.TensorProductCell(ufl.Cell("triangle"), ufl.Cell("interval"))
    assert wedge.cellname == "triangle * interval"
    assert wedge.cellname not in {"interval", "triangle", "tetrahedron",
                                  "prism"}


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_extruded_wedge_output_is_unchanged(tmp_path, degree):
    """An extruded wedge still writes the wedge cell types it always wrote.

    vtk_lagrange_wedge_reorder now routes the degree through as_tuple. An
    extruded wedge reports the pair (k, k), which as_tuple leaves alone, so the
    permutation must be the one it always was. A DG output also drives the
    get_sup_element branch this task edits.

    A linear function alone is a weak probe of a permutation, because many
    wrong permutations leave it unchanged. The node place check below is the
    one that answers the permutation question. The degree 1 map through the 6
    corners is exact for the same reason it is exact on a prism mesh: the
    coordinate field has degree 1 and the output space has degree k >= 1, so
    the output space contains it. The rule is about degree, not about whether
    the cell is affine.
    """
    import ufl

    mesh = ExtrudedMesh(UnitSquareMesh(3, 3), 3)
    # An extruded triangle is a wedge, which is the cell under guard here.
    assert mesh.ufl_cell() == ufl.TensorProductCell(ufl.Cell("triangle"),
                                                    ufl.Cell("interval"))
    x, y, z = SpatialCoordinate(mesh)
    reference = _vtk_wedge_reference_points(degree)
    for family in ("CG", "DG"):
        V = FunctionSpace(mesh, family, degree)
        f = Function(V, name="linear").interpolate(
            2.0 * x - 3.0 * y + 5.0 * z + 1.0)
        _, vtu = _write_pvd(tmp_path, f, name=f"wedge_{family}{degree}")
        points, cells, types, point_data = _read_vtu(vtu)
        expected_type = VTK_WEDGE if degree == 1 else VTK_LAGRANGE_WEDGE
        assert np.all(types == expected_type)
        assert cells.shape[1] == reference.shape[0]
        assert cells.shape[1] == (degree + 1) * (degree + 2) // 2 * (degree + 1)
        for cell in cells:
            # The corner order, checked against the known shape of a flat
            # extrusion rather than against the written corners. The two
            # triangular faces sit at two constant heights, one above the
            # other, so nodes 0, 1, 2 share a z, nodes 3, 4, 5 share the other
            # z, and node k + 3 sits directly above node k. This does not go
            # vacuous at degree 1, unlike the map check below.
            corners = points[cell[:6]]
            assert np.allclose(corners[:3, 2], corners[0, 2], rtol=0, atol=1e-12)
            assert np.allclose(corners[3:, 2], corners[3, 2], rtol=0, atol=1e-12)
            assert not np.isclose(corners[0, 2], corners[3, 2])
            assert np.allclose(corners[:3, :2], corners[3:, :2],
                               rtol=0, atol=1e-12)
            # Every node sits at the place its VTK local index names. At
            # degree 1 this reduces to got == got, so the corner check above
            # is what carries that case.
            got = points[cell]
            assert np.allclose(got, _prism_map(got[:6], reference),
                               rtol=0, atol=1e-10)
        expected = (2.0 * points[:, 0] - 3.0 * points[:, 1]
                    + 5.0 * points[:, 2] + 1.0)
        assert np.allclose(point_data["linear"], expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_extruded_wedge_reorder_permutation_is_unchanged(degree):
    """The as_tuple change must not move the extruded wedge permutation.

    An extruded wedge reaches vtk_lagrange_wedge_reorder with the degree pair
    (k, k). as_tuple leaves a pair alone, so the permutation must equal the one
    the unchanged code computed from the same pair.

    This guard is for the extruded wedge only. Do not extend it to a prism. A
    prism reports a scalar degree, which is the whole reason the function
    changed, so the inlined old body below raises a TypeError there.
    """
    import finat.ufl
    import ufl
    from firedrake.output.paraview_reordering import (firedrake_local_to_cart,
                                                      invert,
                                                      vtk_lagrange_wedge_reorder,
                                                      vtk_wedge_local_to_cart)

    wedge = ufl.TensorProductCell(ufl.Cell("triangle"), ufl.Cell("interval"))
    element = finat.ufl.TensorProductElement(
        finat.ufl.FiniteElement("CG", ufl.triangle, degree, variant="equispaced"),
        finat.ufl.FiniteElement("CG", ufl.interval, degree, variant="equispaced"),
        cell=wedge)
    assert element.degree() == (degree, degree)
    # The right hand side is the body of the unchanged function.
    assert vtk_lagrange_wedge_reorder(element) == invert(
        vtk_wedge_local_to_cart(element.degree()),
        firedrake_local_to_cart(element))


# ------------------------------------ the PETSc DG1 transitive closure layout
#
# PETSc represents the coordinates of a periodic mesh with a DG1 element whose
# dofs follow the order of the vertices in the transitive closure of the cell.
# _get_firedrake_plex_permutation_dg_transitive_closure in
# firedrake/cython/dmcommon.pyx holds one permutation per cell type, and
# transform_vec_from_firedrake_to_petsc reads it as
#
#   petsc[petsc_offset + perm[i]] = firedrake[firedrake_offset + i]
#
# so perm[i] is the slot that the dof of FIAT vertex i takes.
#
# The 6 vertices of a TRI_PRISM fill the closure positions 15 to 20.
# _reorder_plex_closure sends the FIAT vertices 0 to 5 to the closure positions
# 15, 18, 16, 20, 17, 19, so the slot of FIAT vertex i is that position less the
# base 15.
#
#   FIAT vertex       0   1   2   3   4   5
#   closure position 15  18  16  20  17  19
#   PETSc DG1 slot    0   3   1   5   2   4
#
# The same subtraction on the hexahedron, whose vertices fill the closure
# positions 19 to 26 and whose FIAT vertices 0 to 7 go to 19, 23, 20, 26, 22,
# 24, 21, 25, gives (0, 4, 1, 7, 3, 5, 2, 6), which is the entry that the table
# already held. The tests below repeat neither arithmetic: they read both orders
# from the plex.
PRISM_DG_PERM = (0, 3, 1, 5, 2, 4)
HEXAHEDRON_DG_PERM = (0, 4, 1, 7, 3, 5, 2, 6)


def _dg_closure_vertex_perm(mesh):
    """Derive the PETSc DG1 vertex permutation from the plex of a mesh.

    :arg mesh: A mesh whose cell closure comes from _reorder_plex_closure,
        which is a hexahedral mesh or a prism mesh.
    :returns: The set of the permutations of the cells of this process.

    perm[i] is the place of FIAT vertex i in the list of the vertices of the
    transitive closure of the cell. This reads the FIAT order from
    ``cell_closure`` and the closure order from the plex, so it repeats no
    constant of dmcommon.pyx. The rule of _reorder_plex_closure is fixed, so
    every cell must give the same permutation and the set holds one entry.
    """
    dm = mesh.topology_dm
    cell_numbering = mesh.topology._cell_numbering
    cell_closure = mesh.topology.cell_closure
    vStart, vEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    perms = set()
    for plex_cell in range(cStart, cEnd):
        closure, _ = dm.getTransitiveClosure(plex_cell)
        closure_vertices = [p for p in closure if vStart <= p < vEnd]
        cell = cell_numbering.getOffset(plex_cell)
        fiat_vertices = cell_closure[cell][:len(closure_vertices)]
        perms.add(tuple(closure_vertices.index(v) for v in fiat_vertices))
    return perms


@pytest.mark.parametrize("meshname", MESHNAMES)
def test_prism_dg_permutation_table(meshname):
    from firedrake.cython import dmcommon

    mesh = Mesh(str(MESHDIR / meshname))
    ndofs, perm, perm_offsets = \
        dmcommon._get_firedrake_plex_permutation_dg_transitive_closure(
            mesh.topology_dm)
    assert list(ndofs) == [6, 0, 0, 0]
    assert list(perm_offsets) == [0, 6, 6, 6, 6]
    assert tuple(perm) == PRISM_DG_PERM
    # The table must agree with the plex, which is the independent derivation.
    assert _dg_closure_vertex_perm(mesh) == {PRISM_DG_PERM}


def test_prism_dg_permutation_is_not_the_identity():
    """The negative control of the table test above.

    A round trip that writes with perm and reads with perm gives the input back
    for every perm, so a test of that kind passes with the identity in the
    table. The entry is not the identity, so the tests must read the layout
    itself, which test_prism_plex_dg_coordinates_are_in_closure_order does.
    """
    from firedrake.cython import dmcommon

    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    _, perm, _ = dmcommon._get_firedrake_plex_permutation_dg_transitive_closure(
        mesh.topology_dm)
    assert sorted(perm) == list(range(6))
    assert tuple(perm) != tuple(range(6))


def test_hexahedron_dg_permutation_table_is_unchanged():
    """The control of the prism table test.

    _dg_closure_vertex_perm must reproduce the hexahedron entry, which was
    correct before this task. If it does, it derives the prism entry correctly
    too, because both cell types take the same fixed closure rule.
    """
    from firedrake.cython import dmcommon

    mesh = UnitCubeMesh(1, 1, 1, hexahedral=True)
    ndofs, perm, perm_offsets = \
        dmcommon._get_firedrake_plex_permutation_dg_transitive_closure(
            mesh.topology_dm)
    assert list(ndofs) == [8, 0, 0, 0]
    assert list(perm_offsets) == [0, 8, 8, 8, 8]
    assert tuple(perm) == HEXAHEDRON_DG_PERM
    assert _dg_closure_vertex_perm(mesh) == {HEXAHEDRON_DG_PERM}


def _localise_plex_coordinates(mesh):
    """Give the plex of a mesh the PETSc DG coordinates, in place.

    This is the body of _postprocess_periodic_mesh in
    firedrake/utility_meshes.py, which is the one caller of
    _set_dg_coordinates. _set_dg_coordinates is the writer that reads the
    permutation table.
    """
    from firedrake import FiniteElement, VectorFunctionSpace
    from firedrake.cython import dmcommon

    # The variant must be equispaced, as it is in _postprocess_periodic_mesh.
    # The permutation gives a slot to each FIAT VERTEX, so the nodes of the
    # element must sit at the vertices. The default variant puts them at the
    # Gauss points instead.
    element = FiniteElement("DG", mesh.ufl_cell(), 1, variant="equispaced")
    V = VectorFunctionSpace(mesh, element)
    coords = Function(V).interpolate(SpatialCoordinate(mesh))
    dmcommon._set_dg_coordinates(mesh.topology_dm, V.finat_element,
                                 V.dm.getLocalSection(), coords.dat._vec)


def _assert_plex_dg_coordinates_are_in_closure_order(mesh):
    """Check the PETSc DG coordinate layout of a mesh against its plex.

    PETSc orders the dofs of its DG1 coordinate element by the order of the
    vertices in the transitive closure of the cell, so slot k of a cell must
    hold the coordinates of closure vertex k. The expected values come from the
    CG coordinate section of the plex, which the permutation never touches, so
    this is an independent check of the layout.

    A round trip through _set_dg_coordinates and back is NOT such a check: both
    directions read the same table, so they agree for every permutation. This
    test fails when the table is wrong, because a wrong slot then holds the
    coordinates of a different vertex.
    """
    dm = mesh.topology_dm
    gdim = dm.getCoordinateDim()
    cg_section = dm.getCoordinateSection()
    cg_coords = dm.getCoordinatesLocal().array_r.reshape(-1, gdim).copy()
    vStart, vEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    closure_vertices = {}
    for plex_cell in range(cStart, cEnd):
        closure, _ = dm.getTransitiveClosure(plex_cell)
        closure_vertices[plex_cell] = [p for p in closure
                                       if vStart <= p < vEnd]
    _localise_plex_coordinates(mesh)
    cell_section = dm.getCellCoordinateSection()
    cell_coords = dm.getCellCoordinatesLocal().array_r.reshape(-1, gdim)
    assert cStart < cEnd
    for plex_cell, vertices in closure_vertices.items():
        offset = cell_section.getOffset(plex_cell) // gdim
        for k, vertex in enumerate(vertices):
            expected = cg_coords[cg_section.getOffset(vertex) // gdim]
            assert np.allclose(cell_coords[offset + k], expected,
                               rtol=0, atol=1e-13)


@pytest.mark.parametrize("meshname", MESHNAMES)
def test_prism_plex_dg_coordinates_are_in_closure_order(meshname):
    _assert_plex_dg_coordinates_are_in_closure_order(
        Mesh(str(MESHDIR / meshname)))


def test_hexahedron_plex_dg_coordinates_are_in_closure_order():
    """The control of the prism layout test.

    The hexahedron entry was correct before this task, so this pins that the
    check itself passes on a cell type whose answer is known.
    """
    _assert_plex_dg_coordinates_are_in_closure_order(
        UnitCubeMesh(1, 1, 1, hexahedral=True))


# --------------------------------------------------------------- checkpoint io
#
# A checkpoint stores the cones of the plex and not the cell types, so PETSc
# infers the type of every cell on load. A cell of 2 triangles and 3
# quadrilaterals is either a DM_POLYTOPE_TRI_PRISM or a
# DM_POLYTOPE_TRI_PRISM_TENSOR, and the PETSc default picks the tensor type,
# which Firedrake does not support. firedrake/checkpointing.py calls
# dmcommon._relabel_tensor_prisms_from_checkpoint after topologyLoad to correct
# that.
#
# CAUTION. These tests cover prism checkpointing. They are NOT evidence about
# the PETSc DG1 permutation, and no review may cite them as such. A round trip
# applies the permutation table in both directions, so a wrong table cancels
# itself. This is measured, not argued: with perm set to 0, 3, 1, 5, 4, 2 every
# test below still passed, while test_prism_dg_permutation_table and
# test_prism_plex_dg_coordinates_are_in_closure_order failed on all four prism
# meshes. The second of those is the test that covers the permutation.

CHECKPOINT_MESH_NAME = "prism_checkpoint_mesh"


def _checkpoint_expr(mesh, vector):
    """An expression whose value differs at every node of a prism mesh.

    A field that a permutation of the vertices leaves alone proves nothing, so
    the coefficients are incommensurable and none of the meshes has a symmetry
    that maps one node onto another with the same value.
    """
    from firedrake import as_vector

    x, y, z = SpatialCoordinate(mesh)
    base = 1.0 + np.pi * x + np.e * y + np.sqrt(2.0) * z + 0.25 * x * y
    if not vector:
        return base
    return as_vector([base, 2.0 * base + x * z, 3.0 - base + y * z])


def _assert_checkpoint_round_trip(tmp_path, meshname, degree, vector=False):
    """Save a mesh and a function, load them back, and compare."""
    from firedrake import VectorFunctionSpace
    from firedrake.checkpointing import CheckpointFile
    from pyop2.mpi import COMM_WORLD

    filename = COMM_WORLD.bcast(str(Path(tmp_path) / "prism_checkpoint.h5"),
                                root=0)
    mesh = Mesh(str(MESHDIR / meshname), name=CHECKPOINT_MESH_NAME)
    make_space = VectorFunctionSpace if vector else FunctionSpace
    V = make_space(mesh, "CG", degree)
    f = Function(V, name="f").interpolate(_checkpoint_expr(mesh, vector))
    if COMM_WORLD.size == 1:
        rows = f.dat.data_ro.reshape(f.dat.data_ro.shape[0], -1)
        assert len(np.unique(rows, axis=0)) == rows.shape[0], \
            "the field must differ at every node, or a wrong slot stays hidden"
    with CheckpointFile(filename, "w", comm=COMM_WORLD) as chk:
        chk.save_mesh(mesh)
        chk.save_function(f)
    with CheckpointFile(filename, "r", comm=COMM_WORLD) as chk:
        loaded_mesh = chk.load_mesh(CHECKPOINT_MESH_NAME)
        g = chk.load_function(loaded_mesh, "f")
    assert loaded_mesh.ufl_cell().cellname == "prism"
    assert loaded_mesh.topology.dm_cell_types == \
        (PETSc.DM.PolytopeType.TRI_PRISM,)
    expected = Function(g.function_space()).interpolate(
        _checkpoint_expr(loaded_mesh, vector))
    # The loaded coordinates equal the saved ones to the last bit, so the two
    # fields agree to the last bit and the squared difference is zero.
    assert assemble(inner(g - expected, g - expected) * dx) < 1e-18
    if COMM_WORLD.size == 1:
        # A load on one process keeps the distribution and the numbering, so
        # the dofs must agree one by one.
        assert np.allclose(g.dat.data_ro, f.dat.data_ro, rtol=0, atol=1e-14)


@pytest.mark.parametrize("meshname", MESHNAMES)
def test_prism_checkpoint_round_trips_a_cg1_function(tmp_path, meshname):
    _assert_checkpoint_round_trip(tmp_path, meshname, 1)


@pytest.mark.parametrize("degree", [2, 3])
def test_prism_checkpoint_round_trips_a_higher_degree_function(tmp_path,
                                                               degree):
    _assert_checkpoint_round_trip(tmp_path, "prism_slab.msh", degree)


@pytest.mark.parametrize("degree", [1, 2])
def test_prism_checkpoint_round_trips_a_vector_function(tmp_path, degree):
    _assert_checkpoint_round_trip(tmp_path, "prism_slab.msh", degree,
                                  vector=True)


def test_prism_checkpoint_round_trips_the_warped_mesh(tmp_path):
    """prism_warped.msh has non-affine cells whose axes tilt differently."""
    _assert_checkpoint_round_trip(tmp_path, "prism_warped.msh", 1)


@pytest.mark.parallel(nprocs=[2, 3])
def test_prism_checkpoint_round_trips_in_parallel(tmp_path):
    _assert_checkpoint_round_trip(tmp_path, "prism_slab.msh", 1)


@pytest.mark.parallel(nprocs=[2, 3])
def test_prism_checkpoint_round_trips_cg2_in_parallel(tmp_path):
    _assert_checkpoint_round_trip(tmp_path, "prism_slab.msh", 2)


def test_relabel_tensor_prisms_from_checkpoint_leaves_a_prism_mesh_alone(meshname):
    from firedrake.cython import dmcommon

    mesh = Mesh(str(MESHDIR / meshname))
    dm = mesh.topology_dm
    dmcommon._relabel_tensor_prisms_from_checkpoint(dm)
    cStart, cEnd = dm.getHeightStratum(0)
    for c in range(cStart, cEnd):
        assert dm.getCellType(c) == PETSc.DM.PolytopeType.TRI_PRISM


def test_relabel_tensor_prisms_from_checkpoint_leaves_a_hexahedral_mesh_alone():
    from firedrake.cython import dmcommon

    mesh = UnitCubeMesh(1, 1, 1, hexahedral=True)
    dm = mesh.topology_dm
    dmcommon._relabel_tensor_prisms_from_checkpoint(dm)
    cStart, cEnd = dm.getHeightStratum(0)
    for c in range(cStart, cEnd):
        assert dm.getCellType(c) == PETSc.DM.PolytopeType.HEXAHEDRON


def test_prism_checkpoint_load_agrees_with_the_cell_type_label(tmp_path):
    """The cell type cache and the cell type label of a loaded plex must agree.

    labelsLoad calls DMRemoveLabel on the cell type label, which leaves the
    cell type cache of the plex behind. A relabel that runs after labelsLoad
    therefore corrects the label alone and leaves the cache holding the tensor
    type. getCellType reads the cache and getCellTypeLabel reads the label, so
    comparing the two catches that.
    """
    from firedrake.checkpointing import CheckpointFile
    from pyop2.mpi import COMM_WORLD

    filename = COMM_WORLD.bcast(str(Path(tmp_path) / "prism_celltype.h5"),
                                root=0)
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"), name=CHECKPOINT_MESH_NAME)
    with CheckpointFile(filename, "w", comm=COMM_WORLD) as chk:
        chk.save_mesh(mesh)
    with CheckpointFile(filename, "r", comm=COMM_WORLD) as chk:
        loaded_mesh = chk.load_mesh(CHECKPOINT_MESH_NAME)
    dm = loaded_mesh.topology_dm
    label = dm.getCellTypeLabel()
    cStart, cEnd = dm.getHeightStratum(0)
    assert cEnd > cStart
    for c in range(cStart, cEnd):
        assert dm.getCellType(c) == PETSc.DM.PolytopeType.TRI_PRISM
        assert label.getValue(c) == int(PETSc.DM.PolytopeType.TRI_PRISM)


# -------------------------------------------------------- mixed function spaces
#
# A mixed space stacks the dofs of its subspaces into one vector. Every
# subspace keeps its own local numbering, and the offset that separates the
# fields lives in the local to global map of the mixed dof dataset. The tests
# below check that offset, and then solve a coupled system that is only exact
# if the offset is right.

MIXED_DEGREES = ((2, 1), (3, 2))


def _mixed_field_offsets(W):
    """The start of each field inside the local vector of a mixed space.

    :arg W: A mixed :class:`~.FunctionSpace`.
    :returns: A pair (starts, owned). starts[f] is the first local index of
        field f. owned[f] is the number of dofs of field f that this rank owns.
        The local block of a field holds its owned dofs first, then its halo.
    """
    total = [W.dof_dset[f].total_size for f in range(len(W))]
    owned = [W.dof_dset[f].size for f in range(len(W))]
    starts = np.concatenate([[0], np.cumsum(total)[:-1]]).astype(int)
    return starts, owned


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degrees", MIXED_DEGREES)
def test_prism_mixed_function_space_dimension(degrees):
    """W.dim() is the sum of the dimensions of the subspaces."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degrees[0])
    Q = FunctionSpace(mesh, "CG", degrees[1])
    W = V * Q
    assert W.dim() == V.dim() + Q.dim()
    assert W.dim() == _expected_cg_dim(degrees[0]) + _expected_cg_dim(degrees[1])


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degrees", MIXED_DEGREES)
def test_prism_mixed_owned_dofs_are_one_contiguous_block_per_field(degrees):
    """The owned dofs of the fields sit next to each other, in field order.

    A rank owns one contiguous block of the global vector. Field 0 takes the
    front of that block and field 1 follows it. An offset that counted the
    halo, or that counted the other field twice, moves the second field.
    """
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degrees[0])
    Q = FunctionSpace(mesh, "CG", degrees[1])
    W = V * Q
    comm = mesh.comm
    starts, owned = _mixed_field_offsets(W)
    lgmap = W.dof_dset.lgmap.indices
    # The first global index this rank owns, over both fields.
    base = comm.scan(sum(owned)) - sum(owned)
    expected = base
    for field in range(len(W)):
        block = lgmap[starts[field]:starts[field] + owned[field]]
        assert np.array_equal(block, expected + np.arange(owned[field]))
        expected += owned[field]


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degrees", MIXED_DEGREES)
def test_prism_mixed_cell_node_map_reaches_every_dof_once(degrees):
    """The cell node maps of the subspaces, offset, cover the mixed space.

    Each subspace map holds its own local node numbers. The global index of
    one of them is lgmap[start of the field + local node]. Over all cells and
    all ranks those global indices must be exactly 0 to W.dim() - 1. A wrong
    offset either leaves a gap or makes two fields collide, and both show up
    here.
    """
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degrees[0])
    Q = FunctionSpace(mesh, "CG", degrees[1])
    W = V * Q
    starts, _ = _mixed_field_offsets(W)
    lgmap = W.dof_dset.lgmap.indices
    submaps = list(W.cell_node_map())
    reached = np.concatenate([
        lgmap[starts[field] + np.unique(submaps[field].values)]
        for field in range(len(W))
    ])
    # Do the collectives before any assert. An assert that fails on one rank
    # only would otherwise leave the other ranks in the collective, and the
    # run would hang in place of a failure. The first W.dim() is collective.
    gathered = np.concatenate(mesh.comm.allgather(reached))
    dim = W.dim()
    # The subspace maps are the maps of the standalone spaces. The offset is
    # not folded into them.
    assert np.array_equal(submaps[0].values, V.cell_node_map().values)
    assert np.array_equal(submaps[1].values, Q.cell_node_map().values)
    assert np.unique(gathered).size == dim
    assert gathered.min() == 0
    assert gathered.max() == dim - 1


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degrees", MIXED_DEGREES)
def test_prism_mixed_poisson_is_exact(degrees):
    """A coupled two field system reproduces a solution of the mixed space.

    The system is

        (grad u, grad v) + (p, v) = (f, v)
        (u, q) - (p, q)           = (g, q)

    with f = -div(grad(u)) + p and g = u - p. Both exact fields are
    polynomials of the corresponding space, so the discrete solution is the
    exact one. The block is [[A, M], [M, -M]], which is invertible, so the
    solution is unique and the comparison is meaningful.
    """
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "CG", degrees[0])
    Q = FunctionSpace(mesh, "CG", degrees[1])
    W = V * Q
    x, y, z = SpatialCoordinate(mesh)
    u_exact = 1.0 + x + 2.0 * y + x**2 + y**2 - 2.0 * z**2
    p_exact = 1.0 + 2.0 * x - y + 0.5 * z
    f = -div(grad(u_exact)) + p_exact
    g = u_exact - p_exact
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a = (inner(grad(u), grad(v)) * dx + inner(p, v) * dx
         + inner(u, q) * dx - inner(p, q) * dx)
    L = inner(f, v) * dx + inner(g, q) * dx
    w = Function(W)
    bc = DirichletBC(W.sub(0), u_exact, "on_boundary")
    solve(a == L, w, bcs=[bc],
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                             "mat_type": "aij"})
    uh, ph = w.subfunctions
    u_error = float(np.sqrt(abs(assemble(inner(uh - u_exact,
                                               uh - u_exact) * dx))))
    p_error = float(np.sqrt(abs(assemble(inner(ph - p_exact,
                                               ph - p_exact) * dx))))
    assert u_error < 1e-10
    assert p_error < 1e-10


@pytest.mark.parallel([1, 2, 3])
def test_prism_mixed_vector_scalar_system_is_exact():
    """A Stokes like system on a vector subspace and a scalar subspace.

    The first field is a vector space, so its subspace carries three dofs per
    node and the field offset is not the node count. The pressure block is
    -(p, q), which keeps the system invertible for any element pair, so the
    test measures the prism and not the inf-sup constant of the pair.
    """
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    assert W.dim() == 3 * _expected_cg_dim(2) + _expected_cg_dim(1)
    x, y, z = SpatialCoordinate(mesh)
    u_exact = as_vector([x * y - z**2, y * z + x**2, x * z - y**2])
    p_exact = 1.0 + 2.0 * x - y + 0.5 * z
    f = -div(grad(u_exact)) - grad(p_exact)
    g = div(u_exact) - p_exact
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a = (inner(grad(u), grad(v)) * dx + inner(p, div(v)) * dx
         + inner(div(u), q) * dx - inner(p, q) * dx)
    L = inner(f, v) * dx + inner(g, q) * dx
    w = Function(W)
    bc = DirichletBC(W.sub(0), u_exact, "on_boundary")
    solve(a == L, w, bcs=[bc],
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                             "mat_type": "aij"})
    uh, ph = w.subfunctions
    u_error = float(np.sqrt(abs(assemble(inner(uh - u_exact,
                                               uh - u_exact) * dx))))
    p_error = float(np.sqrt(abs(assemble(inner(ph - p_exact,
                                               ph - p_exact) * dx))))
    assert u_error < 1e-10
    assert p_error < 1e-10


# ---------------------------------------------------------- convergence order
#
# Every other solver test on a prism asks for exactness: the exact solution is
# a member of the finite element space, so the discrete solution reproduces it
# and the error is zero. Exactness does not see a geometry error or a
# quadrature error that is itself exact on the space. The order of convergence
# for a solution OUTSIDE the space does see both, so the tests below refine a
# sequence of meshes and measure that order.
#
# The meshes are warped, so no prism is affine and the Jacobian varies inside
# every cell. Their element size halves exactly from one level to the next;
# see tests/firedrake/meshes/prism/make_prism_mesh.py for why that matters.

ORDER_MESHNAMES = ("prism_order_r0.msh", "prism_order_r1.msh",
                   "prism_order_r2.msh")


def _poisson_order_step(meshname, degree):
    """Solve Poisson for a non-polynomial solution on one mesh of the sequence.

    :arg meshname: The file name of the mesh.
    :arg degree: The degree of the CG space.
    :returns: A pair (h, error). h is the largest cell diameter of the mesh.
        error is the L2 norm of the difference from the exact solution.

    The exact solution is smooth and is not a polynomial, so it is in no CG
    space of the sequence. The right hand side is its own Laplacian, and the
    boundary condition is the exact solution itself, so the exact solution of
    the continuous problem is the same on every mesh of the sequence even
    though the meshes describe slightly different polyhedra.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", degree)
    x, y, z = SpatialCoordinate(mesh)
    exact = sin(pi * x) * cos(pi * y) * exp(z)
    f = -div(grad(exact))
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, exact, "on_boundary")
    solve(inner(grad(u), grad(v)) * dx - inner(f, v) * dx == 0, u, bcs=bc,
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    error = float(np.sqrt(abs(assemble(inner(u - exact, u - exact) * dx))))
    diameters = Function(FunctionSpace(mesh, "DG", 0))
    diameters.interpolate(CellDiameter(mesh))
    h = float(diameters.dat.data_ro.max())
    return h, error


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prism_poisson_converges_at_the_expected_order(degree):
    """The L2 error of a CG space of degree k falls as h to the power k+1."""
    steps = [_poisson_order_step(name, degree) for name in ORDER_MESHNAMES]
    orders = []
    for (h_coarse, e_coarse), (h_fine, e_fine) in zip(steps[:-1], steps[1:]):
        assert e_fine < e_coarse
        orders.append(np.log(e_coarse / e_fine) / np.log(h_coarse / h_fine))
    # The lower bound is the test. The upper bound catches an error that
    # collapsed into round-off instead of converging.
    for order in orders:
        assert degree + 1 - 0.3 < order < degree + 1 + 0.5, \
            f"degree {degree} orders {orders} from {steps}"


# ----------------------------------------------------- point location, .at
#
# Point location asks a different question of the cell than assembly does. It
# inverts the coordinate map of the cell, so it needs the reference cell of a
# prism, its bounding box in the rtree, and a rule that says whether a
# reference point is inside. None of that is exercised by an integral.

# The vertices of a prism in FIAT order. The element is a tensor product, so
# vertex i is triangle vertex i // 2 at interval vertex i % 2.
# test_prism_reference_vertices_are_in_tensor_product_order pins this.
PRISM_TRIANGULAR_FACE_VERTICES = ((0, 2, 4), (1, 3, 5))
PRISM_QUADRILATERAL_FACE_VERTICES = ((0, 1, 2, 3), (2, 3, 4, 5), (0, 1, 4, 5))
PRISM_EDGE_VERTICES = ((0, 1), (2, 3), (4, 5),
                       (0, 2), (2, 4), (0, 4),
                       (1, 3), (3, 5), (1, 5))
PRISM_VERTEX_GROUPS = (PRISM_TRIANGULAR_FACE_VERTICES
                       + PRISM_QUADRILATERAL_FACE_VERTICES
                       + PRISM_EDGE_VERTICES
                       + tuple((i,) for i in range(6)))

# How far a probe point moves from the centre of the cell towards a face, an
# edge or a vertex of it. 0.98 puts the point close to the boundary of the
# cell and keeps it inside, on every mesh of MESHNAMES.
PROBE_WEIGHT = 0.98


def _at(function, points, **kwargs):
    """Function.at, without the warning that it is deprecated.

    Function.at is deprecated in favour of PointEvaluator, but the two are
    different code paths. at uses the compiled evaluation kernel and the rtree
    of the mesh. PointEvaluator builds a VertexOnlyMesh. Both are tested here.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return function.at(points, **kwargs)


def _cell_vertices(mesh):
    """The vertex coordinates of every cell, in FIAT order.

    :arg mesh: The mesh.
    :returns: An array of shape (ncells, 6, gdim).
    """
    coordinates = mesh.coordinates
    cell_nodes = coordinates.function_space().cell_node_map().values
    return coordinates.dat.data_ro_with_halos[cell_nodes]


def _probe_points(vertices):
    """Interior points of one cell, near its centre, faces, edges and vertices.

    :arg vertices: The (6, gdim) vertex coordinates of the cell.
    :returns: An array of points inside the cell.
    """
    centre = vertices.mean(axis=0)
    targets = [centre]
    targets += [vertices[list(group)].mean(axis=0)
                for group in PRISM_VERTEX_GROUPS]
    return centre + PROBE_WEIGHT * (np.asarray(targets) - centre)


def test_prism_reference_vertices_are_in_tensor_product_order():
    """Vertex i of a prism is triangle vertex i // 2 at interval vertex i % 2.

    The face and edge groups above read the vertices in that order, so a
    change to it must fail here rather than move a probe point outside its
    cell without saying so.
    """
    mesh = Mesh(str(MESHDIR / "prism_reference.msh"))
    triangle = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    expected = np.array([[triangle[i // 2][0], triangle[i // 2][1], i % 2]
                         for i in range(6)])
    assert np.allclose(_cell_vertices(mesh)[0], expected)


def test_prism_locate_cell_finds_the_cell_of_every_interior_point(meshname):
    """locate_cell returns the cell that a point of that cell belongs to.

    The points cover the centre of each cell and points at 98 per cent of the
    way to each face, each edge and each vertex, so a cell whose reference map
    is inverted wrongly near its boundary is found here.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    vertices = _cell_vertices(mesh)
    for cell in range(vertices.shape[0]):
        for point in _probe_points(vertices[cell]):
            assert mesh.locate_cell(point) == cell, \
                f"{meshname} cell {cell} point {point}"


def test_prism_point_evaluation_is_exact(meshname):
    """A quadratic is evaluated exactly at interior points of every cell.

    A quadratic of the physical coordinates pulls back into the degree 2 space
    of a prism, because the coordinate map is degree 1 on the triangle and
    degree 1 on the interval. The interpolant is therefore the quadratic
    itself, and point evaluation must return its value.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", 2)
    x, y, z = SpatialCoordinate(mesh)
    u = Function(V).interpolate(1.0 + x + 2.0 * y - 0.5 * z + x * y - z * z)

    def exact(p):
        return 1.0 + p[0] + 2.0 * p[1] - 0.5 * p[2] + p[0] * p[1] - p[2]**2

    vertices = _cell_vertices(mesh)
    points = np.concatenate([_probe_points(vertices[cell])
                             for cell in range(vertices.shape[0])])
    expected = np.array([exact(p) for p in points])
    assert np.allclose(_at(u, points), expected, rtol=0, atol=1e-12)
    assert np.allclose(PointEvaluator(mesh, points).evaluate(u), expected,
                       rtol=0, atol=1e-12)


def test_prism_point_evaluation_outside_the_mesh_fails(meshname):
    """A point outside the mesh is reported, not answered with a number."""
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", 2)
    x, y, z = SpatialCoordinate(mesh)
    u = Function(V).interpolate(x + y + z)
    outside = np.array([5.0, 5.0, 5.0])
    assert mesh.locate_cell(outside) is None
    with pytest.raises(PointNotInDomainError):
        _at(u, outside)
    assert _at(u, outside, dont_raise=True) is None
    with pytest.raises(VertexOnlyMeshMissingPointsError):
        PointEvaluator(mesh, [outside]).evaluate(u)


# ------------------------------------------------------ facet integrals, ds
#
# A prism has three quadrilateral facets and two triangular facets, and TSFC
# compiles one kernel per facet shape. compile_form splits a plain ds into one
# exterior_facet_quad and one exterior_facet_tri integral. The mesh gives each
# kernel the facets of its shape inside the subdomain, and the local facet
# number of each facet WITHIN its shape group. A correct total can hide two
# errors that cancel, so the tests below measure each facet, or each marker,
# separately.

REFERENCE_MARKED_MESHNAME = "prism_reference_marked.msh"
MIXED_MARKER_MESHNAME = "prism_slab_mixed_marker.msh"

# The five facets of prism_reference_marked.msh, by physical group: the
# area and the outward unit normal. See tests/firedrake/meshes/prism/make_prism_mesh.py.
REFERENCE_FACETS = {
    1: (0.5, (0.0, 0.0, -1.0)),
    2: (0.5, (0.0, 0.0, 1.0)),
    3: (1.0, (0.0, -1.0, 0.0)),
    4: (1.0, (-1.0, 0.0, 0.0)),
    5: (np.sqrt(2.0), (np.sqrt(0.5), np.sqrt(0.5), 0.0)),
}
REFERENCE_SURFACE_AREA = 3.0 + np.sqrt(2.0)

# The area of each physical group of prism_slab.msh (1 bottom, 2 top,
# 3 sides) and of prism_slab_mixed_marker.msh (2 top, 3 three sides, 4 bottom
# and the side y = 0). The slab is the unit square times [0, 0.6].
SLAB_HEIGHT = 0.6
SLAB_MARKER_AREAS = {1: 1.0, 2: 1.0, 3: 4.0 * SLAB_HEIGHT}
MIXED_MARKER_AREAS = {2: 1.0, 3: 3.0 * SLAB_HEIGHT, 4: 1.0 + SLAB_HEIGHT}
SLAB_SURFACE_AREA = 2.0 + 4.0 * SLAB_HEIGHT


def _ds_integral_types(form):
    """The integral type of each kernel that compile_form makes for a form."""
    from firedrake.tsfc_interface import compile_form

    return sorted(k.kinfo.integral_type for k in compile_form(form, "form"))


def _read_gmsh22_boundary_facets(path):
    """Read the marked boundary facets of a gmsh 2.2 ASCII prism file.

    :arg path: The path of the .msh file.
    :returns: A list of pairs (marker, vector area). The vector area is the
        integral of the outward unit normal over the facet.

    This is independent of Firedrake. The integral of the normal over a
    surface depends only on the boundary curve of the surface: it is half the
    sum of the cross products of consecutive vertices. That holds for the
    bilinear quadrilateral facets of a non-affine prism too. The sign comes
    from the adjacent prism: the outward normal points away from its centroid.
    """
    coords, cells = _read_gmsh22_prisms(path)
    lines = Path(path).read_text().split("\n")
    tag_to_row = {}
    i = lines.index("$Nodes")
    for row in range(int(lines[i + 1])):
        tag_to_row[int(lines[i + 2 + row].split()[0])] = row
    cell_of_vertex_set = {}
    for cell in cells:
        for face in ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)):
            cell_of_vertex_set[frozenset(cell[list(face)])] = cell
    j = lines.index("$Elements")
    facets = []
    for row in range(int(lines[j + 1])):
        fields = [int(x) for x in lines[j + 2 + row].split()]
        # Type 2 is the 3-node triangle, type 3 the 4-node quadrilateral.
        if fields[1] not in (2, 3):
            continue
        ntags = fields[2]
        marker = fields[3]
        vertices = [tag_to_row[t] for t in fields[3 + ntags:]]
        points = coords[vertices]
        area = 0.5 * sum(np.cross(points[k], points[(k + 1) % len(points)])
                         for k in range(len(points)))
        cell = cell_of_vertex_set[frozenset(vertices)]
        outward = points.mean(axis=0) - coords[cell].mean(axis=0)
        facets.append((marker, area if np.dot(area, outward) > 0 else -area))
    return facets


def test_prism_ds_compiles_to_one_kernel_per_facet_shape():
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    assert _ds_integral_types(Constant(1.0) * ds(domain=mesh)) == \
        ["exterior_facet_quad", "exterior_facet_tri"]
    assert _ds_integral_types(Constant(1.0) * ds(3, domain=mesh)) == \
        ["exterior_facet_quad", "exterior_facet_tri"]


@pytest.mark.parallel([1, 2, 3])
def test_prism_ds_measures_each_facet_separately():
    """The total surface area and the area of each of the five facets."""
    mesh = Mesh(str(MESHDIR / REFERENCE_MARKED_MESHNAME))
    total = assemble(Constant(1.0) * ds(domain=mesh))
    assert np.isclose(total, REFERENCE_SURFACE_AREA, rtol=0, atol=1e-12)
    for marker, (area, _) in REFERENCE_FACETS.items():
        got = assemble(Constant(1.0) * ds(marker, domain=mesh))
        assert np.isclose(got, area, rtol=0, atol=1e-12), f"facet {marker}"


@pytest.mark.parallel([1, 2, 3])
def test_prism_ds_integrates_a_coordinate_on_each_facet():
    """The integral of x + 2 y + 4 z on each facet, from the facet centroid.

    The integrand is linear, so its integral is the area times the value at
    the centroid. Unlike a constant, it depends on WHERE each quadrature
    point goes, so a kernel that maps the reference facet onto the wrong
    physical facet fails here even when the areas are right.
    """
    mesh = Mesh(str(MESHDIR / REFERENCE_MARKED_MESHNAME))
    x, y, z = SpatialCoordinate(mesh)
    centroids = {1: (1 / 3, 1 / 3, 0.0), 2: (1 / 3, 1 / 3, 1.0),
                 3: (0.5, 0.0, 0.5), 4: (0.0, 0.5, 0.5), 5: (0.5, 0.5, 0.5)}
    for marker, (area, _) in REFERENCE_FACETS.items():
        cx, cy, cz = centroids[marker]
        got = assemble((x + 2 * y + 4 * z) * ds(marker, domain=mesh))
        assert np.isclose(got, area * (cx + 2 * cy + 4 * cz), rtol=0, atol=1e-12), \
            f"facet {marker}"


@pytest.mark.parallel([1, 2, 3])
def test_prism_ds_marker_composes_with_the_facet_shape():
    """ds(marker) on a marker that holds one facet shape only."""
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    for marker, area in SLAB_MARKER_AREAS.items():
        got = assemble(Constant(1.0) * ds(marker, domain=mesh))
        assert np.isclose(got, area, rtol=0, atol=1e-12), f"marker {marker}"
    total = assemble(Constant(1.0) * ds(domain=mesh))
    by_marker = assemble(Constant(1.0) * (ds(1, domain=mesh) + ds(2, domain=mesh)
                                          + ds(3, domain=mesh)))
    assert np.isclose(total, SLAB_SURFACE_AREA, rtol=0, atol=1e-12)
    assert np.isclose(by_marker, total, rtol=0, atol=1e-12)
    two = assemble(Constant(1.0) * ds((1, 3), domain=mesh))
    assert np.isclose(two, SLAB_MARKER_AREAS[1] + SLAB_MARKER_AREAS[3], rtol=0, atol=1e-12)


@pytest.mark.parallel([1, 2, 3])
def test_prism_ds_marker_that_holds_both_facet_shapes():
    """Marker 4 holds the triangles of the bottom and the quadrilaterals of y = 0."""
    mesh = Mesh(str(MESHDIR / MIXED_MARKER_MESHNAME))
    form = Constant(1.0) * ds(4, domain=mesh)
    assert _ds_integral_types(form) == ["exterior_facet_quad", "exterior_facet_tri"]
    for marker, area in MIXED_MARKER_AREAS.items():
        got = assemble(Constant(1.0) * ds(marker, domain=mesh))
        assert np.isclose(got, area, rtol=0, atol=1e-12), f"marker {marker}"
    x, y, z = SpatialCoordinate(mesh)
    # The bottom contributes only through the triangles, and the side y = 0
    # only through the quadrilaterals: int x over the bottom is 1/2, and
    # int z over the side is H^2 / 2.
    got = assemble((x + z) * ds(4, domain=mesh))
    assert np.isclose(got, 0.5 + 0.5 * SLAB_HEIGHT**2 + 0.5 * SLAB_HEIGHT, rtol=0, atol=1e-12)


@pytest.mark.parallel([1, 2, 3])
def test_prism_ds_otherwise_excludes_the_marked_facets():
    """ds + ds(4) makes an "otherwise" kernel for the facets outside marker 4.

    UFL adds the everywhere integrand to the ds(4) integrand, and gives the
    rest of the boundary an "otherwise" integral. Marker 4 holds triangles and
    quadrilaterals, so each shape kernel must remove it from "otherwise".
    """
    mesh = Mesh(str(MESHDIR / MIXED_MARKER_MESHNAME))
    got = assemble(Constant(1.0) * ds(domain=mesh) + Constant(1.0) * ds(4, domain=mesh))
    assert np.isclose(got, SLAB_SURFACE_AREA + MIXED_MARKER_AREAS[4], rtol=0, atol=1e-12)


def _subdomain_data_error(mesh):
    """The error of a ds with subdomain data, which only a cell integral supports.

    The subdomain data of an integral is an iteration set. Here it is the
    subset of the exterior facets in marker 1.
    """
    data = mesh.topology.exterior_facets.subset(1)
    with pytest.raises(NotImplementedError) as error:
        assemble(Constant(1.0) * ds(domain=mesh, subdomain_data=data))
    return str(error.value)


def test_prism_ds_with_subdomain_data_is_rejected_as_on_other_meshes():
    """The subdomain data of a prism ds must not be dropped at the split.

    The assembler holds the form from before the split, whose subdomain data
    is under "exterior_facet". A missed lookup would run the kernels over every
    facet and ignore the data.
    """
    prism = _subdomain_data_error(Mesh(str(MESHDIR / "prism_slab.msh")))
    tetrahedron = _subdomain_data_error(UnitCubeMesh(1, 1, 1))
    assert prism == tetrahedron == "subdomain_data only supported with cell integrals"


@pytest.mark.parallel([1, 2, 3])
def test_prism_ds_facet_subsets_hold_the_facets_of_one_shape():
    """Each kernel iterates over the facets of its shape in the marker only.

    The counts are summed over the processes. prism_slab_mixed_marker.msh has
    26 triangles on the bottom and 6 quadrilaterals on the side y = 0 in marker
    4, 26 triangles in marker 2, and 18 quadrilaterals in marker 3.
    """
    mesh = Mesh(str(MESHDIR / MIXED_MARKER_MESHNAME))
    topology = mesh.topology
    expected = {("exterior_facet_tri", 4): 26, ("exterior_facet_quad", 4): 6,
                ("exterior_facet_tri", 2): 26, ("exterior_facet_quad", 2): 0,
                ("exterior_facet_tri", 3): 0, ("exterior_facet_quad", 3): 18,
                ("exterior_facet_tri", "everywhere"): 52,
                ("exterior_facet_quad", "everywhere"): 24}
    for (integral_type, marker), count in expected.items():
        subset = topology.measure_set(integral_type, marker)
        assert mesh.comm.allreduce(subset.size) == count, (integral_type, marker)


def test_prism_ds_local_facet_number_is_the_position_in_the_shape_group():
    """The kernel of a facet shape needs the position, not the FIAT number.

    For the triangles the entity list is [3, 4], so FIAT facet 3 must reach the
    kernel as 0 and FIAT facet 4 as 1. For the quadrilaterals the list is
    [0, 1, 2] and the two numbers agree.
    """
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    facets = mesh.topology.exterior_facets
    fiat = facets.local_facet_dat.data_ro_with_halos.reshape(-1)
    position = facets.shape_local_facet_dat.data_ro_with_halos.reshape(-1)
    shape_indices, _ = facets._facet_shape_groups
    triangles = shape_indices["exterior_facet_tri"]
    quadrilaterals = shape_indices["exterior_facet_quad"]
    assert len(triangles) == 52 and len(quadrilaterals) == 24
    assert set(fiat[triangles]) == {3, 4}
    assert np.array_equal(position[triangles], fiat[triangles] - 3)
    assert set(fiat[quadrilaterals]) <= {0, 1, 2}
    assert np.array_equal(position[quadrilaterals], fiat[quadrilaterals])


@pytest.mark.parallel([1, 2, 3])
def test_prism_ds_outward_normal_on_each_reference_facet():
    """Each component of the integral of n over each facet, and n . n."""
    mesh = Mesh(str(MESHDIR / REFERENCE_MARKED_MESHNAME))
    n = FacetNormal(mesh)
    assert np.isclose(assemble(dot(n, n) * ds(domain=mesh)), REFERENCE_SURFACE_AREA,
                      rtol=0, atol=1e-12)
    for marker, (area, normal) in REFERENCE_FACETS.items():
        assert np.isclose(assemble(dot(n, n) * ds(marker, domain=mesh)), area,
                          rtol=0, atol=1e-12), f"facet {marker}"
        for i in range(3):
            got = assemble(n[i] * ds(marker, domain=mesh))
            assert np.isclose(got, area * normal[i], rtol=0, atol=1e-12), \
                f"facet {marker} component {i}"


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", ["prism_slab.msh", "prism_warped.msh",
                                      MIXED_MARKER_MESHNAME])
def test_prism_ds_outward_normal_matches_the_gmsh_geometry(meshname):
    """The integral of each component of n over each marker, from the gmsh file.

    prism_warped.msh has no affine cell, a curved top, and tilted sides, so
    every component of the normal is non-zero somewhere.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    n = FacetNormal(mesh)
    expected = {}
    for marker, area in _read_gmsh22_boundary_facets(MESHDIR / meshname):
        expected[marker] = expected.get(marker, 0.0) + area
    for marker, area in expected.items():
        for i in range(3):
            got = assemble(n[i] * ds(marker, domain=mesh))
            assert np.isclose(got, area[i], rtol=0, atol=1e-12), \
                f"marker {marker} component {i}"


def _neumann_exact(mesh, degree):
    x, y, z = SpatialCoordinate(mesh)
    if degree == 1:
        return 1.0 + x + 2.0 * y + 3.0 * z
    return 1.0 + x + x**2 + 2.0 * y**2 + 3.0 * z**2 + x * y * z


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("meshname,neumann,dirichlet",
                         [("prism_slab.msh", (1, 3), 2),
                          (MIXED_MARKER_MESHNAME, 4, (2, 3))])
def test_prism_poisson_with_a_neumann_condition(meshname, neumann, dirichlet, degree):
    """A manufactured solution in the FE space is reproduced exactly.

    The Neumann boundary holds triangular facets and quadrilateral facets, so
    both kernels of the ds contribute to the right hand side. A missing or
    wrong contribution from either one moves the solution.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", degree)
    exact = _neumann_exact(mesh, degree)
    n = FacetNormal(mesh)
    u = Function(V)
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) * dx - inner(-div(grad(exact)), v) * dx
         - inner(dot(grad(exact), n), v) * ds(neumann, domain=mesh))
    bc = DirichletBC(V, exact, dirichlet)
    solve(F == 0, u, bcs=bc, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    error = float(np.sqrt(abs(assemble(inner(u - exact, u - exact) * dx))))
    assert error < 1e-10


def test_prism_ds_matrix_rows_match_a_vector_assembly():
    """A boundary mass matrix times one equals the assembled ds of the test function."""
    mesh = Mesh(str(MESHDIR / "prism_warped.msh"))
    V = FunctionSpace(mesh, "CG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(inner(u, v) * ds(domain=mesh))
    b = assemble(inner(Constant(1.0), v) * ds(domain=mesh))
    ones, Ax = A.petscmat.createVecs()
    ones.set(1.0)
    A.petscmat.mult(ones, Ax)
    with b.dat.vec_ro as bvec:
        assert np.allclose(Ax.array_r, bvec.array_r, rtol=0, atol=1e-13)
    assert np.isclose(Ax.sum(), assemble(Constant(1.0) * ds(domain=mesh)), rtol=0, atol=1e-12)


def test_prism_slate_facet_integral_is_rejected():
    """Slate on a prism ds fails with a message, not a KeyError or a number."""
    from firedrake import Tensor

    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    with pytest.raises(NotImplementedError, match="Slate does not support facet integrals on prism"):
        assemble(Tensor(inner(u, v) * ds(domain=mesh)))


@pytest.mark.parametrize("facet", ["triangle", "quadrilateral"])
def test_prism_ds_with_a_quadrature_rule_object_is_rejected(facet):
    """A QuadratureRule object is for one facet shape, so a prism ds rejects it.

    The split gives the same metadata to the two shape kernels. Without the
    check, a triangle rule on the quadrilateral facets gives half of their
    area and no error.
    """
    from FIAT.reference_element import UFCQuadrilateral, UFCTriangle
    from finat.quadrature import make_quadrature

    ref = {"triangle": UFCTriangle, "quadrilateral": UFCQuadrilateral}[facet]()
    rule = make_quadrature(ref, 1)
    mesh = Mesh(str(MESHDIR / "prism_reference.msh"))
    match = "QuadratureRule object is not supported"
    with pytest.raises(NotImplementedError, match=match):
        assemble(Constant(1.0) * ds(domain=mesh, metadata={"quadrature_rule": rule}))
    with pytest.raises(NotImplementedError, match=match):
        assemble(Constant(1.0) * ds(domain=mesh),
                 form_compiler_parameters={"quadrature_rule": rule})
    # A degree or a scheme name is not specific to one facet shape.
    for metadata in ({"quadrature_degree": 2}, {"quadrature_rule": "default"}):
        got = assemble(Constant(1.0) * ds(domain=mesh, metadata=metadata))
        assert np.isclose(got, REFERENCE_SURFACE_AREA, rtol=0, atol=1e-12)


def test_non_prism_ds_with_a_quadrature_rule_object_still_works():
    """The check is for a cell with more than one facet shape only."""
    from FIAT.reference_element import UFCTriangle
    from finat.quadrature import make_quadrature

    rule = make_quadrature(UFCTriangle(), 1)
    got = assemble(Constant(1.0) * ds(domain=UnitCubeMesh(1, 1, 1),
                                      metadata={"quadrature_rule": rule}))
    assert np.isclose(got, 6.0, rtol=0, atol=1e-12)


# The regression guard: a mesh whose facets all have one shape keeps ONE kernel
# for one ds, with the old integral type, and the same values. The values are
# exact: the surface area, and the divergence theorem for x . n.
NON_PRISM_MESHES = {
    "triangle": (lambda: UnitSquareMesh(3, 4), 4.0, 2.0),
    "quadrilateral": (lambda: UnitSquareMesh(3, 4, quadrilateral=True), 4.0, 2.0),
    "tetrahedron": (lambda: UnitCubeMesh(2, 2, 3), 6.0, 3.0),
    "hexahedron": (lambda: UnitCubeMesh(2, 2, 3, hexahedral=True), 6.0, 3.0),
}


@pytest.mark.parametrize("cellname", sorted(NON_PRISM_MESHES))
def test_non_prism_ds_still_compiles_to_one_kernel(cellname):
    make_mesh, area, x_dot_n = NON_PRISM_MESHES[cellname]
    mesh = make_mesh()
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    assert _ds_integral_types(Constant(1.0) * ds(domain=mesh)) == ["exterior_facet"]
    assert _ds_integral_types(Constant(1.0) * ds(1, domain=mesh)) == ["exterior_facet"]
    assert np.isclose(assemble(Constant(1.0) * ds(domain=mesh)), area, rtol=0, atol=1e-12)
    assert np.isclose(assemble(dot(x, n) * ds(domain=mesh)), x_dot_n, rtol=0, atol=1e-12)
    assert np.isclose(assemble(Constant(1.0) * ds(1, domain=mesh)), 1.0, rtol=0, atol=1e-12)


def test_extruded_ds_still_compiles_to_one_kernel_per_measure():
    mesh = ExtrudedMesh(UnitSquareMesh(2, 3), 3)
    for measure, integral_type, area in ((ds_v, "exterior_facet_vert", 4.0),
                                         (ds_t, "exterior_facet_top", 1.0),
                                         (ds_b, "exterior_facet_bottom", 1.0)):
        form = Constant(1.0) * measure(domain=mesh)
        assert _ds_integral_types(form) == [integral_type]
        assert np.isclose(assemble(form), area, rtol=0, atol=1e-12)


# ------------------------------------------- Robin and Neumann conditions
#
# A Robin condition puts a ds term into the bilinear form, so the split of a ds
# into one kernel per facet shape reaches the MATRIX, and not only the right
# hand side. A split that is correct for a linear form can still be wrong for
# a bilinear form. Each Robin problem below has no Dirichlet condition, so the
# ds term of the matrix alone makes the system nonsingular.
#
# The exactness tests use a probe of total degree k in the physical
# coordinates. The coordinate map is degree 1 on the triangle and degree 1 on
# the interval, so each of x, y and z is in P1(triangle) x P1(interval) on
# every prism, affine or not. The probe is therefore in the CG space of degree
# k on prism_warped.msh too.

# The vertices of each facet of prism_reference_marked.msh, by physical group.
# A triangle is (p0, p1, p2). A parallelogram is (p0, p0 + e1, p0 + e2).
REFERENCE_FACET_VERTICES = {
    1: ("triangle", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    2: ("triangle", (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
    3: ("parallelogram", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    4: ("parallelogram", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    5: ("parallelogram", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
}

# The Robin coefficient of each marker. The values are all different, so a
# facet whose matrix entries come from the wrong marker moves the solution.
# On prism_reference_marked.msh this does NOT detect a kernel that integrates
# over the wrong facet of the one cell: the data are manufactured from the
# exact solution, so they are consistent for any coefficient on any facet.
# test_prism_ds_boundary_mass_matrix_on_each_reference_facet detects that.
ROBIN_ALPHA_BY_MARKER = {
    REFERENCE_MARKED_MESHNAME: {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0},
    "prism_slab.msh": {1: 1.0, 2: 2.0, 3: 3.0},
    MIXED_MARKER_MESHNAME: {2: 2.0, 3: 3.0, 4: 4.0},
}


def _planar_facet_integral(shape, points, function):
    """Integrate a function over a planar facet, independently of Firedrake.

    :arg shape: ``"triangle"`` or ``"parallelogram"``.
    :arg points: The three points of the facet, as in REFERENCE_FACET_VERTICES.
    :arg function: A function of a point (an array of length 3).
    :returns: The integral. The 6 point Gauss rule on each axis, with the Duffy
        transform on the triangle, is exact for a polynomial of degree 11.
    """
    p0, p1, p2 = (np.asarray(p) for p in points)
    e1, e2 = p1 - p0, p2 - p0
    area = np.linalg.norm(np.cross(e1, e2))
    s, ws = _gauss(6)
    total = 0.0
    for i, j in itertools.product(range(6), repeat=2):
        if shape == "triangle":
            a, b, jac = s[i], s[j] * (1.0 - s[i]), 1.0 - s[i]
        else:
            a, b, jac = s[i], s[j], 1.0
        total += ws[i] * ws[j] * jac * area * function(p0 + a * e1 + b * e2)
    return total


def _total_degree_exact(mesh, degree):
    """A polynomial of total degree ``degree`` in the physical coordinates."""
    x, y, z = SpatialCoordinate(mesh)
    exact = 1.0 + x + 2.0 * y + 3.0 * z
    if degree >= 2:
        exact += x**2 + 2.0 * y**2 - z**2 + x * y - y * z + 0.5 * x * z
    if degree >= 3:
        exact += x * y * z + x**3 - y**2 * z + z**3
    return exact


def _l2_error(u, exact):
    return float(np.sqrt(abs(assemble(inner(u - exact, u - exact) * dx))))


@pytest.mark.parallel([1, 2, 3])
def test_prism_ds_boundary_mass_matrix_on_each_reference_facet():
    """f^T A g for the boundary mass matrix A of each facet, from numpy.

    f and g are in the CG1 space, and their product is different on each
    facet. A matrix entry that goes to the wrong facet, or to the wrong
    vertex of the correct facet, changes f^T A g for at least one facet.
    """
    mesh = Mesh(str(MESHDIR / REFERENCE_MARKED_MESHNAME))
    V = FunctionSpace(mesh, "CG", 1)
    x, y, z = SpatialCoordinate(mesh)
    f = Function(V).interpolate(1.0 + x + 2.0 * y)
    g = Function(V).interpolate(1.0 + 4.0 * z + 3.0 * x * z)
    u, v = TrialFunction(V), TestFunction(V)

    def fg(p):
        return (1.0 + p[0] + 2.0 * p[1]) * (1.0 + 4.0 * p[2] + 3.0 * p[0] * p[2])

    for marker, (shape, *points) in REFERENCE_FACET_VERTICES.items():
        A = assemble(inner(u, v) * ds(marker, domain=mesh)).petscmat
        with f.dat.vec_ro as fvec, g.dat.vec_ro as gvec:
            Ag = A.createVecLeft()
            A.mult(gvec, Ag)
            got = fvec.dot(Ag)
        expected = _planar_facet_integral(shape, points, fg)
        assert np.isclose(got, expected, rtol=0, atol=1e-12), f"facet {marker}"


def _solve_robin(mesh, degree, exact, robin, neumann=(), dirichlet=None):
    """Solve a Poisson problem with Robin, Neumann and Dirichlet conditions.

    :arg robin: A list of pairs (alpha, measure). Each adds
        alpha u v to the bilinear form on that measure, and the matching
        alpha exact + grad(exact) . n to the right hand side.
    :arg neumann: A list of measures with the Neumann condition of exact.
    :arg dirichlet: The subdomain of the Dirichlet condition, or None.
    :returns: The discrete solution.
    """
    V = FunctionSpace(mesh, "CG", degree)
    n = FacetNormal(mesh)
    u, v = TrialFunction(V), TestFunction(V)
    flux = dot(grad(exact), n)
    a = inner(grad(u), grad(v)) * dx
    L = inner(-div(grad(exact)), v) * dx
    for alpha, measure in robin:
        a += alpha * inner(u, v) * measure
        L += inner(flux + alpha * exact, v) * measure
    for measure in neumann:
        L += inner(flux, v) * measure
    bcs = [] if dirichlet is None else [DirichletBC(V, exact, dirichlet)]
    uh = Function(V)
    solve(a == L, uh, bcs=bcs,
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    return uh


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("meshname", ["prism_slab.msh", "prism_warped.msh"])
def test_prism_poisson_with_a_robin_condition_is_exact(meshname, degree):
    """A Robin condition on the whole boundary, with a variable coefficient.

    The plain ds holds both facet shapes, so both kernels contribute to the
    matrix. The coefficient 1 + x^2 varies over each facet, so it must also
    reach each kernel at the correct quadrature points.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    x, _, _ = SpatialCoordinate(mesh)
    exact = _total_degree_exact(mesh, degree)
    uh = _solve_robin(mesh, degree, exact, [(1.0 + x**2, ds(domain=mesh))])
    assert _l2_error(uh, exact) < 1e-10


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("meshname", sorted(ROBIN_ALPHA_BY_MARKER))
def test_prism_robin_coefficient_on_each_marker_is_exact(meshname, degree):
    """A Robin condition with a different coefficient on each ds(marker).

    Every marker of the mesh has a Robin term, so the whole boundary is
    covered, and there is no Dirichlet condition. On prism_slab_mixed_marker.msh
    marker 4 holds triangles and quadrilaterals. On prism_reference_marked.msh
    each marker is one facet of one cell.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    exact = _total_degree_exact(mesh, degree)
    robin = [(Constant(alpha), ds(marker, domain=mesh))
             for marker, alpha in ROBIN_ALPHA_BY_MARKER[meshname].items()]
    uh = _solve_robin(mesh, degree, exact, robin)
    assert _l2_error(uh, exact) < 1e-10


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prism_robin_on_a_marker_with_both_facet_shapes(degree):
    """Robin on marker 4, Neumann on marker 3, Dirichlet on marker 2.

    Marker 4 holds the triangles of the bottom and the quadrilaterals of the
    side y = 0, so the Robin matrix term of ds(4) needs both kernels, each
    over the facets of its shape in marker 4 only.
    """
    mesh = Mesh(str(MESHDIR / MIXED_MARKER_MESHNAME))
    exact = _total_degree_exact(mesh, degree)
    uh = _solve_robin(mesh, degree, exact, [(Constant(2.5), ds(4, domain=mesh))],
                      neumann=[ds(3, domain=mesh)], dirichlet=2)
    assert _l2_error(uh, exact) < 1e-10


def _boundary_condition_order_step(meshname, degree, condition):
    """Solve for a non-polynomial solution on one mesh of the order sequence.

    :arg condition: ``"neumann"`` puts a Neumann condition on the bottom and
        the sides (markers 1 and 3), so on both facet shapes, and a Dirichlet
        condition on the top. ``"robin"`` puts a Robin condition with the
        coefficient 1 + x^2 on the whole boundary.
    :returns: A triple (h, L2 error, H1 seminorm error). h is the largest
        cell diameter of the mesh, over all the processes.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    x, y, z = SpatialCoordinate(mesh)
    exact = sin(pi * x) * cos(pi * y) * exp(z)
    if condition == "neumann":
        uh = _solve_robin(mesh, degree, exact, [],
                          neumann=[ds((1, 3), domain=mesh)], dirichlet=2)
    else:
        uh = _solve_robin(mesh, degree, exact, [(1.0 + x**2, ds(domain=mesh))])
    e = uh - exact
    l2 = float(np.sqrt(abs(assemble(inner(e, e) * dx))))
    h1 = float(np.sqrt(abs(assemble(inner(grad(e), grad(e)) * dx))))
    diameters = Function(FunctionSpace(mesh, "DG", 0))
    diameters.interpolate(CellDiameter(mesh))
    h = mesh.comm.allreduce(float(diameters.dat.data_ro.max(initial=0.0)),
                            op=MPI.MAX)
    return h, l2, h1


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("condition", ["neumann", "robin"])
def test_prism_boundary_condition_converges_at_the_expected_order(condition, degree):
    """The L2 error falls as h^(k+1) and the H1 seminorm error as h^k.

    The meshes are warped, so the facet normal varies over each quadrilateral
    facet and the top is not flat. The flux is the exact gradient dotted with
    the discrete normal, so the continuous problem on each mesh has the exact
    solution.

    The test is serial only. The order does not depend on the partition, and
    the ds exactness tests run at 2 and 3 ranks. At 2 and 3 ranks this test
    passes and then hangs in the PETSc teardown at exit, which stops the run.
    """
    steps = [_boundary_condition_order_step(name, degree, condition)
             for name in ORDER_MESHNAMES]
    for (h0, l2_0, h1_0), (h1, l2_1, h1_1) in zip(steps[:-1], steps[1:]):
        ratio = np.log(h0 / h1)
        l2_order = np.log(l2_0 / l2_1) / ratio
        h1_order = np.log(h1_0 / h1_1) / ratio
        # The lower bound is the test. The upper bound catches an error that
        # collapsed into round-off instead of converging.
        assert degree + 1 - 0.3 < l2_order < degree + 1 + 0.5, (l2_order, steps)
        assert degree - 0.3 < h1_order < degree + 0.5, (h1_order, steps)


# ------------------------------------------------- interior facet integrals, dS
#
# compile_form splits a prism dS into one interior_facet_tri and one
# interior_facet_quad kernel, as it splits a ds. An interior facet has two
# sides, and each side maps the reference facet points through its own cell.
# The two lists of physical points must be in the same order. On a prism the
# closure keeps the plex cone order, so the two cells can see a shared facet
# in different orientations, and each side permutes its points into the order
# of the plex cone of the facet.
#
# Three rules decide which tests can detect an error in that order:
#
# 1. A value that is constant along a facet cannot. An area, the normal of a
#    flat facet, FacetArea and DG0 data do not change when the points move.
# 2. A form that reads one side only cannot. It moves the points and the
#    weights of that side together. Only a product of a '+' value and a '-'
#    value can.
# 3. A field that is constant along the facets of one shape hides that shape.
#
# gmsh gives both sides of every triangular facet of a z-aligned mesh the
# same orientation, so a missing permutation on the triangles gives the right
# answer on those meshes. The *_scrambled.msh meshes permute the node list of
# each prism, and test_prism_dS_scrambled_mesh_presents_every_orientation
# makes sure that they stay scrambled.
#
# The '+' cell of a facet on a partition boundary can change with the number
# of processes. So every parallel test below is symmetric in '+' and '-'.

INTERIOR_MESHNAME = "prism_interior_marked.msh"
SCRAMBLED_INTERIOR_MESHNAME = "prism_interior_marked_scrambled.msh"
INTERIOR_MESHNAMES = (INTERIOR_MESHNAME, SCRAMBLED_INTERIOR_MESHNAME)
SCRAMBLED_MESHNAMES = (SCRAMBLED_INTERIOR_MESHNAME, "prism_warped_scrambled.msh",
                       "prism_order_r0_scrambled.msh", "prism_order_r1_scrambled.msh",
                       "prism_order_r2_scrambled.msh")
SCRAMBLED_ORDER_MESHNAMES = SCRAMBLED_MESHNAMES[2:]

# The marked interior surfaces of prism_interior_marked.msh. Marker 10 is the
# plane z = 0.5 and holds triangles only. Marker 20 is the plane x = 0 and
# holds quadrilaterals only. Marker 30 is the two together.
INTERIOR_MARKER_AREAS = {10: 1.0, 20: 1.0, 30: 2.0}


def _plane_distance(mesh, marker):
    """The signed distance from the marked plane, positive on one side."""
    x, y, z = SpatialCoordinate(mesh)
    return {10: z - 0.5, 20: x}[marker]


def _read_gmsh22_interior_facets(path):
    """Read the interior facets of a gmsh 2.2 ASCII prism file.

    :arg path: The path of the .msh file.
    :returns: A pair (areas, marked). areas is an array with the area of each
        facet that two prisms share. marked maps each physical group of the
        surface elements to an array of the areas of its facets that two
        prisms share.

    This is independent of Firedrake. It is correct for planar facets only:
    the area is the norm of half the sum of the cross products of
    consecutive vertices.
    """
    coords, cells = _read_gmsh22_prisms(path)

    def area(vertices):
        points = coords[list(vertices)]
        return np.linalg.norm(0.5 * sum(np.cross(points[k], points[(k + 1) % len(points)])
                                        for k in range(len(points))))

    faces = {}
    for cell in cells:
        for face in ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)):
            vertices = tuple(cell[list(face)])
            faces.setdefault(frozenset(vertices), [vertices, 0])[1] += 1
    interior = {key: vertices for key, (vertices, count) in faces.items() if count == 2}
    lines = Path(path).read_text().split("\n")
    tag_to_row = {}
    i = lines.index("$Nodes")
    for row in range(int(lines[i + 1])):
        tag_to_row[int(lines[i + 2 + row].split()[0])] = row
    j = lines.index("$Elements")
    marked = {}
    for row in range(int(lines[j + 1])):
        fields = [int(x) for x in lines[j + 2 + row].split()]
        # Type 2 is the 3-node triangle, type 3 the 4-node quadrilateral.
        if fields[1] not in (2, 3):
            continue
        ntags = fields[2]
        vertices = [tag_to_row[t] for t in fields[3 + ntags:]]
        if frozenset(vertices) in interior:
            marked.setdefault(fields[3], []).append(area(vertices))
    areas = np.array([area(vertices) for vertices in interior.values()])
    return areas, {marker: np.array(values) for marker, values in marked.items()}


def _point_mismatch(mesh, measure):
    """The integral of the squared distance between the two sides' points."""
    x = SpatialCoordinate(mesh)
    return assemble(inner(x('+') - x('-'), x('+') - x('-')) * measure)


def _sipg_terms(V, alpha):
    """The symmetric interior penalty terms of a DG Laplacian.

    :arg V: The DG function space.
    :arg alpha: The penalty coefficient. The penalty is alpha / h, with h the
        cell diameter.
    :returns: A tuple (u, v, interior, boundary). interior is the integrand on
        the interior facets, and boundary the Nitsche integrand on the
        exterior facets, for a Dirichlet condition u = 0.
    """
    mesh = V.mesh()
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    interior = (-inner(avg(grad(u)), jump(v, n)) - inner(jump(u, n), avg(grad(v)))
                + alpha / avg(h) * inner(jump(u, n), jump(v, n)))
    boundary = (-inner(grad(u), n) * v - inner(grad(v), n) * u + alpha / h * u * v)
    return u, v, interior, boundary


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", INTERIOR_MESHNAMES)
def test_prism_dS_measures_each_marked_surface(meshname):
    """The area of each marked interior surface, and of all interior facets.

    Marker 10 holds triangles only, so a triangle kernel that gets the FIAT
    facet number in place of the position in [3, 4] measures zero here.
    Markers 30 and (10, 20) hold both shapes. dS + dS(10) makes an
    "otherwise" kernel for the facets outside marker 10.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    areas, _ = _read_gmsh22_interior_facets(MESHDIR / meshname)
    got = {marker: assemble(Constant(1.0) * dS(marker, domain=mesh))
           for marker in INTERIOR_MARKER_AREAS}
    both = assemble(Constant(1.0) * dS((10, 20), domain=mesh))
    total = assemble(Constant(1.0) * dS(domain=mesh))
    otherwise = assemble(Constant(1.0) * dS(domain=mesh) + Constant(1.0) * dS(10, domain=mesh))
    for marker, area in INTERIOR_MARKER_AREAS.items():
        assert np.isclose(got[marker], area, rtol=0, atol=1e-12), f"marker {marker}"
    assert np.isclose(both, 2.0, rtol=0, atol=1e-12)
    assert np.isclose(total, areas.sum(), rtol=0, atol=1e-12)
    assert np.isclose(otherwise, areas.sum() + 1.0, rtol=0, atol=1e-12)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", SCRAMBLED_MESHNAMES + ("prism_two_perpendicular.msh",))
def test_prism_dS_points_of_the_two_sides_agree(meshname):
    """|x('+') - x('-')|^2 over the interior facets is zero.

    Each side maps its own facet quadrature points to physical space. The
    integral is zero only if the two lists are in the same order, so this
    catches an error in the point permutation of either side. On
    prism_interior_marked_scrambled.msh it is measured for each facet shape
    too. prism_two_perpendicular.msh is the only mesh whose cells see a
    shared quadrilateral facet with different extrinsic orientations.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    got = {"all": _point_mismatch(mesh, dS(domain=mesh))}
    if meshname == SCRAMBLED_INTERIOR_MESHNAME:
        got.update({marker: _point_mismatch(mesh, dS(marker, domain=mesh))
                    for marker in (10, 20)})
    for key, value in got.items():
        assert abs(value) < 1e-26, f"{key}: {value}"


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("meshname", [SCRAMBLED_INTERIOR_MESHNAME, "prism_warped_scrambled.msh",
                                      "prism_two_perpendicular.msh"])
def test_prism_dS_jump_of_a_continuous_function_is_zero(meshname, degree):
    """jump(u)^2 over the interior facets is zero for u in CG of degree k.

    u is the interpolant of a smooth function that changes along every
    facet, so a point order error or a facet dof order error moves the value.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    x, y, z = SpatialCoordinate(mesh)
    u = Function(FunctionSpace(mesh, "CG", degree))
    u.interpolate(sin(3 * x) * cos(2 * y) * exp(z))
    got = assemble(jump(u)**2 * dS(domain=mesh))
    assert abs(got) < 1e-20, got


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prism_dS_discontinuous_interpolant_of_a_polynomial(degree):
    """The traces of u = I p in DG of degree k agree with p, on each facet shape.

    p has total degree k, so it is in the space, and the two traces of u are
    equal to p on every interior facet. p changes along the facets of both
    markers, so the tests on marker 10 (triangles) and marker 20
    (quadrilaterals) see a point order error on each shape.
    """
    mesh = Mesh(str(MESHDIR / SCRAMBLED_INTERIOR_MESHNAME))
    p = _total_degree_exact(mesh, degree)
    u = Function(FunctionSpace(mesh, "DG", degree)).interpolate(p)
    got = {}
    for marker in (10, 20):
        got[f"avg, marker {marker}"] = assemble((avg(u) - p('+'))**2 * dS(marker, domain=mesh))
        got[f"jump, marker {marker}"] = assemble(jump(u)**2 * dS(marker, domain=mesh))
    got["jump, all"] = assemble(jump(u)**2 * dS(domain=mesh))
    for key, value in got.items():
        assert abs(value) < 1e-20, f"{key}: {value}"


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", [SCRAMBLED_INTERIOR_MESHNAME, "prism_warped_scrambled.msh",
                                      "prism_two_perpendicular.msh"])
def test_prism_dS_normals_of_the_two_sides_are_opposite(meshname):
    """n('+') + n('-') is zero and n('+') has unit length on every interior facet.

    The quadrilateral facets of prism_warped_scrambled.msh are not planar, so
    there the normal changes along each facet and a point order error moves
    the first integral too. A normal from the wrong facet of the cell moves
    both.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    n = FacetNormal(mesh)
    opposite = assemble(inner(n('+') + n('-'), n('+') + n('-')) * dS(domain=mesh))
    unit = assemble(inner(n('+'), n('+')) * dS(domain=mesh))
    area = assemble(Constant(1.0) * dS(domain=mesh))
    assert abs(opposite) < 1e-24, opposite
    assert np.isclose(unit, area, rtol=1e-13, atol=0)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", INTERIOR_MESHNAMES)
def test_prism_dS_normal_of_each_marked_plane(meshname):
    """The normal of the plane z = 0.5 is +-e_z, and of the plane x = 0 is +-e_x.

    The square makes the value independent of the side that is '+'.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    n = FacetNormal(mesh)
    got = {10: assemble(n('+')[2]**2 * dS(10, domain=mesh)),
           20: assemble(n('+')[0]**2 * dS(20, domain=mesh))}
    for marker, value in got.items():
        assert np.isclose(value, INTERIOR_MARKER_AREAS[marker], rtol=0, atol=1e-12), \
            f"marker {marker}"


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", INTERIOR_MESHNAMES)
def test_prism_dS_facet_area_on_each_shape(meshname):
    """The integral of FacetArea over a set of facets is the sum of the squared areas."""
    from firedrake import FacetArea

    mesh = Mesh(str(MESHDIR / meshname))
    areas, marked = _read_gmsh22_interior_facets(MESHDIR / meshname)
    facet_area = FacetArea(mesh)
    got = {marker: assemble(facet_area * dS(marker, domain=mesh))
           for marker in INTERIOR_MARKER_AREAS}
    total = assemble(facet_area * dS(domain=mesh))
    for marker, value in got.items():
        assert np.isclose(value, (marked[marker]**2).sum(), rtol=1e-13, atol=0), \
            f"marker {marker}"
    assert np.isclose(total, (areas**2).sum(), rtol=1e-13, atol=0)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("marker", [10, 20])
@pytest.mark.parametrize("meshname", INTERIOR_MESHNAMES)
def test_prism_dS_surface_source_is_exact(meshname, marker, degree):
    """A source of strength 2 on a marked interior plane gives u = |s|.

    s is the signed distance from the plane. -div(grad(u)) = 2 delta(s) with
    u = |s| on the boundary has the solution |s|, which is in CG1 because the
    plane is a union of facets.

    This is the linear form path of a dS(marker), and it catches a wrong facet
    position. It reads the '+' side only, so it is blind to the point order.
    Do not count it as a test of the orientations.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    V = FunctionSpace(mesh, "CG", degree)
    exact = abs(_plane_distance(mesh, marker))
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == -2.0 * v('+') * dS(marker, domain=mesh), uh,
          bcs=DirichletBC(V, exact, "on_boundary"),
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    error = _l2_error(uh, exact)
    assert error < 1e-10, error


def _contact_conductance_problem(mesh, degree, marker, hc=4.0):
    """A DG Laplacian with a contact conductance on a marked interior plane.

    :returns: A triple (a, L, exact). a has the SIPG terms on every interior
        facet but the marked ones, and the contact conductance term
        hc * jump(u, n) . jump(v, n) on the marked ones. UFL makes the first
        an "otherwise" integral.

    The exact solution is s + w below the plane (s < 0), and s + w + 1/hc
    above it. w is linear and does not change with s, so the flux through the
    plane is 1 on both sides and equals hc times the jump. w changes along the
    plane, so a point order error on the marked facets moves the solution:
    with w = 0 the solution is constant along the triangular facets of
    marker 10, and a missing triangle permutation gives no error.
    """
    x, y, z = SpatialCoordinate(mesh)
    s = _plane_distance(mesh, marker)
    w = {10: 0.3 * (x - 2.0 * y), 20: 0.3 * (y - 2.0 * z)}[marker]
    exact = s + w + conditional(gt(s, 0), 1.0 / hc, 0.0)
    V = FunctionSpace(mesh, "DG", degree)
    alpha = Constant(10.0 * (degree + 1)**2)
    u, v, sipg, nitsche = _sipg_terms(V, alpha)
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    interface = hc * inner(jump(u, n), jump(v, n))
    a = (inner(grad(u), grad(v)) * dx + sipg * dS(domain=mesh)
         + (interface - sipg) * dS(marker, domain=mesh) + nitsche * ds(domain=mesh))
    L = (-inner(grad(v), n) * exact + alpha / h * exact * v) * ds(domain=mesh)
    return a, L, exact


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("marker", [10, 20])
def test_prism_dS_contact_conductance_is_exact(marker, degree):
    """SIPG with a contact conductance on marker 10 or 20 reproduces a piecewise linear solution.

    This uses dS(marker) on each facet shape, the "otherwise" kernel, jump,
    avg, FacetNormal and CellDiameter on both sides, and a DG solve.
    """
    mesh = Mesh(str(MESHDIR / SCRAMBLED_INTERIOR_MESHNAME))
    a, L, exact = _contact_conductance_problem(mesh, degree, marker)
    uh = Function(a.arguments()[0].function_space())
    solve(a == L, uh, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    error = _l2_error(uh, exact) / _l2_error(Constant(0.0), exact)
    assert error < 1e-10, error


@pytest.mark.parallel([1, 2, 3])
def test_prism_dS_matrix_matches_the_action():
    """The assembled DG2 matrix times a function equals the assembled action.

    The matrix gets the four blocks of each interior facet, and the action
    gets two vectors. Both use the same kernels, so a difference means that
    the matrix puts a block into the wrong rows or columns.
    """
    from firedrake import action

    mesh = Mesh(str(MESHDIR / SCRAMBLED_INTERIOR_MESHNAME))
    a, _, _ = _contact_conductance_problem(mesh, 2, 20)
    V = a.arguments()[0].function_space()
    x, y, z = SpatialCoordinate(mesh)
    w = Function(V).interpolate(sin(3 * x) * cos(2 * y) * exp(z))
    A = assemble(a).petscmat
    Aw = assemble(action(a, w))
    with w.dat.vec_ro as wvec, Aw.dat.vec_ro as Awvec:
        product = A.createVecLeft()
        A.mult(wvec, product)
        reference = Awvec.norm()
        product.axpy(-1.0, Awvec)
        difference = product.norm()
    assert difference < 1e-12 * reference, (difference, reference)


def _sipg_order_step(meshname, degree):
    """Solve a DG Laplacian by SIPG for a non-polynomial solution on one mesh.

    :returns: A triple (h, L2 error, broken H1 seminorm error). h is the
        largest cell diameter of the mesh, over all the processes.

    The Dirichlet condition is weak (Nitsche), so every facet of the mesh
    carries a facet integral.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    x, y, z = SpatialCoordinate(mesh)
    exact = sin(pi * x) * cos(pi * y) * exp(z)
    V = FunctionSpace(mesh, "DG", degree)
    alpha = Constant(10.0 * (degree + 1)**2)
    u, v, sipg, nitsche = _sipg_terms(V, alpha)
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    a = inner(grad(u), grad(v)) * dx + sipg * dS(domain=mesh) + nitsche * ds(domain=mesh)
    L = (-div(grad(exact)) * v * dx
         + (-inner(grad(v), n) * exact + alpha / h * exact * v) * ds(domain=mesh))
    uh = Function(V)
    solve(a == L, uh, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    e = uh - exact
    l2 = float(np.sqrt(abs(assemble(inner(e, e) * dx))))
    h1 = float(np.sqrt(abs(assemble(inner(grad(e), grad(e)) * dx))))
    diameters = Function(FunctionSpace(mesh, "DG", 0))
    diameters.interpolate(CellDiameter(mesh))
    h = mesh.comm.allreduce(float(diameters.dat.data_ro.max(initial=0.0)),
                            op=MPI.MAX)
    return h, l2, h1


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prism_dS_sipg_converges_at_the_expected_order(degree):
    """The SIPG L2 error falls as h^(k+1) and the broken H1 error as h^k.

    The meshes are warped and scrambled, so no prism is affine and the cells
    see their shared facets in every orientation. Every part of a prism dS
    contributes to the error.

    The test is serial only, as
    test_prism_boundary_condition_converges_at_the_expected_order is. The dS
    exactness tests run at 2 and 3 ranks. Degree 3 uses the first two meshes
    only: the DG3 solve on the finest mesh takes more than 7 minutes.
    """
    meshnames = SCRAMBLED_ORDER_MESHNAMES[:2] if degree == 3 else SCRAMBLED_ORDER_MESHNAMES
    steps = [_sipg_order_step(name, degree) for name in meshnames]
    for (h0, l2_0, h1_0), (h1, l2_1, h1_1) in zip(steps[:-1], steps[1:]):
        ratio = np.log(h0 / h1)
        l2_order = np.log(l2_0 / l2_1) / ratio
        h1_order = np.log(h1_0 / h1_1) / ratio
        # The lower bound is the test. The upper bound catches an error that
        # collapsed into round-off instead of converging.
        assert degree + 1 - 0.3 < l2_order < degree + 1 + 0.5, (l2_order, steps)
        assert degree - 0.3 < h1_order < degree + 0.5, (h1_order, steps)


@pytest.mark.parallel([1, 2, 3])
def test_prism_dS_compiles_to_one_kernel_per_facet_shape():
    mesh = Mesh(str(MESHDIR / INTERIOR_MESHNAME))
    got = {"all": _ds_integral_types(Constant(1.0) * dS(domain=mesh)),
           "marker 10": _ds_integral_types(Constant(1.0) * dS(10, domain=mesh)),
           "markers 10 and 20": _ds_integral_types(Constant(1.0) * dS((10, 20), domain=mesh))}
    both = _ds_integral_types(Constant(1.0) * ds(domain=mesh) + Constant(1.0) * dS(domain=mesh))
    for key, types in got.items():
        assert types == ["interior_facet_quad", "interior_facet_tri"], key
    assert both == ["exterior_facet_quad", "exterior_facet_tri",
                    "interior_facet_quad", "interior_facet_tri"]


@pytest.mark.parametrize("cellname", sorted(NON_PRISM_MESHES))
def test_non_prism_dS_still_compiles_to_one_kernel(cellname):
    make_mesh, _, _ = NON_PRISM_MESHES[cellname]
    mesh = make_mesh()
    assert _ds_integral_types(Constant(1.0) * dS(domain=mesh)) == ["interior_facet"]


def test_extruded_dS_still_compiles_to_one_kernel_per_measure():
    from firedrake import dS_h, dS_v

    mesh = ExtrudedMesh(UnitSquareMesh(2, 3), 3)
    for measure, integral_type in ((dS_v, "interior_facet_vert"),
                                   (dS_h, "interior_facet_horiz")):
        assert _ds_integral_types(Constant(1.0) * measure(domain=mesh)) == [integral_type]


# The unsupported paths of a prism dS must fail with a message, and not give a
# number. These tests are serial: each error comes before any parallel work.
# The Slate and subdomain_data errors are tested in
# test_prism_interior_facets_core.py.

def test_prism_patch_pc_with_an_interior_facet_integral_is_rejected():
    """PatchPC rejects a dS form, as it rejects a ds form on a prism.

    The error comes from a PETSc callback, so PETSc wraps it.
    """
    mesh = Mesh(str(MESHDIR / INTERIOR_MESHNAME))
    V = FunctionSpace(mesh, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(mesh)
    a = inner(grad(u), grad(v)) * dx + 10.0 * inner(jump(u, n), jump(v, n)) * dS + u * v * dx
    uh = Function(V)
    parameters = {"ksp_type": "cg", "ksp_max_it": 3, "ksp_convergence_test": "skip",
                  "pc_type": "python", "pc_python_type": "firedrake.PatchPC",
                  "patch_pc_patch_construct_type": "star",
                  "patch_pc_patch_construct_dim": 0,
                  "patch_sub_ksp_type": "preonly", "patch_sub_pc_type": "lu"}
    with pytest.raises(PETSc.Error) as error:
        solve(a == v * dx, uh, solver_parameters=parameters)
    assert isinstance(error.value.__cause__, NotImplementedError)
    assert "Only for cell, interior facet, or exterior facet integrals" in str(error.value.__cause__)


@pytest.mark.parametrize("facet", ["triangle", "quadrilateral"])
def test_prism_dS_with_a_quadrature_rule_object_is_rejected(facet):
    """A QuadratureRule object is for one facet shape, so a prism dS rejects it."""
    from FIAT.reference_element import UFCQuadrilateral, UFCTriangle
    from finat.quadrature import make_quadrature

    ref = {"triangle": UFCTriangle, "quadrilateral": UFCQuadrilateral}[facet]()
    rule = make_quadrature(ref, 1)
    mesh = Mesh(str(MESHDIR / INTERIOR_MESHNAME))
    with pytest.raises(NotImplementedError, match="QuadratureRule object is not supported"):
        assemble(Constant(1.0) * dS(domain=mesh, metadata={"quadrature_rule": rule}))
    # A degree or a scheme name is not specific to one facet shape.
    for metadata in ({"quadrature_degree": 2}, {"quadrature_rule": "default"}):
        got = assemble(Constant(1.0) * dS(10, domain=mesh, metadata=metadata))
        assert np.isclose(got, 1.0, rtol=0, atol=1e-12)


def test_prism_dS_with_a_rule_that_has_no_point_permutation_is_rejected():
    """The canonical triangle rule of degree 4 is not symmetric, so it has no point permutation.

    Without the permutation the two sides of a triangular facet do not agree
    on the points, and the answer is wrong with no error. So the kernel must
    fail, and the message must name the integral type, the scheme and the
    degree.
    """
    mesh = Mesh(str(MESHDIR / INTERIOR_MESHNAME))
    metadata = {"quadrature_rule": "canonical", "quadrature_degree": 4}
    with pytest.raises(NotImplementedError) as error:
        assemble(Constant(1.0) * dS(domain=mesh, metadata=metadata))
    message = str(error.value)
    for word in ("interior_facet_tri", "canonical", "4"):
        assert word in message, message


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("meshname", SCRAMBLED_MESHNAMES)
def test_prism_dS_scrambled_mesh_presents_every_orientation(meshname):
    """The '-' sides of the interior facets take all 6 triangle orientations and all 4 io values.

    The orientation of a quadrilateral facet is o = 4 eo + io. Without this
    test a change in the gmsh reader could align the scrambled meshes again,
    and every orientation test above would pass without any permutation.
    """
    mesh = Mesh(str(MESHDIR / meshname))
    facets = mesh.topology.interior_facets
    local_facets = facets.local_facet_dat.data_ro_with_halos.reshape(-1, 2)
    orientations = facets.local_facet_orientation_dat.data_ro_with_halos.reshape(-1, 2)
    # A facet in the outer halo can have one absent cell.
    present = (facets.facet_cell != -1).all(axis=1)
    triangle = local_facets[:, 1] >= 3
    local = (set(int(o) for o in orientations[present & triangle, 1]),
             set(int(o) % 4 for o in orientations[present & ~triangle, 1]))
    # Do the collective before any assert. A rank can hold no triangular facet.
    gathered = mesh.comm.allgather(local)
    triangle_orientations = set().union(*(t for t, _ in gathered))
    quadrilateral_io = set().union(*(q for _, q in gathered))
    assert triangle_orientations == set(range(6))
    assert quadrilateral_io == set(range(4))


def _shell_partition(comm, meshname):
    """A partition of prism_interior_marked*.msh that cuts along the marked planes.

    :arg comm: The communicator, of size 2 or 3.
    :arg meshname: The file name of the mesh.
    :returns: The value of the "partition" distribution parameter: the number
        of cells of each rank, and the cells in rank order. Rank 0 gives both;
        the other ranks give empty arrays.

    At 2 ranks the cut is the plane z = 0.5. At 3 ranks rank 0 gets z > 0.5,
    and the cut x = 0 divides z < 0.5 between ranks 1 and 2. The cell numbers
    are the order of the prisms in the gmsh file, as the reader keeps it.
    """
    coords, cells = _read_gmsh22_prisms(MESHDIR / meshname)
    centroids = coords[cells].mean(axis=1)
    above = centroids[:, 2] > 0.5
    if comm.size == 2:
        owner = np.where(above, 1, 0)
    else:
        owner = np.where(above, 0, np.where(centroids[:, 0] < 0.0, 1, 2))
    if comm.rank == 0:
        sizes = np.array([(owner == r).sum() for r in range(comm.size)], dtype=np.int32)
        points = np.concatenate([np.flatnonzero(owner == r)
                                 for r in range(comm.size)]).astype(np.int32)
    else:
        sizes = np.zeros(comm.size, dtype=np.int32)
        points = np.empty(0, dtype=np.int32)
    return sizes, points


@pytest.mark.parallel([2, 3])
@pytest.mark.parametrize("degree", [2, 3])
def test_prism_dS_on_a_partition_boundary(degree):
    """dS is exact on triangular and quadrilateral facets that two ranks share.

    The default partitioner puts almost no triangular facet on a partition
    boundary, so the other parallel tests do not see a triangle whose two
    cells are on different ranks. This partition cuts along the marked
    planes. All 136 triangles of marker 10 are on the partition boundary, and
    at 3 ranks the 21 quadrilaterals of marker 20 below z = 0.5 are too. The
    test asserts these counts, so that a change of the partition cannot make
    it pass without any shared facet.
    """
    comm = MPI.COMM_WORLD
    distribution = _shell_partition(comm, SCRAMBLED_INTERIOR_MESHNAME)
    mesh = Mesh(str(MESHDIR / SCRAMBLED_INTERIOR_MESHNAME),
                distribution_parameters={"partition": distribution})
    facets = mesh.topology.interior_facets
    cells = np.asarray(facets.facet_cell).reshape(-1, 2)
    # An owned facet on the partition boundary has one owned cell and one
    # halo cell. Each such facet is counted once, on the rank that owns it.
    owned = np.arange(len(cells)) < facets.set.size
    shared = owned & ((cells != -1).all(axis=1)) & ((cells < mesh.cell_set.size).sum(axis=1) == 1)
    local = np.array([shared[facets.subset(marker).indices].sum() for marker in (10, 20)])
    # Do the collectives before any assert.
    on_boundary = comm.allreduce(local)
    p = _total_degree_exact(mesh, degree)
    u = Function(FunctionSpace(mesh, "DG", degree)).interpolate(p)
    got = {}
    for marker in (10, 20):
        got[f"points, marker {marker}"] = _point_mismatch(mesh, dS(marker, domain=mesh))
        got[f"avg, marker {marker}"] = assemble((avg(u) - p('+'))**2 * dS(marker, domain=mesh))
        got[f"jump, marker {marker}"] = assemble(jump(u)**2 * dS(marker, domain=mesh))
    assert on_boundary[0] == 136, on_boundary
    assert on_boundary[1] == (21 if comm.size == 3 else 0), on_boundary
    for key, value in got.items():
        assert abs(value) < 1e-20, f"{key}: {value}"
