"""The global numbering of every cell type that is not a prism.

Task 2b changed the global numbering to key the number of dofs of a plex point
on the DMPlex polytope type of the point, not on its topological dimension. The
reason is the prism: a prism mesh holds both triangular and quadrilateral faces,
and a Lagrange space of degree 2 or more gives those two different numbers of
dofs.

Every other cell type has exactly one polytope type per dimension, so the new
path has to reproduce the old numbering EXACTLY for all of them. That is what
this file guards. The expected values below were measured on the commit before
Task 2b, ``c9007a76e``, with the script that this file replaces.
"""
from pathlib import Path

import numpy as np
import pytest

from firedrake import (ExtrudedMesh, FunctionSpace, Mesh, UnitCubeMesh,
                       UnitSquareMesh, VectorFunctionSpace)
from firedrake.petsc import PETSc


# Measured on commit c9007a76e, before Task 2b changed the numbering.
# Key: (mesh name, degree, "scalar" or "vector").
# Value: (V.dim(), V.node_set.sizes, nodes_per_entity, node classes).
BEFORE_TASK_2B = {
    ("triangle", 1, "scalar"): (25, (25, 25, 25), (1, 0, 0), (25, 25, 25)),
    ("triangle", 1, "vector"): (50, (25, 25, 25), (1, 0, 0), (25, 25, 25)),
    ("triangle", 3, "scalar"): (169, (169, 169, 169), (1, 2, 1), (169, 169, 169)),
    ("triangle", 3, "vector"): (338, (169, 169, 169), (1, 2, 1), (169, 169, 169)),
    ("quadrilateral", 1, "scalar"): (25, (25, 25, 25), (1, 0, 0), (25, 25, 25)),
    ("quadrilateral", 1, "vector"): (50, (25, 25, 25), (1, 0, 0), (25, 25, 25)),
    ("quadrilateral", 3, "scalar"): (169, (169, 169, 169), (1, 2, 4), (169, 169, 169)),
    ("quadrilateral", 3, "vector"): (338, (169, 169, 169), (1, 2, 4), (169, 169, 169)),
    ("tetrahedron", 1, "scalar"): (27, (27, 27, 27), (1, 0, 0, 0), (27, 27, 27)),
    ("tetrahedron", 1, "vector"): (81, (27, 27, 27), (1, 0, 0, 0), (27, 27, 27)),
    ("tetrahedron", 3, "scalar"): (343, (343, 343, 343), (1, 2, 1, 0), (343, 343, 343)),
    ("tetrahedron", 3, "vector"): (1029, (343, 343, 343), (1, 2, 1, 0), (343, 343, 343)),
    ("hexahedron", 1, "scalar"): (27, (27, 27, 27), (1, 0, 0, 0), (27, 27, 27)),
    ("hexahedron", 1, "vector"): (81, (27, 27, 27), (1, 0, 0, 0), (27, 27, 27)),
    ("hexahedron", 3, "scalar"): (343, (343, 343, 343), (1, 2, 4, 8), (343, 343, 343)),
    ("hexahedron", 3, "vector"): (1029, (343, 343, 343), (1, 2, 4, 8), (343, 343, 343)),
    ("extruded", 1, "scalar"): (48, (48, 48, 48), ((1, 0), (0, 0), (0, 0)), (48, 48, 48)),
    ("extruded", 1, "vector"): (144, (48, 48, 48), ((1, 0), (0, 0), (0, 0)), (48, 48, 48)),
    ("extruded", 3, "scalar"): (700, (700, 700, 700), ((1, 2), (2, 4), (1, 2)), (700, 700, 700)),
    ("extruded", 3, "vector"): (2100, (700, 700, 700), ((1, 2), (2, 4), (1, 2)), (700, 700, 700)),
}

MESH_MAKERS = {
    "triangle": lambda: UnitSquareMesh(4, 4),
    "quadrilateral": lambda: UnitSquareMesh(4, 4, quadrilateral=True),
    "tetrahedron": lambda: UnitCubeMesh(2, 2, 2),
    "hexahedron": lambda: UnitCubeMesh(2, 2, 2, hexahedral=True),
    "extruded": lambda: ExtrudedMesh(UnitSquareMesh(3, 3), 2),
}

SPACE_MAKERS = {"scalar": FunctionSpace, "vector": VectorFunctionSpace}


@pytest.mark.parametrize("key", sorted(BEFORE_TASK_2B))
def test_numbering_is_unchanged_for_a_single_polytope_per_dimension(key):
    mesh_name, degree, kind = key
    mesh = MESH_MAKERS[mesh_name]()
    V = SPACE_MAKERS[kind](mesh, "CG", degree)
    topology = mesh.topology
    nodes_per_entity = tuple(
        topology.make_dofs_per_plex_entity(V.finat_element.entity_dofs()))
    got = (
        V.dim(),
        tuple(int(size) for size in V.node_set.sizes),
        nodes_per_entity,
        tuple(int(count) for count in topology.node_classes(nodes_per_entity)),
    )
    assert got == BEFORE_TASK_2B[key]


@pytest.mark.parametrize("mesh_name", sorted(MESH_MAKERS))
def test_every_dimension_holds_one_polytope_type(mesh_name):
    """These meshes take the fast path: one numbering stratum per dimension."""
    mesh = MESH_MAKERS[mesh_name]()
    topology = mesh.topology
    assert all(types == (None,) for types in topology._plex_polytope_types)
    assert topology._numbering_strata == tuple(
        (dim, None) for dim in range(len(topology._plex_polytope_types)))
    assert np.array_equal(topology._entity_classes_per_stratum,
                          topology._entity_classes)


@pytest.mark.parallel([2, 3])
@pytest.mark.parametrize("mesh_name", sorted(MESH_MAKERS))
@pytest.mark.parametrize("degree", [1, 3])
def test_node_classes_count_every_node_once_in_parallel(mesh_name, degree):
    """The owned node counts of all ranks add up to V.dim().

    This is partition independent, so it stays true when the partitioner
    changes. The serial test above pins the exact numbers, and
    test_node_classes_split_matches_the_plex below pins the split.
    """
    mesh = MESH_MAKERS[mesh_name]()
    V = FunctionSpace(mesh, "CG", degree)
    owned = mesh.comm.allreduce(int(V.node_set.size))
    assert owned == V.dim()


# ------------------------------------- the core / owned / ghost split, pinned

PRISM_MESH = Path(__file__).parent.parent / "meshes" / "prism" / "prism_slab.msh"


def _stratum_points(topology):
    """The plex points of every numbering stratum, as a list of sets."""
    plex = topology.topology_dm
    points = []
    for dim, polytope_type in topology._numbering_strata:
        start, end = plex.getDepthStratum(dim)
        points.append({point for point in range(start, end)
                       if polytope_type is None
                       or plex.getCellType(point) == polytope_type})
    return points


def _classes_from_the_plex_labels(topology):
    """The core / owned / ghost split of every stratum, counted in Python.

    This does not restate the implementation. ``get_entity_classes_per_stratum``
    walks the three label index sets and asks which stratum each point falls
    in. This walks the strata and asks which label each point carries. The two
    derivations meet only at the labels, which Task 2b did not touch.
    """
    plex = topology.topology_dm
    rows = []
    for points in _stratum_points(topology):
        counts = []
        for label in ("pyop2_core", "pyop2_owned", "pyop2_ghost"):
            if plex.getStratumSize(label, 1) > 0:
                labelled = set(plex.getStratumIS(label, 1).indices)
            else:
                labelled = set()
            counts.append(len(points & labelled))
        rows.append(np.cumsum(counts))
    return np.array(rows)


def _leaves_per_stratum(topology):
    """The number of point-SF leaves in every stratum.

    A leaf is a point that another rank owns. That is the definition of a
    ghost point, and it does not go through the PyOP2 labels at all.
    """
    _, local, _ = topology.topology_dm.getPointSF().getGraph()
    leaves = set() if local is None else set(local)
    return [len(points & leaves) for points in _stratum_points(topology)]


def _check_the_split(topology):
    got = topology._entity_classes_per_stratum
    points_per_stratum = _stratum_points(topology)
    # Do the collective before any assert. An assert that fails on one rank
    # only would otherwise leave the other ranks in the collective, and the
    # run would hang in place of a failure.
    ghosts = sum(int(row[2] - row[1]) for row in got)
    ghosts_on_all_ranks = topology.comm.allreduce(ghosts)

    assert np.array_equal(got, _classes_from_the_plex_labels(topology))

    # Every point of a stratum carries exactly one of the three labels, so the
    # cumulative total is the size of the stratum.
    for row, points in zip(got, points_per_stratum):
        assert int(row[2]) == len(points)

    # The ghost count of a stratum is the number of its points that another
    # rank owns, taken from the point SF rather than from the labels.
    assert [int(row[2] - row[1]) for row in got] == _leaves_per_stratum(topology)

    # The strata of one dimension add up to the per-dimension split, which
    # Task 2b did not change.
    for dim, per_dimension in enumerate(topology._entity_classes):
        rows = [row for row, (d, _) in zip(got, topology._numbering_strata)
                if d == dim]
        assert np.array_equal(np.sum(rows, axis=0), per_dimension)

    # Guard against a vacuous pass: with more than one rank some stratum must
    # hold ghost points, or none of the assertions above says anything.
    assert ghosts_on_all_ranks > 0


@pytest.mark.parallel([2, 3])
@pytest.mark.parametrize("mesh_name", sorted(MESH_MAKERS))
def test_entity_classes_per_stratum_match_the_plex(mesh_name):
    """Every one of these meshes has one stratum per dimension."""
    _check_the_split(MESH_MAKERS[mesh_name]().topology)


@pytest.mark.parallel([2, 3])
def test_entity_classes_per_stratum_on_a_split_dimension():
    """The prism mesh is the case the per-stratum split exists for.

    Dimension 2 holds triangles and quadrilaterals, so it becomes two strata,
    and each needs its own core / owned / ghost counts.
    """
    topology = Mesh(str(PRISM_MESH)).topology
    assert topology._numbering_strata == (
        (0, None),
        (1, None),
        (2, PETSc.DM.PolytopeType.TRIANGLE),
        (2, PETSc.DM.PolytopeType.QUADRILATERAL),
        (3, None),
    )
    _check_the_split(topology)


# ------------------------------------- the core / owned / ghost node split

def _nodes_per_stratum(topology, nodes_per_entity):
    """One node count per numbering stratum.

    An extruded mesh reports a pair per base mesh stratum: the nodes ON the
    entity of each of its ``layers`` levels, and the nodes ABOVE it in each of
    the ``layers - 1`` cells of the column. Every other mesh reports one number
    per stratum already.
    """
    nodes = np.asarray(nodes_per_entity)
    if nodes.ndim == 1:
        return [int(count) for count in nodes]
    layers = topology.layers
    return [int(count) for count in
            sum(nodes[:, i] * (layers - i) for i in range(2))]


def _node_classes_from_the_plex(topology, nodes_per_stratum):
    """The core / owned / ghost node counts, cumulative, counted point by point.

    This does not restate ``node_classes``, which multiplies the node counts by
    ``_entity_classes_per_stratum``. It walks the plex points of each stratum
    and asks which class each one belongs to, taking the ghosts from the point
    SF rather than from the PyOP2 labels.
    """
    plex = topology.topology_dm
    _, local, _ = plex.getPointSF().getGraph()
    leaves = set() if local is None else {int(point) for point in local}
    if plex.getStratumSize("pyop2_core", 1) > 0:
        core_points = {int(point) for point in plex.getStratumIS("pyop2_core", 1).indices}
    else:
        core_points = set()

    core = owned = ghost = 0
    for nodes, points in zip(nodes_per_stratum, _stratum_points(topology)):
        for point in points:
            if point in leaves:
                ghost += nodes
            elif point in core_points:
                core += nodes
            else:
                owned += nodes
    return (core, core + owned, core + owned + ghost)


def _check_the_node_split(mesh, degree):
    topology = mesh.topology
    V = FunctionSpace(mesh, "CG", degree)
    nodes_per_entity = tuple(
        topology.make_dofs_per_plex_entity(V.finat_element.entity_dofs()))
    got = tuple(int(count) for count in topology.node_classes(nodes_per_entity))
    # Do the collectives before any assert, as in _check_the_split. The first
    # call of V.dim() is collective too.
    owned_on_all_ranks = mesh.comm.allreduce(got[1])
    core_on_all_ranks = mesh.comm.allreduce(got[0])
    dim = V.dim()

    assert got == _node_classes_from_the_plex(
        topology, _nodes_per_stratum(topology, nodes_per_entity))
    # The PyOP2 Set that the space is built on reports the same three numbers.
    assert got == tuple(int(size) for size in V.node_set.sizes)
    assert owned_on_all_ranks == dim

    # Guard against a vacuous pass. In serial the three numbers are all V.dim(),
    # which says nothing, so every rank must see ghost nodes and some rank must
    # hold core nodes. A rank whose whole partition touches the halo has none.
    assert got[2] > got[1]
    assert core_on_all_ranks > 0


@pytest.mark.parallel([2, 3])
@pytest.mark.parametrize("mesh_name", sorted(MESH_MAKERS))
@pytest.mark.parametrize("degree", [1, 3])
def test_node_classes_split_matches_the_plex(mesh_name, degree):
    """The core / owned / ghost node counts, not just their total.

    In serial all three are V.dim(), so only a parallel run says anything.
    """
    _check_the_node_split(MESH_MAKERS[mesh_name](), degree)


@pytest.mark.parallel([2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_node_classes_split_on_a_split_dimension(degree):
    """The prism mesh gives its two dimension 2 strata different node counts.

    A per-dimension split would give the triangular and the quadrilateral
    faces the same count, so this is the case the per-stratum split exists for.
    """
    mesh = Mesh(str(PRISM_MESH))
    nodes_per_entity = tuple(mesh.topology.make_dofs_per_plex_entity(
        FunctionSpace(mesh, "CG", degree).finat_element.entity_dofs()))
    triangle, quadrilateral = nodes_per_entity[2], nodes_per_entity[3]
    assert triangle == (degree - 1) * (degree - 2) // 2
    assert quadrilateral == (degree - 1) ** 2
    _check_the_node_split(mesh, degree)
