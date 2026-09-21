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
import numpy as np
import pytest

from firedrake import (ExtrudedMesh, FunctionSpace, UnitCubeMesh, UnitSquareMesh,
                       VectorFunctionSpace)


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
    changes. The serial test above pins the exact numbers.
    """
    mesh = MESH_MAKERS[mesh_name]()
    V = FunctionSpace(mesh, "CG", degree)
    owned = mesh.comm.allreduce(int(V.node_set.size))
    assert owned == V.dim()
