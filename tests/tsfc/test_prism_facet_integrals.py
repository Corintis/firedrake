"""Element layer tests for the two facet integral types of a prism.

A prism has two facet shapes: three quadrilaterals (FIAT faces 0, 1 and 2) and
two triangles (FIAT faces 3 and 4). TSFC fixes one reference facet cell and one
quadrature rule per kernel, so one exterior facet integral becomes two
integrals, one per shape. These tests cover the element layer only. The
Firedrake side of the split is a later task.

The kernel selects among the entities of an integral type by POSITION in the
list that `lower_integral_type` returns, not by FIAT entity number. For the
triangular facets the list is [3, 4], so the kernel expects the runtime values
0 and 1. The tests below pass the position, and they check each facet with
three probes whose values are different on every facet. A probe that measures
only the area cannot tell facet 1 from facet 2, or facet 3 from facet 4.
"""
import ctypes
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import gem
import loopy
import tsfc
import ufl
import finat.ufl
from FIAT.reference_element import (UFCInterval, UFCTriangle, UFCQuadrilateral,
                                    UFCTetrahedron, UFCHexahedron, UFCPrism,
                                    TensorProductCell, make_affine_mapping)
from tsfc.fem import get_quadrature_rule, make_cell_facet_jacobian
from tsfc.kernel_interface.common import lower_integral_type


PRISM = UFCPrism()

# The FIAT reference prism, in FIAT vertex order.
REF = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1]],
               dtype=float)

# One row per FIAT facet: the integral type, the position in the entity list,
# and the three probe values (the area, the integral of x, the integral of z).
SQRT2 = np.sqrt(2.0)
FACETS = [
    # facet, integral type, position, area, integral of x, integral of z
    (0, "exterior_facet_quad", 0, SQRT2, 0.5 * SQRT2, 0.5 * SQRT2),
    (1, "exterior_facet_quad", 1, 1.0, 0.0, 0.5),
    (2, "exterior_facet_quad", 2, 1.0, 0.5, 0.5),
    (3, "exterior_facet_tri", 0, 0.5, 1.0 / 6.0, 0.0),
    (4, "exterior_facet_tri", 1, 0.5, 1.0 / 6.0, 0.5),
]


# ------------------------------------------------------- the integral types
@pytest.mark.parametrize(("integral_type", "entity_ids"),
                         [("exterior_facet_tri", [3, 4]),
                          ("exterior_facet_quad", [0, 1, 2])])
def test_prism_facet_integral_type(integral_type, entity_ids):
    dim, ids = lower_integral_type(PRISM, integral_type)
    assert dim == 2
    assert ids == entity_ids


def test_prism_entity_list_is_the_positional_contract():
    """State the contract that the Firedrake side must honour.

    The kernel reads the facet number as a POSITION in this list. The caller
    must therefore pass the index of the facet within its shape group, not the
    FIAT facet number.
    """
    for facet, integral_type, position, _, _, _ in FACETS:
        _, entity_ids = lower_integral_type(PRISM, integral_type)
        assert entity_ids[position] == facet


def test_prism_exterior_facet_is_rejected():
    """A plain exterior facet integral on a prism must not compile.

    Two facet shapes need two quadrature rules. If this raise is removed, the
    kernel uses the rule of the first facet on every facet, and the answers on
    the other shape are wrong with no message.
    """
    with pytest.raises(ValueError) as excinfo:
        lower_integral_type(PRISM, "exterior_facet")
    assert "more than one shape" in str(excinfo.value)


def test_prism_interior_facet_is_not_supported():
    """dS on a prism is out of scope. A split by facet shape does not fix it."""
    with pytest.raises(NotImplementedError, match="not supported on prism meshes"):
        lower_integral_type(PRISM, "interior_facet")


@pytest.mark.parametrize("cell", [UFCTriangle(), UFCTetrahedron(), UFCHexahedron()])
@pytest.mark.parametrize("integral_type", ["exterior_facet_tri", "exterior_facet_quad"])
def test_shape_restricted_types_need_two_facet_shapes(cell, integral_type):
    with pytest.raises(ValueError):
        lower_integral_type(cell, integral_type)


@pytest.mark.parametrize("integral_type", ["exterior_facet_tri", "exterior_facet_quad"])
def test_shape_restricted_types_reject_a_tensor_product_cell(integral_type):
    """An extruded cell separates its facets by direction, not by shape.

    Without this guard the dimension arithmetic gives a bare TypeError,
    because the dimension of a tensor product cell is a tuple.
    """
    cell = TensorProductCell(UFCTriangle(), UFCInterval())
    with pytest.raises(ValueError) as excinfo:
        lower_integral_type(cell, integral_type)
    assert "TensorProductCell" in str(excinfo.value)


@pytest.mark.parametrize(("cell", "integral_type", "dim", "entity_ids"), [
    (UFCInterval(), "cell", 1, [0]),
    (UFCInterval(), "exterior_facet", 0, [0, 1]),
    (UFCTriangle(), "cell", 2, [0]),
    (UFCTriangle(), "exterior_facet", 1, [0, 1, 2]),
    (UFCTriangle(), "interior_facet", 1, [0, 1, 2]),
    (UFCTriangle(), "vertex", 0, [0, 1, 2]),
    (UFCQuadrilateral(), "exterior_facet", 1, [0, 1, 2, 3]),
    (UFCTetrahedron(), "exterior_facet", 2, [0, 1, 2, 3]),
    (UFCHexahedron(), "exterior_facet", 2, [0, 1, 2, 3, 4, 5]),
    (UFCPrism(), "cell", 3, [0]),
    (UFCPrism(), "vertex", 0, [0, 1, 2, 3, 4, 5]),
    (TensorProductCell(UFCTriangle(), UFCInterval()), "exterior_facet_bottom", (2, 0), [0]),
    (TensorProductCell(UFCTriangle(), UFCInterval()), "exterior_facet_top", (2, 0), [1]),
    (TensorProductCell(UFCTriangle(), UFCInterval()), "exterior_facet_vert", (1, 1), [0, 1, 2]),
    (TensorProductCell(UFCTriangle(), UFCInterval()), "interior_facet_horiz", (2, 0), [0, 1]),
])
def test_other_cells_are_unchanged(cell, integral_type, dim, entity_ids):
    got_dim, got_ids = lower_integral_type(cell, integral_type)
    assert got_dim == dim
    assert got_ids == entity_ids


# ------------------------------------------------------ the reference cells
def test_tsfc_builds_one_quadrature_rule_per_facet_shape():
    """The TSFC path must reach the entity aware FIAT method.

    The two rules must also be distinct. `get_quadrature_rule` is cached, so a
    cache key that drops the entity returns the rule of the first facet shape
    for both.
    """
    tri = get_quadrature_rule(PRISM, 2, 2, "default", 3)
    quad = get_quadrature_rule(PRISM, 2, 2, "default", 0)
    # A rule on a quadrilateral sits on the underlying tensor product cell, as
    # it does for a quadrilateral mesh, and it holds its weights in factors.
    # Compare the reference volume, not the class.
    assert isinstance(tri.ref_el, UFCTriangle)
    assert np.isclose(tri.ref_el.volume(), 0.5)
    assert np.isclose(quad.ref_el.volume(), 1.0)
    assert np.isclose(sum(tri.weights), 0.5)
    assert np.isclose(np.prod([f.weights.sum() for f in quad.factors]), 1.0)
    assert tri is not quad


def test_prism_needs_the_entity_to_build_a_facet_cell():
    with pytest.raises(NotImplementedError):
        PRISM.construct_subcomplex(2)


@pytest.mark.parametrize("facet", [0, 1, 2, 3, 4])
def test_cell_facet_jacobian_covers_every_prism_facet(facet):
    jacobian = make_cell_facet_jacobian(PRISM, 2, facet)
    assert np.asarray(jacobian).shape == (3, 2)


@pytest.mark.parametrize(("facet", "wrong_cell"),
                         [(0, UFCTriangle()), (3, UFCQuadrilateral())])
def test_the_two_facet_reference_cells_differ_by_a_reflection(facet, wrong_cell):
    """Record why one mutation of `make_cell_facet_jacobian` cannot be caught.

    The first three vertices of the reference quadrilateral are the three of
    the reference triangle with the last two exchanged. `make_cell_facet_jacobian`
    builds its affine map from the first three vertices only, so the wrong facet
    reference cell gives the correct matrix with its two COLUMNS exchanged. That
    is the reflection u <-> v. Both reference cells are symmetric under it, so
    the integration domain and its measure do not change, and no scalar facet
    integral can see the substitution.

    The entity must still reach FIAT. Without it the call raises, which is the
    failure this task clears. This test pins the reason the mutation check for
    that call site reports the substitution as unobservable rather than caught.
    """
    correct = np.asarray(make_cell_facet_jacobian(PRISM, 2, facet))
    vertices = PRISM.get_vertices_of_subcomplex(PRISM.get_topology()[2][facet])
    forced, _ = make_affine_mapping(wrong_cell.get_vertices()[:3], vertices[:3])
    assert np.allclose(np.asarray(forced), correct[:, ::-1])


# ------------------------------------------------------------- the kernels
def _prism_mesh():
    cell = ufl.Cell("prism")
    return ufl.Mesh(finat.ufl.VectorElement("Lagrange", cell, 1, dim=3))


def _integrand(name, mesh):
    """Return the integrand of one probe.

    The probes are the constant 1, the two coordinates that separate facets of
    equal area, and the three components of the outward normal.
    """
    if name == "one":
        return ufl.as_ufl(1.0)
    if name in ("x", "z"):
        return ufl.SpatialCoordinate(mesh)[{"x": 0, "z": 2}[name]]
    return ufl.FacetNormal(mesh)[int(name[1])]


_KERNELS = {}


def _kernel(integral_type, probe):
    """Compile one probe on one integral type, once per session."""
    key = (integral_type, probe)
    if key not in _KERNELS:
        mesh = _prism_mesh()
        form = _integrand(probe, mesh) * ufl.Measure(integral_type, domain=mesh)
        _KERNELS[key] = _compile(form)
    return _KERNELS[key]


def _compile(form):
    """Compile one form to a ctypes callable and its argument names."""
    kernel, = tsfc.compile_form(form, parameters={"mode": "spectral"})
    names = [a.name for a in kernel.ast.default_entrypoint.args]
    directory = tempfile.mkdtemp()
    source = os.path.join(directory, "kernel.c")
    library = os.path.join(directory, "kernel.so")
    with open(source, "w") as f:
        f.write(loopy.generate_code_v2(kernel.ast).device_code())
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", library, source], check=True)
    function = getattr(ctypes.CDLL(library), kernel.name)
    function.restype = None
    return function, names


def _run(function, names, coords, facet):
    """Run a compiled facet kernel and return the single output value."""
    buffers = {
        "A": np.zeros(1, dtype=float),
        "coords_0": np.ascontiguousarray(coords, dtype=float).ravel(),
        "facet_0": np.array([facet], dtype=np.uint32),
        "entity_orientations_0": np.zeros(1, dtype=gem.uint_type),
    }
    unknown = set(names) - set(buffers)
    assert not unknown, f"the kernel takes an argument this test does not supply: {unknown}"
    function(*[buffers[name].ctypes.data_as(ctypes.c_void_p) for name in names])
    return buffers["A"][0]


needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")


@pytest.mark.parametrize("integral_type", ["exterior_facet_tri", "exterior_facet_quad"])
@needs_cc
def test_prism_facet_kernel_signature(integral_type):
    _, names = _kernel(integral_type, "one")
    assert names == ["A", "coords_0", "facet_0"]


@pytest.mark.parametrize(("facet", "integral_type", "position", "area", "x", "z"), FACETS)
@needs_cc
def test_prism_each_facet_separately(facet, integral_type, position, area, x, z):
    """Measure each facet on its own with three probes.

    A correct total can hide two errors that cancel. The area alone cannot tell
    facet 1 from facet 2, or facet 3 from facet 4, because those pairs have the
    same area. The integrals of x and of z separate them.
    """
    for probe, expected in (("one", area), ("x", x), ("z", z)):
        function, names = _kernel(integral_type, probe)
        got = _run(function, names, REF, position)
        assert np.isclose(got, expected, atol=1e-12), \
            f"facet {facet}, probe {probe}: got {got}, want {expected}"


@needs_cc
def test_prism_facet_areas_sum_to_the_surface_area():
    total = 0.0
    for integral_type in ("exterior_facet_tri", "exterior_facet_quad"):
        function, names = _kernel(integral_type, "one")
        _, entity_ids = lower_integral_type(PRISM, integral_type)
        for position in range(len(entity_ids)):
            total += _run(function, names, REF, position)
    assert np.isclose(total, 2.0 + SQRT2 + 1.0)


@pytest.mark.parametrize("integral_type", ["exterior_facet_tri", "exterior_facet_quad"])
def test_prism_facet_normal_compiles(integral_type):
    mesh = _prism_mesh()
    cell = ufl.Cell("prism")
    V = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("Lagrange", cell, 2))
    normal = ufl.FacetNormal(mesh)
    form = ufl.inner(ufl.grad(ufl.Coefficient(V)), normal) * ufl.TestFunction(V) \
        * ufl.Measure(integral_type, domain=mesh)
    kernels = tsfc.compile_form(form, parameters={"mode": "spectral"})
    assert len(kernels) == 1


@needs_cc
def test_prism_facet_normal_points_outward():
    """Integrate the outward normal over the whole boundary.

    The divergence theorem gives zero for a closed surface. A normal that
    points inward on one facet breaks this.
    """
    total = np.zeros(3)
    for integral_type in ("exterior_facet_tri", "exterior_facet_quad"):
        _, entity_ids = lower_integral_type(PRISM, integral_type)
        for component in range(3):
            function, names = _kernel(integral_type, f"n{component}")
            for position in range(len(entity_ids)):
                total[component] += _run(function, names, REF, position)
    assert np.allclose(total, 0.0, atol=1e-12)
