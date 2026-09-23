"""Core tests of the interior facet integrals (dS) on unstructured prism meshes.

Firedrake splits a prism dS into one kernel per facet shape
(``interior_facet_tri`` and ``interior_facet_quad``). These tests check the
split, the data that the kernels get, and the paths that stay unsupported.
The main exactness tests, on scrambled meshes, are in test_prism.py.
"""
from pathlib import Path

import numpy as np
import pytest

from firedrake import (Constant, FacetNormal, Function, FunctionSpace, Mesh,
                       SpatialCoordinate, TestFunction, TrialFunction,
                       UnitCubeMesh, assemble, cos, dS, exp, inner, jump, sin)
from firedrake.tsfc_interface import compile_form
from tsfc.kernel_interface.common import lower_integral_type
from finat.element_factory import as_fiat_cell

MESHDIR = Path(__file__).parent.parent / "meshes" / "prism"


def _kernel_types(form):
    return sorted(k.kinfo.integral_type for k in compile_form(form, "form"))


def test_prism_dS_gives_one_kernel_per_facet_shape():
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    assert _kernel_types(Constant(1.0) * dS(domain=mesh)) == ["interior_facet_quad", "interior_facet_tri"]


def test_other_cells_keep_one_dS_kernel():
    mesh = UnitCubeMesh(1, 1, 1)
    assert _kernel_types(Constant(1.0) * dS(domain=mesh)) == ["interior_facet"]


@pytest.mark.parametrize("meshfile", ["prism_slab.msh", "prism_warped.msh", "prism_two_perpendicular.msh"])
def test_prism_interior_facet_positions(meshfile):
    """Each side of an interior facet gets the POSITION of its FIAT facet.

    The kernel selects its facet by position in the entity list of its
    integral type. The FIAT facet number stays in local_facet_dat.
    """
    mesh = Mesh(str(MESHDIR / meshfile))
    facets = mesh.interior_facets
    fiat_cell = as_fiat_cell(mesh.ufl_cell())
    shape_indices, positions = facets._facet_shape_groups
    local = facets.local_facet_dat.data_ro_with_halos.reshape(-1, 2)
    present = np.asarray(facets.facet_cell).reshape(-1, 2) != -1
    assert positions.shape == (len(facets.facets), 2)
    both = np.concatenate([shape_indices[t] for t in sorted(shape_indices)])
    assert sorted(both) == list(range(len(facets.facets)))
    for integral_type, indices in shape_indices.items():
        _, entity_ids = lower_integral_type(fiat_cell, integral_type)
        for side in range(2):
            ok = present[indices, side]
            got = np.asarray(entity_ids)[positions[indices, side][ok]]
            assert (got == local[indices, side][ok]).all()
    assert (facets.shape_local_facet_dat.data_ro_with_halos.reshape(-1, 2) == positions).all()


@pytest.mark.parametrize("meshfile", ["prism_warped.msh", "prism_two_perpendicular.msh"])
def test_prism_dS_sides_agree(meshfile):
    """The two sides of each interior facet give the same physical points.

    A constant along the facet cannot detect a point order error, so the
    integrands are the coordinates and a smooth CG function.
    """
    mesh = Mesh(str(MESHDIR / meshfile))
    x = SpatialCoordinate(mesh)
    assert abs(assemble(inner(x('+') - x('-'), x('+') - x('-')) * dS)) < 1e-26
    V = FunctionSpace(mesh, "CG", 3)
    u = Function(V).interpolate(sin(3 * x[0]) * cos(2 * x[1]) * exp(x[2]))
    assert abs(assemble(jump(u) ** 2 * dS)) < 1e-26
    n = FacetNormal(mesh)
    assert abs(assemble(inner(n('+') + n('-'), n('+') + n('-')) * dS)) < 1e-26


def test_prism_dS_with_a_quadrature_rule_object_is_rejected():
    from FIAT.reference_element import UFCTriangle
    from finat.quadrature import make_quadrature

    rule = make_quadrature(UFCTriangle(), 1)
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    with pytest.raises(NotImplementedError, match="interior facet integral"):
        assemble(Constant(1.0) * dS(domain=mesh, metadata={"quadrature_rule": rule}))


def test_prism_dS_with_a_triangle_rule_that_is_not_symmetric_is_rejected():
    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    x = SpatialCoordinate(mesh)
    metadata = {"quadrature_rule": "canonical", "quadrature_degree": 4}
    with pytest.raises(NotImplementedError, match="scheme 'canonical' at degree 4"):
        assemble(x[0]('+') * dS(domain=mesh, metadata=metadata))


def test_prism_dS_subdomain_data_gives_the_error_of_other_meshes():
    def error(mesh):
        data = mesh.interior_facets.measure_set("interior_facet", "everywhere")
        with pytest.raises(NotImplementedError) as excinfo:
            assemble(Constant(1.0) * dS(domain=mesh, subdomain_data=data))
        return str(excinfo.value)

    assert error(Mesh(str(MESHDIR / "prism_slab.msh"))) == error(UnitCubeMesh(1, 1, 1))


def test_prism_slate_dS_is_rejected_and_names_dS():
    from firedrake import Tensor

    mesh = Mesh(str(MESHDIR / "prism_slab.msh"))
    V = FunctionSpace(mesh, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    with pytest.raises(NotImplementedError, match=r"interior_facet, measure dS"):
        assemble(Tensor(jump(u) * jump(v) * dS))
