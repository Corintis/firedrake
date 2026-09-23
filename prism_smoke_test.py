#!/usr/bin/env python
"""Prism smoke test for unstructured Firedrake meshes.

Run this after every change while you implement
`prism_unstructured_implementation_plan.md`:

    python prism_smoke_test.py

The test has 7 stages. Each stage isolates one failure class, so a failure
tells you which part of the plan broke.

Stages 1 to 4 run today, against the prototype of Appendix A. They already
pass. Use them as a fixed target: the real implementation must reproduce the
same numbers.

Stages 5 to 7 need a real prism mesh. They report SKIP until Phase A lands.

The test uses the prototype patch ONLY when Firedrake cannot build a prism mesh
by itself. Once Phase A lands, the test drops the patch and exercises the real
code path. Nothing needs to change in this file.
"""
import os
import sys
import collections
import numpy as np

PASS, FAIL, SKIP = [], [], []


def check(stage, name, got, want, tol=1e-12):
    ok = np.allclose(got, want, atol=tol, rtol=0) if not isinstance(got, (list, tuple)) \
         or all(isinstance(x, (int, np.integer)) for x in got) is False else list(got) == list(want)
    if isinstance(got, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in got):
        ok = list(got) == list(want)
    (PASS if ok else FAIL).append(f"[{stage}] {name}")
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}")
    if not ok:
        print(f"        got  {got}")
        print(f"        want {want}")
    return ok


def skip(stage, name, why):
    SKIP.append(f"[{stage}] {name}")
    print(f"  SKIP  {name}   ({why})")


# ---------------------------------------------------------------- setup
def firedrake_builds_prisms():
    """True once Phase A lands. False today.

    The probe loads a gmsh file, which gives DM_POLYTOPE_TRI_PRISM. It must not
    use plex_from_cell_list, which gives DM_POLYTOPE_TRI_PRISM_TENSOR: that is
    the variant Firedrake rejects, so such a probe can never return True.
    """
    try:
        from firedrake import Mesh
        here = os.path.dirname(os.path.abspath(__file__))
        Mesh(os.path.join(here, "prism_meshes", "prism_reference.msh"))
        return True
    except Exception:
        return False


NATIVE = firedrake_builds_prisms()
print(f"Firedrake builds prism meshes natively: {NATIVE}")
print("Using the Appendix A prototype for the element layer.\n" if not NATIVE else "")

import FIAT
import FIAT.reference_element as RE
import ufl
import finat
import finat.ufl

if not NATIVE:
    from FIAT.reference_element import (Cell, TensorProductCell, UFCTriangle,
                                        UFCInterval, flatten_entities,
                                        compute_unflattening_map)
    PRISM_SHAPE = 111222

    class UFCPrism(RE.Hypercube):
        def __init__(self):
            product = TensorProductCell(UFCTriangle(), UFCInterval())
            Cell.__init__(self, PRISM_SHAPE, product.get_vertices(),
                          flatten_entities(product.get_topology()))
            self.dimension = 3
            self.shape = PRISM_SHAPE
            self.product = product
            self.unflattening_map = compute_unflattening_map(product.get_topology())

        def _key(self):
            return ("prism",)

    import finat.cube
    import finat.quadrature
    import finat.element_factory as EF
    import tsfc.fem
    import tsfc.kernel_interface.common as KC
    import tsfc.kernel_interface.firedrake_loopy as KFL

    EF.ufc_cell = lambda c: UFCPrism() if (c if isinstance(c, str) else c.cellname) == "prism" \
        else RE.ufc_cell(c)
    finat.cube.FlattenedDimensions.cell = property(
        lambda self: UFCPrism() if len(getattr(self.product.cell, "cells", ())) == 2
        and self.product.cell.get_spatial_dimension() == 3 else RE.UFCHexahedron())

    _mq = finat.quadrature.make_quadrature

    def _mq_prism(r, d, **kw):
        return _mq(r.product, d, **kw) if isinstance(r, UFCPrism) else _mq(r, d, **kw)
    tsfc.fem.make_quadrature = _mq_prism

    _prism_tpc = ufl.TensorProductCell(ufl.triangle, ufl.interval)
    _orig_convert = EF.convert.dispatch(finat.ufl.FiniteElement)

    @EF.convert.register(finat.ufl.FiniteElement)
    def _convert_prism(element, **kw):
        if element.cell.cellname == "prism":
            e = finat.ufl.TensorProductElement(
                finat.ufl.FiniteElement(element.family(), ufl.triangle, element.degree()),
                finat.ufl.FiniteElement(element.family(), ufl.interval, element.degree()),
                cell=_prism_tpc)
            fe, deps = EF._create_element(e, **kw)
            return finat.cube.FlattenedDimensions(fe), deps
        return _orig_convert(element, **kw)

    SHAPE = {"which": "quad"}

    def _sub(self, dim):
        # dim == 3 must return the prism itself. Falling through to
        # RE.Cell.construct_subcomplex here recurses forever, because that
        # method calls construct_subelement straight back.
        if dim == 3:
            return self
        if dim == 2:
            return RE.UFCQuadrilateral() if SHAPE["which"] == "quad" else RE.UFCTriangle()
        return RE.Cell.construct_subcomplex(self, dim)
    UFCPrism.construct_subcomplex = _sub
    UFCPrism.construct_subelement = _sub

    _orig_lit = KC.lower_integral_type

    def _lit(fiat_cell, integral_type):
        if isinstance(fiat_cell, UFCPrism) and integral_type == "exterior_facet":
            return 2, ([0, 1, 2] if SHAPE["which"] == "quad" else [3, 4])
        return _orig_lit(fiat_cell, integral_type)
    for _m in (KC, tsfc.fem, KFL):
        _m.lower_integral_type = _lit

    def prism_cell():
        return UFCPrism()
else:
    SHAPE = {"which": "quad"}

    def prism_cell():
        return FIAT.ufc_cell("prism")


# ------------------------------------------------- stage 1: reference cell
print("Stage 1  reference cell")
p = prism_cell()
top = p.get_topology()
check(1, "entity counts (v, e, f, c)",
      [len(top[0]), len(top[1]), len(top[2]), len(top[3])], [6, 9, 5, 1])
check(1, "faces 0,1,2 are quads; 3,4 are triangles",
      [len(top[2][i]) for i in range(5)], [4, 4, 4, 3, 3])
check(1, "reference volume", p.volume(), 0.5)
# FIAT normalises reference normals in the INFINITY norm, not the 2-norm.
check(1, "reference normals",
      np.array([p.compute_reference_normal(2, e) for e in range(5)]),
      np.array([[1, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1]], dtype=float))

# ------------------------------- stage 2: gmsh meshes and closure structure
print("\nStage 2  gmsh prism meshes and the closure permutation  (plan section 5)")
MESHDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prism_meshes")
MESHES = ("prism_reference.msh", "prism_slab.msh", "prism_warped.msh")

# The permutation that _reorder_plex_closure must implement.
# fiat_closure[i] = plex_closure[PERM[i]]
#   entries  0..5   FIAT vertices
#   entries  6..14  FIAT edges
#   entries 15..19  FIAT faces (0,1,2 quads; 3,4 triangles)
#   entry   20      the cell
PERM = [15, 18, 16, 20, 17, 19, 13, 14, 12, 7, 10, 8, 9, 6, 11, 4, 3, 5, 1, 2, 0]

try:
    from firedrake.petsc import PETSc
    from firedrake import COMM_WORLD
    missing = [m for m in MESHES if not os.path.exists(os.path.join(MESHDIR, m))]
    if missing:
        raise FileNotFoundError(f"run prism_meshes/make_prism_mesh.py first; missing {missing}")

    top = p_topology = prism_cell().get_topology()
    seen, ncells, celltypes = collections.Counter(), 0, set()

    for name in MESHES:
        plex = PETSc.DMPlex().createFromFile(os.path.join(MESHDIR, name), comm=COMM_WORLD)
        vS, vE = plex.getDepthStratum(0)
        cS, cE = plex.getHeightStratum(0)
        for c in range(cS, cE):
            celltypes.add(int(plex.getCellType(c)))
            clo, _ = plex.getTransitiveClosure(c)
            pos = {int(q): i for i, q in enumerate(clo)}

            def vset(q):
                vs, _ = plex.getTransitiveClosure(q)
                return frozenset(int(x) for x in vs if vS <= x < vE)

            fv = [int(clo[k]) for k in (15, 18, 16, 20, 17, 19)]
            by_v = {vset(q): int(q) for q in clo if not (vS <= q < vE)}
            row = [pos[v] for v in fv]
            for d in (1, 2):
                for i in range(len(top[d])):
                    row.append(pos[by_v[frozenset(fv[v] for v in top[d][i])]])
            row.append(pos[int(c)])
            seen[tuple(row)] += 1
            ncells += 1

    # 8 = DM_POLYTOPE_TRI_PRISM. 9 = DM_POLYTOPE_TRI_PRISM_TENSOR, which has a
    # different cone ordering and would invalidate PERM.
    check(2, "gmsh cells are TRI_PRISM (8), not TRI_PRISM_TENSOR (9)",
          sorted(celltypes), [8])
    check(2, f"one closure permutation across all {ncells} cells", len(seen), 1)
    check(2, "the permutation equals PERM", list(next(iter(seen))), PERM)
except Exception as exc:
    FAIL.append("[2] gmsh meshes and closure permutation")
    print(f"  FAIL  gmsh meshes and closure permutation: {type(exc).__name__}: {exc}")

# ------------------------------------------------- kernel helpers
import tsfc
import loopy
import ctypes
import subprocess
import tempfile
import os

CELL = ufl.Cell("prism")
UFLMESH = ufl.Mesh(finat.ufl.VectorElement("Lagrange", CELL, 1, dim=3))
DP = ctypes.POINTER(ctypes.c_double)
UP = ctypes.POINTER(ctypes.c_uint32)
# FIAT prism vertex order, matching p.get_vertices()
REF = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1]], float)


def facet_measure(which):
    """The measure over the facets of one shape.

    Phase B gives TSFC one integral type per facet shape, so the native path
    selects the shape with the measure. The Appendix A prototype patch reads
    SHAPE["which"] inside lower_integral_type instead, so it keeps ds.
    """
    if not NATIVE:
        return ufl.ds(domain=UFLMESH)
    name = "exterior_facet_quad" if which == "quad" else "exterior_facet_tri"
    return ufl.Measure(name, domain=UFLMESH)


def build(form):
    k, = tsfc.compile_form(form, parameters={"mode": "spectral"})
    d = tempfile.mkdtemp()
    src, so = os.path.join(d, "k.c"), os.path.join(d, "k.so")
    open(src, "w").write(loopy.generate_code_v2(k.ast).device_code())
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", so, src], check=True)
    fn = getattr(ctypes.CDLL(so), k.name)
    fn.restype = None
    return fn


# --------------------------------------------------- stage 3: cell kernel
print("\nStage 3  cell integral kernel")
try:
    fn = build(ufl.as_ufl(1.0) * ufl.dx(domain=UFLMESH))

    def volume(v):
        A = np.zeros(1)
        c = np.ascontiguousarray(v, dtype=float).ravel()
        fn(A.ctypes.data_as(DP), c.ctypes.data_as(DP))
        return A[0]
    scaled = REF.copy(); scaled[:, 0] *= 2.0; scaled[:, 2] *= 3.0
    sheared = REF.copy(); sheared[:, 0] += 0.7 * sheared[:, 2]
    warped = REF.copy(); warped[[1, 3, 5], 0] += 0.4; warped[[1, 3, 5], 1] -= 0.2
    check(3, "reference prism volume", volume(REF), 0.5)
    check(3, "scaled x2 z3 volume", volume(scaled), 3.0)
    check(3, "sheared volume", volume(sheared), 0.5)
    check(3, "non-affine top volume", volume(warped), 0.5)
except Exception as exc:
    FAIL.append("[3] cell kernel")
    print(f"  FAIL  cell kernel: {type(exc).__name__}: {exc}")

# -------------------------------------------------- stage 4: facet kernels
print("\nStage 4  facet integral kernels  (each facet separately)")
print("         A correct TOTAL can hide two errors that cancel.")
try:
    areas = {}
    for which, ents, want in (("quad", [0, 1, 2], [np.sqrt(2), 1.0, 1.0]),
                              ("tri", [3, 4], [0.5, 0.5])):
        SHAPE["which"] = which
        if hasattr(tsfc.fem.get_quadrature_rule, "cache_clear"):
            tsfc.fem.get_quadrature_rule.cache_clear()
        fn = build(ufl.as_ufl(1.0) * facet_measure(which))
        for pos, ent in enumerate(ents):
            A = np.zeros(1)
            c = np.ascontiguousarray(REF, dtype=float).ravel()
            # NOTE: `pos`, not `ent`. gem.select_expression indexes POSITIONALLY.
            f = np.array([pos], dtype=np.uint32)
            fn(A.ctypes.data_as(DP), c.ctypes.data_as(DP), f.ctypes.data_as(UP))
            areas[ent] = A[0]
        for ent, w in zip(ents, want):
            check(4, f"facet {ent} area ({which})", areas[ent], w)
    check(4, "total surface area", sum(areas.values()), 2.0 + np.sqrt(2) + 1.0)
    # FacetNormal must at least compile.
    SHAPE["which"] = "quad"
    V = ufl.FunctionSpace(UFLMESH, finat.ufl.FiniteElement("Lagrange", CELL, 2))
    n = ufl.FacetNormal(UFLMESH)
    tsfc.compile_form(ufl.inner(ufl.grad(ufl.Coefficient(V)), n) * ufl.TestFunction(V)
                      * facet_measure(SHAPE["which"]),
                      parameters={"mode": "spectral"})
    check(4, "FacetNormal form compiles", True, True)
except Exception as exc:
    FAIL.append("[4] facet kernels")
    print(f"  FAIL  facet kernels: {type(exc).__name__}: {exc}")

# ----------------------------------------- stages 5 to 7: need a real mesh
print("\nStage 5  one-prism mesh")
print("Stage 6  two-prism dof matching")
print("Stage 7  eo agreement on a shared quad face   (plan section 8.6)")
if not NATIVE:
    for s, name in ((5, "mass matrix row sums"),
                    (6, "two-prism interpolation exactness, CG1 to CG4"),
                    (7, "eo agrees between neighbours on a shared quad face")):
        skip(s, name, "Phase A not landed; Firedrake cannot build a prism mesh")
else:
    from firedrake import (Mesh, FunctionSpace, Function, SpatialCoordinate,
                           assemble, dx, TrialFunction, TestFunction, inner, COMM_WORLD)
    from firedrake.mesh import plex_from_cell_list
    m1 = Mesh(os.path.join(MESHDIR, "prism_reference.msh"))
    for deg in (1, 2, 3):
        W = FunctionSpace(m1, "CG", deg)
        M = assemble(inner(TrialFunction(W), TestFunction(W)) * dx).M.values
        check(5, f"CG{deg} mass matrix total equals volume", M.sum(), 0.5)
    check(5, "prism_slab.msh volume equals 0.6",
          float(assemble(1.0 * dx(domain=Mesh(os.path.join(MESHDIR, "prism_slab.msh"))))), 0.6)

    # 52 prisms, unstructured in plane, with shared bases AND shared quad faces.
    # prism_warped.msh additionally has no affine cell and no common axis.
    for meshname in ("prism_slab.msh", "prism_warped.msh"):
        m2 = Mesh(os.path.join(MESHDIR, meshname))
        for deg in (1, 2, 3, 4):
            W = FunctionSpace(m2, "CG", deg)
            x, y, z = SpatialCoordinate(m2)
            e = x**deg + 2 * y**deg + 3 * z**deg + x * y * z
            u = Function(W).interpolate(e)
            err = float(np.sqrt(abs(assemble((u - e)**2 * dx))))
            check(6, f"{meshname} CG{deg} interpolation exact", err, 0.0, tol=1e-9)

    # Stage 7: the highest-risk claim in the plan.
    m3 = Mesh(os.path.join(MESHDIR, "prism_warped.msh"))
    plex3 = m3.topology.topology_dm
    closure = m3.topology.cell_closure
    orient = m3.topology.entity_orientations
    # Flat closure layout: 6 vertices, 9 edges, 5 faces, 1 cell -> faces at 15..19
    shared, eos = None, []
    for c in range(2):
        for i, col in enumerate(range(15, 20)):
            pt = closure[c, col]
            if plex3.getConeSize(pt) == 4:          # a quad face
                sup = plex3.getSupport(pt)
                if len(sup) == 2:                    # shared by both prisms
                    shared = pt
                    eos.append(int(orient[c, col]) // 4)
    if shared is None:
        skip(7, "eo agreement", "no shared quad face found; check the closure layout")
    else:
        check(7, "eo agrees between the two prisms", eos, [eos[0]] * len(eos))

# --------------------------------------------------------------- summary
print("\n" + "=" * 62)
print(f"PASS {len(PASS)}   FAIL {len(FAIL)}   SKIP {len(SKIP)}")
for f in FAIL:
    print("  FAILED:", f)
print("=" * 62)
sys.exit(1 if FAIL else 0)
