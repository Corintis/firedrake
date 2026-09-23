# Corrections to `prism_unstructured_implementation_plan.md`

The plan is a local, untracked file. It is NOT in the repository. Each item below
quotes the claim of the plan that it corrects, so this file is complete without it.

Every item below was established by running code, not by reading it. Each says
what the plan claims, what is actually true, and the evidence. To be folded into
the plan document when the branch is finished.

---

## 1. Section 8.5 / 8.6 — the highest-risk question was the wrong question

**Plan says.** The largest unknown is whether neighbouring prisms agree on the
extrinsic orientation `eo` of a shared quadrilateral face. If they agree, the
4-entry FInAT table suffices at every degree. Test this first; a wrong `eo`
gives silently wrong answers.

**Actually.** Neighbours do NOT always agree, and agreement was never the real
issue. On the three shipped meshes every quad face gives `eo = 1` and all 66
interior faces agree — but every prism in those meshes has its axis along `z`.
On a mesh with two prisms whose axes are perpendicular, the two cells give
`eo = 0` and `eo = 1` on the same face.

The deeper point: a face's cone is fixed once, when the face is created, and it
records the base edges at slots 0 and 2 *relative to the prism that created it*.
A neighbour whose axis runs differently through that same face sees those slots
holding its own axis edges. So an unstructured prism mesh genuinely presents all
8 quadrilateral orientations, while FInAT supplies 4.

**Evidence.** `tests/firedrake/meshes/prism/prism_two_perpendicular.msh`:

```
SHARED quad face 14: (cell, shipped, transposed) = [(0, 1, 5), (1, 6, 2)]
  eo as shipped = [0, 1]  -> neighbours DISAGREE
  transposed    = [5, 2]  -> 5 is outside the FInAT range 0..3
```

**Consequence.** This is an expressiveness gap in the ELEMENT, not a convention
mismatch in Firedrake, and it cannot be fixed on the Firedrake side. An
intermediate proposal — transposing the quad-face cone so `eo` becomes 0 — is
dead for two independent reasons: the cells disagree, and the transposed value
is still out of range. That proposal passed every test available at the time and
would have produced silently wrong answers on any mesh with non-parallel axes.

## 2. Section 8.5 / Phase C — the axis-consistency validator is unnecessary

**Plan says.** Build a validator rejecting meshes whose prisms disagree about
which edges run along the axis. Estimated 2 to 3 days.

**Actually.** Section 8.7's option C2 — teaching FIAT to emit all 8 orientations
for a prism quad face — is about four lines in
`FIAT/orientation_utils.py::_make_axis_perms_tensorproduct`, needs no Cython
change, and leaves the quadrilateral, hexahedron and interval tables
byte-identical. It handles perpendicular-axis meshes correctly instead of
rejecting them.

The change also affects the EXTRUDED wedge, `TensorProductCell(triangle, interval)`.
Its vertical-face entity permutation dict at dimension (1, 1) grows from 4 to 8
keys: `(0..1, 0..1, 0..1)` in place of `(0, 0..1, 0..1)`. The 4 old entries do not
change, and `symmetry_group_size` does not change. The change is inert, because an
extruded mesh always gives extrinsic part 0, so it reads only the old entries. The
generated code of the extruded kernels is byte-identical (Tasks 5 and 8).

**Consequence. Phase C should be deleted, not rescheduled.** The validator
exists only to reject meshes the real fix accepts.

**Note.** The name "Phase C" now means other work: interior facet integrals (`dS`)
on prisms. The user needs them for weak interior-facet conditions.
`phaseC_interior_facets_plan.md` gives that plan. The validator stays cancelled.
Phase C is done. Section 8 of the handover records it, and section 7 below
corrects its plan.

## 3. Section 4 — a missing layer: the global numbering

**Plan says.** The missing work is FIAT (one class), FInAT (three branches),
TSFC (two edits), mesh topology, the assembler facet split, and VTK.

**Actually.** A layer is missing from that list. `make_dofs_per_plex_entity`,
`create_section` and `node_classes` all assume every plex point of a given
DIMENSION carries the same dof count — true for every cell type Firedrake
supported before, false for a prism, which has both quadrilateral and
triangular faces at dimension 2.

**Evidence.** Before the fix, `prism_slab.msh` at CG2 gave `V.dim() = 403` with
only 325 nodes referenced and 78 empty stiffness rows — exactly the 78
triangular facets. Every solve above CG1 was singular.

**Consequence.** "Phase A delivers a usable solver on its own" is true only at
CG1 without this work. The fix touches numbering shared by every mesh type.

## 4. Section 12.5 — the Phase A exit criterion is not usable

**Plan says.** Phase A is done when `prism_smoke_test.py` reports PASS 18+,
FAIL 0, SKIP 0 for stages 1 to 6.

**Actually.** Unreachable and, where reachable, not diagnostic. Three
independent reasons:

1. **Stage 4 could not pass until Phase B. CLOSED.** It needed a facet entity
   argument that is Phase B work, and it passed at first only because the
   Appendix A.1 prototype patch was applied. Phase B (the `ds` split) landed, and
   stage 4 then passed with no prototype.
2. **Stage 6's `prism_warped.msh` assertions — the original claim here was
   WRONG, and is retracted.** It said a non-affine mesh cannot support an
   exactness test at any degree, because the space holds functions polynomial in
   REFERENCE coordinates while the probe is polynomial in PHYSICAL coordinates.
   Measured errors say otherwise:

   ```
   prism_warped.msh  CG3 = 3.375215309428213e-16
   prism_warped.msh  CG4 = 6.288223712270024e-16
   ```

   That is machine round-off, identical to the affine slab — not "below the
   tolerance".

   The correct rule: with a DEGREE-1 coordinate map, each of `x`, `y` and `z`
   lies in `P1(triangle) x P1(interval)` on ANY prism, affine or not. So a
   monomial of total degree `m` in physical coordinates lies in the degree-`k`
   space whenever `m <= k`. Geometry does not enter it.

   That rule explains all eight stage-6 entries exactly. The probe
   `x**k + 2*y**k + 3*z**k + x*y*z` is limited by its CUBIC `x*y*z` term:
   - CG1 fails on BOTH meshes — `x*y*z` needs degree 3.
   - CG2 fails on the WARPED mesh only. On the slab the axis is vertical, so `x`
     and `y` depend only on the triangle coordinates and `z` only on the
     interval one, placing `x*y*z` in `P2(triangle) x P1(interval)`. On the
     warped mesh all three mix, so it needs `P3 x P3`.
   - CG3 and CG4 are exact on both.

   So the only invalid entry is warped CG2, and it is invalid because of the
   PROBE's degree, not because of the geometry. `prism_warped.msh` is a
   perfectly good mesh for exactness tests at CG3 and above.

3. **Stage 6 is structurally blind to numbering defects.** Measured by reverting
   the fix, rebuilding, running, restoring and rebuilding: the smoke-test output
   is byte-identical either side of a change that moves `V.dim()` from 403 to
   325. Unreferenced dofs never enter an L2 interpolation error.

**Consequence.** The real Phase A gate is
`tests/firedrake/regression/test_prism.py` — `V.dim()`, the empty mass rows and
the CG2 Poisson solve — not the smoke test.

**Status.** `prism_smoke_test.py` is removed from the repository. pytest collected
it through the pattern `*_test.py`, and it stopped the collection with
INTERNALERROR. The regression tests replace it. The last version is
`git show c9a0f9fe7:prism_smoke_test.py`.

## 5. Section 5.4 / stage 7 — the shipped `eo` check proves nothing

**Plan says.** Stage 7 of the smoke test verifies the `eo` agreement claim.
(The script is removed now; see section 4.)

**Actually.** As written it iterates Firedrake cells 0 and 1 assuming they are
adjacent (RCM reordering breaks that), collects an `eo` from every interior quad
face of both cells rather than from one shared face, and compares that list to
its own first element. It can pass while proving nothing — and it does pass
today. It also never checks whether the orientation is in RANGE, which is the
actual defect.

## 6. Section 9 — an unlisted hazard that dominated the schedule

Not a plan error, but absent from it and expensive. A pre-existing PETSc at-exit
deadlock — `Py_FinalizeEx` -> `PetscGarbageCleanup` -> `PetscCommDuplicate` ->
`MPI_Comm_dup`, a collective the ranks do not all reach — stalls these suites.
It fires on SUCCESS as well as on failure, so it is not a symptom of a failing
test.

Practical guidance, each learned the hard way:

- Run each suite as its own `pytest` invocation. A deadlock in one otherwise
  destroys every other suite's results.
- Use `--timeout-method=signal`. `thread` cannot interrupt the blocking
  `waitpid`, so pytest-timeout escalates and kills the whole run.
- Reap orphaned MPI processes between suites, by the name of the script only,
  never with a global `pkill -f prterun`. Leaked ranks from a stuck teardown
  do consume cores, so this is worth doing — but do not over-credit it. A large
  apparent speedup between a combined run and per-suite runs on this project was
  mostly a warm TSFC kernel cache, not the reaping.
- Never run two MPI jobs at once; it manufactures a symptom indistinguishable
  from a real hang.
- If something looks stuck, `sample <pid> 5` two ranks BEFORE killing anything.
  A `PetscGarbageCleanup` stack means the body already passed. Symmetric 100%
  CPU proves nothing either way, because OpenMPI busy-polls.
- `os._exit(0)` at the end of a diagnostic script skips finalisation and avoids
  the deadlock entirely.

---

## 7. Corrections to `phaseC_interior_facets_plan.md`

The Phase C plan is also a local, untracked file. It came from a serial prototype
(`/tmp/phaseC/patch_dS.py`), and most of its claims were correct: the real code
gives the prototype values to 17 digits. Each item below quotes a claim that was
wrong or incomplete.

### 7.1 Plan 4.3 (C-F1): the lazy map broke every stored map

**Plan says.** Put the triangle map in
`AbstractQuadratureRule.intrinsic_orientation_permutation_map_tuple`, lazily. A
lazy property changes nothing for a kernel that does not ask for it.

**Actually.** The laziness is correct, but the property is not specific to the
triangle. Each kernel with the canonical point order calls it, and that includes
every hexahedron facet kernel and every prism `interior_facet_quad` kernel. The
first version (fiat `0029f52a`) tested for a missing map with
`io_ornt_map_tuple == (None,)`. A Python tuple compare compares the entries before
the lengths, so for a rule with a stored numpy map it evaluates `array == None`.
The truth value of that array is ambiguous, so Python raises `ValueError`. Thus
every hexahedron `dS` and `ds` kernel and every prism `interior_facet_quad` kernel
failed.

**Evidence.** A driver called the bodies of the `nprocs=2` tests of
`test_integral_hex.py` on one rank: 10 of 10 fail with the check of `0029f52a`,
and 10 of 10 pass with the fix. The fix, `cc839a77`, uses
`len(t) == 1 and t[0] is None`. `a6979469` adds
`test_stored_intrinsic_orientation_map`, which fails 5 of 5 on `0029f52a`.

**Why no test found it.** The FInAT suite passed (433 tests), because no FInAT
test calls this property. Its only caller is TSFC. The FIAT orientation tests use
a different property of the same name in `FIAT/quadrature.py`. The new tests, the
prototype comparison and the repr check used triangle and tetrahedron rules only.

**Lesson.** When a change is on a shared property, test each rule family that
calls it. Also run a caller from the consumer (TSFC or Firedrake), not only the
suite of the library.

### 7.2 Plan 3.5, 8.2 test 12: the metadata key is `"quadrature_rule"`

**Plan says.** A triangle rule that is not symmetric is `scheme="canonical"`, and
test 12 asks for an error with `scheme="canonical"` at degree 4.

**Actually.** The metadata key of the scheme is `"quadrature_rule"`
(`set_quad_rule` in `tsfc/kernel_interface/common.py`). A `"scheme"` key has no
effect. Measured at HEAD: `dS(10, metadata={"scheme": "canonical",
"quadrature_degree": 4})` gives the area 0.9999999999999984 with no error, because
the default rule is used. The test uses
`{"quadrature_rule": "canonical", "quadrature_degree": 4}`.

### 7.3 Plan 4.2 (C-T5): the error comes from `set_quad_rule`

**Plan says.** `_make_quad_multiindex_permuted` catches the `ValueError` and
raises a `NotImplementedError` with the integral type, the scheme and the degree.

**Actually.** That function does not know the scheme or the degree. `set_quad_rule`
knows them, so it asks for the map and raises the error. The catch in
`_make_quad_multiindex_permuted` stays as a second guard for a direct caller.

### 7.4 Contract item 6 and plan 4.2 (C-T4): the exterior types need the order in C8

**Plan says.** Do NOT turn on the canonical point order for the exterior shape
types.

**Actually.** That is correct for a kernel on one mesh, and the prism `ds`
kernels on one mesh stay byte-identical. But C8 couples a prism facet with the
cell of a facet submesh. The two see the facet in different orientations, so
`use_canonical_quadrature_point_ordering` is also on for an exterior shape type of
a prism when the kernel has more than one domain.

### 7.5 Plan 1.3 and 9.1 (C8): more than `trans_mesh_entity_map`

**Plan says.** A cross-mesh `dS` needs `trans_mesh_entity_map` for the shape
types, and the submesh cell side uses the triangle map.

**Actually.** C8 also changed `_split_facet_integrals_by_shape` (the submesh cell
fixes the facet shape, so the split keeps one shape),
`submesh_map_child_parent` (it gives the shape type, and it uses the facets of the
same kind when no rank holds a facet of the shape), the facet arguments in
`firedrake/assemble.py` (the other mesh takes its own integral type and
`shape_local_facet_dat`), and the canonical point order of 7.4.

The plan also did not list a limit that C8 found: an untagged cross-mesh measure
on the parent reads the map value -1 and gives a wrong answer with no error. This
is pre-existing and is on all cells. See section 8.3 of the handover.

### 7.6 Plan 8.2 test 8: the tag 10 variant is not in DG1

**Plan says.** On tag 10 use `u = z + 0.3*x*y` below the plane.

**Actually.** `x*y` is not in DG1. The test adds a linear harmonic term that
changes along the plane: `0.3 (x - 2y)` on tag 10 and `0.3 (y - 2z)` on tag 20.
It has the same teeth for DG1 and DG2: the triangle permutation mutation makes the
tag 10 case fail.

### 7.7 Plan 8.2 test 10: DG3 on the finest mesh costs too much

**Plan says.** SIPG convergence, DG1-3, on `prism_order_r{0,1,2}_scrambled.msh`.

**Actually.** The DG3 solve on `r2` took 454 s and 10 GB. Degree 3 uses `r0` and
`r1` only. Degrees 1 and 2 use all three meshes.

### 7.8 Plan 5.3 and 8.2: the parallel tests did not share a triangle

**Plan says.** Mark the exactness tests `parallel([1, 2, 3])`. A facet on the
partition boundary has nothing prism-specific.

**Actually.** The default partitioner put 0 to 2 triangles on a partition boundary
at 2 and 3 ranks. So the parallel tests of the plan did not test a triangle whose
two cells are on different ranks. C6 found this gap. `8b6155581` adds
`test_prism_dS_on_a_partition_boundary`, which uses a shell partition and asserts
that all 136 triangles of marker 10 are on the partition boundary.
