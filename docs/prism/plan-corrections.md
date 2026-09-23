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
