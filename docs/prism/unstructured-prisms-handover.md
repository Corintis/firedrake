# Unstructured prisms in Firedrake — implementation handover

Branch: `prism-unstructured` (this repository) and `prism-unstructured` in `../fiat`.
Base: `458649bba` here, `d0f9589d` there.
Design document: `prism_unstructured_implementation_plan.md`, with corrections in section 3 below.

This document records what was built, what the design document got wrong, and the
things that cost the most time. It is written for whoever picks this up next.

---

## 1. What works

An unstructured prism mesh — one produced by gmsh, with no extrusion, where each
prism may point in its own direction — loads, assembles and solves.

| Capability | State |
|---|---|
| CG Lagrange on prisms, any degree | works |
| Cell integrals (`dx`) | works |
| Strong Dirichlet conditions | works |
| Mixed function spaces, `aij` and `nest` | works, at 1, 2 and 3 ranks |
| Convergence at the theoretical order | works, 2.03 / 3.00 / 4.07 at degrees 1, 2, 3 |
| `locate_cell`, `Function.at`, `PointEvaluator` | works |
| Mixed-cell meshes | unchanged from base (deliberately) |
| VTK output, linear and higher-order | works |
| Checkpoint save and load | works |
| MATIS assembly (`mat_type="is"`) of a mixed space | **broken, pre-existing, not prism-specific** — see below |
| Facet integrals (`ds`) | Phase B, not yet done |
| `dS`, H(div)/H(curl) on prisms, mixed tet/prism meshes | out of scope |

**Read the mixed-space row as qualified, because it is.** Prism mixed spaces
work with `aij` and `nest`; the `mat_type="is"` (MATIS) path is broken. The
break is pre-existing and is not cell-type specific, but a reader who takes
"mixed function spaces: works" unqualified and then assembles a MATIS matrix
will hit it.

The reproducer, so the caveat is actionable:

```python
mesh = Mesh("prism_meshes/prism_slab.msh")
W = VectorFunctionSpace(mesh, "CG", 1) * FunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(W), TestFunction(W)
bcs = [DirichletBC(c, 0, (1, 2))
       for c in [W.sub(0).sub(j) for j in range(3)] + [W.sub(1)]]
assemble(inner(grad(u), grad(v))*dx, bcs=bcs, mat_type="is", sub_mat_type="is")
# IndexError: index 80 is out of bounds for axis 0 with size 80
```

Drop the boundary conditions and it assembles. The failure is at
`firedrake/functionspaceimpl.py:916`, in `FunctionSpace.local_to_global_map`,
where `bc.nodes * block_size + component` overruns the sub-block map — the same
line that fails `test_assemble_matis[True-mixed-is]` on a simplex mesh, so the
prism is incidental. `assemble.py:2100` indexes the mixed space and calls the
non-mixed method per block, which is why the `MixedFunctionSpace` method is not
the one that raises.

Separately, `MixedFunctionSpace.local_to_global_map` at
`functionspaceimpl.py:1213` raises `NotImplementedError("Not for mixed maps
right now sorry!")` for every mixed space on every cell. Neither defect is ours
to fix, but together they block a monolithic mixed preconditioner on a prism
mesh.

Two things found while adding the output and checkpoint support, both of which the
design document treated as small:

- **VTK was not a two-key table addition.** `vtk_lagrange_wedge_reorder` took
  `degree()` as a pair, which is what an extruded wedge reports, while a prism
  reports a scalar. Every prism above degree 1 raised
  `TypeError: 'int' object is not subscriptable`. `get_sup_element` also asked for
  `"DQ"`, which is registered on hypercubes only and raises on a prism; prisms now
  take `"DG"`, which builds `P_k(triangle) x P_k(interval)`.
- **Checkpoint LOAD was broken independently of any permutation question.**
  `DMPlexTopologyLoad` restores cones only, so PETSc infers cell types, and its
  default makes a 2-triangle/3-quadrilateral cell a `DM_POLYTOPE_TRI_PRISM_TENSOR`,
  which Firedrake rejects. `dmcommon.relabel_tensor_prisms` corrects that
  inference — a checkpoint carries the cones unchanged, so a loaded prism always
  has `TRI_PRISM` cone order and the tensor type is a wrong answer to a question
  the cones alone cannot settle. **The call must precede `labelsLoad`**, which
  removes the "celltype" label but leaves the `mesh->cellTypes` cache that
  `DMPlexGetCellType` reads first; a fix placed after it repairs the label and
  leaves the cache stale — silently on an optimised PETSc, as `PETSC_ERR_PLIB` on a
  debug build.

Test meshes, all tracked in `prism_meshes/`:

| File | Contents | Why it exists |
|---|---|---|
| `prism_reference.msh` | 1 cell at the FIAT reference coordinates | hand-written; gmsh will not place a single prism at chosen coordinates |
| `prism_slab.msh` | 52 cells, axes along +z, affine | the only mesh that can carry an exactness test |
| `prism_warped.msh` | 52 cells, non-affine, axes tilt | non-affine geometry |
| `prism_two_perpendicular.msh` | 2 cells whose axes are perpendicular | **the counterexample that disproved the first orientation fix** |
| `prism_order_r0/r1/r2.msh` | 64, 512, 4096 cells, non-affine, each level halves the element size | the convergence order test; a Delaunay base at half the target size does not halve the element size, so these use a transfinite base |

`prism_meshes/*.msh` is tracked via a `!` exception at `.gitignore:71`; the repository
otherwise ignores `*.msh`.

---

## 2. The two defects the design document did not predict

Both blocked every degree above CG1, and neither appears in the plan.

### 2.1 The global numbering assumed one dof count per dimension

`make_dofs_per_plex_entity`, `create_section` and `node_classes` took the dof count
from `entity_dofs[d][0]` — the first entity of dimension *d*. A prism has BOTH
quadrilateral and triangular faces at dimension 2, with different interior dof counts.

Measured before the fix: `prism_slab.msh` at CG2 reported `V.dim() = 403` with only
325 nodes referenced and 78 empty stiffness rows — exactly the 78 triangular facets.
Every solve above CG1 was singular.

Fixed by keying the dof count on the DMPlex polytope type of the point rather than on
its dimension stratum. Every cell type that existed before has one polytope type per
dimension and therefore takes a byte-identical fast path.

### 2.2 A prism quad face genuinely needs all 8 orientations

`o = (2**dim) * eo + io`. FIAT emitted only the 4 orientations with `eo = 0` for a
prism quadrilateral face, because a prism quad is (triangle edge) x (interval) and
those two factors are distinct. The mesh presents values of 4 and 6. The lookup
therefore read past the end of the permutation table — a genuine out-of-bounds read of
a `PetscInt[::1]` memoryview under `@cython.boundscheck(False)`, not a clean exception.

Fixed in `FIAT/orientation_utils.py::_make_axis_perms_tensorproduct` by deciding on the
SUB-ENTITIES rather than the component cells: a prism quad face is interval x interval
and admits the axis swap, while the prism cell and its triangular bases do not.

---

## 3. Where the design document is wrong

Full detail, with evidence, in the corrections file kept alongside this one. The
load-bearing errors:

1. **Section 8.6 asked the wrong question.** It framed the highest risk as "do
   neighbouring prisms agree on `eo`?" They do not always — and agreement was never
   the issue. The real defect was that the required orientation lies outside the range
   the element supplies.
2. **Section 8.5's axis-consistency validator should not be built.** It exists to
   reject exactly the meshes the section 8.7 fix handles correctly. Phase C is
   cancelled, not deferred.
3. **Section 4 omits the numbering layer entirely** (see 2.1).
4. **Section 12.5's Phase A exit criterion is unusable.** `prism_smoke_test.py` cannot
   gate this work: stage 4 needs Phase B, and stage 6 is structurally blind to
   numbering defects — its output is byte-identical either side of a change that moves
   `V.dim()` from 403 to 325. The real gate is
   `tests/firedrake/regression/test_prism.py`.

---

## 4. The decision most worth understanding

An earlier fix for 2.2 transposed the quadrilateral face's cone so that `eo` became 0,
keeping the existing 4-entry table. **On all three shipped test meshes it was a
complete fix: zero dof mismatches at CG2, CG3 and CG4.**

It was wrong. Every prism in those three meshes has its axis along `z`. Two prisms
whose axes are PERPENDICULAR can share a quadrilateral face, and then the two cells
disagree:

```
prism_two_perpendicular.msh, shared quad face 14
  (cell, shipped, transposed) = [(0, 1, 5), (1, 6, 2)]
  eo as shipped = [0, 1]   -> the neighbours disagree
  transposed    = [5, 2]   -> 5 is still outside the range 0..3
```

Had it shipped, it would have passed every test that existed and produced silently
wrong answers on any mesh with non-parallel prism axes.

The general lesson, which applies beyond this feature: a property measured across 105
cells of three meshes is not a structural law if all three meshes share a symmetry.

### Keep `prism_two_perpendicular.msh`, and keep it in the test matrix

That mesh earned its keep twice. It killed the cone-transposition approach before
anyone built it, and it is **the only mesh in the set that exercises extrinsic part
0 on a quadrilateral face** — the three original meshes present only `{4, 6}`,
i.e. extrinsic part 1 throughout.

The practical consequence: any future change to
`FIAT/orientation_utils.py::_make_axis_perms_tensorproduct` that reverts to
comparing the component cells instead of the sub-entities will **pass on all three
original meshes and fail only on that one**. If it is ever dropped from the matrix
as redundant, that regression becomes invisible again.

---

## 5. Two test weaknesses worth knowing about

Both were found by the task that wrote the tests, in its own tests, after the
implementation was already passing. They are recorded because both would have let
a real defect through.

**A sorted-list comparison cannot detect a collapse.** A dof-identity test compared
the two cells' shared-face nodes with `sorted(a) == sorted(b)`. The defect this
branch fixes gives the SAME global node to several dofs of a face — so a sorted
list containing a repeat still equals the other cell's sorted list containing the
same repeat. Both cells agree, and both are wrong. The test now asserts each cell's
face nodes are DISTINCT before comparing them. Without that, the CG3 collapse of
`[12, 12, 12, 12]` would have passed.

**A hand-written geometry map silently assumes degree 1.** A helper computed
physical dof positions from six vertex coordinates, which is the correct map only
for a degree-1 coordinate field. On a higher-order coordinate field it would have
computed wrong points and compared nonsense against nonsense. It now asserts the
coordinate cell node map has 6 columns.

---

## 6. Running the test suites here

This tree has a pre-existing PETSc at-exit deadlock:
`Py_FinalizeEx -> PetscGarbageCleanup -> PetscCommDuplicate -> MPI_Comm_dup`, a
collective that the ranks do not all reach. **It fires on success as well as on
failure**, so it is not a symptom of a failing test — the ranks print
`1 passed in 0.64s` and the parent then hangs.

What actually works:

- **One `pytest` invocation per suite.** A deadlock in one otherwise destroys the
  results of every other suite in the same invocation.
- **`--timeout-method=signal`, never `thread`.** `thread` cannot interrupt the
  blocking `waitpid`, so pytest-timeout escalates and kills the whole run.
- **Reap orphaned MPI processes between suites**
  (`pkill -9 -f "_PYTEST_MPI_CHILD_PROCESS"; pkill -9 -f prterun`).
- **Never run two MPI jobs at once.** It manufactures a symptom indistinguishable
  from a real hang, and cost roughly three hours here.
- **`sample <pid> 5` on two ranks before killing anything.** A `PetscGarbageCleanup`
  stack means the body already passed and only teardown is stuck. Symmetric 100% CPU
  proves nothing either way — OpenMPI busy-polls, so that is what both real work and
  a deadlock look like.
- **Count failures with `grep -c "^FAILED"`** on the `-rf` summary. Counting `F`
  characters matches the `F` in "False" in pytest-timeout's header, and produced two
  false reports here.
- **The deadlock is reproducible on a given test, but which test it hits varies
  between runs.** Both halves are measured, from different sources. Reproducible:
  the same parametrisation of `test_io_backward_compat_base_load` failed in both a
  full-suite run and an isolation run. Varies: `test_io_freeze_dist_perm_base[triangle_small]`
  was the failure in one suite run and `test_io_backward_compat_base_load` in
  another, and a base-versus-branch comparison hit different
  `test_covariance_operator` variants on each side.

  So the `output` suite has no stable expected-failure identity. Confirm by STACK,
  not by test name and not by count — a run with the same count and a different
  test failing looks identical to a clean match. Note that the count is not stable
  either: runs have produced both 318/1 and 317/2.
- **Drop `-p no:randomly` on `tests/firedrake/multigrid`.** That flag's ordering
  triggers a GC segfault at `pmg.py:163 destroy`. Running that suite one file at a
  time is order-independent and also works.
- **`os._exit(0)`** at the end of a diagnostic script skips finalisation and avoids
  the deadlock entirely.
- **`pytest -k` matches MARKER names, not just test names.** `-k "checkpoint and not
  parallel"` deselects every test carrying `@pytest.mark.parallel`, which on this
  branch meant all 202. An empty selection reports as a clean run, so this reads as
  success rather than as a mistake.
- **Never edit a tracked source file to take a diagnostic baseline while anything
  else may be running.** Disabling one call in `checkpointing.py` for twenty minutes
  produced 14 failures in another run, which were then blamed on a stale build. Do
  the experiment in a copy or a `git worktree`. A source edit is as disruptive as a
  job, and it is harder to see: no process list shows it.
- **Check the built extension is newer than `dmcommon.pyx` before trusting a
  failure.** A stale `.so` produced 14 spurious checkpoint failures here. `make ext`
  after every `.pyx` edit, and never rebuild while another process has the
  extension mapped — overwriting a mapped `.so` can corrupt the running process.

Seven pre-existing failure modes are catalogued in project memory under
`firedrake-preexisting-test-failures`. The branch base produces 17 failures in
`tests/firedrake/regression`; this branch produces 15.

Reference counts, confirmed three times independently:

```
extrusion   581 passed,  0 failed
submesh     387 passed,  0 failed
slate       550 passed,  1 xfailed, 0 failed
output      318 passed,  1 failed    (the teardown deadlock)
multigrid   396 passed, 63 skipped, 1 failed
```

---

## 7. Tests that passed against broken code

Five tests on this branch were found to pass whether the implementation was right or
wrong. Each was caught by asking "would this fail if the code were wrong?" rather
than "does this pass?". They are listed because the pattern recurs, not to
criticise anyone — three were found by the people who wrote them.

**A checkpoint round trip cannot verify a permutation.** A round trip applies the
permutation table in both directions, so a wrong table cancels itself. Measured:
with a deliberately corrupted `perm`, every checkpoint round-trip test still passed,
while the one-directional layout checks failed 8 of 8. Verify a permutation against
an independent oracle — here, the PETSc DG coordinate array against the plex
transitive closure — not by writing and reading it back.

**A sorted-list comparison cannot detect a collapse.** Comparing two cells' shared
face nodes with `sorted(a) == sorted(b)` misses the exact defect it was written for:
a face reading the wrong permutation block gives the SAME global node to several
dofs, and a sorted list with a repeat still equals the other cell's sorted list with
the same repeat. Assert the nodes are DISTINCT before comparing.

**`"DG"` and `"DQ"` are the same element on a tensor-product cell.**
`canonical_element_description` rewrites one to the other, so a guard asserting the
family selection was unchanged could not fail. No element-level guard can
discriminate there; assert the branch condition instead.

**A linear probe is invariant under many wrong permutations.** Reading back a linear
function verifies far less than it appears to. Use a geometric oracle — map each
node's reference position through the cell's own degree-1 map — or check a
polynomial of high enough degree.

**Mirror permutations pass symmetric checks.** `[0, 2, 4, 1, 3, 5]` and
`[0, 4, 2, 1, 5, 3]` differ only in handedness, and every VTK test passed under both
until an orientation assertion was added comparing the sign of the FIAT and VTK
frame determinants.

The cheap general defence, used throughout the later tasks: **mutate the
implementation deliberately and confirm the test fails.** It costs one rebuild and
it is the only thing that distinguishes a test from a ritual. One hexahedron control
added this way caught a bug in its own test rather than in the code — which is also
the argument for keeping controls on cell types the change should not affect.
