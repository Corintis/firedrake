# Unstructured prisms in Firedrake — implementation handover

Branch: `prism-unstructured` (this repository) and `prism-unstructured` in `../fiat`.
Base: `458649bba` here, `d0f9589d` there.
Design document: `prism_unstructured_implementation_plan.md`, with corrections in section 3 below.
That file is NOT in the repository. It is a local, untracked file. The corrections in
`plan-corrections.md` are complete without it.

This document records what was built, what the design document got wrong, and the
things that cost the most time. It is written for whoever picks this up next.

Phase C (interior facet integrals, `dS`) has its own plan,
`phaseC_interior_facets_plan.md`. That file is also local and untracked. Section 8
of this document and section 7 of `plan-corrections.md` are complete without it.
The Phase C commits are `6873b3278..a3e55ab08` here and `08064fb5..a6979469` in
`../fiat`.

---

## 1. What works

An unstructured prism mesh — one produced by gmsh, with no extrusion, where each
prism may point in its own direction — loads, assembles and solves.

| Capability | State |
|---|---|
| CG Lagrange on prisms, any degree | works |
| DG and Real on prisms | works |
| Other element families (for example "HDiv Trace", "Bernstein") | `NotImplementedError` from FInAT `convert_finiteelement`, which names the family |
| Cell integrals (`dx`) | works |
| Strong Dirichlet conditions | works |
| Mixed function spaces, `aij` and `nest` | works, at 1, 2 and 3 ranks |
| Convergence at the theoretical order | works, 2.03 / 3.00 / 4.07 at degrees 1, 2, 3 |
| `locate_cell`, `Function.at`, `PointEvaluator` | works |
| Mixed-cell meshes | unchanged from base, with one exception: a mixed mesh that holds a pyramid or a `TRI_PRISM_TENSOR` cell now raises `NotImplementedError` (see 2.1) |
| VTK output, linear and higher-order | works in serial; parallel output is not tested |
| Checkpoint save and load | works |
| MATIS assembly (`mat_type="is"`) of a mixed space | **broken, pre-existing, not prism-specific** — see below |
| Exterior facet integrals (`ds`), with markers, "otherwise", Robin and Neumann conditions | works, at 1, 2 and 3 ranks (Phase B) |
| Interior facet integrals (`dS`, `dS(tag)`, `dS((a, b))`, and `dS + dS(tag)`, which gives an "otherwise" kernel) | works in serial and at 2 and 3 ranks (Phase C, section 8). A marker can hold triangles, quadrilaterals or both |
| `jump`, `avg`, `FacetNormal('±')`, `CellVolume('±')`, `CellDiameter('±')`, `FacetArea` in a `dS` integrand | works |
| DG methods on `dS`, for example SIPG | works; DG1-3 converge at the theoretical order on non-affine, scrambled meshes |
| Weak interface conditions on `dS(tag)`: a surface source, a contact conductance | works; exact on each facet shape |
| `dS` on a facet submesh, `Submesh(mesh, 2, tag)` (its interior edges) | works (C7), in serial and at 2 and 3 ranks. Before C7 it gave a wrong answer with no error. In parallel, a quadrilateral submesh needs a `RIDGE` or `VERTEX` overlap of the parent (section 8.3) |
| An integral over a prism mesh and its facet submesh: `dS(tag)` or `ds(tag)` on the prism mesh with `dx` on the submesh, in either direction (C8) | works in serial and at 2 and 3 ranks; a Lagrange multiplier on an interior surface gives the exact solution |
| Interpolation from a submesh to its parent in parallel | works for a codim-1 submesh (tested at 1, 2 and 3 ranks). For a codim-0 submesh a CG space can get wrong values in parallel. This is a pre-existing Firedrake defect (issue 4483, section 8.3) |
| `ds` or `dS` with a `QuadratureRule` object in the metadata or the form compiler parameters | `NotImplementedError` from `_split_facet_integrals_by_shape`; use `quadrature_degree` or a scheme name. A C8 coupling that keeps one facet shape accepts the object |
| `dS` with a triangle rule that has no point permutation (`"quadrature_rule": "canonical"` at degree 2 or more, or the default rule above degree 50) | `NotImplementedError` from `set_quad_rule`, which names the integral type, the scheme and the degree |
| `ds(subdomain_data=...)`, `dS(subdomain_data=...)` | the same `NotImplementedError` as on the other meshes |
| Slate facet integrals (`ds` and `dS`) | `NotImplementedError`, which names prisms, the integral type and the measure (Slate `kernel_builder.py`) |
| `PatchPC` with `ds` or `dS` | `NotImplementedError("Only for cell, interior facet, or exterior facet integrals")`, inside a PETSc error |
| `MinFacetEdgeLength` on a prism | `Exception: Cell type prism not supported.` (not a Phase C change; no test) |
| A facet integral on a prism mesh coupled with `dx` of a prism submesh (codim 0) | `NotImplementedError` ("more than one shape"). Integrate on the submesh instead (section 8.3) |
| A cross-mesh measure on the parent with no tag | **a wrong answer with no error**, pre-existing on all cells (section 8.3) |
| A mixed-degree prism element, `TensorProductElement(triangle element, interval element)`, for example P2 x P1 | works on an axis-consistent mesh, at 1, 2 and 3 ranks (section 9). The design document (its scope table and section 6.2) made separate base and axis degrees "analysis only" |
| A mixed-degree element on a mesh that is not axis-consistent | `NotImplementedError` on all ranks (section 9) |
| H(div)/H(curl) on prisms, mixed tet/prism meshes | out of scope |

**Read the mixed-space row as qualified, because it is.** Prism mixed spaces
work with `aij` and `nest`; the `mat_type="is"` (MATIS) path is broken. The
break is pre-existing and is not cell-type specific, but a reader who takes
"mixed function spaces: works" unqualified and then assembles a MATIS matrix
will hit it.

The reproducer, so the caveat is actionable:

```python
mesh = Mesh("tests/firedrake/meshes/prism/prism_slab.msh")
W = VectorFunctionSpace(mesh, "CG", 1) * FunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(W), TestFunction(W)
bcs = [DirichletBC(c, 0, (1, 2))
       for c in [W.sub(0).sub(j) for j in range(3)] + [W.sub(1)]]
assemble(inner(grad(u), grad(v))*dx, bcs=bcs, mat_type="is", sub_mat_type="is")
# IndexError: index 80 is out of bounds for axis 0 with size 80
```

Drop the boundary conditions and it assembles. The failure is in
`FunctionSpace.local_to_global_map` (`firedrake/functionspaceimpl.py`), at the
line `indices[nodes] = -1`, where `bc.nodes * block_size + component` overruns
the sub-block map — the same line that fails `test_assemble_matis[True-mixed-is]`
on a simplex mesh, so the prism is incidental. `ParloopBuilder.collect_lgmaps`
(`firedrake/assemble.py`) indexes the mixed space and calls the non-mixed method
per block, which is why the `MixedFunctionSpace` method is not the one that
raises.

Separately, `MixedFunctionSpace.local_to_global_map` raises
`NotImplementedError("Not for mixed maps right now sorry!")` for every mixed
space on every cell. Neither defect is ours
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
  which Firedrake rejects. `dmcommon._relabel_tensor_prisms_from_checkpoint`
  corrects that inference — a checkpoint carries the cones unchanged, so a loaded
  prism always has `TRI_PRISM` cone order and the tensor type is a wrong answer to
  a question the cones alone cannot settle. The name carries its precondition:
  call it only on a dm a Firedrake checkpoint supplied.

  **The call must precede `labelsLoad`.** `labelsLoad` restores the saved
  "celltype" label, whose tensor stratum is empty, so a call placed after it does
  nothing at all, and the stale `mesh->cellTypes` cache that `DMPlexGetCellType`
  reads first survives. Measured by moving the call, on this optimised PETSc
  build: PETSc returns the stale tensor type with no error, and the failure
  surfaces only because Firedrake's `_ufl_cell` guard then rejects the tensor
  prism with `NotImplementedError`. So the misplacement is loud in practice, but
  it is Firedrake's guard that makes it loud, not PETSc — remove that guard and
  the same misplacement would be silent.

  The relabel costs one `DMPlexComputeCellTypes` pass over the chart on every
  checkpoint load of every mesh type. No guard can avoid it, because testing for
  the label computes it.

Test meshes, all tracked in `tests/firedrake/meshes/prism/`:

| File | Contents | Why it exists |
|---|---|---|
| `prism_reference.msh` | 1 cell at the FIAT reference coordinates | hand-written; gmsh will not place a single prism at chosen coordinates |
| `prism_reference_marked.msh` | the same cell, each facet in its own physical group | the `ds` tests measure each facet separately |
| `prism_slab.msh` | 52 cells, axes along +z, affine | the exactness tests at every degree (`plan-corrections.md` section 4 gives the degree rule for the warped mesh) |
| `prism_slab_mixed_marker.msh` | the `prism_slab.msh` cells; marker 4 holds triangular and quadrilateral facets | the `ds` tests of a marker with both facet shapes, and of "otherwise" |
| `prism_warped.msh` | 52 cells, non-affine, axes tilt | non-affine geometry |
| `prism_two_perpendicular.msh` | 2 cells whose axes are perpendicular | **the counterexample that disproved the first orientation fix** |
| `prism_order_r0/r1/r2.msh` | 64, 512, 4096 cells, non-affine, each level halves the element size | the convergence order test; a Delaunay base at half the target size does not halve the element size, so these use a transfinite base |
| `prism_interior_marked.msh` | the box `[-0.5, 0.5]^2 x [0, 1]`; marker 10 = the plane `z = 0.5` (136 triangles), marker 20 = the plane `x = 0` (42 quadrilaterals), marker 30 = markers 10 and 20 | the `dS` tests (Phase C) and the submesh tests (C7, C8) |
| `prism_interior_marked_scrambled.msh`, `prism_warped_scrambled.msh`, `prism_order_r0/r1/r2_scrambled.msh` | the same geometry as the source mesh; each prism's node list is changed by one of the 6 orientation-preserving prism symmetries | **the only meshes that show a missing triangle point permutation** (section 8.4) |

The repository ignores `*.msh`. These files are tracked with `git add -f`, as are
the other meshes in `tests/firedrake/meshes/`. `make_prism_mesh.py` in the same
directory writes all of them. It writes the two `prism_reference*.msh` files from
hand-written text, and the others through gmsh. `scramble` writes the scrambled
copies, with the fixed seeds in `SCRAMBLED`. A missing mesh makes the prism tests
fail, not skip.

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
dimension and therefore takes a byte-identical fast path, EXCEPT a mixed-cell mesh.

A mixed-cell mesh, for example `tests/firedrake/meshes/mixed_cell_unit_square.msh`,
has triangles and quadrilaterals at dimension 2. So `_plex_polytope_types` splits
that dimension, and `create_section`, `get_entity_classes_per_stratum` and
`make_dofs_per_plex_entity` take the per-stratum path. Its section does not change,
because the element gives each cell stratum the same dof count. The submesh
mixed-cell tests pass (Step 1 of the end-of-branch run: `tests/firedrake/submesh`,
387 passed, 0 failed). A mixed mesh that holds a pyramid or a `TRI_PRISM_TENSOR`
cell now raises `NotImplementedError`.

On the per-stratum path, `make_dofs_per_plex_entity` raises `NotImplementedError`
when an element gives the entities of one polytope type different dof counts. The
prism edges are the exception: since `3d4a93574` they take one count for each
edge role, axis or base (section 9). This
check is new for a dimension that has one polytope type. Before it, the count of
entity 0 went to the whole stratum with no error, and "HDiv Trace" 0 on a prism gave
a space of dimension 0.

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
   reject exactly the meshes the section 8.7 fix handles correctly. The validator
   (the "Phase C" of the design document) is cancelled, not deferred. The name
   "Phase C" now means other work: interior facet integrals (`dS`) on prisms,
   which the user needs. `phaseC_interior_facets_plan.md` gives that plan.
   Section 8 records what Phase C did.
3. **Section 4 omits the numbering layer entirely** (see 2.1).
4. **Section 12.5's Phase A exit criterion is unusable.** `prism_smoke_test.py` could
   not gate this work: stage 4 needed Phase B, and stage 6 is structurally blind to
   numbering defects — its output is byte-identical either side of a change that moves
   `V.dim()` from 403 to 325. The real gate is
   `tests/firedrake/regression/test_prism.py`. The script is removed from the
   repository, because pytest collected it and stopped with INTERNALERROR. The last
   version is `git show c9a0f9fe7:prism_smoke_test.py`.

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
  (`pkill -9 -f "_PYTEST_MPI_CHILD_PROCESS"`). Kill a `prterun` by the name of
  its script only. A global `pkill -f prterun` also kills the MPI jobs of other
  users or agents.
- **A parallel test that passes can still stop the run.** At 2 and 3 ranks,
  `test_prism_boundary_condition_converges_at_the_expected_order` passed and then
  hung in the teardown. It is now serial only. To test a body in parallel, call it
  from a direct `mpiexec` driver that ends with `os._exit(0)`.
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

These failure modes are pre-existing. Each one also occurs at the base commit, or
does not depend on the cell type:

1. The PETSc at-exit teardown deadlock (above). It fires on a passing test.
2. A garbage collection re-entrancy segfault in `destroy` of `fdm.py` or
   `facet_split.py`. It hits a different test on each run. `test_fdm.py` also has
   5 `KeyError` failures.
3. A tinyasm segfault at `preconditioners/asm.py`: `test_linesmoother.py` and
   `test_star_pc.py`.
4. The `EnrichedElement` dual basis: `multigrid/test_hiptmair.py::test_pmg_hiptmair_hcurl`.
5. A missing optional dependency: `test_netgen.py` (`netgen`).
6. A `petsctools` API mismatch: `multigrid/test_adaptive_multigrid.py` and
   `multigrid/test_embedded_transfer.py`.
7. A tolerance: `test_interior_elements.py::test_vanish_on_bdy` (1.79e-14 against 1e-14).
8. MATIS mixed assembly: `test_assemble.py::test_assemble_matis[*-mixed*-is]`.
9. A `RecursionError` in `tests/tsfc/test_dual_evaluation.py` (5 cases), identical
   with FIAT at its base commit.

The end-of-branch Step 1 run was stopped by the user before it finished. Its
partial results, at commit f96d72298, before the fix wave:

```
prism files   513 passed, 6 failed   (each failure: the sampled teardown deadlock)
tsfc          369 passed, 5 failed   (the count of test_dual_evaluation.py; not checked at base)
extrusion     581 passed, 0 failed
submesh       387 passed, 0 failed
```

The output, multigrid, slate and regression suites were not run. One full run
after Phase C replaces these numbers. That run is **pending**.

One partial result comes before it. The `tests/firedrake/submesh` suite ran in
parallel on a snapshot of `a3e55ab08` (after C8): 376 passed and 3 `F`. Each `F`
is a sampled at-exit teardown deadlock after the body passed. The run stopped at
test 380, a 6-rank test that did live work on a loaded machine. So about 7 tests
did not run. The full run covers them.

The earlier counts (17 regression failures at
the base, 15 on the branch) come from before Tasks 7 to 9 and are not current.

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

---

## 8. Phase C: interior facet integrals (`dS`)

Phase C makes `dS` work on an unstructured prism mesh. The user writes one `dS`,
as on any other mesh. The user needs it for weak conditions on interior surfaces
and for DG methods.

Commits here: `67916c527` (TSFC), `f3a11f158` (Firedrake), `25d030000` (the Slate
message), `bd4780c6a` and `b52574371` (meshes and tests), `4fc173324` (C7),
`8b6155581` (the partition-boundary test), `a3e55ab08` (C8). Commits in `../fiat`:
`0029f52a` (the triangle map), `cc839a77` (its fix), `a6979469` (a regression test).

### 8.1 How the two sides of a facet agree

A `dS` kernel maps the facet quadrature points to physical space from each of the
two cells. The two lists of points must be in the same order. Firedrake gets this
in three ways:

| Cell | Method |
|---|---|
| Simplex | `dmcommon.closure_ordering` sorts the vertices by global number, so both cells list a shared facet in one order |
| Quadrilateral (2D) | `dmcommon.quadrilateral_closure_ordering` orients the cells (Homolya-McRae), so each shared edge has one direction |
| Hexahedron | the closure keeps the PETSc cone order; each side reads its facet orientation from `local_facet_orientation_dat` and permutes its points into the canonical order |

`MeshTopology.cell_closure` sends a prism to `dmcommon.create_cell_closure`, as a
hexahedron. So a prism uses the hexahedron method on both facet shapes. The
prototype measured the naive order (point `k` against point `k`): it fails on 42
of the 66 interior quadrilaterals of `prism_slab.msh`, and on 83 % of the
triangles of a scrambled mesh.

### 8.2 The design

**Two interior shape types.** `interior_facet_tri` (FIAT facets `[3, 4]`, measure
name `dS_tri`) and `interior_facet_quad` (FIAT facets `[0, 1, 2]`, `dS_quad`).
`_split_facet_integrals_by_shape` (`firedrake/tsfc_interface.py`) replaces each
prism `dS` with one integral of each type. Each integral keeps its subdomain id.
`_Facets._shape_subset` (`firedrake/mesh.py`) intersects the marker subset with
the facets of the shape. Both names start with `"interior_facet"`, so the many
`startswith("interior_facet")` tests in UFL and TSFC need no change. A plain
`interior_facet` integral on a prism still raises `NotImplementedError` in
`lower_integral_type`. Only a direct TSFC caller can send one.

**A separate dict.** The two types are in `interior_shape_facet_types`, in
`tsfc/kernel_interface/common.py`, beside `shape_facet_types`. The same module
registers them with `ufl.measure.register_integral_type` and gives them the value
`"+"` in `ufl.algorithms.apply_restrictions.default_restriction_map`. A missing
key there is a bare `KeyError`.

**Warning.** Do not add the interior types to `shape_facet_types`. The exterior
facet paths read that dict: `exterior_facet_types` in `firedrake_loopy.py`,
`_as_parloop_arg_exterior_facet` and `MeshTopology.measure_set`. The mutation
makes 54 of 73 test cases fail, but a change that is only a little different can
give wrong answers with no error.

**The position contract (rank 2).** This is the point where Phase C can fail with
no error:

| Data | Indexed by | Holds |
|---|---|---|
| `_Facets.shape_local_facet_dat` | facet, side | the POSITION of the FIAT facet in the entity list of `lower_integral_type` for its shape. The `dS` kernel selects its tables by position |
| `_Facets.local_facet_dat` | facet, side | the FIAT facet number. Unchanged |
| `_Facets.local_facet_orientation_dat` | facet, side | the orientation of the FIAT facet in that cell. Unchanged; the kernel uses it directly |

`_Facets._facet_shape_groups` computes the positions for both kinds of facets.
The array has the shape `(nfacets, 1)` for exterior facets and `(nfacets, 2)` for
interior facets. A side whose cell is absent (`facet_cell == -1`, an outer halo
facet) gets position 0 with no check, because no iteration set holds that facet.
FIAT facets 3 and 4 are positions 0 and 1. The quadrilateral positions are equal
to their FIAT numbers. So a FIAT number in place of the position ("trap A" of
Phase B) is wrong on the triangles only.

**The canonical point order, on the prism interior types only.**
`ContextBase.use_canonical_quadrature_point_ordering` (`tsfc/fem.py`) returns
`True` for the two interior shape types on a `UFCPrism`. It stays off for the
exterior shape types of one mesh, because a `ds` kernel reads one side only. So
the Phase B prism `ds` kernels are byte-identical. C8 adds one case: an exterior
shape type on a prism in a kernel with more than one domain (see the C8 map
changes below).

**The lazy FInAT triangle map.** The permutation needs a point map for each facet
orientation. A Gauss-Legendre line rule has one. A triangle rule does not: its
stored map is `(None,)`. In `../fiat/finat/quadrature.py`,
`AbstractQuadratureRule.intrinsic_orientation_permutation_map_tuple` is now a
`cached_property`. For a triangle rule with the stored map `(None,)`, it calls
`_triangle_intrinsic_orientation_permutation_map`. Row `o` of the map uses entry
`o` of `sorted(itertools.permutations(range(3)))` on the barycentric coordinates
of the points. This is the convention of
`FIAT.orientation_utils.make_entity_permutations_simplex`, and a FInAT test checks
it against that function. The map checks the points AND the weights (tolerance
1e-12), and raises `ValueError` if the rule is not symmetric.

The map is lazy because of the cache keys. `QuadratureRule.__repr__` includes the
stored map, and the repr is the hash and the cache key of the rule. A map that the
constructor stores changes the key of every triangle and tetrahedron facet rule in
every Firedrake form. A lazy property changes nothing for a kernel that does not
ask for it. Measured: the repr and the hash of 144 triangle and tetrahedron rules
do not change, and the generated code of the non-prism kernels is byte-identical.

When a triangle rule has no map, `set_quad_rule`
(`tsfc/kernel_interface/common.py`) raises `NotImplementedError`, which names the
integral type, the scheme and the degree. `_make_quad_multiindex_permuted`
(`tsfc/fem.py`) catches the `ValueError` too, as a second guard for a direct
caller. Without the error, the naive order gives a wrong answer with no error.

**The C7 closure rule.** `MeshTopology.cell_closure`: a codim-1 submesh of a prism
mesh does not inherit the closure of its parent. The prism closure keeps the cone
order, so the two cells of a submesh facet saw the points in different orders.
Triangle cells now take `dmcommon.closure_ordering` and quadrilateral cells take
`dmcommon.quadrilateral_closure_ordering`. A codim-0 submesh still inherits.
Measured: `|x('+') - x('-')|**2 * dS` on `Submesh(mesh, 2, 10)` fell from 8.4e-2 to
3.8e-32. The interpolation between the parent and the submesh is exact in both
directions, before and after the change.

**The C8 map changes.** C8 couples the facets of a prism mesh with the cells of a
facet submesh:

1. `_split_facet_integrals_by_shape` also reads `extra_domain_integral_type_map`.
   The cells of a triangle or a quadrilateral submesh fix the facet shape, so the
   split keeps one shape type, and gives it to each prism domain of the integral.
   A coupling with another integral type, for example a prism `ds` with `dx` of a
   prism submesh, raises `NotImplementedError`.
2. `MeshTopology.submesh_map_child_parent` and `trans_mesh_entity_map`
   (`firedrake/mesh.py`) accept the shape types. A facet type on a prism target
   becomes the shape type of the source facets, or of the submesh cells. If no
   rank holds a facet of the shape in the subdomain, the map uses the facets of
   the same kind, because the kernel then iterates over an empty set.
3. `_as_parloop_arg_exterior_facet` and `_as_parloop_arg_interior_facet`
   (`firedrake/assemble.py`) take the integral type of the other mesh from
   `trans_mesh_entity_map`. On a prism mesh they pass `shape_local_facet_dat`.
4. `use_canonical_quadrature_point_ordering` is on for an exterior shape type of
   a prism when the kernel has more than one domain. The submesh cell permutes its
   points by its own orientation, with the triangle map above.

Measured: the generated code of 17 kernels (non-prism cross-mesh kernels, and
prism `ds`, `dS` and `dx` on one mesh) is byte-identical before and after C8.

### 8.3 The limits

Each limit is an explicit error, except the interpolation from a submesh to
its parent and the last one.

- **Slate** with a `ds` or `dS` on a prism: `NotImplementedError` from
  `firedrake/slate/slac/kernel_builder.py`. The message names prisms, the integral
  type and the measure. Slate gives the kernel the FIAT facet number of its own
  loop, so the shape types in Slate would give wrong answers on the triangles.
- **PatchPC** with a `ds` or `dS` on a prism:
  `NotImplementedError("Only for cell, interior facet, or exterior facet integrals")`
  from `firedrake/preconditioners/patch.py`, inside a `PETSc.Error`.
- **A `QuadratureRule` object** in the metadata or the parameters of a prism `ds`
  or `dS`: `NotImplementedError` from `_split_facet_integrals_by_shape`. A rule is
  for one facet shape only. Use `quadrature_degree` or a scheme name. A C8
  coupling that keeps one shape accepts the object.
- **A triangle rule that is not symmetric**: the metadata key
  `"quadrature_rule": "canonical"` at degree 2 or more, or the default rule above
  degree 50 (collapsed Gauss). `NotImplementedError`, as in section 8.2. The
  default rule is symmetric at degrees 0 to 50, `"KMV"` at degrees 1 to 6.
  **Caution:** the key is `"quadrature_rule"`. A `"scheme"` key has no effect: the
  default rule is used with no error.
- **`MinFacetEdgeLength`** on a prism: `Exception: Cell type prism not supported.`
  Phase C did not change it, and no test covers it.
- **`dS(subdomain_data=...)`**: the same `NotImplementedError` as on other meshes.
  `_subdomain_data_integral_type` (`firedrake/assemble.py`) maps the shape types
  back to `"interior_facet"`. Without that map the data is ignored with no error.
- **A prism facet integral coupled with `dx` of a codim-0 prism submesh**:
  `NotImplementedError` ("more than one shape"). Integrate on the submesh
  instead, for example
  `Measure("ds", sub, intersect_measures=(Measure("ds", mesh),))`.
- **A quadrilateral submesh in parallel with the default `FACET` overlap**:
  `NotImplementedError` on every rank, from `_check_quadrilateral_submesh_halo`
  (`firedrake/mesh.py`, commit `72db2f225`). Load the parent with a `RIDGE` (or
  `VERTEX`) overlap, for example
  `distribution_parameters={"overlap_type": (DistributedMeshOverlapType.RIDGE, 1)}`.
  With `RIDGE` the submesh is correct. This is a generic Firedrake limit, not a
  prism defect. The quadrilateral orientation algorithm exchanges one value for
  each edge between an owned cell and a halo cell, so the two ranks must list the
  same edges. With a `FACET` overlap, two quadrilaterals on the plane `x = 0` share
  an edge, but their prisms share no facet. So one rank sees the halo cell of the
  other rank, and the other rank does not. Before the check, `Submesh(mesh, 2, 20)`
  waited forever at 2 and 3 ranks in `dmcommon.quadrilateral_facet_orientations`.
  The hexahedron mesh of `test_submesh_facet_corner_case_1` with a `FACET` overlap
  waits at the same place on main (`458649bba`). The `Submesh` docstring already
  asks for a `VERTEX` or `RIDGE` overlap for a codim-1 submesh. A triangle submesh
  does not need the exchange, so it works with the default overlap.
- **The interpolation from a submesh to its parent in parallel**: a CG space can
  get wrong values (3.8e-4 in place of 0 for a codim-0 prism submesh at 3 ranks).
  The value is the same before Phase C. This is Firedrake issue 4483. The test of
  the codim-0 case stays serial (section 8.5). The codim-1 case is correct in
  parallel.
- **MATIS mixed assembly**: broken, pre-existing, on all cells (section 1).
- **An untagged cross-mesh measure on the parent: a silent wrong answer.**
  `dS(parent, intersect_measures=(dx(sub),))` with no tag reads the map value -1
  for each parent facet that the submesh does not have. On a tetrahedron mesh it
  gave 7.99 and 11.49 in two runs of the same form. On a prism codim-0 case it
  gave a bus error. This defect is pre-existing and is on all cells. **Always give
  the measure on the parent the tag of the submesh**, for example `dS(10)`.

A marker that holds one facet shape still runs the kernel of the other shape over
an empty subset. The result is correct. Phase B accepts the same small cost.

### 8.4 Lessons for the tests (plan 7.4)

The prototype had four mutation switches. Each one removes one part of the design.
The test tables of `.superpowers/sdd/phaseC-tests-report.md` record which tests
each mutation fails. These are the lessons:

1. **An aligned mesh hides the triangle permutation.** In each measured gmsh mesh
   whose prism axes are along `z`, both sides of every triangle give `o = 2`, so
   the naive order is correct by chance. With the triangle permutation removed, every
   check on `prism_interior_marked.msh` passes. The scrambled meshes are the only
   meshes that show the defect. `test_prism_dS_scrambled_mesh_presents_every_orientation`
   asserts that the `'-'` sides of a scrambled mesh have all 6 triangle
   orientations and all 4 quadrilateral `io` values. Without it, a change in the
   gmsh reader could align the meshes again, and each orientation test would pass
   with no permutation. This is the lesson of section 4 again.
2. **A value that is constant along a facet cannot find a point order error.**
   Areas, the normals of flat facets, `FacetArea` and DG0 data do not change when
   the points change order.
3. **A form that reads one side only cannot find it either.** A surface source
   `v('+') * dS(tag)` permutes the points and the weights of one side together.
   Only a product of a `'+'` value and a `'-'` value finds a mismatch.
4. **A field that is constant along the facets of one shape hides that shape.**
   The first contact conductance test used `u = u(z)` on marker 10, and each
   triangle of that marker is a plane `z = const`. It passed with no triangle
   permutation. `test_prism_dS_contact_conductance_is_exact` now adds a linear
   term that changes along the plane.
5. **Use `|x('+') - x('-')|**2 * dS` on a scrambled mesh.** It finds every
   orientation mutation, for one assembly. It does not find trap A, because a
   wrong FIAT facet selects the same wrong facet on both sides. The area test
   (`test_prism_dS_measures_each_marked_surface`) finds trap A.
6. **Make every parallel test symmetric in `'+'` and `'-'`.** The `'+'` cell of a
   partition-boundary facet can change with the number of ranks. Use `jump`,
   `avg` or `|x('+') - x('-')|**2`, not a bare `u('+')` against a fixed number.
7. **The default partitioner puts almost no triangle on a partition boundary.** At
   2 and 3 ranks it gave 0 to 2 such triangles. So the 12 parallel `dS` tests did
   not see a triangle whose two cells are on different ranks.
   `test_prism_dS_on_a_partition_boundary` (`parallel([2, 3])`) uses a shell
   partition that cuts along the marked planes. Its guard asserts the counts: all
   136 triangles of marker 10 on the partition boundary, and 21 quadrilaterals of
   marker 20 at 3 ranks (0 at 2 ranks). So a change of the partition cannot make
   the test pass with no shared facet. The test does its collectives before its
   first assert.
8. **Some tests cost too much for the suite.** The SIPG DG3 solve on
   `prism_order_r2_scrambled.msh` took 454 s and 10 GB. So
   `test_prism_dS_sipg_converges_at_the_expected_order` uses `r0` and `r1` only at
   degree 3. Degrees 1 and 2 use all three meshes. The test is serial only.
9. **Test a shared property through each caller.** `0029f52a` broke every
   hexahedron facet kernel and every prism `interior_facet_quad` kernel, and the
   FInAT suite passed. See `plan-corrections.md` section 7.1.

The plan gives, for each mutation, the tests that must fail. On the real code each
of these tests fails (in-memory patches, one fresh kernel cache for each run).

### 8.5 The parallel evidence

The parallel evidence comes from direct `mpiexec` drivers, not from pytest
(section 6). The code was a `git archive` snapshot, with fiat `a6979469`. The
`dS` rows used `b52574371` (the code of C7, before C8). The submesh rows used
`72db2f225` (C8 and the halo check of section 8.3).

| Evidence | Scope | Result |
|---|---|---|
| The 12 `parallel([1, 2, 3])` tests of the `dS` section of `test_prism.py`, plus 4 bodies of `test_prism_interior_facets_core.py` | 57 cases | pass at 2 and at 3 ranks |
| Rank independence: 6 meshes, the default partitioner at 1, 2 and 3 ranks, a shell partition at 2 and 3 ranks | the facet number and the orientation of each side, the vertex order of each cell, the areas, the point and jump checks, and a DG2 matrix and solve | the facet data is identical at 1, 2 and 3 ranks; the largest relative difference of a functional is 7.9e-13 |
| Mutations at 3 ranks | the orientation of `'-'` replaced by that of `'+'`; the triangle permutation removed | 25 of 57 cases fail; the functionals change by about 1e-2 |
| `test_prism_dS_on_a_partition_boundary` (`8b6155581`) | degrees 2 and 3 | pass at 2 and at 3 ranks; fails at 3 ranks with the triangle permutation removed |
| The 13 `parallel([1, 2, 3])` functions of `test_prism_submesh.py` (C7 and C8) | 62 cases, the default and the shell partition | pass at 1, 2 and 3 ranks with both partitions |
| Rank independence of C7 and C8 | 138 values for each mesh: the submesh `dS` and `ds` checks, the interpolation errors, the cross-mesh integrals and matrices, the Lagrange multiplier solution | the largest relative difference is 1.1e-13; the values that are exactly 0 stay below 1.4e-12 |
| A mutation at 3 ranks, shell partition | no point permutation on the submesh cell | 18 of 62 cases fail (15 of 62 in serial) |
| The `tests/firedrake/submesh` suite at `a3e55ab08` | the run stopped at test 380; about 7 tests did not run | 376 passed, 3 `F`; each `F` is a sampled teardown deadlock after the body passed (section 6) |

Not covered in parallel: `test_prism_dS_sipg_converges_at_the_expected_order`
(serial by design), and the error tests (serial; an error does not depend on the
partition). `test_prism_interior_facets_core.py` has no parallel mark.

`test_prism_submesh_interpolate_submesh_to_parent` has the mark
`parallel([1, 2, 3])` since `20d34d044`. Its asserts are global now. It passes at
1, 2 and 3 ranks with the default and the shell partitions. The error on the
surface dofs is at most 8.9e-16 in all parallel runs.

One function of `test_prism_submesh.py` stays serial,
`test_prism_codim0_submesh_interpolate`. At 3 ranks with the shell partition
  the value is wrong (3.8e-4, exact 0). The value is the same on `6873b3278`,
  before Phase C. So it is not a Phase C defect (Firedrake issue 4483).

### 8.6 How to run the Phase C tests

**Caution.** Use a new `FIREDRAKE_TSFC_KERNEL_CACHE_DIR` and `PYOP2_CACHE_DIR` for
a check whose result depends on the generated code. A cached kernel can hide a
change.

Serial. Run one `pytest` command for each file:

```
source ../venv-firedrake/bin/activate
python -m pytest -m "parallel[1] or not parallel" --timeout-method=signal \
    tests/firedrake/regression/test_prism.py
```

Do the same for `tests/firedrake/regression/test_prism_interior_facets_core.py`,
`tests/firedrake/regression/test_prism_submesh.py` and
`tests/tsfc/test_prism_facet_integrals.py`. In `../fiat`, run
`python -m pytest test/finat`. `test_prism.py` and the core file take about 12 to
14 minutes together in serial. Do not select with `-k "not parallel"`: `-k` also
matches marker names (section 6).

Parallel. The at-exit teardown deadlock of section 6 also stops these tests after
they pass. So call the test bodies from a direct driver:

- Load the test module by path (`importlib.util.spec_from_file_location`) and call
  each test function with each parameter set. A fixture (for example
  `interior_mesh` of `test_prism_submesh.py`) does not run, so make its value in
  the driver.
- After each case, `allgather` the status of each rank. A case passes only if it
  passes on all ranks.
- End with a barrier and `os._exit(0)`. Print with `flush=True`, because
  `os._exit` does not flush the output buffers. `prterun` then reports "exiting
  improperly" and exit code 1. This is expected.
- Start it with `mpiexec -n 2 python -u driver.py`, then with `-n 3`. Run one MPI
  job at a time. Kill a stuck job by the name of its script only, never with
  `pkill -f prterun`.

---

## 9. Mixed-degree prism elements

A prism element can now have one degree on the triangle and a different degree on
the axis, for example P2 x P1:

```python
from finat.ufl import FiniteElement, TensorProductElement
element = TensorProductElement(FiniteElement("CG", triangle, 2),
                               FiniteElement("CG", interval, 1))
V = FunctionSpace(prism_mesh, element)
```

The design document made separate base and axis degrees "analysis only" (its
scope table and section 6.2). They now work on an unstructured prism mesh that is
axis-consistent. Commits: fiat `4d67fe02` and `59e96582`; here `3d4a93574`,
`6b36e5458`, `fa080182b` and `a37f2a335`.

### 9.1 The design

**FInAT.** `convert_tensorproductelement` (`finat/element_factory.py`) accepts a
`TensorProductElement` on the unstructured `prism` cell. It wraps the product in
`FlattenedDimensions`, as the other prism elements are.
`FlattenedDimensions.degree` (`finat/cube.py`) is now the largest factor degree,
so the two factors can have different degrees.

**The numbering.** The 9 edges of a prism are one DMPlex polytope type. With
different degrees, the axis edges and the base edges carry different dof counts.
So a count for each polytope type (section 2.1) cannot hold them. The mesh gives
each edge a role, axis or base:

- `make_dofs_per_plex_entity` takes this path for a pure prism mesh when the dof
  counts of the edges differ. It returns one count for each of the vertices, the
  axis edges (FIAT edges 0 to 2), the base edges (FIAT edges 3 to 8), the
  triangles, the quadrilaterals and the cell. A count that changes inside one role
  raises `NotImplementedError`. An element with the same count on all edges keeps
  the old path, so it does not need an axis-consistent mesh.
- `_prism_edge_roles` reads the roles from `cell_closure`. The FIAT closure of a
  prism holds the 3 axis edges in positions 6 to 8 and the 6 base edges in
  positions 9 to 14. Two `np.bincount` calls give the role of every edge.
- `_prism_mixed_degree_point_dofs` gives the dof count of every point of the
  chart as one array. `create_section` and `node_classes` use it.

**Axis consistency.** The role of an edge must be the same in all of its prisms.
If an edge is an axis edge in one prism and a base edge in another, the element
cannot be conforming. Then `_prism_edge_roles` raises `NotImplementedError`
("consistent base or axis role"). An `allreduce` makes every rank raise, not only
the rank that sees the edge. Without it, one rank did not raise at 3 ranks.
`prism_two_perpendicular.msh` is the example of a mesh that is not
axis-consistent. The other test meshes, which gmsh makes by extrusion, and their
scrambled copies are axis-consistent.

**Performance.** On 144 000 prisms, a P2 x P1 space builds in 0.09 s. The first
version, a Python loop over the cells, took 2.01 s.

### 9.2 The limits

- **A restricted space** (`RestrictedFunctionSpace`) of a mixed-degree element:
  `NotImplementedError` from `create_section`. Measured at HEAD.
- **A factor that is not Lagrange or Discontinuous Lagrange**, for example a Real
  factor: `NotImplementedError` ("needs factors of the families ...") from
  `convert_tensorproductelement` (fiat `59e96582`). Before this guard,
  `CG2 x Real` and `Real x CG1` built a P2 x DG0 space with no error, so the Real
  factor did not give one global value.
- **An RT or an HDiv product** fails with an unclear message. Measured at HEAD:
  `TensorProductElement(RT1, DG0)` gives `ValueError: Unsupported mapping:
  undefined`, and `HDiv` of that product gives a bare `AssertionError`. Both
  errors come from UFL, before FInAT sees the element, so the guard above does
  not apply.
  H(div) on prisms is out of scope (section 1).

### 9.3 The tests

In `tests/firedrake/regression/test_prism.py`:

- `test_prism_p2_base_p1_axis_interpolation_and_dS`: P2 x P1 on `prism_slab.msh`
  (dimension 195) and on `prism_interior_marked_scrambled.msh`. A field in the
  space interpolates exactly, and the `jump(u)**2 * dS` of the interpolant is 0.
- `test_prism_mixed_degree_rejects_axis_inconsistent_mesh`: P2 x P1 on
  `prism_two_perpendicular.msh` raises `NotImplementedError`.
- `test_prism_tensor_product_with_a_real_factor_is_rejected`: `CG2 x Real` and
  `Real x CG1` raise `NotImplementedError`. This test is serial.
- The dof count test asserts the edge-role counts of a product of two
  "HDiv Trace" 0 elements, and that a count that changes inside one role raises.

The first two tests have the mark `parallel([1, 2, 3])`. They pass at 1, 2 and 3
ranks, measured with a direct driver.
