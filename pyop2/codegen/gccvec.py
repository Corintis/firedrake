"""A loopy C target that emits GCC/Clang vector extensions for VectorArrayDimTag.

Mirrors loopy's OpenCL vectorization mechanism but for the plain GNU-C target:
- vector dtypes reuse loopy's `vec` numpy structured dtypes (double2, ...);
- a preamble typedefs them via __attribute__((vector_size(...)));
- vector lane access is plain subscript v[i] (a GCC vector-extension feature).
"""
import numpy as np
import pymbolic.primitives as prim
from loopy.target.c import CWithGNULibcTarget, CWithGNULibcASTBuilder
from loopy.target.opencl import vec, _register_vector_types
from loopy.target.c.compyte.dtypes import (DTypeRegistry, fill_registry_with_c_types)
from loopy.target.c import CompyteDTypeRegistryWrapper
from loopy.types import NumpyType
from loopy.target.c.codegen.expression import (
    ExpressionToCExpressionMapper as _BaseExprMapper)


def _vecsize_bytes(dtype, count):
    return int(np.dtype(dtype).itemsize) * count


def cell_vec_width():
    """The requested cross-element vector width (0/1 = disabled)."""
    import os
    try:
        return int(os.environ.get("PYOP2_CELL_VEC_WIDTH", "0"))
    except ValueError:
        return 0


def _debug(msg):
    import os
    if os.environ.get("PYOP2_CELL_VEC_DEBUG"):
        import sys
        print(f"[cell-vec] {msg}", file=sys.stderr, flush=True)


def syntax_check_ok(code, include_dirs=()):
    """Cheap ``-fsyntax-only`` check of generated C code.

    Used to validate vectorized wrappers so that any structure the
    transform mishandled degrades to the scalar wrapper instead of a hard
    compilation failure.  Returns True when the check cannot be run (the
    real compiler will decide then).
    """
    import os
    import shlex
    import subprocess
    import tempfile

    import petsctools

    try:
        cc = shlex.split(petsctools.get_petscvariables()["CC"])
        cppargs = (*petsctools.get_petsc_dirs(prefix="-I", subdir="include"),
                   *(f"-I{d}" for d in include_dirs),
                   f"-I{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w",
                                         delete=False) as f:
            f.write(code)
            path = f.name
        try:
            rc = subprocess.run([*cc, "-fsyntax-only", *cppargs, path],
                                capture_output=True).returncode
        finally:
            os.unlink(path)
        return rc == 0
    except Exception:
        return True


def _vec_ctype_name(npd):
    base, count = vec.type_to_scalar_and_count[npd]
    return "%s%d" % (_BASE_C[np.dtype(base)], count), np.dtype(base), count


class GCCVecCASTBuilder(CWithGNULibcASTBuilder):
    def add_vector_access(self, access_expr, index):
        # GCC/Clang vector extensions allow plain subscripting v[i].
        return access_expr[index]

    def emit_assignment(self, codegen_state, insn):
        # Clang's GCC vector-extension types reject scalar->vector assignment
        # (`double2 acc = 0.0;`).  Inside a vectorized loop, an assignment
        # whose RHS does not depend on the vec iname produces a uniform scalar
        # that must be broadcast across all lanes.  We do that with the splat
        # trick `(rhs) + (T){0, ...}`, which clang accepts (scalar OP vector).
        vinfo = codegen_state.vectorization_info
        if vinfo is not None and not insn.atomicity:
            from loopy.symbolic import get_dependencies
            assignee_deps = get_dependencies(insn.assignee)
            rhs_deps = get_dependencies(insn.expression)
            if vinfo.iname in assignee_deps and vinfo.iname not in rhs_deps:
                return self._emit_broadcast_assignment(codegen_state, insn,
                                                       vinfo)
        return super().emit_assignment(codegen_state, insn)

    def _emit_broadcast_assignment(self, codegen_state, insn, vinfo):
        from cgen import Assign
        from loopy.expression import dtype_to_type_context
        from loopy.target.c.codegen.expression import PREC_NONE
        kernel = codegen_state.kernel
        ecm = codegen_state.expression_to_code_mapper
        name, = insn.assignee_var_names()
        base_dtype = kernel.get_var_descriptor(name).dtype
        tname, _, count = _vec_ctype_name(
            self.target.vector_dtype(base_dtype, vinfo.length).numpy_dtype)
        lhs_code = ecm(insn.assignee, prec=PREC_NONE, type_context=None)
        base_ctx = dtype_to_type_context(kernel.target, base_dtype)
        rhs_code = ecm(insn.expression, prec=PREC_NONE, type_context=base_ctx,
                       needed_dtype=base_dtype)
        zero = "(%s){%s}" % (tname, ", ".join(["0"] * count))
        return Assign(str(lhs_code), "(%s) + %s" % (rhs_code, zero))

    def get_array_base_declarator(self, ary):
        from loopy.kernel.array import (FixedStrideArrayDimTag,
                                        VectorArrayDimTag)
        from loopy.target.c import POD
        dtype = ary.dtype
        vec_size = ary.vector_length()
        if vec_size > 1:
            dtype = self.target.vector_dtype(dtype, vec_size)
        if ary.dim_tags:
            for dim_tag in ary.dim_tags:
                if isinstance(dim_tag, (FixedStrideArrayDimTag,
                                        VectorArrayDimTag)):
                    pass
                else:
                    raise NotImplementedError(
                        f"{type(self).__name__} does not understand axis tag "
                        f"'{type(dim_tag)}.")
        return POD(self, dtype, ary.name)

    def preamble_generators(self):
        return [*super().preamble_generators(), _gcc_vec_preamble_generator]


# C base-type names matching loopy's vec dtype registry (OpenCL-style names:
# the int32/int64 vectors are registered as int2/long2, not int32_t2/...).
_BASE_C = {np.dtype(np.float64): "double", np.dtype(np.float32): "float",
           np.dtype(np.int32): "int", np.dtype(np.int64): "long"}


def _gcc_vec_preamble_generator(preamble_info):
    # Emit GCC vector-extension typedefs for any array/temp tagged with a
    # VectorArrayDimTag (vector_length() > 1).  The vector-ness lives in the
    # dim-tags, not the dtype, so we key off vector_length().
    knl = preamble_info.kernel
    pairs = set()  # (base_numpy_dtype, count)

    def consider(dtype, vl):
        base = getattr(dtype, "numpy_dtype", None)
        if base is None:
            return
        # Case 1: dtype is already a vec structured dtype (post-lowering, the
        # vec dim has been folded into the dtype and the tag dropped).
        if base in vec.type_to_scalar_and_count:
            sbase, count = vec.type_to_scalar_and_count[base]
            if np.dtype(sbase) in _BASE_C:
                pairs.add((np.dtype(sbase), int(count)))
        # Case 2: scalar dtype carrying a VectorArrayDimTag (pre-lowering).
        elif vl and vl > 1 and np.dtype(base) in _BASE_C:
            pairs.add((np.dtype(base), int(vl)))

    for dtype in preamble_info.seen_dtypes:
        consider(dtype, 1)
    for ary in list(knl.args) + list(knl.temporary_variables.values()):
        try:
            vl = ary.vector_length()
        except Exception:
            vl = 1
        consider(getattr(ary, "dtype", None), vl)

    decls = []
    for base, count in sorted(pairs, key=lambda p: (str(p[0]), p[1])):
        name = "%s%d" % (_BASE_C[base], count)
        # Typedef at key 10 (before any use); math helpers/macros at keys 98/99
        # so they land AFTER every #include (headers are at key "0"/"50_cmath").
        decls.append((f"10_vec_{name}",
                      f"typedef {_BASE_C[base]} {name} "
                      f"__attribute__((vector_size({_vecsize_bytes(base, count)})));"))
        if base.kind == "f":
            decls.extend(_vec_math_decls(name, count))
    yield from decls


# Math functions used by TSFC kernels.  `fabs` has a direct clang elementwise
# builtin (-> fabs.2d); the rest use a per-lane fallback that clang's loop
# vectorizer turns into the corresponding NEON vector op (e.g. fsqrt.2d).
# We dispatch via C11 _Generic so `fn(x)` keeps working for BOTH scalar and
# vector arguments in the same translation unit (the merged wrapper is scalar).
_ELEMENTWISE = {"fabs": "__builtin_elementwise_abs", "abs": "__builtin_elementwise_abs"}
_UNARY = ["fabs", "abs", "sqrt", "exp", "log", "log10", "sin", "cos", "tan",
          "sinh", "cosh", "tanh", "asin", "acos", "atan", "ceil", "floor",
          "erf", "erfc"]
_BINARY = ["pow", "atan2", "fmax", "fmin", "copysign"]


def _vec_math_decls(vname, count):
    out = []
    for fn in _UNARY:
        helper = f"cvec_{fn}_{vname}"
        if fn in _ELEMENTWISE:
            sig = (f"static inline {vname} {helper}({vname} x)"
                   f"{{ return {_ELEMENTWISE[fn]}(x); }}")
        else:
            lanes = "; ".join(f"r[{i}] = __builtin_{fn}(x[{i}])"
                              for i in range(count))
            sig = (f"static inline {vname} {helper}({vname} x)"
                   f"{{ {vname} r; {lanes}; return r; }}")
        out.append((f"98_vecmath_{vname}_{fn}_def", sig))
        out.append((f"99_vecmath_{vname}_{fn}",
                    f"#undef {fn}\n#define {fn}(x) "
                    f"_Generic((x), {vname}: {helper}, default: __builtin_{fn})(x)"))
    for fn in _BINARY:
        helper = f"cvec_{fn}_{vname}"
        lanes = "; ".join(f"r[{i}] = __builtin_{fn}(x[{i}], y[{i}])"
                          for i in range(count))
        sig = (f"static inline {vname} {helper}({vname} x, {vname} y)"
               f"{{ {vname} r; {lanes}; return r; }}")
        out.append((f"98_vecmath_{vname}_{fn}_def", sig))
        out.append((f"99_vecmath_{vname}_{fn}",
                    f"#undef {fn}\n#define {fn}(a, b) "
                    f"_Generic((a), {vname}: {helper}, default: __builtin_{fn})((a), (b))"))
    return out


class GCCVecTarget(CWithGNULibcTarget):
    def get_device_ast_builder(self):
        return GCCVecCASTBuilder(self)

    def get_dtype_registry(self):
        result = DTypeRegistry()
        fill_registry_with_c_types(result, respect_windows=False, include_bool=True)
        _register_vector_types(result)
        return CompyteDTypeRegistryWrapper(result)

    def is_vector_dtype(self, dtype):
        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec.types.values()))

    def vector_dtype(self, base, count):
        return NumpyType(vec.types[base.numpy_dtype, count])


# ---------------------------------------------------------------------------
# Cross-element vectorization transform (Sun et al., IJHPCA 2020,
# arXiv:1903.08243).  Process VL cells per iteration with the cell index mapped
# to the SIMD lane and per-cell data laid out interleaved across lanes, so the
# (gather-free) local compute vectorises trivially.  The indirect gather/scatter
# stays scalar (per-lane, ~3% of the work) since loopy cannot vector-index it.
# ---------------------------------------------------------------------------

def cross_element_vectorize(wrapper, local_kernel_name, vl, cell_iname="n"):
    """Return *wrapper* transformed for VL-wide cross-element vectorization.

    Raises on any unsupported structure; callers should fall back to the
    scalar wrapper on failure.  The iteration space is padded to a multiple of
    VL and the dummy lanes are predicated out of all global memory access, so
    arbitrary (odd) cell counts and arbitrary [start, end) ranges are handled
    correctly.
    """
    import loopy as lp
    import pymbolic.primitives as prim
    from loopy.kernel.instruction import CInstruction
    from loopy.symbolic import SubstitutionMapper, get_dependencies
    from pymbolic.mapper.substitutor import make_subst_func

    ep = local_kernel_name and [e for e in wrapper.entrypoints][0]

    # Inline the local (compute) kernel so the whole per-cell computation lives
    # in the wrapper's cell loop.
    wrapper = lp.inline_callable_kernel(wrapper, local_kernel_name)
    wk = wrapper[ep]

    # TSFC adds a no-op CInstruction that touches every kernel argument so
    # that unused arguments survive into the generated code's signature (the
    # caller passes them regardless; pruning them shifts every later
    # argument — e.g. an unused Constant in one block of a mixed form).
    # After inlining, its read set still names the *callee's* variables
    # (loopy's inliner does not rewrite CInstruction reads), so it no longer
    # keeps anything alive.  As a "C instruction" it also blocks
    # vectorization.  Hoist it out of the cell loop and point its reads at
    # every wrapper argument explicitly.
    arg_names = frozenset(a.name for a in wk.args)
    wk = wk.copy(instructions=[
        i.copy(within_inames=frozenset(), depends_on=frozenset(),
               read_variables=arg_names, assignees=())
        if isinstance(i, CInstruction) else i
        for i in wk.instructions])

    # Classify gather/scatter instructions: those touching the global Dats/maps.
    global_names = {a.name for a in wk.args
                    if getattr(a, "shape", None) is not None}
    def touches_global(insn):
        names = set()
        for a in insn.assignees:
            names |= get_dependencies(a)
        names |= insn.read_dependency_names()
        return bool(names & global_names)
    gs_ids = {i.id for i in wk.instructions if touches_global(i)}
    if not gs_ids:
        raise ValueError("no gather/scatter instructions found")

    # Bail out on kernels containing conditionals / comparisons: vectorizing a
    # comparison yields a vector-of-bool, and combining those with `&&`/`||` is
    # invalid C (C++-only).  Plain arithmetic Poisson/mass/Helmholtz kernels have
    # none; such kernels fall back to the scalar wrapper.
    from loopy.symbolic import WalkMapper as _WalkMapper

    class _CondFinder(_WalkMapper):
        found = False

        def map_comparison(self, expr, *args):
            self.found = True

        def map_logical_and(self, expr, *args):
            self.found = True

        def map_logical_or(self, expr, *args):
            self.found = True

        def map_if(self, expr, *args):
            self.found = True

    finder = _CondFinder()
    for insn in wk.instructions:
        if hasattr(insn, "expression"):
            finder(insn.expression)
    if finder.found:
        raise ValueError("kernel contains conditionals; not vectorizable here")

    # Split the cell loop into outer * VL + lane.
    outer = cell_iname + "_outer"
    wk = lp.split_iname(wk, cell_iname, vl, inner_iname="lane", outer_iname=outer)

    # Force the lane axis to the full width [0, VL) for every outer iteration
    # (vectorization requires a full vector), padding the iteration space to a
    # multiple of VL.  The dummy lanes in the first/last partial groups are made
    # safe below by predicating their gather/scatter on the true cell range.
    import islpy as isl
    new_domains = []
    for dom in wk.domains:
        if "lane" in dom.get_var_names(isl.dim_type.set):
            bs, = isl.BasicSet(
                "[start, end] -> {{ [{o}, lane] : 0 <= lane < {vl} "
                "and {vl}*{o} < end and {vl}*{o} + {vl} > start }}"
                .format(o=outer, vl=vl)).get_basic_sets()
            new_domains.append(bs)
        else:
            new_domains.append(dom)
    wk = wk.copy(domains=new_domains)

    # Give every per-cell temporary a trailing lane axis (one slot per lane).
    # Exclude read-only / constant-initialized temporaries (tabulation tables):
    # they are shared across cells, stay scalar, and broadcast-multiply with the
    # per-cell vector data (a scalar*vector splat, which clang vectorizes).
    tempnames = [t.name for t in wk.temporary_variables.values()
                 if not t.read_only and t.initializer is None]
    wk = lp.privatize_temporaries_with_inames(wk, "lane", only_var_names=tempnames)

    # Expand each gather/scatter into VL copies with the lane index replaced by a
    # compile-time constant (so the buffer's vec axis is indexed by a constant),
    # dropping `lane` from their iname set.  The remaining (compute) instructions
    # keep `lane`, which we tag "vec".
    # Predicate each expanded lane on the true cell range so that the padded
    # dummy lanes never touch global memory (no OOB gather, no double scatter).
    outer_var = prim.Variable(outer)
    start_var = prim.Variable("start")
    end_var = prim.Variable("end")
    expanded_ids = set()
    new_insns = []
    for insn in wk.instructions:
        if insn.id in gs_ids and "lane" in insn.within_inames:
            expanded_ids.add(insn.id)
            for L in range(vl):
                sm = SubstitutionMapper(make_subst_func({"lane": L}))
                cell = vl * outer_var + L
                pred = frozenset({
                    prim.Comparison(cell, "<", end_var),
                    prim.Comparison(cell, ">=", start_var)})
                new_insns.append(insn.copy(
                    id=f"{insn.id}_lane{L}",
                    within_inames=insn.within_inames - {"lane"},
                    predicates=insn.predicates | pred,
                    assignee=sm(insn.assignee),
                    expression=sm(insn.expression)))
        else:
            new_insns.append(insn)
    # Reroute dependencies onto the expanded instruction ids.
    fixed = []
    for insn in new_insns:
        dep = set()
        for d in insn.depends_on:
            if d in expanded_ids:
                dep |= {f"{d}_lane{L}" for L in range(vl)}
            else:
                dep.add(d)
        fixed.append(insn.copy(depends_on=frozenset(dep)))
    wk = wk.copy(instructions=fixed)

    # Disjoint-lane writes need no ordering between them.
    wk = wk.copy(options=wk.options.copy(enforce_variable_access_ordered=False))
    wk = lp.tag_inames(wk, {"lane": "vec"})

    # Tag the trailing (lane) axis of each PRIVATIZED buffer as the vector axis.
    # Restrict to the privatized temporaries: a non-privatized array whose last
    # axis happens to equal VL (e.g. a 2D gradient when VL==2) must not be tagged.
    privatized = set(tempnames)
    for tv in list(wk.temporary_variables.values()):
        if tv.name in privatized and tv.shape and tv.shape[-1] == vl:
            spec = ",".join(["c"] * (len(tv.shape) - 1) + ["vec"])
            wk = lp.tag_array_axes(wk, tv.name, spec)

    wrapper = wrapper.with_kernel(wk)
    wrapper = lp.register_preamble_generators(wrapper, [_gcc_vec_preamble_generator])
    wrapper = wrapper.copy(target=GCCVecTarget())
    return wrapper
