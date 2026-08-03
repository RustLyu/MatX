## MatX

MatX is a modular linear algebra and numerical compute library written in **C17** (core) with **C++20** build support (tools/io/tests). It uses CMake 3.20+ and targets Windows/Linux.

All third-party dependencies are fetched via CMake **FetchContent** at configure time — no submodules, no vendored code in-tree.

## Module layout

- `types`: exported ABI types, version info, public opaque handle definitions.
- `tools`: logging and timing utilities (`spdlog` based).
- `core`: allocators, dense/sparse/vector container create/destroy/wrap/dup/fill.
- `io`: matrix/vector MTX-format print and read helpers.
- `vec_blas`: vector compute wrappers (`scal/copy/swap/dot/nrm2/asum/iamax/axpy/norm`) with vtable backend dispatch.
- `dense_blas`: dense compute wrappers (`gemm/gemv/geadd/ger/trsv/trsm/syrk/herk/transpose/conj_transpose/norm`) with vtable backend dispatch.
- `sparse_blas`: sparse compute wrappers (`spmv/spmm/dsp2md/transpose/conj/mat_norm/spadd`) with vtable backend dispatch.
- `dense_solve`: dense linear factor + solve APIs (f64/c64), backed by OpenBLAS / libFLAME.
- `sparse_solve`: sparse linear factor + solve APIs (f64/c64), backed by SuiteSparse (KLU/UMFPACK), SuperLU, and **MUMPS**.
- `tests`: GoogleTest-based unit tests.
- `cmake/`: CMake module files for platform detection, options, CPU vendor auto-detection, backend validation, and per-dependency FetchContent management.

### Module Dependency Graph

```
types  (leaf — version, ABI types, opaque handle definitions)
tools  → types          (spdlog logging, timing)
core   → types, tools   (allocators, container create/destroy/wrap/dup/fill)
io     → types, core    (MTX format print/read)
vec_blas      → types, core, tools (+ OpenBLAS/BLIS at link)
dense_blas    → types, core, tools (+ OpenBLAS/BLIS/libFLAME)
sparse_blas   → types, core, tools (+ GraphBLAS, AOCL_SPARSE)
dense_solve   → types, core, tools (+ OpenBLAS/libFLAME via CBLAS/LAPACK)
sparse_solve  → types, tools, core (+ SuiteSparse, SuperLU, MUMPS)
tests         → all above + GTest + OpenMP
```

### Code Conventions

- **Commit messages**: lowercase, terse English sentence fragments
- **Branch naming**: lowercase with underscores (e.g., `del_3party`)
- **Core is pure C**, tools/io are C++ (for spdlog integration), tests are C++ with GTest
- Public API functions follow `matx_<module>_<op>_<precision>` naming (e.g., `matx_gemm_d_i8`)
- Header-only public API: all headers under `include/matx/`

## Architecture

### Opaque Handle + Vtable Dispatch

Public types are **opaque pointers** (`typedef struct *_opaque_t*`). Concrete structs live in internal headers. Each compute module uses a **vtable dispatch** pattern:

- A `*_backend_kind_t` enum (`REFERENCE`, `OPENBLAS`, `BLIS`, etc.)
- A vtable struct of function pointers for every operation (separate `_d_i8` and `_z_i8` variants)
- Factory: `matx_*_default()` auto-selects; `matx_*_by_type(kind)` for explicit choice
- Public API functions take a `const matx_*_backend_t*` and dispatch through the vtable

### Numeric Types

Every operation has two precision variants: `_d_i8` (double real) and `_z_i8` (double complex). Complex type is defined as an aligned `{double real, imag}` struct — not `double _Complex`.

### Memory Layout

Column-major is the default/optimized layout for solve and most compute paths. Set at object creation time via `MATX_COL_MAJOR` / `MATX_ROW_MAJOR`.

### Allocator System

`matx_alloc_t` provides custom malloc/free with user-data pointer. All object creation accepts an optional allocator for user-controlled memory management.

## Backend auto-selection

Top-level CMake detects CPU vendor via a generated `try_run` program and applies defaults unless manually overridden:

| Vendor | BLAS Backend | libFLAME | AOCL-Sparse | SuiteSparse | GraphBLAS | mumps	   | superlu   |	
|--------|-------------|----------|--------------|-------------|-----------|-----------|-----------|
| AMD    | BLIS        | ON       |  ON          |ON           | ON        | ON        | ON        |
| Intel  | OpenBLAS    | OFF      |  OFF         |ON           | ON        | ON        | ON        |
| Other  | OpenBLAS    | OFF      |  OFF         |ON           | ON        | ON        | ON        |

Exactly one dense BLAS provider (OpenBLAS / BLIS) is enforced at configure time.

## CMake options

| Option | Default | Purpose |
|--------|---------|---------|
| `-DMATX_BACKEND` | `OPENBLAS` | Dense BLAS backend: `OPENBLAS` / `AMD_AOCL` / `AUTO` |
| `-DMATX_BUILD_TESTS` | `ON` | Build GoogleTest-based unit tests |
| `-DMATX_ENABLE_SUITESPARSE` | `ON` | SuiteSparse sparse solvers (KLU, UMFPACK) |
| `-DMATX_ENABLE_OPENBLAS` | `ON` | OpenBLAS dense + sparse BLAS |
| `-DMATX_ENABLE_BLIS` | `OFF` | BLIS dense BLAS (AMD path) |
| `-DMATX_ENABLE_LIBFLAME` | `OFF` | libFLAME LAPACK replacement (AMD path) |
| `-DMATX_ENABLE_GRAPHBLAS` | `OFF` | GraphBLAS sparse operations |
| `-DMATX_ENABLE_AOCL_SPARSE` | `OFF` | AOCL-Sparse sparse operations |
| `-DBUILD_SHARED_LIBS` | `OFF` | Build shared libraries instead of static |

## Build and test

**Windows (Visual Studio x64, PowerShell):**

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build --config Release -- /m
ctest --test-dir build -C Release --output-on-failure
```

**Linux:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

**Run a single test:** use `ctest -R <filter>` or execute `./build/tests/matx_tests` directly.

## Public API snapshot

### Dense compute
`matx_gemm_*`, `matx_gemv_*`, `matx_geadd_*`, `matx_ger_*`, `matx_zgerc_*`, `matx_trsv_*`, `matx_trsm_*`, `matx_syrk_*`, `matx_dsyr2k_*`, `matx_herk_*`, `matx_zher2k_*`, `matx_hadamard_*`, `matx_transpose_*`, `matx_conj_transpose_*`, `matx_mat_norm*`, `matx_inv_dense_*`, `matx_expm_dense_d_i8`, `matx_diag_create_*`, `matx_get_diag_*`
### Dense element-wise math
`matx_dense_*_{exp,log,sqrt,sin,cos,abs,pow}`, `matx_dense_*_{add,sub,mul,div}`, `matx_dense_*_add_scalar`, `matx_dense_*_mul_scalar`

### Vector compute
`matx_vec_scal_*`, `matx_vec_copy_*`, `matx_vec_swap_*`, `matx_vec_dot_*`, `matx_vec_nrm2_*`, `matx_vec_asum_*`, `matx_vec_iamax_*`, `matx_vec_axpy_*`, `matx_vec_norm1_*`, `matx_vec_norm2_*`, `matx_vec_norminf_*`, `matx_vec_cross_*`
### Vector element-wise math
`matx_vec_*_{exp,log,sqrt,sin,cos,abs,pow}`, `matx_vec_*_{add,sub,mul,div}`, `matx_vec_*_add_scalar`, `matx_vec_*_mul_scalar`, `matx_vec_*_cumsum`

### Sparse compute
`matx_spmv_*`, `matx_spmm_*`, `matx_dsp2md_*`, `matx_zsp2md_*`, sparse-sparse to dense, transpose/conjugate, sparse matrix norms, sparse addition, row/column NNZ counts (`spnnz_rows`, `spnnz_cols`), row/column sums (`sprowsums`, `spcolsums`), diagonal extraction (`spdiag`), row/column scaling (`scale_rows`, `scale_cols`), COO helpers (`matx_coo_*_get_row`, `matx_coo_*_get_col`, `matx_coo_*_to_dense`)

### Dense solve
Factor + solve and one-shot solve for d_i8 / z_i8. Eigenvalue solvers (`matx_syev_*`, `matx_geev_*`), QR factorization (`matx_qr_*`), determinant (`matx_det_dense_*`), condition number (`matx_cond_dense_*`).

### Sparse solve
Factor + solve and one-shot solve for COO/CSC pathways (LU and Cholesky). Backends: **KLU**, **UMFPACK**, **SuperLU**, **MUMPS**.

### Random
`matx_vec_rand_uniform_*`, `matx_vec_rand_normal_*`, `matx_dense_rand_uniform_*`

## Test coverage

The `tests` target covers:

- Core object lifecycle (create/destroy/dup/fill) for dense, sparse, and vector containers
- Dense compute (real + complex): gemm, gemv, geadd, ger, zgerc, trsv, trsm, syrk, dsyr2k, herk, zher2k, hadamard, transpose, conj_transpose, norms
- Vector compute (real + complex)
- Sparse compute (real + complex)
- Dense/sparse solve paths (all backends)
- Print/read roundtrip checks

Some tests allow backend-dependent `MATX_ERR_NOT_SUPPORTED` responses for optional paths.

## CI/CD

GitHub Actions runs on push/PR to all branches: configure (Release) → build. See `.github/workflows/ci.yml`.

## Known constraints

- Several solve and compute code paths assume/optimize for column-major memory layout
- Sparse solve workflows rely on COO→CSC conversion cached inside COO handle fields
- Certain complex sparse solve functions are API placeholders depending on backend availability

## License

MatX is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full text.
