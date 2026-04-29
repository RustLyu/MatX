## MatX

MatX is a modular linear algebra and numerical compute library written in C/C++, built with CMake for Windows/Linux.

This README is aligned with the current codebase structure (`master`, Apr 2026 snapshot) and focuses on practical build/use information plus current backend behavior.

## Current module layout

- `types`: exported ABI types, version, public opaque handles.
- `tools`: logging/time utilities (`spdlog` based).
- `core`: allocators, dense/sparse/vector containers, status/error helpers.
- `io`: matrix/vector print and read helpers.
- `vec_blas`: vector compute wrappers (`scal/copy/swap/dot/nrm2/asum/iamax/axpy/norm`) with backend dispatch.
- `dense_blas`: dense compute wrappers (`gemm/gemv/geadd/ger/trsv/trsm/syrk/herk/transpose/norm`) with backend dispatch.
- `sparse_blas`: sparse compute wrappers (`spmv/spmm/dsp2md/transpose/conj/mat_norm/spadd`) with backend dispatch.
- `dense_solve`: dense linear solve/factor APIs.
- `sparse_solve`: sparse linear solve/factor APIs (SuiteSparse KLU path).
- `tests`: GoogleTest-based unit tests.
- `3party`: vendored dependency packages (Windows/Linux split under subfolders).

## Backend auto-selection (current CMake logic)

Top-level `CMakeLists.txt` detects CPU vendor using a generated `try_run` program and applies defaults unless manually overridden:

- AMD:
  - `MATX_ENABLE_BLIS=ON`
  - `MATX_ENABLE_LIBFLAME=ON`
  - `MATX_ENABLE_GRAPHBLAS=ON`
  - `MATX_ENABLE_AOCL_SPARSE=ON`
  - `MATX_ENABLE_OPENBLAS=OFF`
- Intel:
  - `MATX_ENABLE_OPENBLAS=ON`
  - `MATX_ENABLE_GRAPHBLAS=ON`
  - `MATX_ENABLE_AOCL_SPARSE=ON`
  - `MATX_ENABLE_BLIS=OFF`
  - `MATX_ENABLE_LIBFLAME=OFF`
- Other:
  - `MATX_ENABLE_OPENBLAS=ON`
  - `MATX_ENABLE_GRAPHBLAS=OFF`

Also, build configuration enforces exactly one dense BLAS provider enabled among OpenBLAS/BLIS/MKL.

## Main CMake options

- `-DMATX_BUILD_TESTS=ON|OFF` (default `ON`)
- `-DMATX_ENABLE_OPENBLAS=ON|OFF`
- `-DMATX_ENABLE_BLIS=ON|OFF`
- `-DMATX_ENABLE_LIBFLAME=ON|OFF`
- `-DMATX_ENABLE_MKL=ON|OFF`
- `-DMATX_ENABLE_SUITESPARSE=ON|OFF` (default `ON`)
- `-DMATX_ENABLE_GRAPHBLAS=ON|OFF`
- `-DMATX_ENABLE_AOCL_SPARSE=ON|OFF`
- `-DMATX_DENSE_BLAS_BACKEND=AUTO|OPENBLAS|BLIS|REFERENCE`
- `-DMATX_SPARSE_BLAS_BACKEND=AUTO|OPENBLAS|BLIS|REFERENCE`
- `-DBUILD_SHARED_LIBS=ON|OFF`

## Build and test

Windows (Visual Studio x64, PowerShell):

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build --config Release -- /m
ctest --test-dir build -C Release --output-on-failure
```

Linux:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## Public capability snapshot

- Dense compute APIs: `matx_gemm_*`, `matx_gemv_*`, `matx_geadd_*`, `matx_ger_*`, `matx_trsv_*`, `matx_trsm_*`, `matx_syrk_*`, `matx_herk_*`, `matx_transpose_*`, `matx_conj_transpose_*`, `matx_mat_norm*`.
- Vector compute APIs: `matx_vec_scal_*`, `matx_vec_copy_*`, `matx_vec_swap_*`, `matx_vec_dot_*`, `matx_vec_nrm2_*`, `matx_vec_asum_*`, `matx_vec_iamax_*`, `matx_vec_axpy_*`, `matx_vec_norm1_*`, `matx_vec_norm2_*`, `matx_vec_norminf_*`.
- Sparse compute APIs: `matx_spmv_*`, `matx_spmm_*`, `matx_dsp2md_*`, sparse-sparse to dense, transpose/conjugate, sparse matrix norms, sparse addition.
- Dense solve APIs: factor + solve and one-shot solve for `f64/c64`.
- Sparse solve APIs: factor + solve and one-shot solve for COO/CSC pathways.
- IO APIs: print/read dense, sparse, and vectors to/from text files.

## Current test coverage (high level)

The `tests` target includes:

- core object create/destroy and basic behavior
- dense compute (real + complex)
- vector compute (real + complex)
- sparse compute (real + complex)
- dense/sparse solve paths
- print/read roundtrip style checks

Some tests intentionally allow backend-dependent `MATX_ERR_NOT_SUPPORTED` responses for optional paths.

## Known constraints

- Several solve and compute code paths assume/optimize for column-major memory layout.
- Sparse solve workflows often rely on COO->CSC conversion cached in handle fields.
- Certain complex sparse solve functions are present as API placeholders depending on backend availability.

## Optimization roadmap

Actionable optimization items are maintained in `TODO.md` with priorities and expected outcomes.

