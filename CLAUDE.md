# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatX is a modular linear algebra and numerical compute library written in **C17** (core) with **C++20** build support (tools/io/tests). It uses CMake 3.20+ and targets Windows/Linux.

All third-party dependencies are fetched via CMake **FetchContent** at configure time — no submodules, no vendored code in-tree. The one exception is **AGMG** (algebraic multigrid), which requires a manual download due to license terms.

## Build Commands

```bash
# Linux (GCC/Clang)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Windows (PowerShell, VS x64)
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build --config Release -- /m
ctest --test-dir build -C Release --output-on-failure

# Windows (MSYS2 MinGW-w64 UCRT64)
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

**Single test:** `ctest --test-dir build -R <filter> --output-on-failure` or run `./build/tests/matx_tests` directly (the test binary accepts no arguments; all tests run).

**Benchmarks (optional, requires GraphBLAS):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON -DMATX_ENABLE_GRAPHBLAS=ON -DMATX_BUILD_BENCHMARKS=ON
cmake --build build --target matx_benchmarks -j$(nproc)
./build/benchmarks/matx_benchmarks > results.csv          # default scale sweep
./build/benchmarks/matx_benchmarks --quick                 # shorter run
./build/benchmarks/matx_benchmarks --coo-conversion-only   # isolated COO conversion
```

### Compiler Flags (per `MatXPlatform.cmake`)

| Toolchain | Release Flags |
|-----------|--------------|
| GCC / Clang / AppleClang | `-O3 -DNDEBUG -march=native -mtune=native -flto` (+ LTO link) |
| MSVC | `/O2 /DNDEBUG /arch:AVX2` |

### Key CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `-DMATX_BACKEND` | `OPENBLAS` | Dense BLAS backend: `OPENBLAS` / `AMD_AOCL` / `AUTO` |
| `-DMATX_BUILD_TESTS` | `ON` | Build the test binary (`matx_tests`) |
| `-DMATX_BUILD_BENCHMARKS` | `OFF` | Build performance benchmarks (requires `MATX_ENABLE_GRAPHBLAS=ON`) |
| `-DMATX_ENABLE_SUITESPARSE` | `ON` | SuiteSparse: KLU, UMFPACK, CXSparse, GraphBLAS |
| `-DMATX_ENABLE_UMFPACK` | `ON` | SuiteSparse UMFPACK (requires `MATX_ENABLE_SUITESPARSE`) |
| `-DMATX_ENABLE_GRAPHBLAS` | `ON` | GraphBLAS sparse ops (requires `MATX_ENABLE_SUITESPARSE`) |
| `-DMATX_ENABLE_SUPERLU` | `ON` | SuperLU sparse solver (auto-fetched) |
| `-DMATX_ENABLE_MUMPS` | `ON` | MUMPS sparse solver (auto-fetched) |
| `-DMATX_ENABLE_AGMG` | `OFF` | AGMG multigrid solver (manual download from agmg.eu) |
| `-DBUILD_SHARED_LIBS` | `OFF` | Static libraries by default |

### Backend Auto-Selection (CPU Vendor)

At configure time, a `try_run` program detects the CPU vendor via `cpuid` and applies defaults:

| Vendor | BLAS | libFLAME | AOCL-Sparse | SuiteSparse | GraphBLAS | MUMPS | SuperLU |
|--------|------|----------|-------------|-------------|-----------|-------|---------|
| AMD | BLIS | ON | ON | ON | ON | ON | ON |
| Intel | OpenBLAS | OFF | OFF | ON | ON | ON | ON |
| Other | OpenBLAS | OFF | OFF | ON | ON | ON | ON |

Exactly one dense BLAS provider is enforced. Override with `-DMATX_CPU_PLATFORM=AMD|INTEL|OTHER`.

## Architecture

### Flat Module Layout (no `src/` root)

Each module lives in its own top-level directory with `include/matx/` and `src/` subdirectories:

```
types/          → matx_types.h, matx_types_internal.h (public + internal types)
tools/          → matx_log.h, matx_tm.h (spdlog logging, timing)
core/           → matx_func.h (allocators, container create/destroy/wrap/dup/fill)
io/             → matx_write.h, matx_read.h, matx_print.h (MTX + CSV I/O)
vec_blas/       → matx_vec_compute.h (vector BLAS: scal/copy/swap/dot/nrm2/asum/iamax/axpy/norm + element-wise math)
dense_blas/     → matx_dense_compute.h (dense BLAS: gemm/gemv/geadd/ger/trsv/trsm/syrk/herk + element-wise + norms)
sparse_blas/    → matx_sparse_compute.h (sparse: spmv/spmm/spadd/transpose/norms/nnz counts/row-col sums/scale)
dense_solve/    → matx_dense_solve.h (LU, Cholesky, QR, eigenvalues, inverse, expm, det, cond)
sparse_solve/   → matx_sparse_solve.h (LU + Cholesky factor/solve via KLU/UMFPACK/SuperLU/MUMPS/AGMG)
tests/          → 16 test files + custom harness, single binary
benchmarks/     → CSV-output performance sweeps covering all API surfaces
cmake/          → Per-dependency FetchContent modules + platform/option detection
```

### Module Dependency Graph

```
types  (leaf — version, ABI types, opaque handle definitions)
tools  → types          (spdlog logging, timing)
core   → types, tools   (allocators, container create/destroy/wrap/dup/fill)
io     → types, core    (MTX format print/read/write, CSV)
vec_blas      → types, core, tools (+ OpenBLAS/BLIS at link)
dense_blas    → types, core, tools (+ OpenBLAS/BLIS/libFLAME)
sparse_blas   → types, core, tools (+ GraphBLAS, AOCL_SPARSE)
dense_solve   → types, core, tools (+ OpenBLAS/libFLAME via CBLAS/LAPACK)
sparse_solve  → types, tools, core (+ SuiteSparse, SuperLU, MUMPS, AGMG)
tests         → all above + OpenMP
```

### Opaque Handle + Vtable Dispatch Pattern

Public types are **opaque pointers** (`typedef struct *_opaque_t*`). Concrete structs live in `matx_types_internal.h`. Each compute module uses a **vtable dispatch** pattern:

- A `*_backend_kind_t` enum (e.g., `REFERENCE`, `OPENBLAS`, `BLIS`)
- A vtable struct of function pointers for every operation (separate `_d_i8` and `_z_i8` variants)
- Factory functions: `matx_*_default()` auto-selects; `matx_*_by_type(kind)` for explicit choice
- Public API functions take a `const matx_*_backend_t*` and dispatch through the vtable

### Numeric Types

Every operation has **two precision variants**: `_d_i8` (double real, 8 bytes) and `_z_i8` (double complex, 16 bytes). Complex type is `matx_complex_d_t`: an aligned `{double real, imag}` struct — not `double _Complex`.

### Memory Layout

Column-major is the default/optimized layout for solve and most compute paths. Set at object creation time via `MATX_COL_MAJOR` / `MATX_ROW_MAJOR`.

### Allocator System

`matx_alloc_t` provides custom `malloc`/`free` with a user-data pointer. All object creation functions accept an optional allocator for user-controlled memory management. Default: `matx_alloc_default()`.

## Code Conventions

- **Commit messages**: lowercase, terse English sentence fragments (no conventional commit prefixes)
- **Branch naming**: lowercase with underscores (e.g., `del_3party`)
- **Core is pure C**, tools/io are C++ (for spdlog integration), tests are C++ with a custom harness
- Public API functions follow `matx_<module>_<op>_<precision>` naming (e.g., `matx_gemm_d_i8`, `matx_spmv_z_i8`)
- Header-only public API: all headers under `*/include/matx/`

## CMake Modules (`cmake/`)

| File | Purpose |
|------|---------|
| `MatXPlatform.cmake` | OS detection, compiler flags, C17/C++20 standards |
| `MatXCPUDetect.cmake` | CPU vendor auto-detect via `cpuid` try_run |
| `MatXOptions.cmake` | Backend selection, feature toggles, build options |
| `MatXBackendValidate.cmake` | Enforce exactly one BLAS backend, set install prefix paths |
| `DepSpdlog.cmake` | FetchContent for spdlog |
| `DepOpenBLAS.cmake` | FetchContent for OpenBLAS (ILP64 interface) |
| `DepAOCL.cmake` | FetchContent for AMD AOCL (BLIS + libFLAME + AOCL-Sparse) |
| `DepSuiteSparse.cmake` | FetchContent for SuiteSparse (KLU, UMFPACK, CXSparse, GraphBLAS) |
| `DepSuperLU.cmake` | FetchContent for SuperLU |
| `DepMUMPS.cmake` | FetchContent for MUMPS (with SCOTCH ordering) |
| `DepAGMG.cmake` | Find pre-built AGMG (manual download, LP64 BLAS bridge) |

### AGMG Solver (Special Dependency)

AGMG (Algebraic Multigrid) requires manual download from https://agmg.eu (free for academic use). Steps:
1. Download Fortran 90 + C wrapper source
2. Build: `gfortran -c -O3 dagmg.f90 zagmg.f90 && ar rcs libagmg.a *.o`
3. Set `AGMG_ROOT` to the install prefix and enable with `-DMATX_ENABLE_AGMG=ON`

AGMG uses LP64 (32-bit int) BLAS ABI, isolated from MatX's ILP64 OpenBLAS via a bridge DLL.

## Test System

### Custom Test Harness (`tests/matx_test_harness.h`)

Tests use a **lightweight custom framework**, not Google Test. The harness provides a GTest-compatible syntax:

```cpp
TEST(suite, case_name)           // registers via static global constructor
ASSERT_EQ(lhs, rhs)              // fatal on failure (returns from test)
ASSERT_NEAR(lhs, rhs, tolerance) // floating-point comparison, fatal
EXPECT_EQ / EXPECT_NE / EXPECT_NEAR / EXPECT_TRUE  // non-fatal variants
```

**255 test cases** across 16 `.cpp` files in a single `matx_tests` executable. No fixtures, no SetUp/TearDown — each test manages its own MatX objects via explicit create/destroy calls.

### Testing Patterns

```cpp
TEST(core_dense, create_destroy) {
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 4, NULL), MATX_OK);
    ASSERT_NE(M->data, nullptr);
    EXPECT_EQ(M->nrows, 3u);
    matx_dense_d_i8_destroy(&a, M);
}
```

**Backend-not-supported skipping** (used when optional backends may be absent):
```cpp
matx_status_t st = matx_vec_axpy_d_i8(&vblas, 2.0, x, y);
if (st == MATX_ERR_NOT_SUPPORTED) { /* clean up and return — test passes */ }
```

### Test File Organization

| File | Area | Count |
|------|------|-------|
| `test_core_dense.cpp` | Dense container lifecycle | 26 |
| `test_core_vec.cpp` | Vector container lifecycle | 12 |
| `test_core_math.cpp` | Element-wise math ops | 36 |
| `test_core_sparse.cpp` | Sparse container lifecycle | 14 |
| `test_compute_dgemm.cpp` | Dense GEMM | 1 |
| `test_compute_vec.cpp` | Vector element-wise | 2 |
| `test_compute_vec_blas.cpp` | Vector BLAS (axpy, dot, nrm2, etc.) | 23 |
| `test_compute_dense.cpp` | Dense BLAS (gemm, trsm, syrk, etc.) | 41 |
| `test_compute_sparse.cpp` | Sparse BLAS (spmv, spmm, spadd, etc.) | 35 |
| `test_solve.cpp` | Dense + sparse solve | 20 |
| `test_sparse_conversion.cpp` | COO↔CSR/CSC conversions | 9 |
| `test_extended_numeric.cpp` | Eigen, QR, det, cond, expm | 16 |
| `test_print_mtx.cpp` | MTX print | 6 |
| `test_read_mtx.cpp` | MTX read | 6 |
| `test_write_mtx.cpp` | MTX write | 6 |
| `test_csv.cpp` | CSV read/write | 2 |

## Sparse Solve Backends

Six backends, selected via `matx_sparse_linsolve_backend_kind_t`:

| Enum Value | Backend | Method | Source |
|-----------|---------|--------|--------|
| `SUITESPARSE_KLU` (0) | KLU | LU | SuiteSparse (FetchContent) |
| `UMFPACK` (1) | UMFPACK | LU | SuiteSparse (FetchContent) |
| `CXSPARSE` (2) | CXSparse | LU | SuiteSparse (FetchContent) |
| `SUPERLU` (3) | SuperLU | LU | Auto-fetched |
| `MUMPS` (4) | MUMPS | LU (distributed) | Auto-fetched |
| *(pending)* | AGMG | Multigrid | Manual download (in progress — bridge not yet wired) |

Cholesky factorization is available for SPD/HPD matrices where the backend supports it.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) on push/PR to all branches: checkout → install deps → configure (Release) → build → run `matx_tests`.

## Header Organization

There is **no umbrella `matx.h` header**. Each module exports its own headers and must be included separately:

```c
#include "matx/matx_types.h"          // types, status codes, enums
#include "matx/matx_func.h"           // core: allocators, containers, element-wise math
#include "matx/matx_vec_compute.h"    // vector BLAS Level 1
#include "matx/matx_dense_compute.h"  // dense BLAS Level 2/3 + norms + element-wise
#include "matx/matx_sparse_compute.h" // sparse COO ops (spmv, spmm, spadd, norms, etc.)
#include "matx/matx_dense_solve.h"    // dense LU/Cholesky/LDL/QR/eigen/SVD/det/cond
#include "matx/matx_sparse_solve.h"   // sparse direct solvers (KLU/UMFPACK/SuperLU/MUMPS)
#include "matx/matx_write.h"          // MTX + CSV write
#include "matx/matx_read.h"           // MTX + CSV read
#include "matx/matx_print.h"          // pretty-print (C++ references — C++ only)
#include "matx/matx_log.h"            // spdlog-based logging
#include "matx/matx_tm.h"             // high-resolution timing
```

Only `matx_types_internal.h` is private (concrete structs behind opaque handles).

## Known Constraints &amp; Gotchas

- Several solve/compute paths assume or optimize for column-major layout
- Sparse solve relies on COO→CSC conversion cached inside COO handle fields
- Certain complex sparse solve functions are API placeholders depending on backend availability
- Benchmarks require `MATX_ENABLE_GRAPHBLAS=ON`
- AGMG uses a separate LP64 BLAS bridge to avoid symbol conflicts with MatX's ILP64 OpenBLAS
- The custom test harness does not support `--gtest_filter`; all tests always run
- COO↔CSC conversion helpers (`coo_to_csc_*`) lack the `matx_` prefix, unlike all other public API
- `matx_print_*` functions take C++ references — the print header is C++-only despite `extern "C"` wrappers
- `expm_dense_z_i8` exists in the dense backend vtable but has no public wrapper function
- `matx_gemm_d_i8` is declared twice in `matx_dense_compute.h` (duplicate declaration)
- SuiteSparse builds use ILP64 OpenBLAS; SuperLU auto-disables on OPENBLAS backend due to LP64/ILP64 ABI conflict

## License

BSD 3-Clause. See [LICENSE](LICENSE).