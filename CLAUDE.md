# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatX is a modular linear algebra and numerical compute library written in **C17** (core) with **C++20** build support (tools/io/tests). It uses CMake 3.20+ and targets Windows/Linux.

## Build Commands

```bash
# Linux
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Windows (PowerShell, VS x64)
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build --config Release -- /m
ctest --test-dir build -C Release --output-on-failure

# Single test run: use ctest with -R filter or run the matx_tests executable directly from build dir
```

### Key CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `-DMATX_BACKEND` | `OPENBLAS` | `OPENBLAS` / `AMD_AOCL` / `AUTO` — selects dense BLAS backend |
| `-DMATX_BUILD_TESTS` | `ON` | Build GoogleTest-based tests |
| `-DMATX_ENABLE_SUITESPARSE` | `ON` | SuiteSparse sparse solvers |
| `-DBUILD_SHARED_LIBS` | `OFF` | Static libs by default |

All third-party dependencies are fetched via CMake **FetchContent** at configure time — no submodules, no vendored code in-tree.

## Architecture

### Opaque Handle + Vtable Dispatch Pattern

All public types are **opaque pointers** (`typedef struct *_opaque_t*`). Concrete structs live in `matx_types_internal.h`. This ensures ABI stability and allows clean separation between public API and internals.

Each compute module (vec_blas, dense_blas, sparse_blas) uses a **vtable dispatch** pattern:
- A `*_backend_kind_t` enum (`REFERENCE`, `OPENBLAS`, `BLIS`, etc.)
- A vtable struct of function pointers for every operation (separate `_d_i8` and `_z_i8` variants)
- Factory: `matx_*_default()` auto-selects; `matx_*_by_type(kind)` for explicit choice
- Public API functions take a `const matx_*_backend_t*` and dispatch through the vtable

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

### Numeric Types

Every operation has **two precision variants**: `_d_i8` (double real, 8 bytes) and `_z_i8` (double complex, 16 bytes). Complex type is defined as an aligned `{double real, imag}` struct in `matx_types.h`, not `double _Complex`.

### Memory Layout

Column-major is the default/optimized layout for solve and most compute paths. The `MATX_COL_MAJOR` / `MATX_ROW_MAJOR` enum is set at object creation time.

### Allocator System

`matx_alloc_t` in `matx_types.h` provides custom malloc/free with user-data pointer. All object creation accepts an optional allocator for user-controlled memory management.

## Code Conventions

- **Commit messages**: lowercase, terse English sentence fragments (no conventional commit prefixes)
- **Branch naming**: lowercase with underscores (e.g., `del_3party`)
- **Core is pure C**, tools/io are C++ (for spdlog integration), tests are C++ with GTest
- Public API functions follow `matx_<module>_<op>_<precision>` naming (e.g., `matx_gemm_d_i8`)
- Header-only public API: all headers under `include/matx/`

## Known Constraints

- Some solve/compute paths assume column-major layout
- Sparse solve relies on COO→CSC conversion cached inside COO handle fields
- Certain complex sparse solve functions are API placeholders depending on backend availability
- CI test execution step is currently commented out (only configure + build runs)
