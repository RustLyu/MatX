# Optional iterative sparse solvers

MatX keeps its existing direct sparse solvers and exposes optional iterative
backends through `matx/matx_sparse_iterative.h`. When enabled, CMake obtains and
builds pinned upstream sources: Ginkgo v1.11.0 and hypre v2.33.0 are configured
as CMake subprojects, while PETSc v3.26.0 uses its documented `configure` and
`make install` flow through CMake `ExternalProject`. The source checkouts,
build trees, and install artifacts remain inside the MatX build directory.
The upstream projects are open source: [Ginkgo (BSD 3-Clause)](https://github.com/ginkgo-project/ginkgo),
[PETSc (2-Clause BSD)](https://gitlab.com/petsc/petsc/-/blob/main/doc/install/license.md), and
[hypre (MIT or Apache-2.0)](https://github.com/hypre-space/hypre#license).

## Build options

Enable one or more backends; CMake will build their open-source dependencies
from the pinned upstream tags:

```sh
-DMATX_ENABLE_GINKGO=ON
-DMATX_ENABLE_PETSC=ON
-DMATX_ENABLE_HYPRE=ON
```

Then build and install through the normal CMake targets:

```sh
cmake --build build-iterative -j
cmake --install build-iterative
```

The default install prefix is `build-iterative/install`. To use an existing
package instead, disable its source option (for example,
`-DMATX_BUILD_GINKGO_FROM_SOURCE=OFF`) and provide its CMake package or
pkg-config metadata. Ginkgo enables reference and OpenMP CPU executors; CUDA,
HIP, and SYCL are disabled. MatX builds the source version of Ginkgo as shared
CPU libraries without LTO to keep the installed package and downstream link
step manageable. The standalone hypre build is serial with OpenMP,
which matches MatX's current local COO adapter. PETSc is serial, real double
precision, and shared; its configure step downloads the open-source f2c
BLAS/LAPACK implementation. Building PETSc with `MATX_ENABLE_PETSC_HYPRE=ON`
also asks PETSc's configure script to build its hypre support.

An installed package exports `MatX::matx_sparse_solve` and installs the
source-built dependencies beside it. A downstream CMake project can consume
the iterative API like this:

```cmake
find_package(MatX CONFIG REQUIRED)
target_link_libraries(my_solver PRIVATE MatX::matx_sparse_solve)
```

Configure that project with
`-DCMAKE_PREFIX_PATH=/path/to/build-iterative/install`, then include
`<matx/matx_sparse_iterative.h>` in the source file.

`MATX_ENABLE_PETSC_HYPRE=ON` enables PETSc's `PCHYPRE`/BoomerAMG
preconditioner and includes hypre in a source-built PETSc. With an installed
PETSc package, PETSc must already have hypre support. This option does not
require the standalone hypre backend; use `MATX_ENABLE_HYPRE=ON` separately to
enable that adapter.

All iterative options default to OFF, so an existing MatX build gains no new
third-party requirement. The direct sparse solve APIs remain available.

## API flow

1. Start with `matx_sparse_iterative_options_default()` and select a backend,
   Krylov method, preconditioner, tolerance, and iteration limit.
2. Create a solver and call `matx_sparse_iterative_setup_d_i8` or
   `matx_sparse_iterative_setup_z_i8` once for a square COO matrix.
3. Call the matching `matx_sparse_iterative_solve_*` for each right-hand side.
   The matrix is snapshotted at setup, and the handle can be reused for more
   right-hand sides.
4. Check `result.reason`. `MATX_OK` means the backend call completed; it does
   not mean the target residual was reached. MatX computes `||A*x-b||_2` from
   the original COO snapshot with a scaled norm and compares it with
   `max(atol, rtol*||b||_2)`.
5. Destroy the handle with `matx_sparse_iterative_solver_destroy`.

## Current adapter coverage

| Backend | Methods | Preconditioners | Precision and execution |
| --- | --- | --- | --- |
| Ginkgo | CG, BiCGSTAB, GMRES, MINRES | none, Jacobi, ILU | real/complex double; reference or OMP CPU executor |
| PETSc KSP | CG, BiCGSTAB, GMRES, MINRES | none, Jacobi, ILU; AMG when enabled | real double; complex double only with complex PETSc |
| standalone hypre | CG, BiCGSTAB, GMRES | none, diagonal scaling, BoomerAMG | real double; local single-process matrix |

Unsupported combinations return `MATX_ERR_NOT_SUPPORTED` during setup. CG
requires a matrix and preconditioner compatible with a Hermitian positive
definite system; MatX does not infer or verify those mathematical properties.
MINRES likewise requires a symmetric/Hermitian matrix. Iterative methods can
fail to converge, so applications should inspect the reported residual and
retain a direct-solver fallback for cases that require a solution regardless
of conditioning.

There is no universally fastest, most stable choice for every sparse matrix.
For a verified SPD elliptic system, start with CG and BoomerAMG (PETSc+hypre or
standalone hypre). For a general nonsymmetric system, start with GMRES and ILU
(Ginkgo or PETSc), then compare solve time and the reported true residual on
representative matrices. The default GMRES+Jacobi configuration is a portable
starting point, not a performance claim.

PETSc currently uses `PETSC_COMM_SELF`, and standalone hypre uses
`MPI_COMM_SELF`. MatX's present COO API has no distributed row partition, so
these adapters do not distribute a matrix across MPI ranks. PETSc and hypre
initialize their process-global runtimes on first use. Destroy each solver
handle, then call `matx_sparse_iterative_runtime_finalize()` once during
process shutdown. It finalizes only runtimes started by MatX; runtimes started
by the application remain the application's responsibility. Creating another
iterative solver after this call returns `MATX_ERR_NOT_SUPPORTED`. With hypre
older than 2.29, its public API does not support this runtime finalization call,
so MatX returns `MATX_ERR_NOT_SUPPORTED` and leaves the process runtime active.
