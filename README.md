## MatX

MatX is a modular linear-algebra & numerical-compute library written in C/C++.
It is CMake-based and targets Windows/Linux. The project supports dense BLAS/LAPACK
backends (OpenBLAS / AOCL BLIS+libFLAME / MKL) and SuiteSparse backends for sparse
operations (KLU and SuiteSparse:GraphBLAS).

### Repository layout

- `tools`: logging, timers, shared utilities.
- `core`: core types (dense/sparse matrices, vectors, allocators, error codes).
- `dense_blas`: dense compute APIs and backend adapters.
- `sparse_blas`: sparse compute APIs and backend adapters (GraphBLAS / MKL).
- `dense_solve`: dense linear solves (LU factor/solve via LAPACK).
- `sparse_solve`: sparse linear solves (KLU factor/solve, real+complex).
- `tests`: unit tests using GoogleTest.
- `3party/`: vendored third-party packages used by top-level CMake.

### Backends & auto-selection

Top-level CMake performs a **CPU vendor detection** and (unless overridden) selects:

- **AMD**: AOCL **BLIS** + **libFLAME** (+ GraphBLAS enabled)
- **Intel**: **OpenBLAS** (+ GraphBLAS enabled)
- **Other/unknown**: OpenBLAS (GraphBLAS disabled)

You can override this by toggling the backend options below.

### CMake options (high level)

- **Tests**: `-DMATX_BUILD_TESTS=ON|OFF` (default: ON)
- **Dense BLAS/LAPACK**:
  - `-DMATX_ENABLE_OPENBLAS=ON|OFF`
  - `-DMATX_ENABLE_BLIS=ON|OFF`
  - `-DMATX_ENABLE_LIBFLAME=ON|OFF`
  - `-DMATX_ENABLE_MKL=ON|OFF`
  - `-DMATX_DENSE_BLAS_BACKEND=AUTO|OPENBLAS|BLIS|REFERENCE`
- **Sparse**:
  - `-DMATX_ENABLE_SUITESPARSE=ON|OFF`
  - `-DMATX_ENABLE_GRAPHBLAS=ON|OFF`
  - `-DMATX_SPARSE_BLAS_BACKEND=AUTO|OPENBLAS|BLIS|REFERENCE` (backend selector for sparse compute module)

### Build

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
ctest --test-dir build -C Release --output-on-failure
```

### Notes

- MatX currently assumes **column-major** for some solver paths (notably dense LU factor/solve). See module headers for exact constraints.
- Sparse COO inputs are lazily converted to CSC and cached in `matx_coo_*::handle_csc` for SuiteSparse KLU, to avoid repeated conversions when re-solving.

