## MatX

`MatX` is a C-based linear algebra library split into 4 subprojects:

- **core**: basic data types (dense matrices, allocators, error codes)
- **compute**: numerical compute on core types (BLAS backend: reference / OpenBLAS / BLIS)
- **solve**: linear solves on core types (temporarily via SuiteSparse; extensible backend interface)
- **tests**: unit tests using GoogleTest

### Build (Windows / Linux)

# MatX

Overview
--------
MatX is a modular linear-algebra and numerical-compute library written in C/C++.
It splits functionality into focused subprojects and supports multiple dense
and sparse backends (OpenBLAS, MKL, BLIS, SuiteSparse, etc.). The build system
is CMake-based and lets you select or auto-detect available backends.

Repository layout
-----------------
- `core` — core data types and utilities (dense matrices, allocators, error codes).
- `dense_blas` — dense BLAS interfaces and backend adapters (OpenBLAS / BLIS / MKL / reference).
- `sparse_blas` — sparse matrix numeric interfaces and backends.
- `dense_solve`, `sparse_solve` — solvers that plug into the numeric backends.
- `tests` — unit tests using GoogleTest.
- `3party/` — local copies of third-party libraries (this repository includes vendors such as OpenBLAS, MKL, SuiteSparse, AOCL, etc.).

Key features
------------
- Modular design for easy backend replacement and extension.
- Support for multiple dense and sparse backends with CMake feature flags.
- Cross-platform (Windows / Linux). Uses C17 and C++20 language standards.

Dependencies & CMake options
----------------------------
- Required: `CMake >= 3.20`.
- Optional/Pluggable backends: OpenBLAS, MKL, BLIS, LIBFLAME, SuiteSparse, GraphBLAS, etc.
- Common CMake options:
  - `-DMATX_BUILD_TESTS=ON|OFF` — build unit tests (default: ON).
  - `-DMATX_ENABLE_OPENBLAS=ON|OFF`, `-DMATX_ENABLE_MKL=ON|OFF`, `-DMATX_ENABLE_SUITESPARSE=ON|OFF` — enable/disable specific backends.
  - `-DMATX_DENSE_BLAS_BACKEND=AUTO|OPENBLAS|BLIS|REFERENCE`
  - `-DMATX_SPARSE_BLAS_BACKEND=AUTO|OPENBLAS|BLIS|REFERENCE`

Quick build examples
--------------------
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

Notes
-----
- The project includes several third-party libraries under `3party/` (for example `openblas-0.3.30/`, `mkl-2025.3.0/`, `suitesparse-v7.11.0/`). The top-level CMakeLists attempts to find and use these local packages when available.
- If a high-performance backend is not found, MatX falls back to reference implementations for core dense operations. Some sparse-solver features may return a "not supported" error when SuiteSparse or other sparse backends are not available.

Contributing & testing
----------------------
- Please run the test suite locally before submitting changes. Use the `cmake` and `ctest` commands shown above.

License
-------
- If a `LICENSE` file exists in this repository, consult it for licensing terms. If not, contact the project maintainers to clarify project licensing.

Changed file
------------
- Updated: [README.md](README.md)

If you want, I can add API usage examples, a quick-start tutorial, or CI steps next.

